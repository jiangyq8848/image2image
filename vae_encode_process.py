# vae_encode_process.py

import pika
import redis
import torch
from diffusers import AutoencoderKL
from PIL import Image
import io
import base64
import pickle
from config import *

def vae_encode_process():
    # 建立RabbitMQ连接
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=INPUT_QUEUE)
    channel.queue_declare(queue=ENCODE_TO_DENOISE_QUEUE)

    # 建立Redis连接
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    prompt = r.get(PROMPT_KEY).decode('utf-8')

    # 加载VAE模型
    vae = AutoencoderKL.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="vae")
    vae.eval()

    def callback(ch, method, properties, body):
        # 从消息中获取图像数据
        message = pickle.loads(body)
        image_id = message['image_id']
        image_data = message['image_data']

        # 将图像数据转换为Tensor
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        image = image.resize((512, 512))
        image_tensor = torch.tensor([list(image.getdata(band=i)) for i in range(3)], dtype=torch.float32) / 255.0
        image_tensor = image_tensor.unsqueeze(0)

        # VAE编码
        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.sample()

        # 添加噪声
        noise = torch.randn_like(latent) * NOISE_STRENGTH
        noisy_latent = latent + noise

        # 发送到下一个队列
        new_message = {'image_id': image_id, 'latent': noisy_latent}
        channel.basic_publish(
            exchange='',
            routing_key=ENCODE_TO_DENOISE_QUEUE,
            body=pickle.dumps(new_message)
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=callback)
    print("VAE Encode 加噪进程已启动，等待消息...")
    channel.start_consuming()