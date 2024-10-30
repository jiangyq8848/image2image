# vae_decode_process.py

import pika
import pickle
import torch
from diffusers import AutoencoderKL
from PIL import Image
import io
import base64
from config import *

def vae_decode_process():
    # 建立RabbitMQ连接
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=DENOISE_TO_DECODE_QUEUE)
    channel.queue_declare(queue=OUTPUT_QUEUE)

    # 加载VAE模型
    vae = AutoencoderKL.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="vae")
    vae.eval()

    def callback(ch, method, properties, body):
        # 从消息中获取latent
        message = pickle.loads(body)
        image_id = message['image_id']
        latent = message['latent']

        # VAE解码
        with torch.no_grad():
            image_tensor = vae.decode(latent).sample

        # 转换为图像格式
        image_array = (image_tensor.squeeze().permute(1, 2, 0) * 255).clamp(0, 255).cpu().numpy().astype('uint8')
        image = Image.fromarray(image_array)

        # 将图像转换为base64字符串
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 发送到输出队列
        new_message = {'image_id': image_id, 'image_data': image_data}
        channel.basic_publish(
            exchange='',
            routing_key=OUTPUT_QUEUE,
            body=pickle.dumps(new_message)
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=DENOISE_TO_DECODE_QUEUE, on_message_callback=callback)
    print("VAE Decode 进程已启动，等待消息...")
    channel.start_consuming()