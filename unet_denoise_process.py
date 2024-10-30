# unet_denoise_process.py

import pika
import pickle
import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import redis
from config import *

def unet_denoise_process():
    # 建立RabbitMQ连接
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=ENCODE_TO_DENOISE_QUEUE)
    channel.queue_declare(queue=DENOISE_TO_DECODE_QUEUE)

    # 建立Redis连接
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    prompt = r.get(PROMPT_KEY).decode('utf-8')

    # 加载UNet模型
    unet = UNet2DConditionModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="unet")
    unet.eval()

    # 加载文本编码器
    tokenizer = CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="text_encoder")
    text_encoder.eval()

    # 对prompt进行编码
    text_inputs = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids)[0]

    def callback(ch, method, properties, body):
        # 从消息中获取latent
        message = pickle.loads(body)
        image_id = message['image_id']
        latent = message['latent']

        # 进行DENOISE_STEPS步去噪
        for _ in range(DENOISE_STEPS):
            with torch.no_grad():
                latent = unet(latent, encoder_hidden_states=text_embeddings).sample

        # 发送到下一个队列
        new_message = {'image_id': image_id, 'latent': latent}
        channel.basic_publish(
            exchange='',
            routing_key=DENOISE_TO_DECODE_QUEUE,
            body=pickle.dumps(new_message)
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=ENCODE_TO_DENOISE_QUEUE, on_message_callback=callback)
    print("UNet Denoise 进程已启动，等待消息...")
    channel.start_consuming()