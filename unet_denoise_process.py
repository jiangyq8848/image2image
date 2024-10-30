# unet_denoise_process.py

import pika
import pickle
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import redis
from config import *

def unet_denoise_process():
    # 建立 RabbitMQ 连接
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=ENCODE_TO_DENOISE_QUEUE)
    channel.queue_declare(queue=DENOISE_TO_DECODE_QUEUE)

    # 建立 Redis 连接
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    prompt = r.get(PROMPT_KEY).decode('utf-8')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 UNet 模型
    unet = UNet2DConditionModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="unet").to(device)
    unet.eval()

    # 加载文本编码器
    tokenizer = CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="text_encoder").to(device)
    text_encoder.eval()

    # 对 prompt 进行编码
    text_inputs = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")

    # 将 inputs 移动到设备上（CPU 或 GPU）
    text_inputs = {key: val.to(device) for key, val in text_inputs.items()}

    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs['input_ids'])[0]

    # 初始化调度器
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(DENOISE_STEPS)

    def callback(ch, method, properties, body):
        # 从消息中获取 latent
        message = pickle.loads(body)
        image_id = message['image_id']
        latent = message['latent']

        # 确保 latent 数据类型和设备正确
        latent = latent.to(device).to(torch.float32)

        # 进行去噪
        for i, timestep in enumerate(scheduler.timesteps):
            timestep = timestep.to(device).long()
            with torch.no_grad():
                # 调用 UNet 获取预测的噪声
                noise_pred = unet(latent, timestep, encoder_hidden_states=text_embeddings).sample
                # 使用调度器更新 latent
                latent = scheduler.step(noise_pred, timestep, latent).prev_sample

        # 将 latent 移动到 CPU，以便序列化
        latent = latent.to('cpu')

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