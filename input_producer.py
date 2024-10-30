# input_producer.py

import pika
import sys
import os
import base64
import pickle
from config import INPUT_QUEUE

def send_images(image_paths):
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=INPUT_QUEUE)

    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        image_id = os.path.basename(image_path)
        message = {'image_id': image_id, 'image_data': image_data}

        channel.basic_publish(
            exchange='',
            routing_key=INPUT_QUEUE,
            body=pickle.dumps(message)
        )
        print(f"已发送 {image_id} 到输入队列")

    connection.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("请提供要发送的图像路径，例如：python input_producer.py image1.png image2.jpg")
    else:
        send_images(sys.argv[1:])