# output_consumer.py

import pika
import pickle
import base64
import os
from config import OUTPUT_QUEUE

def receive_images(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=OUTPUT_QUEUE)

    def callback(ch, method, properties, body):
        message = pickle.loads(body)
        image_id = message['image_id']
        image_data = message['image_data']

        # 保存图像
        image_bytes = base64.b64decode(image_data)
        output_path = os.path.join(output_folder, f"output_{image_id}")
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        print(f"已保存输出图像：{output_path}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=OUTPUT_QUEUE, on_message_callback=callback)
    print("输出结果处理进程已启动，等待输出图像...")
    channel.start_consuming()

if __name__ == '__main__':
    output_folder = '/Users/gaoty/Desktop/纯色背景商品图/rr'
    receive_images(output_folder)