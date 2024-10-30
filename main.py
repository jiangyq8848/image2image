# main.py

import multiprocessing
import redis
from vae_encode_process import vae_encode_process
from unet_denoise_process import unet_denoise_process
from vae_decode_process import vae_decode_process
from config import REDIS_HOST, REDIS_PORT, PROMPT_KEY
import sys

def set_global_prompt(prompt):
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    r.set(PROMPT_KEY, prompt)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("请提供prompt，例如：python main.py 'a beautiful scenery'")
        sys.exit(1)

    prompt = sys.argv[1]
    set_global_prompt(prompt)

    # 启动三个进程
    p1 = multiprocessing.Process(target=vae_encode_process)
    p2 = multiprocessing.Process(target=unet_denoise_process)
    p3 = multiprocessing.Process(target=vae_decode_process)

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()