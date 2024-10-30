# config.py

RABBITMQ_HOST = 'localhost'
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

PROMPT_KEY = 'global_prompt'

INPUT_QUEUE = 'input_queue'
ENCODE_TO_DENOISE_QUEUE = 'encode_to_denoise_queue'
DENOISE_TO_DECODE_QUEUE = 'denoise_to_decode_queue'
OUTPUT_QUEUE = 'output_queue'

NOISE_STRENGTH = 0.5  # 加噪强度
DENOISE_STEPS = 2     # 去噪步数