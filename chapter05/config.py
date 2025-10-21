import torch

CLIP_MODEL_PATH = "/Users/zhangyf/llm/chinese-clip-vit-base-patch16"
# 一张图片的嵌入经过投影转换成10个token的embedding，每个embedding的dim是768
IMAGE_TOKEN_LENGTH = 10 # 图片的token的数量
MAX_LENGTH = 100 # 最大token数量
# clip对接的大语言模型
LLM_PATH = "/Users/zhangyf/llm/gpt2-chinese-cluecorpussmall"
LLM_WORD_EMBD_DIM = 768 # gpt2的词嵌入维度
IMAGE_EMBD_DIM = 512 # clip输出的图像嵌入的维度
device = torch.device("mps")