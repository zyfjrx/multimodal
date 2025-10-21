from PIL import Image
import torch
from transformers import BertTokenizer, ChineseCLIPModel, ChineseCLIPProcessor
from model import ClipCaptionModel
import torch.nn.functional as F
from config import LLM_PATH, CLIP_MODEL_PATH, IMAGE_TOKEN_LENGTH, LLM_WORD_EMBD_DIM, device, MAX_LENGTH


def generate(model, image_embeds, tokenizer):
    """
    :param image_embeds: [B, IMAGE_EMBD_DIM=512]
    """

    b_size = image_embeds.size(0)
    pad_id = tokenizer.pad_token_id
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id
    temperature = 0.7

    cur_len = 0
    caption_ids = []    # 存储生成的caption

    # gpt2模型的输入: inputs_embeds:[B, 图片token的数量为10, gpt2的词嵌入维度768]
    # 先将图片特征投影为10个图片token
    inputs_embeds = model.projection(
        image_embeds
    ).view(-1, IMAGE_TOKEN_LENGTH, LLM_WORD_EMBD_DIM)
    finish_flag = [False] * b_size  # 第i个输入是否完成生成的标志

    while True:
        out = model.gpt2(inputs_embeds=inputs_embeds)
        logits = out.logits  # [B, len, vocab_size]
        # 采样下一个token
        next_token_logits = logits[:, -1, :]    # 取最后一个单词的预测分布
        next_token_logits = next_token_logits / temperature
        next_token_logits[:, unk_id] = -float('Inf')   # 将unk设为无穷小

        # 采样下一个token，多项分布
        next_token_ids = torch.multinomial(
            F.softmax(next_token_logits, dim=-1),
            num_samples=1
        ).squeeze(1).tolist()

        # 分别判断生成图片是否已生成完毕
        # index表示第index张正在生成文本的图片
        for index in range(len(next_token_ids)):
            token_id = next_token_ids[index]
            # 如果第i个句子已经生成结束
            if finish_flag[index]:
                next_token_ids[index] = pad_id
            # 如果第i个句子生成结束，预测到了分隔符
            elif token_id == sep_id:
                finish_flag[index] = True
            # 生成刚开始
            elif cur_len == 0:
                caption_ids.append([token_id])
            else:
                caption_ids[index].append(token_id)
        next_token_ids = torch.tensor(next_token_ids).to(device)
        next_token_embeds = model.gpt2.transformer.wte(
            next_token_ids).to(device).unsqueeze(1)
        # 将生成的next token拼接到上文的后面，继续生成
        inputs_embeds = torch.cat((inputs_embeds, next_token_embeds), dim=1)

        cur_len += 1 # 生成长度+1
        # 如果生成长度大于最大长度，或者所有图片的生成文本都结束了，退出生成过程。
        if cur_len > MAX_LENGTH or False not in finish_flag:
            break

    # 对token_id进行解码
    captions = []
    for caption_id in caption_ids:
        caption = tokenizer.convert_ids_to_tokens(caption_id)
        caption = ''.join(caption)
        captions.append(caption)

    return captions


def main():
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(LLM_PATH)
    # 初始化模型
    model = ClipCaptionModel().to(device)
    # 加载权重
    model.load_state_dict(torch.load(
        "model.pt",
        map_location=device
    ), False)
    model.eval()

    # 加载clip模型
    clip_model = ChineseCLIPModel.from_pretrained(CLIP_MODEL_PATH)
    processor = ChineseCLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    inputs_1 = processor(images=Image.open("1.jpg"), return_tensors="pt")
    inputs_2 = processor(images=Image.open("2.jpg"), return_tensors="pt")
    image_1_features = clip_model.get_image_features(**inputs_1)
    image_2_features = clip_model.get_image_features(**inputs_2)
    image_1_features = image_1_features / \
        image_1_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    image_2_features = image_2_features / \
        image_2_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    # 将两张图片的特征打包成一个批次数据
    data = torch.stack([
        image_1_features,
        image_2_features
    ], dim=0).to(device)
    captions = generate(model, data, tokenizer)
    print(captions)
    captions = generate(model, data, tokenizer)
    print(captions)
    captions = generate(model, data, tokenizer)
    print(captions)
    captions = generate(model, data, tokenizer)
    print(captions)
    captions = generate(model, data, tokenizer)
    print(captions)


if __name__ == '__main__':
    main()