from PIL import Image
import pickle
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from config import CLIP_MODEL_PATH


def main():
    # 加载clip模型
    # clip模型只用来生成图片的嵌入，不进行微调。
    clip_model = ChineseCLIPModel.from_pretrained(CLIP_MODEL_PATH)
    # 加载clip处理器
    processor = ChineseCLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    # 将2张图片进行处理，处理完之后交给clip抽取特征
    inputs_1 = processor(images=Image.open("1.jpg"), return_tensors="pt")
    inputs_2 = processor(images=Image.open("2.jpg"), return_tensors="pt")
    # 获取第一张图片的嵌入（dim: 512）
    image_1_features = clip_model.get_image_features(**inputs_1)
    image_2_features = clip_model.get_image_features(**inputs_2)
    # 除以模长，归一化
    image_1_features = image_1_features / \
        image_1_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    image_2_features = image_2_features / \
        image_2_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    # key：图片id
    # value: 图片的嵌入
    # 下面的字典也可以放在chroma这样的向量数据库
    image_id2embed = {
        1: image_1_features,
        2: image_2_features,
    }
    # 图片的id和图片标题对
    caption_list = [
        (1, "两只狗在雪地里嬉闹"),
        (1, "两条狗在雪中玩耍"),
        (1, "两只小狗在雪地追逐"),
        (1, "两只狗在白雪上打闹"),
        (1, "两条狗在雪地相互追赶"),
        (1, "两只狗在雪地里玩游戏"),
        (1, "两只小狗在雪里翻滚"),
        (1, "两只狗在积雪中奔跑"),
        (1, "两条小狗在雪地打闹嬉戏"),
        (1, "两只狗在雪面上互相追逐"),
        (1, "两只狗在雪地欢快地跑来跑去"),
        (1, "两条狗在雪中你追我赶"),
        (1, "两只小狗在白雪里闹腾"),
        (1, "两只狗在雪地做游戏"),
        (1, "两条小狗在雪地里撒欢儿"),
        (1, "两只狗在雪地上嬉戏打闹"),
        (1, "两只狗在厚雪中蹦跳"),
        (1, "两条狗在雪地里打滚玩耍"),
        (1, "两只小狗在雪地上欢闹"),
        (1, "两只狗在冰天雪地里玩闹"),
        (1, "两条狗在雪地里互相玩闹"),
        (1, "两只小狗在雪地上你追我赶"),
        (1, "两只狗在雪地里跑跳嬉戏"),
        (1, "两条小狗在雪地里追逐嬉戏"),
        (1, "两只狗在雪地里兴奋地玩耍"),
        (2, "一件好看的立领的很特别的紫色风衣"),
        (2, "一件好看的立领的不错的紫色风衣"),
        (2, "一件好看的立领的有点意思紫色风衣"),
        (2, "一件好看的立领的好玩的紫色风衣"),
        (2, "一件好看的立领的奥特曼的紫色风衣"),
        (2, "一件好看的立领的孙悟空的紫色风衣"),
        (2, "一件好看的立领的伊尔亚的紫色风衣"),
        (2, "一件好看的立领的马斯克的紫色风衣"),
        (2, "一件好看的立领的特朗普的紫色风衣"),
        (2, "一件好看的立领的丁真的紫色风衣"),
        (2, "一件好看的立领的OK的紫色风衣"),
        (2, "一件好看的立领的不太好的紫色风衣"),
        (2, "一件好看的立领的淘宝的紫色风衣"),
        (2, "一件好看的立领的拼多多的紫色风衣"),
        (2, "一件好看的立领的尚硅谷的紫色风衣"),
        (2, "一件好看的立领的左元的紫色风衣"),
        (2, "一件好看的立领的策略梯度法的紫色风衣"),
        (2, "一件好看的立领的深度学习的紫色风衣"),
        (2, "一件好看的立领的强化学习的紫色风衣"),
        (2, "一件好看的立领的多模态的紫色风衣"),
        (2, "一件好看的立领的扩散模型的紫色风衣"),
        (2, "一件好看的立领的VAE的紫色风衣"),
        (2, "一件好看的立领的变分下界的紫色风衣"),
    ]

    with open("caption_image.pkl", 'wb') as f:
        pickle.dump([caption_list, image_id2embed], f)

    print(f'图像嵌入的数量:{len(image_id2embed)}')
    print(f'图像文本的数量:{len(caption_list)}')


if __name__ == '__main__':
    main()
