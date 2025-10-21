from modelscope.msdatasets import MsDataset
ds = MsDataset.load("muge", namespace="modelscope", split="train",cache_dir="/Users/zhangyf/PycharmProjects/multimodal/chapter05/datasets/train")
print(ds[0])
print(ds[0:2])