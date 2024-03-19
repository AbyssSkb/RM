# 小车装甲板识别（分类）

## 各个文件的功能
`MyDataset.py`：自定义数据集

`ShuffleNetV2.py`：在 `ericsun99/Shufflenet-v2-Pytorch` [源码](https://github.com/ericsun99/Shufflenet-v2-Pytorch/blob/master/ShuffleNetV2.py)的基础上改小了模型参数量

`train.py`：模型训练以及验证

`test.py`：模型测试

`best.pt`：在验证集上 `Accuracy` 最高的模型参数

`latest.pt`：最新训练完后的模型参数