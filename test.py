import torch
import os
from torchvision.transforms import v2
from thop import profile
from MyDataset import MyDataset
from ShuffleNetV2 import ShuffleNetV2
from torch.utils.data import DataLoader

GPU_Available = torch.cuda.is_available()

data_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]
)

test_folder = os.path.join('dataset', 'test')
ds_test = MyDataset(image_folder=test_folder, transform=data_transforms)
test_loader = DataLoader(ds_test)

model = ShuffleNetV2(n_class=12, input_size=100, width_mult=0.5)
model.load_state_dict(torch.load('latest.pt'))
model.eval()

# 输出模型相关信息
input = torch.randn(1, 3, 100, 100)
flops, params = profile(model, inputs=(input, ))
print('MFLOPs = ' + str(flops / 1000 ** 2))
print('Params = ' + str(params / 1000 ** 2) + 'M')

# 是否使用 GPU
if GPU_Available:
    model = model.cuda()

def test():
    sum = 0
    all = 0
    
    with torch.no_grad(): 
        for data in test_loader:
            input, label, filename = data

            if GPU_Available:
                input = input.cuda()

            output = model(input)
            predict = torch.argmax(output, dim=1)
            if int(predict) != int(label):
                print('Label: %d'%int(label), 'Predict: %d'%int(predict), 'Name: %s'%filename)
                sum += 1

            all += 1

    print('incorrect: %d'%(sum))
    print('All: %d'%(all))
    print('Accuracy: %.3f'%((all - sum) / all))

if __name__ == '__main__':
    test()