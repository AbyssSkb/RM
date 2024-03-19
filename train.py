import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from MyDataset import MyDataset
from ShuffleNetV2 import ShuffleNetV2

LR = 0.01
Epoch = 20
Batch_size = 64
GPU_Available = torch.cuda.is_available()

data_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]
)

ds_train = MyDataset(image_folder='train', transform=data_transforms)
ds_valid = MyDataset(image_folder='valid', transform=data_transforms)

train_loader = DataLoader(ds_train, batch_size=Batch_size, shuffle=True)
valid_loader = DataLoader(ds_valid, batch_size=Batch_size)

def main():
    model = ShuffleNetV2(n_class=12, input_size=100, width_mult=0.5)
    # model.load_state_dict(torch.load("model.pt"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 是否使用 GPU
    if GPU_Available:
        model = model.cuda()
        criterion = criterion.cuda()

    for i in range(Epoch):
        print('Epoch: %d'%i)

        # 训练
        train_loss = 0
        train_accuracy = 0
        model.train()

        print('Training...')
        for data in tqdm(train_loader):
            input, label, _ = data

            # 是否使用 GPU
            if GPU_Available:
                input = input.cuda()
                label = label.cuda()
            
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * input.shape[0]
            predict = torch.argmax(output, dim=1)
            train_accuracy += torch.sum(predict == label)

        # 验证
        valid_loss = 0
        valid_accuracy = 0
        model.eval()

        print('Validating...')
        with torch.no_grad():
            for data in tqdm(valid_loader):
                input, label, _ = data
                if GPU_Available:
                    input = input.cuda()
                    label = label.cuda()
                
                output = model(input)
                loss = criterion(output, label)
                valid_loss += loss.item() * input.shape[0]
                predict = torch.argmax(output, dim=1)
                valid_accuracy += torch.sum(predict == label)
                
        # 打印一些基本信息
        print('Train Loss: %.1e' %(train_loss / ds_train.__len__()), 'Train Accuracy: %.3f' %(train_accuracy / ds_train.__len__()))
        print('Valid Loss: %.1e' %(valid_loss / ds_valid.__len__()), 'Valid Accuracy: %.3f' %(valid_accuracy / ds_valid.__len__()))
        print('')

    # 保存
    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    main()