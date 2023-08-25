import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import timm
import requests
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd

import tome
import torch.cuda.nvtx as nvtx


def test2():
   # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   # print(device)

    print(torch.cuda.device_count())

    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    model = torch.hub.load('./', 'deit_base_patch16_224', pretrained=True, source='local')
    
    tome.patch.timm(model)
    # Set the number of tokens reduced per layer. See paper for details.
    model.r = 0.85

    print(model.__class__)

    
    model.eval()


    paths=[]
    labels=[]
    for dirname, _, filenames in os.walk('../imagenet-mini/val'):
        for filename in filenames:
            if filename[-4:] == 'JPEG':
                paths += [(os.path.join(dirname, filename))]
                label = dirname.split('/')[-1]
                labels += [label]            

    class_names = sorted(set(labels))
    N=list(range(len(class_names)))
    normal_mapping = dict(zip(class_names, N))

    df=  pd.DataFrame(columns=['path','label'])
    df['path'] = paths
    df['label'] = labels
    df['label'] = df['label'].map(normal_mapping)

    class CustomDataset(Dataset):
        def __init__(self, dataframe):
            self.dataframe = dataframe

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            path = self.dataframe.loc[index, 'path']
            label = self.dataframe.loc[index, 'label']
            image = Image.open(path).convert('RGB')

            transform = transforms.Compose([
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            image = transform(image)
            return image, label


    test_set = CustomDataset(df)
    test_loader = DataLoader(test_set, batch_size=1)

    #test_set = torchvision.datasets.ImageNet(root="/home/smh-ewha/imagenet-mini", transform=transform, split='val')
    #test_loader = data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4)

    total = 0
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            outputs = model(images)

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (pred == labels).sum().item()

            _, rank5 = outputs.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

            for k in range(6):
                correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

            correct_top5 += correct_k.item()

            print("step : {} / {}".format(idx + 1, len(test_set) / int (labels.size(0))))
            print("top-1 percentage : {0:0.2f}%".format(correct_top1 / total * 100))
            print("top-5 percentage : {0:0.2f}%".format(correct_top5 / total * 100))

    print("top-1 percentage : {0:0.2f}%".format(correct_top1 / total * 100))
    print("top-5 percentage : {0:0.2f}%".format(correct_top5 / total * 100))


if __name__ == "__main__":
    test2()
