from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import tome
import timm.models

import torch.cuda.nvtx as nvtx

def test2():
    
    # Load a pretrained model, can be any vit / deit model.
    model = torch.hub.load('/home/smh-ewha/OURPROJ/SPViT/SPViT_DeiT', 'deit_base_patch16_224', pretrained=True, source='local')

    #model = timm.create_model("deit_base_patch16_224", pretrained=True)
    # Patch the model with ToMe.
    tome.patch.timm(model)
    # Set the number of tokens reduced per layer. See paper for details.
    model.r = 14

    print(model.__class__)
    model.eval()


    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
    img = transform(img)[None,]

    for i in range(11):
        with torch.no_grad():
            nvtx.range_push("Forward " + str(i))
            out = model(img)
            nvtx.range_pop()
            clsidx = torch.argmax(out)
            print(clsidx.item())


if __name__ == "__main__":
    test2()
