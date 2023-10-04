import timm
import tome
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# We use the timm augreg models here, but you can use any supported implementation.
model_name = "deit_base_patch16_224"
model = timm.create_model(model_name, pretrained=True)

# Source tracing is necessary for visualization!
tome.patch.timm(model, trace_source=True)

input_size = model.default_cfg["input_size"][1]

# Make sure the transform is correct for your model!
transform_list = [
    transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(input_size)
]

# The visualization and model need different transforms
transform_vis  = transforms.Compose(transform_list)
transform_norm = transforms.Compose(transform_list + [
    transforms.ToTensor(),
    transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
])

img = Image.open("examples/images/fox.JPEG")
img_vis = transform_vis(img)
img_norm = transform_norm(img)

img.show()

model.r = 0.00000000000000000001
_ = model(img_norm[None, ...])
source = model._tome_info["source"]

print(f"{source.shape[1]} tokens at the end")
print(f"source shape : {source.shape}")
merged_img = tome.make_visualization(img_vis, source, patch_size=16, class_token=True)
merged_img.show()
merged_img.save('/home/smh-ewha/OURPROJ/ToMato/output/output.JPEG', 'JPEG')


'''
model.r = [8] * 8  # 8 / 24 layers
_ = model(img_norm[None, ...])
source = model._tome_info["source"]

print(f"{source.shape[1]} tokens at the end")
tome.make_visualization(img_vis, source, patch_size=16, class_token=True)

model.r = [8] * 16  # 16 / 24 layers
_ = model(img_norm[None, ...])
source = model._tome_info["source"]

print(f"{source.shape[1]} tokens at the end")
tome.make_visualization(img_vis, source, patch_size=16, class_token=True)

model.r = [8] * 22  # 22 / 24 layers
_ = model(img_norm[None, ...])
source = model._tome_info["source"]

print(f"{source.shape[1]} tokens at the end")
tome.make_visualization(img_vis, source, patch_size=16, class_token=True)
'''
