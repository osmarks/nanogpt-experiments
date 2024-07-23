import torch
from PIL import Image
import open_clip
import numpy as np

model_name = "ViT-SO400M-14-SigLIP-384"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="webli", precision="fp16", device="cuda")
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

print(model)

print("preprocess")
image = preprocess(Image.open("siglip.jpg")).unsqueeze(0).half().cuda()
image.requires_grad = True

print("fwd")
features = model.encode_image(image)
print("bwd")
s = features.abs().sum()
print(s.backward())
# Due to some nonsense, the model actually cuts off exactly six pixels from the right and bottom of the image.
# (6 = 384 - (14*27))
# Those can be varied arbitrarily without affecting the output, but that isn't interesting.
# B C W H, probably
real_grad = image.grad[:, :, :378, :378].abs()

x = torch.min(real_grad, dim=3)
print(x)
y = torch.min(x.values, dim=2)
print(y)
z = torch.min(y.values, dim=1)
print(z)

l_chan = z.indices[0]
l_x = y.indices[0][l_chan]
l_y = x.indices[0][l_chan][l_x]

least_affecting_index = 0, l_chan, l_x, l_y

image.requires_grad = False

print(real_grad[least_affecting_index], image[least_affecting_index])

avgmean = 0
avgmax = 0
n = 500
with torch.no_grad():
    for some_float in np.linspace(-1, 1, n):
        if -1 <= some_float <= 1:
            image[least_affecting_index] = float(some_float)
            altered_features = model.encode_image(image)
            mean_diff = (features - altered_features).abs().mean().item()
            max_diff = (features - altered_features).max().item()
            print(f"{some_float:0.3f}: {mean_diff:3f}, {max_diff:3f}")
            avgmean += mean_diff / n
            avgmax += max_diff / n

print(f"avg mean diff: {avgmean}, avg max diff: {avgmax}")