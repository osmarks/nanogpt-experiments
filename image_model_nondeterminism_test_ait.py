import torch
from PIL import Image
import open_clip
import numpy as np

model_name = "ViT-SO400M-14-SigLIP-384"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="webli", precision="fp16", device="cuda")
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

print(model)

from aitemplate.compiler import Model as AITModel
from aitemplate.testing import detect_target

USE_CUDA = detect_target().name() == "cuda"

state = model.state_dict()
conv_weights = state["visual.trunk.patch_embed.proj.weight"].permute((0, 2, 3, 1)).contiguous().cuda().half()

def load_pretrained():
    params = {}
    for key, value in state.items():
        orig_key = key
        if key.startswith("visual."):
            key = key.removeprefix("visual.") \
                .replace("trunk.patch_embed", "patch_embed") \
                .replace("trunk.blocks", "encoder.layers") \
                .replace(".attn.", ".mha.") \
                .replace(".norm1.", ".ln1.") \
                .replace(".norm2.", ".ln2.") \
                .replace("trunk.pos_embed", "pos_emb_pos_emb") \
                .replace("trunk.norm.", "encoder.ln.") \
                .replace("trunk.attn_pool.latent", "pool.probe") \
                .replace("trunk.attn_pool", "pool") \
                .replace("pool.norm", "pool.ln")
            if "patch_embed.proj.weight" not in key:
                params[key.replace(".", "_")] = value.cuda()
                #print(orig_key, key.replace(".", "_"))

    params["patch_embed_proj_weight"] = conv_weights

    return params

def generate_wrapper(path):
    ait_model = AITModel(path)
    ait_model.set_many_constants_with_tensors(load_pretrained())
    ait_model.fold_constants(sync=True)
    def wrapper(batch):
        xs = [batch.permute((0, 2, 3, 1)).contiguous()]
        ys = []
        for i in range(len(ait_model.get_output_name_to_index_map())):
            shape = ait_model.get_output_maximum_shape(i)
            ys.append(torch.empty(shape).cuda().half())
        ait_model.run_with_tensors(xs, ys)
        return ys[0][:, 0, :]
    return wrapper

encode_image = generate_wrapper("siglip/1.so")

print("preprocess")
image = preprocess(Image.open("siglip.jpg")).unsqueeze(0).half().cuda()

print("fwd")
features = encode_image(image)

avgmean = 0
avgmax = 0
n = 500
with torch.no_grad():
    for _ in range(n):
        altered_features = encode_image(image)
        mean_diff = (features - altered_features).abs().mean().item()
        max_diff = (features - altered_features).max().item()
        print(f"{mean_diff:3f}, {max_diff:3f}")
        avgmean += mean_diff / n
        avgmax += max_diff / n

print(f"avg mean diff: {avgmean}, avg max diff: {avgmax}")