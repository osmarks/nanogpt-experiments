import os
from tqdm import tqdm
import numpy as np
import tiktoken
import json
import gzip
import torch
import random

torch.set_grad_enabled(False)

device = "cuda"

def load_exllama(model_dir):
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2DynamicGenerator
    
    config = ExLlamaV2Config(model_dir)
    model = ExLlamaV2(config)
    model.load()

    tokenizer = ExLlamaV2Tokenizer(config)

    return model, tokenizer

def load_nanogpt(model_dir, ckpt):
    import os
    import pickle
    from contextlib import nullcontext
    import torch
    import tiktoken
    from model import GPTConfig, GPT

    ckpt_path = os.path.join(model_dir, ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    return model, tiktoken.get_encoding("gpt2")

#model, tokenizer = load_exllama("./Llama-3-8B-Instruct-exl2")
model, tokenizer = load_nanogpt("./data-injection-1", "ckpt3000.pt")

def find_closest_tokens(model, tokenizer):
    weights_ = model.modules[0].embedding.weight.data
    weights = torch.zeros_like(weights_, device="cuda")
    weights.copy_(weights_)
    # some are zero, so we can't normalize easily
    #weights /= torch.linalg.norm(weights, dim=-1, keepdim=True)
    vocab_size, dim = weights.shape
    print("copied")

    best = torch.zeros(vocab_size, device="cuda", dtype=torch.int32)
    scores = torch.zeros(vocab_size, device="cuda", dtype=torch.float16)

    CHUNK_SIZE = 1024
    for i in range(0, vocab_size, CHUNK_SIZE):
        print(i)
        similarities = (weights @ weights[i:i+CHUNK_SIZE, :].T)
        # zero similarity to self
        torch.diagonal(similarities, offset=i, dim1=1, dim2=0).fill_(-float("inf"))
        score, ix = torch.max(similarities, dim=0)
        best[i:i+CHUNK_SIZE] = ix
        scores[i:i+CHUNK_SIZE] = score

    scores, indices = torch.sort(scores, descending=True)

    print([ (indices[i].item(), best[indices][i].item(), tokenizer.decode(indices[i:i+1]), tokenizer.decode(best[indices][i:i+1])) for i in range(100) ])  

#find_closest_tokens()

#best_pair = 28217, 76665
#best_pair = 34966, 70467
#best_pair = 48, 57
best_pair = 49704, 50009
COUNT = 1000
for _ in range(COUNT):
    sequence = torch.randint(low=0, high=2, size=(1024,), device="cuda", dtype=torch.int32) * (best_pair[1] - best_pair[0]) + best_pair[0]
    print("---")
    for end_choice in best_pair:
        sequence[-1] = end_choice
        logits = model.forward(sequence.unsqueeze(0))
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.bfloat16() # introduce roundoff error deliberately
        print("Final 10 logits", logits[0, -10:, :])
        #print("Input", tokenizer.decode(sequence.tolist()))
        #print("Predictions", tokenizer.decode(torch.argmax(logits[0], dim=-1).tolist()))
        print("Max", torch.max(logits[0, -1], dim=-1), torch.mean(logits[0, -1], dim=-1)) 