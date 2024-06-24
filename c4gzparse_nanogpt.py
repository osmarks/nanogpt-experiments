import os
from tqdm import tqdm
import numpy as np
import tiktoken
import json
import gzip

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = []

    with gzip.open("c4-train.00000-of-01024.json.gz", "r") as file:
        while line := file.readline():
            try:
                dataset.append(json.loads(line))
            except EOFError:
                pass

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.insert(0, enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = [ process(x) for x in dataset ]
    divider = len(tokenized) // 100
    tokenized = {
        "val": tokenized[:divider],
        "train": tokenized[divider:]
    }

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = sum((d['len'] for d in dset))
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for d in tqdm(dset, desc=f'writing {filename}'):
            arr[idx : idx + d["len"]] = d["ids"]
            idx += d["len"]
        arr.flush()

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
