import numpy as np
import os
data_dir = "."
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
datas = set(data)
vocab = set(range(50257))
unused = vocab - datas
unused = sorted(unused)
print(len(unused))
print(unused)