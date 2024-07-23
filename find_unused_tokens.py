import numpy as np
import os
from collections import Counter
data_dir = "."
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
datas = set(data)
counts = Counter(data)
vocab = set(range(50257))
unused = vocab - datas
unused = sorted(unused)
print(len(unused))
print(unused)
print(counts.most_common(100))