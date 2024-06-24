import torch
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn.functional as F

def compute_differences(m1, m2):
    groups = {"mlp.c_fc.weight": defaultdict(lambda: 0.0), "attn.c_attn.weight": defaultdict(lambda: 0.0)}
    for k, v1 in m1["model"].items():
        for cat in groups.keys():
            if cat in k:
                diff = torch.flatten(v1 - m2["model"][k])
                #groups[cat]["l1"] += torch.linalg.norm(diff, dim=None, ord=1).item()
                groups[cat]["l2"] += torch.linalg.norm(diff, dim=None, ord=2).item()
                #groups[cat]["cosine"] += F.cosine_similarity(v1.flatten(), m2["model"][k].flatten(), dim=-1).item()
    return groups

def gradnorm(m):
    x = 0
    for key, state in m["optimizer"]["state"].items():
        #x += torch.linalg.norm(state["exp_avg"], dim=None, ord=2).item()
        x += torch.mean(state["exp_avg_sq"]).item()
    return x

def flatten(xs, out=None, prefix=""):
    if out is None: out = {}
    for k, v in xs.items():
        longk = (prefix + " " + k).strip()
        if isinstance(v, dict): flatten(v, out, longk)
        else:
            out[longk] = v
    return out

xs = []
ys = defaultdict(list)
for step in range(500, 3500, 500):
    file = f"ckpt{step}.pt"
    m_baseline = torch.load(Path("fixed-seed1") / file)
    m_sameseed = torch.load(Path("fixed-seed1-1") / file)
    m_sameseed2 = torch.load(Path("fixed-seed1-2") / file)
    m_other = torch.load(Path("fixed-seed2") / file)
    m_baseline_resumed = torch.load(Path("fixed-seed1-res1500") / file)
    xs.append(step)

    """
    comparisons = {
        "same seed": compute_differences(m_baseline, m_sameseed),
        "same seed 2": compute_differences(m_baseline, m_sameseed2),
        "same seed resume at 1500": compute_differences(m_baseline, m_baseline_resumed),
        "other seed": compute_differences(m_baseline, m_other),
    }
    """
    comparisons = {
        "baseline": gradnorm(m_baseline),
        "same seed": gradnorm(m_sameseed),
        "same seed 2": gradnorm(m_sameseed2),
        "other seed": gradnorm(m_other)
    }
    
    for k, v in flatten(comparisons).items():
        ys[k].append(v)

plt.figure(figsize=(12, 10))
plt.xlabel("step")
plt.ylabel("gradnorm")
for k, v in ys.items():
    plt.plot(xs, v, label=k)
plt.legend()
plt.savefig("x.png")