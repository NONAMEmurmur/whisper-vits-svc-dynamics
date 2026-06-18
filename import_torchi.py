import torch
from collections import defaultdict

ckpt_path = "vits_pretrain/sovits5.0.pretrain.pth"

print("Loading checkpoint...")
ckpt = torch.load(ckpt_path, map_location="cpu")

# -------- model_g を取り出す --------
if "model_g" in ckpt:
    state_dict = ckpt["model_g"]
    print("\nUsing model_g\n")
elif "model" in ckpt:
    state_dict = ckpt["model"]
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

print(f"Total params: {len(state_dict)}\n")

# -------- 各パラメータ --------
for k, v in state_dict.items():
    shape = tuple(v.shape)
    print(f"{k:60s} {shape}")

# -------- グルーピング --------
print("\n\n=== Grouped by top module ===\n")

groups = defaultdict(list)

for k in state_dict.keys():
    top = k.split('.')[0]
    groups[top].append(k)

for g in sorted(groups.keys()):
    print(f"\n[{g}] ({len(groups[g])} params)")
    for name in groups[g][:10]:
        print(f"  {name}")
    if len(groups[g]) > 10:
        print(f"  ... ({len(groups[g]) - 10} more)")

# -------- 入力層候補 --------
print("\n\n=== Possible input layers ===\n")

for k, v in state_dict.items():
    if "weight" in k and len(v.shape) >= 2:
        print(f"{k:60s} {tuple(v.shape)}")