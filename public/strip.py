# strip.py
import sys, os, torch
src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
sd = {k.replace("module.", ""): v.detach().clone() for k, v in sd.items()}
torch.save(sd, dst)
print(os.path.getsize(src) // 10**6, "MB ->", os.path.getsize(dst) // 10**6, "MB")