"""Undo ComfyUI int8_tensorwise convrot quantization: W = (Q * s) @ R, written as bf16 safetensors.

Streams one tensor at a time so a 40 GB output never sits in RAM.
"""
import json
import struct
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]


def hadamard(group: int) -> torch.Tensor:
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64) / 2
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < group:
        h = torch.kron(h, h4)
    assert h.shape[0] == group, group
    return h


f = open(src, "rb")
n = struct.unpack("<Q", f.read(8))[0]
header = json.loads(f.read(n))
meta = header.pop("__metadata__", {})
base = 8 + n

DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I8": torch.int8, "U8": torch.uint8}


def load(name):
    info = header[name]
    a, b = info["data_offsets"]
    f.seek(base + a)
    raw = bytearray(f.read(b - a))
    return torch.frombuffer(raw, dtype=DT[info["dtype"]]).reshape(info["shape"])


quant = {}
for name in header:
    if name.endswith(".comfy_quant"):
        cfg = json.loads(bytes(load(name).tolist()))
        quant[name[: -len(".comfy_quant")]] = cfg

# plan output: drop comfy_quant / weight_scale of quantized modules, quantized weights become bf16
plan = []
for name, info in header.items():
    module = name.rsplit(".", 1)[0]
    if module in quant and (name.endswith(".comfy_quant") or name.endswith(".weight_scale")):
        continue
    if module in quant and name.endswith(".weight"):
        plan.append((name, "BF16", info["shape"]))
    else:
        plan.append((name, info["dtype"], info["shape"]))

SIZE = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1}
out_header, offset = {}, 0
for name, dtype, shape in plan:
    nbytes = SIZE[dtype]
    for d in shape:
        nbytes *= d
    out_header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
    offset += nbytes
meta = dict(meta)
meta["unconvrot"] = "int8 convrot dequantized and unrotated to bf16 for GGUF requantization"
out_header["__metadata__"] = {k: str(v) for k, v in meta.items()}
blob = json.dumps(out_header).encode()
blob += b" " * ((8 - len(blob) % 8) % 8)

hcache = {}
with open(dst, "wb") as out:
    out.write(struct.pack("<Q", len(blob)))
    out.write(blob)
    for i, (name, dtype, shape) in enumerate(plan):
        module = name.rsplit(".", 1)[0]
        if module in quant and name.endswith(".weight"):
            cfg = quant[module]
            assert cfg["format"] == "int8_tensorwise", cfg
            q = load(name).to(torch.float32)
            s = load(module + ".weight_scale").to(torch.float32).reshape(-1, 1)
            w = q * s
            if cfg.get("convrot"):
                g = int(cfg["convrot_groupsize"])
                if g not in hcache:
                    hcache[g] = hadamard(g).to(torch.float32)
                o, k = w.shape
                w = (w.reshape(o, k // g, g) @ hcache[g]).reshape(o, k)
            t = w.to(torch.bfloat16)
        else:
            t = load(name)
        data = t.contiguous().view(torch.uint8).numpy().tobytes()
        a, b = out_header[name]["data_offsets"]
        assert len(data) == b - a, name
        out.write(data)
        if i % 50 == 0:
            print(f"{i}/{len(plan)} {name}", flush=True)
print("done", dst, flush=True)
