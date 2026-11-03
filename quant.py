"""Leaner weights for a small card: int8 (a scale per output row or channel) and 4-bit
(a scale and a floor per 64 weights) storage, widened to the working dtype inside
each call, which torch.compile fuses into the matmul's prologue. Nothing downloads:
the same checkpoints load and shrink on the CPU on their way to the GPU
(hook_loads), so the card never holds the 16-bit weights. Module.to(dtype) leaves
int8 and uint8 buffers alone, so the pipeline's own .to(device, dtype) just moves them.
Measured on a 5090 (bench.py, 2026-11-02): int8 DiT 1.4 GB less at 9% slower, 4-bit
2.1 GB less, int8 VAE 0.4 GB less; her face the same to the eye over 3 minutes."""
import torch
import torch.nn.functional as F

G = 64          # the 4-bit group, along the input


class Int8Linear(torch.nn.Module):
    def __init__(self, lin):
        super().__init__()
        w = lin.weight.detach().float()
        s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
        self.register_buffer("w8", (w / s).round().to(torch.int8))
        self.register_buffer("scale", s.to(lin.weight.dtype))
        self.bias = lin.bias

    def forward(self, x):
        return F.linear(x, self.w8.to(x.dtype) * self.scale, self.bias)


class Int4Linear(torch.nn.Module):
    """Asymmetric: q in 0..15, w = q * scale + low per group; two weights to a byte."""

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.detach().float()
        n, k = w.shape
        g = w.view(n, k // G, G)
        lo, hi = g.amin(dim=2, keepdim=True), g.amax(dim=2, keepdim=True)
        scale = ((hi - lo) / 15).clamp(min=1e-8)
        q = ((g - lo) / scale).round().clamp(0, 15).to(torch.uint8).view(n, k)
        self.register_buffer("w4", q[:, ::2] | (q[:, 1::2] << 4))
        self.register_buffer("scale", scale.squeeze(2).to(lin.weight.dtype))
        self.register_buffer("low", lo.squeeze(2).to(lin.weight.dtype))
        self.bias = lin.bias

    def forward(self, x):
        n = self.w4.shape[0]
        q = torch.stack((self.w4 & 15, self.w4 >> 4), dim=2).view(n, -1, G).to(x.dtype)
        return F.linear(x, (q * self.scale[:, :, None] + self.low[:, :, None]).view(n, -1), self.bias)


class Int8Conv3d(torch.nn.Module):
    """An nn.Conv3d with zero padding, its weight int8 with a scale per output channel."""

    def __init__(self, c):
        super().__init__()
        w = c.weight.detach().float()
        s = w.abs().amax(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-8) / 127
        self.register_buffer("w8", (w / s).round().to(torch.int8))
        self.register_buffer("scale", s.to(c.weight.dtype))
        self.bias = c.bias
        self.args = (c.stride, c.padding, c.dilation, c.groups)

    @property
    def weight(self):
        return self.w8.to(self.scale.dtype) * self.scale

    def forward(self, x):
        return F.conv3d(x, self.w8.to(x.dtype) * self.scale, self.bias, *self.args)


def swap(root, kind, make, keep=lambda name, mod: False):
    """Every child of that kind under root replaced by make(child), one at a time: never
    two copies of a layer's weights at once. keep(name, mod) True leaves that one as it is."""
    n = 0
    parents = [(pn, m) for pn, m in root.named_modules() if any(isinstance(c, kind) for c in m.children())]
    for pname, parent in parents:
        for name, child in list(parent.named_children()):
            if isinstance(child, kind) and not keep(f"{pname}.{name}", child):
                setattr(parent, name, make(child))
                n += 1
        child = None
    return n


def _small(name, lin):
    """4-bit only for the big block layers: the time and head paths and small layers stay int8."""
    return lin.weight.numel() < 2**20 or lin.in_features % G or "time" in name or "head" in name


def dit(model, bits):
    """The DiT's Linear layers at 8 or 4 bits (16: untouched). -> (4-bit layers, int8 layers)."""
    n4 = swap(model, torch.nn.Linear, Int4Linear, keep=_small) if bits == 4 else 0
    n8 = swap(model, torch.nn.Linear, Int8Linear) if bits in (4, 8) else 0
    return n4, n8


def vae(model):
    """The VAE's 3-D convolutions int8. -> layers."""
    return swap(model, torch.nn.Conv3d, Int8Conv3d, keep=lambda name, c: c.padding_mode != "zeros")


def hook_loads(bits, vae8, log=None):
    """The engine's two from_pretrained calls shrink what they load (on the CPU, in fp32)
    before the pipeline moves it to the GPU. After the engine is imported, before get_pipeline."""
    say = log or (lambda *_: None)

    def wrap(cls, after):
        orig = cls.from_pretrained                      # a bound classmethod: cls is in it already
        cls.from_pretrained = classmethod(lambda c, *a, **k: after(orig(*a, **k)))

    if bits in (4, 8):
        from flash_head.src.modules.flash_head_model import WanModelAudioProject

        def shrink(m):
            n4, n8 = dit(m, bits)
            say(f"DiT weights: {n4} layers 4-bit, {n8} layers int8")
            return m
        wrap(WanModelAudioProject, shrink)
    if vae8:
        from flash_head.ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder

        def shrink_vae(m):
            say(f"VAE weights: {vae(m)} convolutions int8")
            return m
        wrap(CausalVideoAutoencoder, shrink_vae)
