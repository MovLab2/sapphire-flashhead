"""How fast this GPU renders her face, the way the server does it.

    python bench.py                    every variant, each in a fresh process
    python bench.py eager compiled     only these (names below)
    python bench.py --chunks 30        more chunks timed (default 20)
    python bench.py --face F --audio A a face picture and a speech recording for the frame strips
    python bench.py --long 180         also 3 minutes of her running on: does her face hold, or drift?

Run by flashhead-bench.sh / flashhead-bench.bat in the env that renders
(flashhead-server, or sapphire-plugin-flashhead inside Sapphire). Needs the
models already there: it never downloads. Each variant loads the model in its
own process (compiling is per process), warms up, then times real chunks:
0.96 s of speech in, 24 frames out, as the server's own _render and _pack.

Variants (the env decides what can run; one it can't is said and skipped):
  full, lean, smallest   the server's own Video memory presets (plugin.json), the way users get them
  eager       the full preset without torch.compile
  compiled    the full preset (torch.compile as shipped)
  sdpa        compiled, attention forced to PyTorch's own (if the env has a faster one)
  autotune    compiled with mode="max-autotune" (long warm-up)
  decode25    compiled, decodes only the frames kept (25 of 33): rougher joins between chunks (seam 1.27 -> 1.56)
  res384      compiled at 384 px square instead of 512: fewer tokens and pixels everywhere, a touch softer
  int8        compiled, the DiT's Linear weights int8 after the load (quant.py): ~1.4 GB less video memory
  int4        compiled, the DiT's big layers 4-bit, the rest int8: ~2 GB less
  vae8        compiled, the VAE's convolutions int8: ~0.4 GB less
Tried and dead (2026-11-02, don't repeat): 256 px melts to a grey mask by 3 minutes; 2 and 3 sampling steps
smear; CUDA graphs (reduce-overhead) drift and hold more memory. fp16 sums and the bf16 audio encoder
are part of every preset now (the server does them), so there is no variant for them.
Each variant also saves a strip of frames of her speaking: quality is for eyes, not timers.
Every variant reports video memory: the peak while rendering, the peak overall (warm-up, the face taken),
and what the process holds as nvidia-smi sees it (what a Windows card pages out when it is more than it has).
Results print as a table and are saved to user/models/flashhead/bench/<time>.json, strips beside.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "full": {"BENCH_PRESET": "full"},
    "lean": {"BENCH_PRESET": "lean"},
    "smallest": {"BENCH_PRESET": "smallest"},
    "eager": {"FLASHHEAD_COMPILE": "0"},
    "compiled": {},
    "sdpa": {"BENCH_ATTN": "sdpa"},
    "autotune": {"BENCH_COMPILE_MODE": "max-autotune"},
    "decode25": {"BENCH_DECODE25": "1"},
    "res384": {"BENCH_RES": "384"},
    "int8": {"BENCH_INT8": "1"},
    "int4": {"BENCH_INT4": "1"},
    "vae8": {"BENCH_VAE8": "1"},
}
DEFAULT = ["full", "lean", "smallest"]


def _root():
    """Where user/models/flashhead is: SAPPHIRE_ROOT, this folder (standalone), or a Sapphire above it."""
    if os.environ.get("SAPPHIRE_ROOT"):
        return Path(os.environ["SAPPHIRE_ROOT"])
    for p in (HERE, *HERE.parents):
        if (p / "user" / "models" / "flashhead").is_dir():
            return p
    return None


def _settings(root):
    """models_dir as this Sapphire's FlashHead settings name it (the server reads it the same way)."""
    try:
        s = json.loads((root / "user" / "webui" / "plugins" / "flashhead.json").read_text(encoding="utf-8"))
        return {"models_dir": s["models_dir"]} if s.get("models_dir") else {}
    except (OSError, ValueError, KeyError):
        return {}


def child(chunks, face, audio, long_s=0):
    """One variant, in this process: load, warm up, time, save a strip. Prints one BENCH line."""
    import numpy as np
    import torch
    out = {"variant": os.environ.get("BENCH_VARIANT"), "chunks": chunks}
    mode = os.environ.get("BENCH_COMPILE_MODE")
    if mode:
        real = torch.compile
        torch.compile = lambda m, *a, **k: real(m, *a, **{**k, "mode": mode})
    sys.path.insert(0, str(HERE))
    import server
    if not (server.CKPT.is_dir() and server.W2V.is_dir() and server.ENGINE.is_dir()):
        out["error"] = f"no models in {server.MODELS} (run the server once first: it downloads them)"
        print("BENCH " + json.dumps(out))
        return
    if os.environ.get("BENCH_RES"):                     # the server sizes her from its preset before the warm-up
        server.PRESET = dict(server.PRESET, size=int(os.environ["BENCH_RES"]))
    if os.environ.get("BENCH_ATTN") == "sdpa":
        sys.path.insert(0, str(server.ENGINE))
        os.chdir(server.ENGINE)
        server._stubs()
        from flash_head.src.modules import flash_head_model as fm
        if not (fm.SAGE_ATTN_AVAILABLE or fm.FLASH_ATTN_3_AVAILABLE or fm.FLASH_ATTN_2_AVAILABLE):
            out["error"] = "skipped: this env has no faster attention to compare with"
            print("BENCH " + json.dumps(out))
            return
        fm.SAGE_ATTN_AVAILABLE = fm.FLASH_ATTN_3_AVAILABLE = fm.FLASH_ATTN_2_AVAILABLE = False

    t = time.perf_counter()
    server._load()
    if server._state["status"] != "ready":
        out["error"] = server._state["error"] or server._state["status"]
        print("BENCH " + json.dumps(out))
        return
    out["load_s"] = round(time.perf_counter() - t, 1)    # includes the server's own two warm-up chunks
    out["vram_load_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    fh, P = server.P["fh"], server.P
    from flash_head.src.modules import flash_head_model as fm
    if os.environ.get("BENCH_DECODE25"):
        real = server._pipe.vae.decode
        server._pipe.vae.decode = lambda z, *a, **k: real(z[:, 1:], *a, **k)   # latents 1..4 -> 25 frames
        P["motion"] = 1                                                      # of which the last 24 are new
    if os.environ.get("BENCH_INT8") or os.environ.get("BENCH_INT4") or os.environ.get("BENCH_VAE8"):
        import quant                                    # after the load here; the server shrinks on the CPU before it
        dit = getattr(server._pipe.model, "_orig_mod", server._pipe.model)
        if os.environ.get("BENCH_INT8"):
            out["int8_layers"] = quant.dit(dit, 8)[1]
        if os.environ.get("BENCH_INT4"):
            out["int4_layers"], out["int8_layers"] = quant.dit(dit, 4)
        if os.environ.get("BENCH_VAE8"):
            out["vae8_layers"] = quant.vae(server._pipe.vae.model)
        torch._dynamo.reset()                           # compiled again, with the new weights
        torch.cuda.empty_cache()
    out["compiled"] = P.get("compiled")
    out["attention"] = ("sage" if fm.SAGE_ATTN_AVAILABLE else "flash3" if fm.FLASH_ATTN_3_AVAILABLE
                        else "flash2" if fm.FLASH_ATTN_2_AVAILABLE else "sdpa")

    stage = {"embed": [], "model": []}           # time inside the engine's two calls, synced

    def timed(fn, key):
        def run(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize()
            stage[key].append(time.perf_counter() - t0)
            return r
        return run
    fh.get_audio_embedding = timed(fh.get_audio_embedding, "embed")
    fh.run_pipeline = timed(fh.run_pipeline, "model")

    s = server.Session("bench")
    s.want = face or str(server.Path(server.tempfile.gettempdir()) / "flashhead_warmup.png")   # else the server's own warm-up card
    server._apply_face(s)
    speech = (np.sin(np.arange(P["n16"]) * 2 * np.pi * 180 / P["sr"]) * 0.3).astype(np.float32)
    a24 = np.zeros(P["n24"], np.int16)
    for _ in range(3):
        server._render(s, speech)
    for v in stage.values():
        v.clear()
    peak = torch.cuda.max_memory_allocated()             # through the load, warm-up, the face taken and three renders
    torch.cuda.reset_peak_memory_stats()
    render, pack = [], []
    for _ in range(chunks):
        t0 = time.perf_counter()
        frames = server._render(s, speech)
        t1 = time.perf_counter()
        server._pack("speech", frames, a24)
        render.append(t1 - t0)
        pack.append(time.perf_counter() - t1)
    med = lambda v: round(float(np.median(v)), 4)
    whole = med(render)              # the server packs JPEGs beside the next render: render alone sets the pace
    out.update(chunk_s=P["chunk_s"], render=med(render), pack=med(pack), whole=whole,
               embed=med(stage["embed"]), model=med(stage["model"]),
               worst=round(float(max(render)), 4),
               realtime=round(P["chunk_s"] / whole, 2),
               vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
               vram_peak_gb=round(max(peak, torch.cuda.max_memory_allocated()) / 1e9, 2),
               vram_reserved_gb=round(torch.cuda.memory_reserved() / 1e9, 2),
               vram_held_gb=_held())
    out["strip"] = _strip(server, s, audio)
    if long_s:
        out["long"] = _long(server, s, audio, long_s)
    print("BENCH " + json.dumps(out))


def _held():
    """GB this process holds on the GPU as nvidia-smi sees it: weights, activations and the
    allocator's cache together. None when nvidia-smi won't say (some Windows drivers)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            pid, mem = (x.strip() for x in line.split(",")[:2])
            if pid.isdigit() and int(pid) == os.getpid() and mem.isdigit():
                return round(int(mem) / 1024, 2)
    except Exception:
        pass
    return None


def _strip(server, s, audio):
    """Her speaking a few seconds (the recording given, else a hum), every
    sixth frame in a row: one PNG per variant, to put side by side."""
    import numpy as np
    from PIL import Image
    P = server.P
    a16 = server._decode(Path(audio).read_bytes())[0] if audio else \
        (np.sin(np.arange(P["n16"] * 4) * 2 * np.pi * 180 / P["sr"]) * 0.3).astype(np.float32)
    server._reset_motion(s)
    frames, starts = [], []
    for i in range(0, min(len(a16), P["n16"] * 6) - P["n16"] + 1, P["n16"]):
        starts.append(len(frames))
        frames += list(server._render(s, a16[i:i + P["n16"]]))
    step = lambda k: float(np.abs(frames[k].astype(np.float32) - frames[k - 1]).mean())
    seams = [step(k) for k in starts[1:]]
    inside = [step(k) for k in range(1, len(frames)) if k not in starts]
    seam = round(float(np.mean(seams)) / float(np.mean(inside)), 2)   # 1.0: a chunk joins as smoothly as frames inside one
    pick = frames[::6][:12]
    strip = Image.new("RGB", (192 * len(pick), 192))
    for i, f in enumerate(pick):
        strip.paste(Image.fromarray(f).resize((192, 192)), (192 * i, 0))
    out = Path(os.environ["BENCH_OUT"]) / f"strip-{os.environ['BENCH_VARIANT']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out)
    return {"file": str(out), "seam": seam}


def _long(server, s, audio, seconds):
    """Her speaking and sitting on, unbroken, for that long: how far her face
    strays from the picture given (the server's own drift measure), at the
    start and at the end, and her last frame saved. Error that feeds back
    through her motion shows here as a climb, then a melt."""
    import numpy as np
    from PIL import Image
    P = server.P
    talk = server._decode(Path(audio).read_bytes())[0] if audio else np.zeros(0, np.float32)
    server._reset_motion(s)
    n, drift, last = int(seconds / P["chunk_s"]), [], None
    for i in range(n):
        at = (i * P["n16"]) % (2 * max(len(talk), P["n16"]))
        a = talk[at:at + P["n16"]] if at + P["n16"] <= len(talk) else np.zeros(P["n16"], np.float32)   # talking, then quiet, again
        last = server._render(s, a)
        drift.append(float(server._drift(s, [last[0], last[-1]]).mean()))
    tenth = max(1, n // 10)
    out = Path(os.environ["BENCH_OUT"]) / f"long-{os.environ['BENCH_VARIANT']}.png"
    Image.fromarray(last[-1]).save(out)
    return {"seconds": round(n * P["chunk_s"]), "drift_start": round(float(np.mean(drift[:tenth])), 1),
            "drift_end": round(float(np.mean(drift[-tenth:])), 1), "drift_worst": round(max(drift), 1), "last": str(out)}


def machine():
    """What this ran on, asked in a separate process so the parent never touches the GPU."""
    code = ("import json,platform,torch\n"
            "d={'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
            "'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),"
            "'os':platform.platform()}\n"
            "for m in ('triton','sageattention','flash_attn'):\n"
            "    try: d[m]=__import__(m).__version__\n"
            "    except Exception: d[m]=None\n"
            "print(json.dumps(d))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": (r.stderr or "").strip()[-300:]}


def main(argv):
    opts = {"--chunks": "20", "--face": "", "--audio": "", "--long": "0"}
    for k in opts:
        if k in argv:
            i = argv.index(k)
            opts[k] = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
    chunks = int(opts["--chunks"])
    for k in ("--face", "--audio"):
        if opts[k]:
            opts[k] = str(Path(opts[k]).resolve())
            if not Path(opts[k]).is_file():
                print(f"{k}: no such file {opts[k]}")
                return 2
    names = argv or DEFAULT
    bad = [n for n in names if n not in VARIANTS]
    if bad:
        print(f"unknown variant {bad[0]!r}: one of {', '.join(VARIANTS)}")
        return 2
    root = _root()
    if not root:
        print("No models found: run the FlashHead server once first (it downloads them).")
        return 1
    info = machine()
    print(f"FlashHead bench on {info.get('gpu')} | torch {info.get('torch')} (CUDA {info.get('cuda')}) | "
          f"triton {info.get('triton')} | sage {info.get('sageattention')} | flash-attn {info.get('flash_attn')} | "
          f"{info.get('os')}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = root / "user" / "models" / "flashhead" / "bench"
    results = []
    for n in names:
        print(f"\n{n}: loading and warming up (compiled ones take a minute or more)...", flush=True)
        settings = _settings(root)
        if "BENCH_PRESET" in VARIANTS[n]:               # the server applies its own preset: size, weights, the preflight
            settings["memory"] = VARIANTS[n]["BENCH_PRESET"]
        env = {**os.environ, "SAPPHIRE_ROOT": str(root), "BENCH_VARIANT": n, "PYTHONUTF8": "1",
               "BENCH_OUT": str(folder / stamp),
               "SAPPHIRE_PLUGIN_SETTINGS": json.dumps(settings), **VARIANTS[n]}
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(chunks),
                            opts["--face"], opts["--audio"], opts["--long"]],
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith("BENCH ")), None)
        r = json.loads(line[6:]) if line else {"variant": n, "error": (p.stderr or "no output").strip()[-500:]}
        results.append(r)
        print(f"  {r['error']}" if r.get("error") else
              f"  {r['realtime']}x real time: {r['whole']}s per {r['chunk_s']}s chunk "
              f"(model {r['model']}, audio {r['embed']}, frames out {r['pack']}), worst {r['worst']}s, "
              f"{r['vram_gb']} GB, attention {r['attention']}, torch.compile: {r['compiled']}")

    ok = [r for r in results if not r.get("error")]
    if ok:
        print(f"\n{'variant':10} {'real time':>9} {'per chunk':>9} {'model':>7} {'audio':>7} {'jpegs*':>7} {'worst':>7} {'seam':>5} {'load':>6}"
              f" {'vram':>5} {'peak':>5} {'held':>5}")
        for r in ok:
            held = r.get("vram_held_gb")
            print(f"{r['variant']:10} {r['realtime']:>8}x {r['whole']:>9} {r['model']:>7} {r['embed']:>7} "
                  f"{r['pack']:>7} {r['worst']:>7} {r['strip']['seam']:>5} {r['load_s']:>5}s"
                  f" {r['vram_gb']:>5} {r.get('vram_peak_gb', r['vram_gb']):>5} {('-' if held is None else held):>5}")
        print("* JPEGs are made beside the next render, so they don't count in the pace. "
              "seam: picture jump between chunks / within one (1.0 = smooth). "
              "vram: GB at peak while rendering; peak: the most ever allocated (warm-up, the face taken); "
              "held: what nvidia-smi shows for the process.")
        for r in ok:
            if r.get("long"):
                L = r["long"]
                print(f"{r['variant']:10} {L['seconds']} s on: drift {L['drift_start']} at the start, "
                      f"{L['drift_end']} at the end, worst {L['drift_worst']} (her last frame: {L['last']})")
    out = folder / f"{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": info, "results": results}, indent=1), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(int(sys.argv[2]), sys.argv[3] or None, sys.argv[4] or None, float(sys.argv[5]))
    else:
        sys.exit(main(sys.argv[1:]))
