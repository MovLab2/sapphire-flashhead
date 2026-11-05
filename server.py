"""FlashHead server — SoulX-FlashHead Lite talking face in its own conda env.

Runs as a Sapphire plugin service with the sapphire-plugin-flashhead env's
python on SAPPHIRE_SERVICE_PORT. A loader: nothing of theirs ships here. On
first start it fetches the engine (Soul-AILab/SoulX-FlashHead, Apache-2.0) at a
pinned commit and the Lite weights from HuggingFace, into user/models/flashhead/
(backups skip user/models/: it all downloads again if lost).

Needs an NVIDIA GPU (compute 8.0+, ~7 GB free): checked before anything is
fetched; without one the service stays up and says why on /health.

Renders only while a viewer is connected. Each chunk is 0.96 s of audio in,
24 frames out; her speech when it's queued, silence otherwise (a calm, living
idle; music under her read as silent singing).

A worker for several Sapphires (setting `share`): other computers come in
through a TLS door on `share_port`, each with its own key (worker_door.py),
and each is its own session: her face, her motion, her words, her viewers.
One model; her words always go first. This computer's Sapphire is "local",
on 127.0.0.1 with no key. The same calls serve both:

  GET  /health          status, this session's viewers
  POST /face?idle=      {"path": image} (local only) or the image itself
                        (Content-Type image/*) - animate this face; idle=loop|live
  POST /speak?take=1    raw audio (any format PyAV reads), mouthed while anyone watches;
       &device=        {"taken": true} only when asked to take it and a browser or a call plays
                        sound (a board stream is not a listener); not taken, browsers get it
                        with its audio zeroed (it plays elsewhere). device= names the board
                        whose own reply it is: its stream carries the voice whole and that
                        board plays it; every other face mouths it (2026-11-03). sid= core's
                        stream id, carried in each packet's header: a tap on her face names it
  POST /viewer          a browser's numbers for the reply it played (JSON): one log line
  POST /flush?by=       she was stopped: drop what is queued (by=viewer: a tap on her face, logged)
  POST /end             her reply is over: its last words go out now (until then a
                        gap in her words holds the face, it never idles mid-reply)
  GET  /stream?audio=0  packets: >I header length, JSON header, s16 PCM, JPEG frames;
                        audio=0 = a viewer that shows frames only (her screens); of the
                        others the newest plays her voice, the rest get packets without it;
                        call=1 = her webcam carrying her voice into a call: it keeps her
                        voice while it watches, a browser opened meanwhile mouths silently;
                        board=1 = a board stream (screens.py): the whole packet always, never
                        the voice tab; a speech head carries {"mute", "device", "taken"} and
                        the board plays the voice only when device is its own and nobody took it
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import contextlib
import fnmatch
import io
import json
import logging
import os
import queue
import shutil
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

import worker_door                  # a copy of Sapphire's core/worker_door.py: this server also runs alone

if sys.platform != "win32":     # Windows picks 1 itself; more needs a subprocess pool it can't start (pass_fds)
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")   # not 32 idle compiler workers per service

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("flashhead")

PORT = int(os.environ.get("SAPPHIRE_SERVICE_PORT", 5017))
ROOT = Path(os.environ["SAPPHIRE_ROOT"])
ENGINE_REPO = "Soul-AILab/SoulX-FlashHead"
ENGINE_SHA = "9bc03de06bb0de82cd6bc477804512ae06144bf2"     # 2026-05-28, the commit tested here
_START = json.loads(os.environ.get("SAPPHIRE_PLUGIN_SETTINGS") or "{}")


def _models_dir():
    """Her models' folder: settings `models_dir` (the same rule as faces.models_dir),
    else user/models/flashhead: core's cache floor, never in a backup."""
    raw = str(_START.get("models_dir") or "").strip().strip('"')     # Explorer's "Copy as path" adds quotes
    if not raw:
        return ROOT / "user" / "models" / "flashhead"
    p = Path(raw).expanduser()
    return p if p.is_absolute() else ROOT / p


MODELS = _models_dir()
for _var, _sub in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")):
    # compiled kernels beside the models, not under a profile folder (Windows: a long or non-ASCII user name breaks them)
    os.environ.setdefault(_var, str(ROOT / "user" / "models" / "flashhead" / "cache" / _sub))
ENGINE = ROOT / "user" / "models" / "flashhead" / "engine" / ENGINE_SHA[:12]   # fetched code: rebuildable, so cache floor too
CKPT = MODELS / "SoulX-FlashHead-1_3B"
W2V = MODELS / "wav2vec2-base-960h"
PLAY_SR = 24000          # Kokoro's native rate; what the viewer hears
LEAD = 0.5               # seconds rendered ahead of the wall clock (at least 1.5 renders: see _pick)
VIEWER_QUEUE = 8
HOLD_S = 4.0             # mid-reply, her face waits this long for her next words before it idles
BEAT = [0.0]             # when the render loop last came round: /speak takes no words from a stalled one

_state = {"status": "starting", "error": None, "progress": None,
          "warning": None,      # ready, but the GPU is slower than real time
          "note": None}         # a few seconds of work under "ready"
_gpu = threading.Lock()
_sessions = {}           # (where, name) -> Session: ("local", "local") is this computer's Sapphire, ("door", name) the others
_viewers_lock = threading.Lock()
_watching = threading.Event()    # anyone, in any session
_pipe = None
P = {}                   # chunk geometry, filled after the model loads
IDLE_CHUNKS, IDLE_SETTLE, IDLE_KEEP = 14, 3, 5   # loop mode: render 14 silent chunks, skip the 3 that settle, keep the calmest 5
HOME = 0.25              # live idle: how much of her picture goes back into the motion she continues from, each quiet chunk
SETTINGS = ROOT / "user" / "webui" / "plugins" / "flashhead.json"
FACES = ROOT / "user" / "models" / "flashhead" / "faces"     # faces other computers sent: copies, rebuildable
DOOR = ROOT / "user" / "worker_door" / "flashhead"           # its cert and the keys of the computers let in
FACES_KEPT = 16          # per session: her face and her moods
_cfg = {"mtime": None, "idle": "loop"}
PRESETS = {             # setting `memory`: what her face takes of the card (measured on a 5090, 2026-11-02:
    "full": dict(size=512, bits=16, vae8=False, need=7.0),      # 6.1 GB held, 6.7x), need: GB free the preflight asks for
    "lean": dict(size=384, bits=8, vae8=True, need=4.5),        # 3.65 GB, 8.1x
    "smallest": dict(size=384, bits=4, vae8=True, need=3.5),    # 2.97 GB, 8.3x
}
PRESET_NAME = str(_START.get("memory") or "full") if str(_START.get("memory") or "full") in PRESETS else "full"
PRESET = PRESETS[PRESET_NAME]
MEASURED = ROOT / "user" / "flashhead" / "measured.json"    # what each preset took and ran at on this card


class Session:
    """One Sapphire using this worker: her face and motion on the GPU, her
    words waiting, her viewers. The model is one; between chunks the render
    loop points it at the session it serves."""

    def __init__(self, name):
        self.name = name
        self.want = None            # the face file wanted
        self.face = None            # file@mtime of the face taken
        self.t = None               # (colour reference, reference latent, motion latent) on the GPU
        self.dq = None              # her last seconds of audio: the model's window
        self.mode = None            # idle "loop" | "live" as her Sapphire asked; None: this computer's setting
        self.speech = queue.Queue()
        self.pending = {"a16": np.zeros(0, np.float32), "a24": np.zeros(0, np.int16), "tag": None}
        self.lock = threading.Lock()
        self.viewers = {}           # queue -> plays the sound (a browser) or frames only (a board)
        self.boards = set()         # viewers that are board streams (?board=1): the whole packet always, never s.voice; the board routes her voice by "device"
        self.voice = None           # the one viewer that plays her voice (the newest browser): two would echo
        self.call = None            # her webcam's viewer with her voice: it keeps s.voice while it watches
        self.idle = {"key": None, "packets": [], "at": 0}
        self.live = False           # her motion runs on from the last chunk; False: speech starts her afresh
        self.clock = None           # [t0, seconds made] while watched
        self.note = None
        self.gen = 0                # bumped by a flush: a chunk rendered before it is never sent after it
        self.build = None           # her calm loop being made, a chunk per turn
        self.fails = 0              # renders failed in a row, this session's alone
        self.open = False           # a reply is coming in (/speak), until /end, a flush or HOLD_S of quiet
        self.heard = 0.0            # when her words last came
        self.lead = 0.0             # how far ahead of her viewers' clock the last pick was
        self.run = None             # this speech run's numbers, logged in one line when it ends
        self.drops = 0              # packets a slow viewer lost, all time

    def talking(self):
        return not self.speech.empty() or len(self.pending["a16"]) > 0

    def held(self):
        """Mid-reply with nothing whole to say yet: she waits as she is. A
        padded tail and a calm-loop chunk here broke her motion (reset, ~2 s
        to settle) in the middle of a long reply on a GPU near real time."""
        return self.open and self.speech.empty() and time.monotonic() - self.heard < HOLD_S


def _session(name, door=False):
    """Door sessions are keyed apart from this computer's: no client name can ever be hers."""
    key = ("door" if door else "local", name)
    with _viewers_lock:
        return _sessions.get(key) or _sessions.setdefault(key, Session(name))


DONE = ".complete"       # written last: a folder without it is a fetch that did not finish


def _preflight():
    """Why this machine can't run Lite, in words, or None. Before any download."""
    need = (f"needs an NVIDIA GPU (RTX 30-series or newer) with {PRESET['need']:g} GB of free video memory"
            + (" for this Video memory preset" if PRESET_NAME != "full" else
               " (Settings > FlashHead > Video memory has leaner presets)"))
    import torch
    if torch.version.hip:
        return f"FlashHead {need}: this is an AMD (ROCm) build"
    if not torch.cuda.is_available():
        if shutil.which("nvidia-smi"):
            return ("FlashHead found an NVIDIA GPU but CUDA would not start: "
                    "update the NVIDIA driver to 570 or newer (CUDA 12.8), then Retry")
        return f"FlashHead {need}: no CUDA GPU found"
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if major < 8:
        return f"FlashHead {need}: the {name} is compute {major}.{minor}"
    free, total = (x / 1e9 for x in torch.cuda.mem_get_info(0))
    _state["gpu"] = {"name": name, "total_gb": round(total, 1), "free_gb": round(free, 1), "preset": PRESET_NAME}
    if free < PRESET["need"]:
        return f"FlashHead {need}: the {name} has {free:.1f} GB free right now"
    return None


def _fetch_engine():
    """flash_head/ and its LICENSE from the pinned commit, nothing else.
    Unpacked beside the target and renamed into place when whole."""
    import tarfile
    import urllib.request
    if (ENGINE / DONE).exists():
        return
    _state["status"] = "fetching engine"
    part = ENGINE.with_name(ENGINE.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    url = f"https://codeload.github.com/{ENGINE_REPO}/tar.gz/{ENGINE_SHA}"
    with urllib.request.urlopen(url, timeout=120) as r, tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz") as t:
        keep = []
        for m in t.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if rel == "LICENSE" or rel.startswith("flash_head/"):
                m.name = rel
                keep.append(m)
        t.extractall(part, members=keep, filter="data")
    (part / DONE).touch()
    shutil.rmtree(ENGINE, ignore_errors=True)
    part.rename(ENGINE)


def _du(path):
    """Bytes under path so far. Files come and go while HuggingFace renames them: those are skipped."""
    total = 0
    for f in Path(path).rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _snapshot(repo, dest, patterns, label):
    """One HuggingFace snapshot, resumable. A first fetch goes to dest.part and
    is renamed when whole; a folder from before the marker is checked in place
    (files already there are skipped). Progress lands on /health."""
    from huggingface_hub import HfApi, snapshot_download
    if (dest / DONE).exists():
        return
    work = dest if dest.exists() else dest.with_name(dest.name + ".part")
    try:
        files = HfApi().model_info(repo, files_metadata=True).siblings
        total = sum(f.size or 0 for f in files if any(fnmatch.fnmatch(f.rfilename, p) for p in patterns))
    except Exception:
        total = 0
    done = threading.Event()

    def watch():
        while not done.wait(2):
            got = _du(work)
            _state["progress"] = {"done": got, "total": total}
            _state["status"] = f"downloading {label}" + (f" {got / 1e9:.1f} / {total / 1e9:.1f} GB" if total else "")
    _state["status"] = f"downloading {label}"
    threading.Thread(target=watch, daemon=True).start()
    try:
        snapshot_download(repo, local_dir=str(work), allow_patterns=patterns)
    finally:
        done.set()
        _state["progress"] = None
    (work / DONE).touch()
    if work != dest:
        work.rename(dest)


def _download():
    _fetch_engine()
    _snapshot("Soul-AILab/SoulX-FlashHead-1_3B", CKPT,
              ["config.json", "model_index.json", "Model_Lite/*", "VAE_LTX/*"], "the Lite model")
    _snapshot("facebook/wav2vec2-base-960h", W2V, ["*.json", "model.safetensors", "*.txt"], "wav2vec2")


def _stubs():
    """The engine imports xfuser (multi-GPU) and mediapipe (face crop) at the top
    of its modules; one GPU and no crop never call them, so they are not in the
    env. Stand-ins that fail loudly if anything ever does."""
    import importlib.util
    import types

    def absent(name):
        def attr(a):
            if a.startswith("__"):
                raise AttributeError(a)
            def call(*_, **__):
                raise RuntimeError(f"{name}.{a}: not installed (FlashHead runs one GPU, no face crop)")
            return call
        m = types.ModuleType(name)
        m.__getattr__ = attr
        return m
    for top, mods in (("xfuser", ("xfuser", "xfuser.core", "xfuser.core.distributed", "xfuser.core.long_ctx_attention")),
                      ("mediapipe", ("mediapipe",))):
        if top not in sys.modules and importlib.util.find_spec(top) is None:   # find_spec throws on a stand-in already put in (bench)
            for name in mods:
                sys.modules.setdefault(name, absent(name))


def _compile_ok():
    """torch.compile needs Triton (on Windows: triton-windows). Without it, plain eager."""
    if os.environ.get("FLASHHEAD_COMPILE") == "0":
        return False
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def _install_test(why):
    """No GPU to render on, but the install is tried all the same (setting
    install_test): fetch, download, stand-ins, engine import, Triton. For a
    machine (a VM) that can test every step but the last."""
    step = "fetching"
    try:
        _download()
        step = "importing the engine"
        os.chdir(ENGINE)
        if str(ENGINE) not in sys.path:
            sys.path.insert(0, str(ENGINE))
        _stubs()
        from flash_head import inference  # noqa: F401
        from flash_head.src.pipeline import flash_head_pipeline  # noqa: F401
        compiled = "with torch.compile" if _compile_ok() else "without torch.compile (no Triton)"
        msg = (f"Install test passed: engine and models fetched, the engine imports, it would run {compiled}. "
               f"Rendering still {why.split(' ', 1)[1]}")
        logger.info(msg)
        _state.update(status="unsupported", error=msg)
    except Exception as e:
        logger.exception("Install test failed")
        _state.update(status="error", error=f"Install test failed while {step}: {e}")


def _load():
    global _pipe
    try:
        _state.update(status="checking the GPU", error=None)
        why = _preflight()
        if why and _START.get("install_test"):
            return _install_test(why)
        if why:
            logger.warning(why)
            _state.update(status="unsupported", error=why)
            return
        _download()
        _state["status"] = "loading model"
        os.chdir(ENGINE)                      # the engine opens its yaml by relative path
        if str(ENGINE) not in sys.path:
            sys.path.insert(0, str(ENGINE))
        import torch
        _stubs()
        from flash_head import inference as fh
        from flash_head.src.pipeline import flash_head_pipeline as fp
        fh.infer_params["height"] = fh.infer_params["width"] = PRESET["size"]     # read whenever a face is set
        import quant                      # leaner weights for a small card (setting `memory`); torch, so not at import
        quant.hook_loads(PRESET["bits"], PRESET["vae8"], logger.info)
        if not _compile_ok():
            logger.warning("No Triton: running without torch.compile (slower)")
            fp.COMPILE_MODEL = fp.COMPILE_VAE = False
        torch._dynamo.config.suppress_errors = True   # a compile that fails falls back to eager, not to a dead service
        torch.backends.cudnn.benchmark = True        # VAE 3-D convs: ~15% on the 5090
        load = lambda: fh.get_pipeline(world_size=1, ckpt_dir=str(CKPT), model_type="lite", wav2vec_dir=str(W2V))
        _pipe = None
        try:
            _pipe = load()
        except Exception as e:
            if not fp.COMPILE_MODEL:
                raise
            logger.warning(f"torch.compile would not start ({type(e).__name__}: {e}): loading without it (slower)")
            fp.COMPILE_MODEL = fp.COMPILE_VAE = False
        if _pipe is None:                     # out here, the failed load's model is freed before the second
            torch.cuda.empty_cache()
            _pipe = load()
        _fp16()
        _w2v16(fh)
        ip = fh.get_infer_params()
        # the model's window geometry, as in SoulX-FlashHead's generate_video.py stream mode (Apache-2.0)
        slice_len = ip["frame_num"] - ip["motion_frames_num"]
        cache = ip["sample_rate"] * ip["cached_audio_duration"]
        end = ip["cached_audio_duration"] * ip["tgt_fps"]
        P.update(fh=fh, ip=ip, fps=ip["tgt_fps"], sr=ip["sample_rate"],
                 motion=ip["motion_frames_num"], n16=slice_len * ip["sample_rate"] // ip["tgt_fps"],
                 chunk_s=slice_len / ip["tgt_fps"], start=end - ip["frame_num"], end=end)
        P["n24"] = P["n16"] * PLAY_SR // P["sr"]
        P["cache"] = cache

        blank = Path(tempfile.gettempdir()) / "flashhead_warmup.png"
        from PIL import Image
        Image.new("RGB", (512, 512), (128, 128, 128)).save(blank)
        warm = Session("warm-up")
        warm.want = str(blank)
        _apply_face(warm)
        _state["status"] = "warming up (torch.compile: a minute, several the first time)" if fp.COMPILE_MODEL else "warming up"
        _render(warm, np.zeros(P["n16"], np.float32))
        t = time.perf_counter()
        _render(warm, np.zeros(P["n16"], np.float32))          # compiled now: this one is the real speed
        P["render_s"] = time.perf_counter() - t
        speed = P["chunk_s"] / P["render_s"]
        P["compiled"] = _compiled(fp)
        g = _state["gpu"]
        g.update(took_gb=round(g["free_gb"] - torch.cuda.mem_get_info(0)[0] / 1e9, 1), speed=round(speed, 1))
        paging = torch.cuda.memory_reserved() / 1e9 > g["free_gb"] + 0.2    # Windows pages to system RAM instead of failing
        leaner = " A leaner Video memory preset (Settings > FlashHead) takes less and renders faster." if PRESET_NAME != "smallest" else ""
        if paging:
            _state["warning"] = (f"The {g['name']} ran out of video memory ({g['free_gb']} GB was free) and is paging to "
                                 f"system RAM: her face will crawl.{leaner}")
            logger.warning(_state["warning"])
        elif speed < 1.1:
            _state["warning"] = (f"This GPU renders {speed:.1f}x real time: her face may fall behind her voice. "
                                 "Other work on the GPU (a local LLM, a game) slows her too."
                                 + {"no": " It runs without torch.compile (see its log), ~20% slower.",
                                    "partly": " Parts of it run without torch.compile (see its log)."}.get(P["compiled"], "")
                                 + leaner)
            logger.warning(_state["warning"])
        _record(g)
        _state["status"] = "ready"
        logger.info(f"Ready: {P['n16']} samples -> {P['n16'] * P['fps'] // P['sr']} frames per chunk at {PRESET['size']} px, "
                    f"{speed:.1f}x real time, torch.compile: {P['compiled']}; {g['name']}, {g['total_gb']} GB, "
                    f"{g['free_gb']} GB free at start, her face takes {g['took_gb']} GB "
                    f"(Video memory: {PRESET_NAME}, {PRESET['bits']}-bit weights, fp16 sums)")
    except Exception as e:
        logger.exception("Load failed")
        _state.update(status="error", error=str(e))


def _compiled(fp):
    """'yes', 'partly' or 'no': whether torch.compile really took. Its
    failures are turned into plain eager runs (suppress_errors), so this
    counts what compiled instead of trusting that it was asked to."""
    if not fp.COMPILE_MODEL:
        return "no"
    from torch._dynamo.utils import counters
    ok, total = counters["frames"]["ok"], counters["frames"]["total"]
    if ok < total:
        logger.warning(f"torch.compile failed for {total - ok} of {total} parts: those run uncompiled (the reason is above)")
    return "yes" if ok == total else "partly" if ok else "no"


def _record(g):
    """What this preset took of the card and ran at, kept per preset for the settings page
    (its line under Video memory). Entries from another card are dropped."""
    try:
        kept = json.loads(MEASURED.read_text(encoding="utf-8")) if MEASURED.exists() else {}
    except (OSError, ValueError):
        kept = {}
    kept = {k: v for k, v in kept.items() if isinstance(v, dict) and v.get("gpu") == g["name"]}
    kept[PRESET_NAME] = {"gpu": g["name"], "took_gb": g["took_gb"], "speed": g["speed"],
                         "size": PRESET["size"], "bits": PRESET["bits"], "at": int(time.time())}
    try:
        MEASURED.parent.mkdir(parents=True, exist_ok=True)
        MEASURED.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    except OSError as e:
        logger.warning(f"{MEASURED.name} not written: {e}")
    _state["measured"] = kept


def _fp16():
    """The DiT in fp16 with fp16 matmul sums: 17% faster on RTX 40/50 (bench fp16acc), her face
    the same over 3-minute runs; the VAE stays bf16, cast at its edges. An overflow shows as
    NaN frames: _render watches for them and goes back to bf16, once, for good."""
    import torch
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    _pipe.model.half()
    _pipe.param_dtype = torch.float16
    dec, enc, bf = _pipe.vae.decode, _pipe.vae.encode, torch.bfloat16
    _pipe.vae.decode = lambda z, *a, **k: dec(z.to(bf), *a, **k)
    _pipe.vae.encode = lambda x, *a, **k: enc(x.to(bf), *a, **k).to(_pipe.param_dtype)
    P["fp16"] = True


def _fp16_off():
    """fp16 overflowed for some face: bf16 from here on, every session's tensors with it."""
    import torch
    logger.warning("fp16 overflowed (NaN frames): the model goes on in bf16, a little slower")
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    _pipe.model.to(torch.bfloat16)
    _pipe.param_dtype = torch.bfloat16
    P["fp16"] = False
    with _viewers_lock:
        for s in _sessions.values():
            if s.t:
                s.t = tuple(t.to(torch.bfloat16) for t in s.t)


def _w2v16(fh):
    """The audio encoder in bf16 (it loads in fp32): 0.2 GB less, 5 -> 3.5 ms a chunk."""
    import torch
    _pipe.audio_encoder.to(torch.bfloat16)
    real = fh.get_audio_embedding

    def emb(*a, **k):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return real(*a, **k)
    fh.get_audio_embedding = emb


def _apply_face(s):
    """Called by the render loop: give the session the face it wants, if changed."""
    path = s.want
    if not path or not os.path.exists(path):
        return
    key = f"{path}@{os.path.getmtime(path)}"     # a re-upload keeps the path
    if key == s.face:
        return
    with _gpu:
        P["fh"].get_base_data(_pipe, cond_image_path_or_dir=path, base_seed=42, use_face_crop=False)
        s.t = (_pipe.original_color_reference, _pipe.ref_img_latent, _pipe.latent_motion_frames)
        s.dq = np.zeros(P["cache"], np.float32)
    s.face, s.live = key, False
    logger.info(f"Face set ({s.name})")


def _reset_motion(s):
    """Start her from the reference again: speech begins here, and its first
    words hide the ~2 s the model takes to settle."""
    with _gpu:
        s.t = (s.t[0], s.t[1], s.t[1][:, :1].clone())
        s.dq = np.zeros(P["cache"], np.float32)


def _home(s):
    """Silence feeds on itself: each chunk goes on from the last, and with no
    words to steer her, her lips run to lipstick and her eyes soften within a
    minute. A quarter of her picture mixed into the motion she goes on from
    holds her (measured: lip colour +4% on average instead of +25%, peak
    +8% instead of +50%; she still blinks and shifts). Speech doesn't drift
    and a pull there dulls her mouth, so it's for quiet chunks only."""
    with _gpu:
        a, ref, m = s.t
        s.t = (a, ref, ((1 - HOME) * m.float() + HOME * ref[:, :m.shape[1]].float()).to(m.dtype))


def _idle_mode(s):
    """'loop' (rendered once, the GPU rests) or 'live' (rendered as she sits,
    the GPU works while anyone watches): as her Sapphire asked, else this
    computer's setting, read again when it changes."""
    if s.mode in ("loop", "live"):
        return s.mode
    try:
        m = SETTINGS.stat().st_mtime
        if m != _cfg["mtime"]:
            _cfg.update(mtime=m, idle=json.loads(SETTINGS.read_text(encoding="utf-8")).get("idle") or "loop")
    except (OSError, ValueError):
        pass
    return _cfg["idle"]


def _drift(s, frames):
    """How far each frame strays from her face as given: small grey middles
    (eyes and mouth), mean difference. Silence slowly lowers her lids and
    tilts her head; this is what sees it."""
    from PIL import Image
    h, w = frames[0].shape[:2]
    box = (int(w * .2), int(h * .2), int(w * .8), int(h * .85))
    small = lambda im: np.asarray(im.crop(box).convert("L").resize((48, 48)), np.float32)
    ref = small(Image.open(s.want).convert("RGB").resize((w, h)))
    return np.array([np.abs(small(Image.fromarray(f)) - ref).mean() for f in frames])


def _idle_step(s):
    """Loop mode: calm idle for the session's face, rendered once. The stretch
    that stays closest to her face is kept and played forward then back, so
    she never nods off and snaps awake at the turn. The GPU then rests.
    Made one chunk per turn of the render loop (shown as she sits meanwhile),
    so anyone's words go in between. -> (frames to send, or a packet made)."""
    if s.idle["key"] == s.face:
        loop = s.idle["packets"]
        s.idle["at"] += 1
        return None, loop[(s.idle["at"] - 1) % len(loop)]
    b = s.build
    if b is None or b["face"] != s.face:
        s.note = "making her calm loop (a few seconds, once per face)"
        _reset_motion(s)
        b = s.build = {"face": s.face, "frames": [], "i": 0}
    f = _render(s, np.zeros(P["n16"], np.float32))
    if b["i"] >= IDLE_SETTLE:
        b["frames"] += list(f)
    b["i"] += 1
    if b["i"] == IDLE_CHUNKS:
        frames, per = b["frames"], len(f)
        keep = IDLE_KEEP * per
        d = _drift(s, frames)
        means = np.convolve(d, np.ones(keep) / keep, mode="valid")    # the mean drift of every window
        at = int(means.argmin())
        win = frames[at:at + keep]
        loop = win + win[::-1]
        s.idle.update(key=s.face, at=0,
                      packets=[_pack("idle", loop[i:i + per], None) for i in range(0, len(loop), per)])
        s.build, s.note = None, None
        logger.info(f"Idle loop made ({s.name}): {len(s.idle['packets'])} chunks from frame {at} of {len(frames)}, "
                    f"drift {means[at]:.1f} (whole render {d.mean():.1f}, worst {d.max():.1f})")
    return f, None


def _render(s, a16):
    """0.96 s of 16 kHz audio -> list of RGB uint8 frames, in the session's face and motion."""
    fh = P["fh"]
    with _gpu:
        _pipe.original_color_reference, _pipe.ref_img_latent, _pipe.latent_motion_frames = s.t
        s.dq = np.concatenate((s.dq[len(a16):], a16))
        emb = fh.get_audio_embedding(_pipe, s.dq, P["start"], P["end"])
        with contextlib.redirect_stdout(io.StringIO()):   # per-step timing prints
            video = fh.run_pipeline(_pipe, emb)
            if P.get("fp16") and not video.isfinite().all():
                _fp16_off()                               # this chunk again in bf16, from the motion it started with
                _pipe.original_color_reference, _pipe.ref_img_latent, _pipe.latent_motion_frames = s.t
                video = fh.run_pipeline(_pipe, emb)
        s.t = (s.t[0], s.t[1], _pipe.latent_motion_frames)
        return video[P["motion"]:].byte().cpu().numpy()   # the engine gives 0-255 floats: trimmed and cast on the GPU, a fifth of the copy


def _decode(data):
    """Any audio bytes -> (16 kHz float32, 24 kHz int16) mono."""
    import av
    outs = {}
    for rate, fmt in ((P["sr"], "flt"), (PLAY_SR, "s16")):
        res = av.AudioResampler(format=fmt, layout="mono", rate=rate)
        parts = []
        with av.open(io.BytesIO(data)) as c:
            for frame in c.decode(audio=0):
                for f in res.resample(frame):
                    parts.append(f.to_ndarray().reshape(-1))
            for f in res.resample(None):
                parts.append(f.to_ndarray().reshape(-1))
        outs[rate] = np.concatenate(parts) if parts else np.zeros(0)
    return outs[P["sr"]].astype(np.float32), outs[PLAY_SR].astype(np.int16)


def _queue_speech(s, data, mute=False, device="", taken=False, sid=""):
    a16, a24 = _decode(data)
    # mute: a device or her speakers play it, so browsers only mouth it (they get the packet with its
    # audio zeroed, as before). A board stream gets it whole, with the tag: the board named plays it,
    # the others mouth it (2026-11-03: a pendant's own replies were silent, its voice zeroed here and
    # core's own sentence skipped for the stream).
    tag = {"mute": bool(mute), "device": device or "", "taken": bool(taken),   # taken: a browser or a call plays it, so a board mouths it
           "sid": sid or ""}                                                   # sid: core's stream, so a tap on her face can stop it at the source
    with s.lock:
        p16 = np.concatenate((s.pending["a16"], a16))
        p24 = np.concatenate((s.pending["a24"], a24))
        while len(p16) >= P["n16"]:
            s.speech.put((p16[:P["n16"]], p24[:P["n24"]], tag))
            p16, p24 = p16[P["n16"]:], p24[P["n24"]:]
        s.pending.update(a16=p16, a24=p24, tag=tag)


def _next_chunk(s):
    """Speech if any is waiting (flushing a partial tail), else silence."""
    try:
        a16, a24, tag = s.speech.get_nowait()
        return "speech", a16, a24, tag
    except queue.Empty:
        pass
    with s.lock:
        if len(s.pending["a16"]):
            a16, a24 = _fit(s.pending["a16"], P["n16"]), _fit(s.pending["a24"], P["n24"])
            tag = s.pending["tag"]
            s.pending.update(a16=np.zeros(0, np.float32), a24=np.zeros(0, np.int16))
            return "speech", a16, a24, tag
    return "idle", np.zeros(P["n16"], np.float32), None, None


def _fit(a, n):
    """Exactly n samples: the tail cut or padded with silence (the two
    resamplers can end a sample apart, and a negative pad threw every turn)."""
    return np.pad(a[:n], (0, max(0, n - len(a))))


def _jpeg(f):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(f).save(b, "JPEG", quality=85)
    return b.getvalue()


def _pack(kind, frames, a24):
    return _packet(kind, [_jpeg(f) for f in frames], a24)


def _packet(kind, jpegs, a24, tag=None):
    audio = a24.tobytes() if a24 is not None else b""
    head = json.dumps({"kind": kind, "fps": P["fps"], "sr": PLAY_SR, "audio": len(audio),
                       "frames": [len(j) for j in jpegs],
                       **({"mute": tag["mute"], "device": tag["device"], "taken": tag["taken"], "sid": tag.get("sid", "")} if tag else {})}).encode()
    return struct.pack(">I", len(head)) + head + audio + b"".join(jpegs)


def _drain(s):
    """Every queued speech chunk, taken off the queue (the render loop may be taking them too)."""
    out = []
    while True:
        try:
            out.append(s.speech.get_nowait())
        except queue.Empty:
            return out


def _flush(s):
    """Stop: drop queued speech here and tell viewers to drop what they
    scheduled. The chunk being rendered right now is dropped too (gen)."""
    head = json.dumps({"kind": "flush"}).encode()
    with s.lock:
        s.gen += 1
        s.open = False
        _drain(s)
        s.pending.update(a16=np.zeros(0, np.float32), a24=np.zeros(0, np.int16))
        _broadcast(s, struct.pack(">I", len(head)) + head)


def _broadcast(s, pkt, quiet=None, hush=None):
    """To every viewer of the session. One that fell behind loses its oldest
    packet, never the newest, so it stays near now and a stop always lands.
    `quiet`: the same without her voice, for every viewer but s.voice (b"":
    nothing for them), so a second tab or window never plays her a beat apart.
    `hush`: the same with her voice zeroed, for s.voice when a device or her own
    speakers play it. A board stream (s.boards) always gets `pkt`, whole: the
    board named in its head plays the voice, the others mouth it."""
    with _viewers_lock:
        for q in s.viewers:
            if q in s.boards:
                p = pkt
            elif quiet is None or s.voice in (None, q):
                p = hush if hush is not None else pkt
            else:
                p = quiet
            if not p:
                continue
            try:
                q.put_nowait(p)
            except queue.Full:
                s.drops += 1
                with contextlib.suppress(queue.Empty, queue.Full):
                    q.get_nowait()
                    q.put_nowait(p)


_out = queue.Queue(maxsize=2)    # rendered chunks waiting for their JPEGs: the GPU starts the next meanwhile


def _packer():
    """Frames to JPEGs and out, in order, beside the render loop."""
    while True:
        s, gen, kind, frames, a24, pkt, tag = _out.get()
        quiet = hush = None
        try:
            if pkt is None:
                jpegs = [_jpeg(f) for f in frames]
                pkt = _packet(kind, jpegs, a24, tag)
                if a24 is not None:
                    quiet = _packet(kind, jpegs, None, tag) if jpegs else b""
                    if tag and tag["mute"]:
                        hush = _packet(kind, jpegs, np.zeros_like(a24), tag)
        except Exception:
            logger.exception("Packing failed")
            continue
        with s.lock:
            if gen == s.gen:                     # not stopped since it was rendered
                _broadcast(s, pkt, quiet, hush)


def _pick():
    """The session to serve next, of those watched and due (less than LEAD
    ahead of their own clock): her words first, then a calm loop (no GPU),
    then a live idle render. A live idle never holds up anyone's words."""
    now, best = time.monotonic(), None
    ahead = max(LEAD, 1.5 * P.get("render_s", 0))   # a talker keeps enough lead to sit out one other render
    with _viewers_lock:                     # a viewer leaving clears s.clock under this same lock
        for s in _sessions.values():
            if not (s.viewers and s.want):
                continue
            if s.clock is None:
                s.clock = [now, 0.0]
            lead = s.clock[1] - (now - s.clock[0])
            if lead < -2.0:                         # fell behind (GPU busy): rebase
                if s.run:
                    s.run["rebases"] += 1
                s.clock, lead = [now, 0.0], 0.0
            if lead > ahead:
                continue
            if s.held():
                if s.run and not s.run["waiting"]:
                    s.run["waits"], s.run["waiting"] = s.run["waits"] + 1, True
                continue
            s.lead = lead
            rank = 0 if s.talking() else 2 if _idle_mode(s) == "live" else 1
            if best is None or (rank, lead) < best[0]:
                best = ((rank, lead), s)
    return best and best[1]


def _loop():
    """The render loop, which never dies: a dead one would swallow every Sapphire's words."""
    while True:
        _watching.wait()
        BEAT[0] = time.monotonic()
        try:
            _turn()
        except Exception:
            logger.exception("Render loop")
            time.sleep(1)


_failing = [0]           # renders failed in a row, any session: past 6 it is the GPU, not one face


def _turn():
    """One chunk for the session most due: rendered here, packed and sent beside."""
    if _state["status"] != "ready":         # a viewer may come while she downloads: wait, never touch the model
        time.sleep(0.5)
        return
    s = _pick()
    if s is None:
        time.sleep(0.02)
        return
    try:
        _apply_face(s)
    except Exception:
        logger.exception(f"Face not taken ({s.name})")
        s.want = None                       # never retried in a loop; her next reply sends it again
        with s.lock:                        # words taken while the face was coming still play, without one
            for _, rest, tag in _drain(s):
                _out.put((s, s.gen, "speech", [], rest, None, tag))
        return
    if not s.face:                          # never animate the grey warm-up card
        time.sleep(0.05)
        return
    kind, a16, a24, tag = _next_chunk(s)
    gen, t = s.gen, time.perf_counter()
    try:
        pkt = None
        if kind == "idle" and s.run:
            _run_end(s)
        if kind == "idle" and _idle_mode(s) == "live":
            frames = _render(s, a16)            # she sits as the model has her; her next words go on from here
            _home(s)
            s.live, s.build = True, None
        elif kind == "idle":
            frames, pkt = _idle_step(s)
            s.live = False
        else:
            r = s.run = s.run or {"n": 0, "render": [], "lead": [], "waits": 0, "waiting": False,
                                  "resets": 0, "rebases": 0, "drops": s.drops}
            if not s.live:
                _reset_motion(s)
                s.live = True
                r["resets"] += 1
            s.build = None                      # her motion moved on: a calm loop half made starts again
            frames = _render(s, a16)
            r["n"] += 1
            r["render"].append(time.perf_counter() - t)
            r["lead"].append(s.lead)
            r["waiting"] = False
    except Exception as e:
        logger.exception(f"Render failed ({s.name})")
        if a24 is not None:
            _out.put((s, gen, kind, [], a24, None, tag))   # her words still reach whoever took them, without a face
        s.fails += 1
        _failing[0] += 1
        if s.fails >= 3:                    # this face won't render: her voice goes back to her own computer
            s.fails, s.want = 0, None
            for _, rest, tag in _drain(s):
                _out.put((s, s.gen, "speech", [], rest, None, tag))
        if _failing[0] >= 6:                # every face fails: the GPU is in trouble, say so
            _state.update(status="error", error=f"Rendering keeps failing ({e}). Her voice plays as usual; Retry starts the face again.")
        time.sleep(1)
        return
    s.fails = _failing[0] = 0
    _out.put((s, gen, kind, frames, a24, pkt, tag))
    c = s.clock                             # a viewer leaving may clear it meanwhile
    if c:
        c[1] += P["chunk_s"]


def _run_end(s):
    """One line per speech run, numbers only (never her words): enough to
    tell a slow GPU (render near chunk_s, lead below 0, rebases) from her
    words arriving late (waits) from a slow viewer (drops) from a reset in
    the middle of a reply (resets > 1)."""
    r, s.run = s.run, None
    if not r or not r["n"]:
        return
    rd, ld = r["render"], r["lead"]
    logger.info(f"speech run ({s.name}): {r['n']} chunks ({r['n'] * P['chunk_s']:.1f}s), "
                f"render {min(rd):.2f}/{sum(rd) / len(rd):.2f}/{max(rd):.2f}s for {P['chunk_s']:.2f}s, "
                f"lead {min(ld):+.2f}..{max(ld):+.2f}s, waits {r['waits']}, resets {r['resets']}, "
                f"rebases {r['rebases']}, drops {s.drops - r['drops']}")


def _viewer_log(s, data):
    """One line per reply from the browser that played it: what this server
    cannot see. Gaps: her voice paused for a late packet (GPU or network).
    Late: frames that missed their moment (a busy CPU or page). Out: the
    audio path's delay, which her mouth waits for."""
    try:
        v = {k: float(x) for k, x in json.loads(data).items() if isinstance(x, (int, float))}
    except (ValueError, AttributeError):
        return
    logger.info(f"viewer ({s.name}): {v.get('packets', 0):.0f} packets, gaps {v.get('gaps', 0):.0f} "
                f"({v.get('gap_s', 0):.2f}s), decode avg/max {v.get('wait_ms', 0):.0f}/{v.get('wait_max', 0):.0f} ms, "
                f"late frames {v.get('late', 0):.0f}, page stalls {v.get('stalls', 0):.0f}, "
                f"audio out {v.get('out', 0):.3f}s ({'the device clock' if v.get('clock') else 'reported'}), {v.get('sr', 0):.0f} Hz"
                + (f", audio paused {v['paused']:.0f}x" if v.get("paused") else ""))


def _inside(path):
    """A face only from inside Sapphire's folder (avatars, uploads): never any picture on the disk."""
    try:
        p = Path(path).resolve()
        return any(p.is_relative_to(r.resolve()) for r in (ROOT, ROOT / "user"))   # user/ may be a link elsewhere
    except (OSError, ValueError, TypeError):
        return False


def _face_bytes(s, data):
    """A face sent as an image (another computer's avatar): checked, re-saved
    as PNG under FACES/<session>/ by its hash, the oldest dropped past FACES_KEPT."""
    import hashlib
    from PIL import Image
    folder = FACES / s.name
    dest = folder / f"{hashlib.sha256(data).hexdigest()[:16]}.png"
    if not dest.exists():
        img = Image.open(io.BytesIO(data)).convert("RGB")
        folder.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        for old in sorted(folder.glob("*.png"), key=lambda f: f.stat().st_mtime)[:-FACES_KEPT]:
            old.unlink(missing_ok=True)         # never touched when sent again: its mtime is part of the face's key
    return str(dest)


class Handler(BaseHTTPRequestHandler):
    def _caller(self, post=False):
        """The session a request speaks for, or None (refused, and answered).
        Through the shared door: the computer its key names. Here on
        127.0.0.1: "local", for Sapphire's own calls only. The Host check
        stops a web page that renames itself to 127.0.0.1 (DNS rebinding)
        from reading her face; the header on writes stops a page's blind
        cross-site POST, which cannot set one without a preflight this server
        never answers."""
        if getattr(self.server, "door", None):
            name = worker_door.admit(self)
            if not name:
                self._json({"error": "unauthorized"}, 401)
                return None
            return _session(name, door=True)
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host in ("127.0.0.1", "localhost") and (not post or self.headers.get("X-FlashHead") == "1"):
            return _session("local")
        self._json({"error": "forbidden"}, 403)
        return None

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        s = self._caller()
        if not s:
            return
        url = urlsplit(self.path)
        if url.path == "/health":
            with _viewers_lock:
                viewers, listeners = len(s.viewers), sum(s.viewers.values())
            info = {**_state, "note": s.note or _state["note"], "face": bool(s.face),
                    "viewers": viewers, "listeners": listeners}
            if getattr(self.server, "door", None) and info["error"]:
                info["error"] = _private(info["error"])
            return self._json(info)
        if url.path != "/stream":
            return self._json({"error": "not found"}, 404)
        q = queue.Queue(maxsize=VIEWER_QUEUE)
        args = parse_qs(url.query)
        with _viewers_lock:
            s.viewers[q] = args.get("audio", ["1"])[0] != "0"
            if args.get("board", ["0"])[0] == "1":
                s.boards.add(q)                  # a board stream: the whole packet always, never the voice tab
            if s.viewers[q] and args.get("call", ["0"])[0] == "1":
                s.call = s.voice = q             # a call hears her: a browser opened mid-call never takes her voice from it
            elif s.viewers[q] and s.call is None and q not in s.boards:
                s.voice = q                      # the newest browser plays her voice; any other mouths it silently
            _watching.set()
        try:                                     # what the browser says its audio path adds (Bluetooth: 0.15-0.3 s)
            lat = f", audio out {float(parse_qs(url.query)['lat'][0]):.3f}s"
        except (KeyError, ValueError, IndexError):
            lat = ""
        logger.info(f"Viewer joined ({s.name}: {len(s.viewers)}{lat}"
                    + (", plays her voice" if s.voice is q and len(s.viewers) > 1 else "") + ")")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        door, checked, gone = getattr(self.server, "door", None), time.monotonic(), False
        try:
            while True:
                try:
                    pkt = q.get(timeout=5)
                except queue.Empty:
                    pkt = struct.pack(">I", 2) + b"{}"       # keepalive
                if door and time.monotonic() - checked > 5:  # removed, or given a new key: its stream ends here
                    checked = time.monotonic()
                    if worker_door.client_of(door, self.headers.get("Authorization")) != s.name:
                        gone = True
                        break
                self.wfile.write(pkt)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _viewers_lock:
                s.viewers.pop(q, None)
                s.boards.discard(q)
                if s.call is q:
                    s.call = None
                if s.voice is q:
                    s.voice = next((v for v, sound in s.viewers.items() if sound and v not in s.boards), None)
                last = not s.viewers
                if last:
                    s.clock = None
                if gone:
                    _sessions.pop(("door", s.name), None)   # its face, motion and words go with it
                if not any(x.viewers for x in _sessions.values()):
                    _watching.clear()
            if last:
                if s.run:
                    _run_end(s)
                with s.lock:                     # nobody to hear it: no old sentence waits for the next viewer
                    s.open = False
                    _drain(s)
                    s.pending.update(a16=np.zeros(0, np.float32), a24=np.zeros(0, np.int16))
            if gone:
                shutil.rmtree(FACES / s.name, ignore_errors=True)
                logger.info(f"{s.name} is no longer let in: its stream is closed and its faces deleted")
            logger.info(f"Viewer left ({s.name}: {len(s.viewers)})")

    def do_POST(self):
        s = self._caller(post=True)
        if not s:
            return
        data = self.rfile.read(max(0, min(int(self.headers.get("Content-Length", 0)), 16 * 2**20)))
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if url.path == "/face":
            try:
                if (self.headers.get("Content-Type") or "").startswith("image/"):
                    path = _face_bytes(s, data)
                elif getattr(self.server, "door", None):
                    return self._json({"error": "send the face as an image"}, 400)
                else:
                    path = json.loads(data).get("path")
                    if not _inside(path):
                        return self._json({"error": "a face must be inside Sapphire's folder"}, 400)
                s.want = path
                if query.get("idle", [""])[0] in ("loop", "live"):
                    s.mode = query["idle"][0]
                return self._json({"ok": True})
            except Exception as e:
                return self._json({"error": str(e)}, 400)
        if url.path == "/flush":
            _flush(s)
            if query.get("by"):                  # a tap on her face, not core's own stop: said beside the run's line
                logger.info(f"Stopped from the {query['by'][0][:24]} ({s.name})")
            return self._json({"ok": True})
        if url.path == "/viewer":
            _viewer_log(s, data)
            return self._json({"ok": True})
        if url.path == "/end":
            s.open = False                       # her last words go out now, then she idles
            return self._json({"ok": True})
        if url.path == "/speak":
            # A face still coming (want, not yet applied) takes her words too: the
            # render loop applies it first. Refusing them split her first reply
            # after Watch between this panel and her own speakers.
            why = ("not ready" if _state["status"] != "ready" else "no viewer" if not s.viewers
                   else "no face" if not s.want or not (s.face or os.path.exists(s.want))
                   else "stalled" if time.monotonic() - BEAT[0] > 15 else "")
            if why:                              # no face to give, or the loop stalled: she speaks at home
                return self._json({"taken": False, "why": why})
            try:
                take = query.get("take", ["0"])[0] == "1"
                taken = take and any(sound for v, sound in s.viewers.items() if v not in s.boards)   # a browser or a call plays it; a board stream is not a listener here (it mouths what others play)
                _queue_speech(s, data, mute=not taken, device=query.get("device", [""])[0], taken=taken,   # played elsewhere: a viewer playing it too was a late echo; a board stream still carries it to the device named
                              sid=query.get("sid", [""])[0][:64])
                s.open, s.heard = True, time.monotonic()
                return self._json({"taken": taken, "why": "" if taken else "no listener" if take else "a device plays it"})
            except Exception as e:
                logger.warning(f"Speak decode failed ({s.name}): {e}")
                return self._json({"taken": False, "error": str(e)})
        return self._json({"error": "not found"}, 404)

    def log_message(self, *args):
        pass


def _private(text):
    """An error as another computer may see it: without this computer's folders or user name."""
    text = str(text).replace(str(ROOT), "<sapphire>")
    return text.replace(str(Path.home()), "~")


def _open_door():
    """Setting `share`: other computers' Sapphires use this GPU through a TLS
    door, each with its own key (worker_door.py, a copy of core's), each its
    own session: her face, her motion, her viewers."""
    port = int(_START.get("share_port") or 5018)
    try:
        srv = worker_door.serve(ThreadingHTTPServer, Handler, "0.0.0.0", port, DOOR)
    except Exception as e:
        logger.error(f"Shared door not opened on port {port}: {e}")
        return
    threading.Thread(target=srv.serve_forever, daemon=True, name="flashhead-door").start()
    logger.info(f"Shared with other computers on port {port} (TLS, {len(worker_door.clients(DOOR))} let in)")


if __name__ == "__main__":
    threading.Thread(target=_load, daemon=True).start()
    threading.Thread(target=_loop, daemon=True).start()
    threading.Thread(target=_packer, daemon=True, name="flashhead-packer").start()
    if _START.get("share"):
        _open_door()
    logger.info(f"FlashHead server on 127.0.0.1:{PORT}, models in {MODELS}")
    class Local(ThreadingHTTPServer):
        allow_reuse_address = sys.platform != "win32"     # Windows lets a second copy share a taken port, silently
    Local(("127.0.0.1", PORT), Handler).serve_forever()
