"""Her voice drives the face.

stream start: pick the face for her current mood and wake her screens. Each
chunk: handed to the server so the face mouths it; the audio is taken (not
played here) only on a lane that plays here and while a viewer plays sound,
so a closed panel never mutes her and a board's voice is never stolen.
Stop: flush what's queued so the face stops when she does (a tap on her face in a
viewer flushes the server itself and stops the pump by the stream id each packet
carries: routes/api.py stop); a reply that ends tells the server so (/end), so a
gap between her sentences holds the face
instead of idling it. One line per reply says where her words went.
Only while the server was last seen ready: a stopped or loading service costs
her voice nothing (looked at again at most every 10 s). The server is this
computer's, or in client mode a worker elsewhere (faces.http()).
"""
import logging
import time

import requests
from urllib.parse import quote

from plugins.flashhead import faces

logger = logging.getLogger(__name__)


_ready = {"ok": False, "at": 0.0}
RECHECK_S = 10
_tally = {}              # stream_id -> {"face": n, "here": {why: n}, "slow": ms}: one log line per reply


def _post(path, timeout=2, **kw):
    return faces.http().post(f"{faces.base_url()}{path}", timeout=timeout, **kw)


def _server_ready():
    now = time.monotonic()
    if _ready["ok"] or now - _ready["at"] > RECHECK_S:
        try:
            ok = faces.http().get(f"{faces.base_url()}/health", timeout=0.5).json().get("status") == "ready"
        except Exception:
            ok = False
            faces.forget()
        _ready.update(ok=ok, at=now)
    return _ready["ok"]


def tts_stream_start(event):
    if not _server_ready():
        return
    try:
        from plugins.flashhead import screens
        screens.wake()
        from core.devices import glass
        device = (event.metadata or {}).get("device") or ""
        for d, _ in glass.av_boards():
            if device and d == device:           # a satellite's own reply: that board's stream carries its voice, so core's sentence to it is skipped from the first
                glass.av_mark(d)
    except Exception as e:
        logger.debug(f"[flashhead] screens not woken: {e}")
    try:
        path, _ = faces.current(chat=getattr(event, "chat_name", None))
        if path:
            url, body, headers = faces.face_body(path)
            _post(url, timeout=3, data=body, headers=headers)
    except Exception as e:
        logger.debug(f"[flashhead] face pick skipped: {e}")


def tts_chunk_audio(event):
    meta = event.metadata or {}
    audio = meta.get("audio") or {}
    data = audio.get("audio_bytes")
    if not data or not _ready["ok"]:
        return
    if len(_tally) > 32:                         # a stream that never ended: forget, never grow
        _tally.clear()
    t = _tally.setdefault(meta.get("stream_id"), {"face": 0, "here": {}, "slow": 0})
    here = meta.get("plays_here", True)      # False: it is a device's voice, never ours to take
    device = meta.get("device") or ""        # whose: the server hands it whole to the board streams (the owner plays it) and hushed to browsers
    t0 = time.monotonic()
    try:
        sid = quote(str(meta.get("stream_id") or ""))   # core's stream: a tap on her face stops it at the source (/api/tts/stop)
        r = _post(f"/speak?take={int(here)}&device={quote(device)}&sid={sid}", data=data,
                  headers={"Content-Type": audio.get("content_type", "audio/ogg")})
        ans = r.json() if r.ok else {"why": f"HTTP {r.status_code}"}
        event.skip_tts = bool(here and ans.get("taken"))
        why = "" if event.skip_tts else ans.get("why") or "refused"
    except requests.exceptions.ReadTimeout:
        # It got there and is (almost surely) queued: playing it here too was
        # the double voice, and dropping the server for the rest of the reply
        # sent every later sentence home without a face (Teri, 2026-10-23).
        event.skip_tts, why = bool(here), ""
        logger.warning(f"[flashhead] /speak slow (over {time.monotonic() - t0:.1f}s): left to the face")
    except Exception as e:
        _ready["ok"] = False                     # never reached it: gone mid-reply, the next reply looks again
        faces.forget()
        why = type(e).__name__
        logger.warning(f"[flashhead] server not taking her voice ({why}): it plays here")
    t["slow"] = max(t["slow"], int((time.monotonic() - t0) * 1000))
    if event.skip_tts:
        t["face"] += 1
    else:
        t["here"][why] = t["here"].get(why, 0) + 1


def tts_stream_end(event):
    meta = event.metadata or {}
    t = _tally.pop(meta.get("stream_id"), None)
    if t and (t["face"] or set(t["here"]) - {"no viewer", "a device plays it"}):   # the panel closed: nothing to say
        here = ", ".join(f"{n} {w}" for w, n in t["here"].items())
        logger.info(f"[flashhead] reply: {t['face']} to the face, {sum(t['here'].values())} played here"
                    + (f" ({here})" if here else "") + f", slowest /speak {t['slow']} ms")
    if not _ready["ok"]:
        return
    try:
        _post("/flush" if meta.get("interrupted") else "/end", timeout=1)
    except Exception:
        pass
