#!/usr/bin/env python3
"""FlashHead viewer — her live talking face from a Sapphire on your network,
in its own window (a Raspberry Pi on a screen, a spare laptop).

The stream is packets of (header, PCM, JPEG frames) for the same 0.96 s, as
the browser player gets them. Each is decoded whole, then its voice and
frames are placed together on the speaker's clock and drawn where the ear
is, so the mouth can't drift from the voice (web/player.js, in Python).

    python flashhead-viewer.py --host 192.168.1.50 --key <API key> [--fullscreen] [--mute]

The key: Sapphire's Settings > System > API Keys. Esc or Q quits, F toggles fullscreen.
A click on her face (or Space) while she speaks stops her: here, at the server, at its source.
"""
import argparse
import hashlib
import json
import os
import ssl
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import sounddevice as sd
import urllib3
from requests.adapters import HTTPAdapter
from PySide6.QtCore import QObject, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget

LEAD = 0.15             # how far ahead of now a packet is placed (the browser's same margin)
SR = 24000              # what the server sends (server.py PLAY_SR)
PINS = Path.home() / ".config" / "flashhead-viewer" / "pins.json"


# ── the clock ────────────────────────────────────────────────────────────────

class Speaker:
    """Her voice, on a timeline of samples. now() is what's been handed to
    the device; heard() is what is coming out of it, read from the device's
    own DAC timestamp (else the latency it reports)."""

    def __init__(self, device):
        self.lock = threading.Lock()
        self.chunks = deque()
        self.pos = 0                    # samples handed to the device so far
        self.end = 0                    # where the last queued sample ends
        self.mark = (0, time.monotonic())
        self.latency = 0.1
        self.stream = sd.OutputStream(samplerate=SR, channels=1, dtype="int16",
                                      device=device, callback=self._fill)
        self.latency = self.stream.latency
        self.stream.start()

    def _fill(self, out, frames, t, status):
        now = time.monotonic()
        dac = now + (t.outputBufferDacTime - t.currentTime if t.outputBufferDacTime and t.currentTime
                     else self.latency)
        with self.lock:
            self.mark = (self.pos, dac)     # this buffer's first sample is heard at dac
            n = 0
            while n < frames and self.chunks:
                c = self.chunks[0]
                k = min(len(c), frames - n)
                out[n:n + k, 0] = c[:k]
                if k == len(c):
                    self.chunks.popleft()
                else:
                    self.chunks[0] = c[k:]
                n += k
            out[n:, 0] = 0
            self.pos += frames

    def now(self):
        return self.pos / SR

    def heard(self):
        pos, dac = self.mark
        return pos / SR + time.monotonic() - dac

    def play(self, at, pcm):
        with self.lock:
            self.end = max(self.end, self.pos)
            start = int(round(at * SR))
            if start > self.end:            # a pause before it: silence keeps her on time
                self.chunks.append(np.zeros(start - self.end, np.int16))
                self.end = start
            self.chunks.append(pcm)
            self.end += len(pcm)

    def flush(self):
        with self.lock:
            self.chunks.clear()
            self.end = self.pos

    def close(self):
        self.stream.close()


class Silent:
    """--mute: no voice here, the wall clock keeps the frames' time."""
    def __init__(self):
        self.t0 = time.monotonic()
        self.latency = 0.0

    def now(self):
        return time.monotonic() - self.t0

    heard = now

    def play(self, at, pcm):
        pass

    def flush(self):
        pass

    def close(self):
        pass


# ── the door ─────────────────────────────────────────────────────────────────

class Pinned(HTTPAdapter):
    """Sapphire's certificate is her own (self-signed, kept 10 years): the one
    seen first is trusted, a different one never is."""
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        super().__init__()

    def init_poolmanager(self, *a, **kw):
        super().init_poolmanager(*a, assert_fingerprint=self.fingerprint, **kw)


class Untrusted(Exception):
    pass


def pin(host, port):
    pem = ssl.get_server_certificate((host, port), timeout=10)
    fp = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()
    try:
        pins = json.loads(PINS.read_text())
    except (OSError, ValueError):
        pins = {}
    key = f"{host}:{port}"
    if key not in pins:
        PINS.parent.mkdir(parents=True, exist_ok=True)
        PINS.write_text(json.dumps({**pins, key: fp}, indent=1))
        print(f"Trusting {key}'s certificate from now on ({fp[:16]}...)")
    elif pins[key] != fp:
        raise Untrusted(f"{key} shows a different certificate than the first time. If you reinstalled "
                        f"Sapphire there, remove its line from {PINS} and start again.")
    return fp


def session(origin, key):
    """Pinned before the key is ever sent: raises while Sapphire can't be reached."""
    s = requests.Session()
    u = urlparse(origin)
    if u.scheme == "https":
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        s.verify = False                    # no authority signs her cert; the pin checks it instead
        s.mount(origin, Pinned(pin(u.hostname, u.port or 443)))
    s.headers["Authorization"] = f"Bearer {key}"
    return s


def words(r):
    try:
        why = r.json()
        why = why.get("error") or why.get("detail")
    except ValueError:
        why = None
    if r.status_code == 401:
        return "Sapphire refused the API key (Settings > System > API Keys)"
    if r.status_code == 404:
        return "FlashHead isn't enabled on that Sapphire"
    return why or f"HTTP {r.status_code}"


def read_exact(raw, n):
    out = bytearray()
    while len(out) < n:
        b = raw.read(n - len(out))
        if not b:
            raise ConnectionError("the stream ended")
        out += b
    return bytes(out)


# ── the player ───────────────────────────────────────────────────────────────

class Player(QObject):
    status = Signal(str)
    still = Signal(QImage)

    def __init__(self, origin, key, clock, mute):
        super().__init__()
        self.origin, self.key, self.clock, self.mute = origin, key, clock, mute
        self.base = f"{origin}/api/plugin/flashhead"
        self.http = None
        self.lock = threading.Lock()
        self.frames = deque()               # (t, QImage) waiting for their moment
        self.next = 0.0
        self.side = 512                     # the face's size on screen, frames are scaled to it here
        self.talk_end = 0.0                 # until when her words are heard, and the stream they are (a click stops it)
        self.sid = ""

    def run(self):
        delay = 2
        while True:
            self.status.emit("connecting...")
            try:
                if not self.http:
                    self.http = session(self.origin, self.key)
                    r = self.http.get(f"{self.base}/face", timeout=10)
                    if r.ok:
                        self.still.emit(QImage.fromData(r.content))
                url = f"{self.base}/stream?lat={self.clock.latency:.3f}" + ("&audio=0" if self.mute else "")
                with self.http.get(url, stream=True, timeout=(10, 30)) as r:
                    if r.status_code != 200:
                        raise ConnectionError(words(r))
                    self.status.emit("")
                    delay = 2
                    self.read(r.raw)
            except Untrusted as e:
                return self.status.emit(str(e))
            except Exception as e:
                why = e.args[0] if isinstance(e, ConnectionError) and e.args and isinstance(e.args[0], str) \
                    else f"can't reach Sapphire ({type(e).__name__})"
                self.status.emit(f"{why}, trying again in {delay} s")
            self.flush()
            time.sleep(delay)
            delay = min(delay * 2, 30)

    def read(self, raw):
        while True:
            hl = struct.unpack(">I", read_exact(raw, 4))[0]
            head = json.loads(read_exact(raw, hl))
            if head.get("kind") == "flush":
                self.flush()
                continue
            if not head.get("frames"):
                continue
            body = read_exact(raw, head["audio"] + sum(head["frames"]))
            self.place(head, body)

    def place(self, head, body):
        # frames decoded BEFORE her voice is placed: a busy CPU pauses her whole, never splits her
        side, p, imgs = self.side, head["audio"], []
        for n in head["frames"]:
            img = QImage.fromData(body[p:p + n], "JPEG")
            p += n
            if not img.isNull():
                imgs.append(img if img.width() == side else
                            img.scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        fps = head["fps"]
        dur = len(head["frames"]) / fps
        t = max(self.next, self.clock.now() + LEAD)
        if head["audio"]:
            pcm = np.frombuffer(body[:head["audio"]], np.int16)
            if head["sr"] != SR:
                pcm = np.interp(np.arange(0, len(pcm), head["sr"] / SR), np.arange(len(pcm)), pcm).astype(np.int16)
            self.clock.play(t, pcm)
        self.next = t + dur
        if head.get("kind") == "speech":
            self.talk_end, self.sid = t + dur, head.get("sid") or ""
        with self.lock:
            self.frames.extend((t + i / fps, img) for i, img in enumerate(imgs))

    def flush(self):                        # she was stopped: cut what's scheduled
        self.talk_end = 0.0
        self.clock.flush()
        now = self.clock.heard()
        with self.lock:
            self.frames = deque(f for f in self.frames if f[0] <= now)
        self.next = 0.0

    def due(self):
        """The newest frame whose moment has come; older ones are skipped."""
        now, show = self.clock.heard(), None
        with self.lock:
            while self.frames and self.frames[0][0] <= now:
                show = self.frames.popleft()[1]
        return show

    def talking(self):
        return self.clock.heard() < self.talk_end

    def hush(self):
        """Her face was clicked: her voice stops here now, then at the server (the rest of
        the reply, every viewer) and at its source (that pump, by the stream id each packet
        carries, so the next sentence never starts her again). After a turn has ended only
        the server's cut is left."""
        sid, http = self.sid, self.http
        self.flush()
        if not http:
            return

        def go():
            try:
                http.post(f"{self.base}/stop", timeout=3)
                if sid:
                    http.post(f"{self.origin}/api/tts/stop", json={"stream_id": sid}, timeout=3)
            except Exception:
                pass
        threading.Thread(target=go, daemon=True).start()


class Face(QWidget):
    def __init__(self, player, fullscreen):
        super().__init__()
        self.player = player
        self.image = None
        self.still = None
        self.talk = False
        self.status = "connecting..."
        self.setWindowTitle("FlashHead")
        self.resize(640, 640)
        self.setStyleSheet("background: black;")
        player.status.connect(self.on_status)
        player.still.connect(self.on_still)
        self.timer = QTimer(self, timeout=self.tick, interval=10)
        self.timer.start()
        self.full(fullscreen)

    def full(self, on):
        self.showFullScreen() if on else self.showNormal()
        self.setCursor(Qt.BlankCursor if on else Qt.ArrowCursor)

    def on_status(self, text):
        self.status = text
        if text:
            self.image = None               # not live: her still shows
        self.update()

    def on_still(self, img):
        self.still = img
        self.update()

    def tick(self):
        img = self.player.due()
        talk = self.player.talking()
        if img is not None:
            self.image = img
        if img is not None or talk != self.talk:
            self.talk = talk
            self.update()

    def resizeEvent(self, e):
        self.player.side = round(min(self.width(), self.height()) * self.devicePixelRatioF())

    def paintEvent(self, e):
        g = QPainter(self)
        g.fillRect(self.rect(), Qt.black)
        side = min(self.width(), self.height())
        box = QRect((self.width() - side) // 2, (self.height() - side) // 2, side, side)
        img = self.image or self.still
        if img is not None:
            g.setRenderHint(QPainter.SmoothPixmapTransform)
            g.setOpacity(1.0 if self.image is not None else 0.5)
            g.drawImage(box, img)
            g.setOpacity(1.0)
        if self.talk:                       # a click on her face stops her
            g.fillRect(QRect(box.right() - side // 12, box.top() + side // 24, side // 24, side // 24), QColor(255, 255, 255, 210))
        if self.status:
            g.setPen(QColor(200, 200, 200))
            g.setFont(QFont("sans", max(10, side // 40)))
            g.drawText(self.rect().adjusted(16, 0, -16, -16), Qt.AlignHCenter | Qt.AlignBottom | Qt.TextWordWrap,
                       self.status)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.close()
        elif e.key() in (Qt.Key_F, Qt.Key_F11):
            self.full(not self.isFullScreen())
        elif e.key() == Qt.Key_Space and self.player.talking():
            self.player.hush()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.player.talking():
            self.player.hush()

    def mouseDoubleClickEvent(self, e):
        self.full(not self.isFullScreen())


def main():
    ap = argparse.ArgumentParser(description="Her live FlashHead face from a Sapphire on your network.")
    ap.add_argument("--host", default=os.environ.get("FLASHHEAD_HOST"),
                    help="Sapphire's address, or a full URL like https://host:8073 (env FLASHHEAD_HOST)")
    ap.add_argument("--port", type=int, default=8073, help="Sapphire's web port (default 8073)")
    ap.add_argument("--key", default=os.environ.get("FLASHHEAD_KEY"),
                    help="an API key from Settings > System > API Keys (env FLASHHEAD_KEY keeps it out of ps)")
    ap.add_argument("--fullscreen", action="store_true", help="start fullscreen (F toggles)")
    ap.add_argument("--mute", action="store_true", help="her face only: her voice stays where it plays now")
    ap.add_argument("--device", default=os.environ.get("FLASHHEAD_DEVICE"),
                    help="sound output, a number or part of a name from --list-devices")
    ap.add_argument("--list-devices", action="store_true", help="list sound outputs and quit")
    a = ap.parse_args()

    if a.list_devices:
        print(sd.query_devices())
        return
    if not a.host or not a.key:
        ap.error("--host and --key are needed (or FLASHHEAD_HOST and FLASHHEAD_KEY)")
    origin = (a.host if "://" in a.host else f"https://{a.host}:{a.port}").rstrip("/")
    device = int(a.device) if a.device and a.device.isdigit() else a.device

    app = QApplication(sys.argv)
    try:
        clock = Silent() if a.mute else Speaker(device)
    except Exception as e:
        sys.exit(f"No sound output ({e}). Pick one from --list-devices, or --mute.")
    player = Player(origin, a.key, clock, a.mute)
    face = Face(player, a.fullscreen)
    threading.Thread(target=player.run, daemon=True).start()
    code = app.exec()
    clock.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
