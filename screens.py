"""Her screens: her face on every board whose picture door reads JPEG.

A frames-only viewer of the FlashHead server (?audio=0: it never makes the
server take her sound). `screens` setting: off, talking (from the start of a
reply until a few seconds after its last word), always (her idle loop too, at
a gentle rate). One pusher per board keeps only the newest frame, so a slow
board drops frames instead of falling behind. A board stays a speaker first:
nothing is sent to one that does not say `screen.jpeg` in its /health.
"""
import collections
import struct
import json
import io
import logging
import threading
import time

from plugins.flashhead import faces

logger = logging.getLogger(__name__)

TALK_FPS, IDLE_FPS = 25, 2      # of the server's 25: every frame when a board's beat asks (glass.takes caps it); IDLE_FPS is the old frame path's idle (a whole frame each), a one-stream board idles at its beat
HOLD_S = 4                      # talking: stay on this long after her last word
AHEAD_S = 0.15                  # the browser starts a packet's sound this far after it lands

_stop = threading.Event()
_wake = threading.Event()
_connected = threading.Event()
_again = threading.Event()      # the server moved (client mode, a new worker): reconnect
_thread = None


def start(plugin_loader, settings):
    global _thread
    _stop.clear()
    _thread = threading.Thread(target=_run, daemon=True, name="flashhead-screens")
    _thread.start()


def stop():
    _stop.set()
    _wake.set()


def on_settings_saved(settings):
    """Core calls this on a save: a stream to where the server was is dropped."""
    _again.set()
    _wake.set()


def wake():
    """A reply starts: be connected before its first words reach the server."""
    if _mode() == "off":
        return
    _wake.set()
    _connected.wait(0.3)


def _mode():
    return faces.settings().get("screens") or "talking"


class _Playout:
    """Frames due on one clock, given to every board's pusher when their time comes. The
    reader thread puts, this one sleeps: neither holds the other up. Behind by a frame or
    more, it skips to the newest frame that is due (steady over eager), and the queue never
    holds more than two seconds ahead of now, so the lag stays bounded."""

    def __init__(self, pushers):
        self.pushers = pushers
        self.items = collections.deque()
        self.cv = threading.Condition()
        self.alive = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="flashhead-playout")
        self.thread.start()

    def put(self, due, frame):
        with self.cv:
            while self.items and due - time.monotonic() > 2.0:
                self.items.popleft()
            self.items.append((due, frame))
            self.cv.notify()

    def close(self):
        with self.cv:
            self.alive = False
            self.items.clear()
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                while self.alive and not self.items:
                    self.cv.wait()
                if not self.alive:
                    return
                wait = self.items[0][0] - time.monotonic()
                if wait > 0:
                    self.cv.wait(wait)                      # a put meanwhile is looked at again
                    continue
                due, frame = self.items.popleft()
                while self.items and self.items[0][0] <= time.monotonic():
                    due, frame = self.items.popleft()      # behind: the newest due frame wins
            for p in self.pushers:
                p.give(frame)


class _AvBuilder:
    """One board that takes her voice and her face as ONE stream (firmware/pendant/main/avatar.h):
    a slice's audio and the frames picked within it go as one packet, each frame as a key (the
    whole face) or as the region that changed since the last one sent, snapped to 16 px. The
    board paints each frame when its audio reaches the frame's moment: one clock, no drift."""

    KEY_EVERY_S = 2.0                # a whole face at least this often: a lost region never lingers
    KEY_SHARE = 0.6                  # a region this big is a key instead
    THRESHOLD = 24                   # a pixel changed: more than this in any channel (of 255)

    def __init__(self, board):
        self.id, self.side = board
        want = int(faces.settings().get("face_px") or 0)
        if 64 <= want < self.side:
            self.side = want // 16 * 16
        self.last = None             # the face as the board has it
        self.key_at = 0.0
        self.sent = self.keys = self.regions = self.still = 0
        try:
            from core.devices import glass
            glass.av_on_stop(self.id, self.stop_at_source)   # a stop at the board reaches the face server's queue
        except Exception as e:
            logger.info(f"[flashhead] {self.id}: stop not registered ({e})")

    def _encode(self, img):
        b = io.BytesIO()
        img.save(b, "JPEG", quality=80)
        return b.getvalue()

    def _frame(self, t_ms, jpeg, force_key=False):
        from PIL import Image, ImageChops
        img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((self.side, self.side))
        now = time.monotonic()
        key = force_key or self.last is None or now - self.key_at > self.KEY_EVERY_S
        x1 = y1 = x2 = y2 = 0
        if not key:
            diff = ImageChops.difference(img, self.last).convert("L").point(lambda v: 255 if v > self.THRESHOLD else 0)
            box = diff.getbbox()
            if box is None:
                self.still += 1
                return None                                  # nothing moved: the board keeps what it has
            x1, y1 = box[0] // 16 * 16, box[1] // 16 * 16
            x2, y2 = min(self.side, -(-box[2] // 16) * 16), min(self.side, -(-box[3] // 16) * 16)
            if (x2 - x1) * (y2 - y1) >= self.KEY_SHARE * self.side * self.side:
                key = True
        if key:
            self.last = img
            self.key_at = now
            self.keys += 1
            data = self._encode(img)
            return {"t": t_ms, "x": 0, "y": 0, "w": self.side, "h": self.side, "n": len(data), "key": True}, data
        region = img.crop((x1, y1, x2, y2))
        self.last.paste(region, (x1, y1))
        self.regions += 1
        data = self._encode(region)
        return {"t": t_ms, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "n": len(data), "key": False}, data

    def flush(self):
        """The face server flushed (she was stopped): what core holds for the board goes, the board drops its queue."""
        from core.devices import glass
        self.last = None                     # the next frame is a key: the board's face may be anywhere after the clear
        glass.av_clear(self.id, upstream=False)

    def stop_at_source(self):
        """Core says the board's person stopped her: the face server drops the rest of her reply (its flush
        comes back down the stream and clears the board through flush())."""
        try:
            faces.http().post(f"{faces.base_url()}/flush", timeout=2)
        except Exception as e:
            logger.info(f"[flashhead] {self.id}: the face server not flushed on a stop ({e})")

    def send(self, head, audio, picked):
        from core.devices import glass
        if audio and (head.get("taken") or head.get("device") != self.id):
            audio = b""                          # her voice plays in ONE place: a browser or a call that took it, the board whose reply it is, else her own speakers; every other face mouths it
        voice, ms, force = bool(audio), int(head.get("ms") or 0), False
        if not voice and not glass.av_shown(self.id):
            # Her tile is not up, or the screen rests: a key alone every KEY_EVERY_S keeps the board's face
            # fresh and asks again whether she is shown; nothing else crosses the air (0.3.0).
            if time.monotonic() - self.key_at < self.KEY_EVERY_S or not picked:
                return
            picked, force, ms = [(0, picked[-1][1])], True, int(self.KEY_EVERY_S * 1000)
        frames, bodies = [], []
        for t_ms, jpeg in picked:
            try:
                out = self._frame(t_ms, jpeg, force)
            except Exception as e:
                logger.info(f"[flashhead] {self.id}: a frame not built ({e})")
                continue
            if out:
                frames.append(out[0]); bodies.append(out[1])
        meta = json.dumps({"kind": head["kind"], "sr": head.get("sr", 24000), "audio": len(audio), "ms": ms, "frames": frames}).encode()
        if (4 + len(meta)) % 4:
            meta += b" " * (4 - (4 + len(meta)) % 4)           # the audio's samples land aligned on the board
        packet = struct.pack(">I", len(meta)) + meta + audio + b"".join(bodies)
        try:
            glass.av_packet(self.id, packet, voice=voice)
            self.sent += 1
        except Exception as e:
            if self.sent % 50 == 0:
                logger.info(f"[flashhead] {self.id}: packet not taken ({e})")

    def close(self):
        try:
            from core.devices import glass
            glass.av_on_stop(self.id, None)
        except Exception:
            pass
        logger.info(f"[flashhead] {self.id}: {self.sent} packets, {self.keys} keys, {self.regions} regions, {self.still} still")


class _Pusher:
    """One board: the newest frame wins, sent one at a time."""

    def __init__(self, board):
        self.id, self.side = board
        want = int(faces.settings().get("face_px") or 0)          # a smaller face than the glass's square: every link of the board's chain scales with pixels
        if 64 <= want < self.side:
            self.side = want // 16 * 16
        self.frame, self.lock, self.ready = None, threading.Lock(), threading.Event()
        self.fails = 0
        threading.Thread(target=self._run, daemon=True, name=f"flashhead-{self.id}").start()

    def give(self, jpeg):
        with self.lock:
            self.frame = jpeg
        self.ready.set()

    def close(self):
        self.give(None)

    def _run(self):
        from PIL import Image
        from core.devices import glass
        while True:
            self.ready.wait()
            self.ready.clear()
            with self.lock:
                jpeg, self.frame = self.frame, None
            if jpeg is None:
                return
            try:
                b = io.BytesIO()
                Image.open(io.BytesIO(jpeg)).resize((self.side, self.side)).save(b, "JPEG", quality=80)
                glass.face_frame(self.id, b.getvalue(), seconds=3)
                self.fails = 0
            except Exception as e:
                self.fails += 1
                if self.fails in (1, 20):
                    logger.info(f"[flashhead] {self.id}: frame not taken ({e})")


def _run():
    while not _stop.is_set():
        mode = _mode()
        if mode != "always":
            _wake.wait(5)
            if not _wake.is_set() or _stop.is_set() or _mode() == "off":
                continue
        _wake.clear()
        try:
            from core.devices import glass
            boards = glass.face_boards()
        except Exception as e:
            logger.warning(f"[flashhead] boards unknown: {e}")
            boards = []
        if not boards:
            _connected.set()                     # nothing to wait for
            time.sleep(1)
            _connected.clear()
            continue
        try:
            _watch(boards, mode)
        except Exception as e:
            logger.warning(f"[flashhead] screens stream ended: {e}")
            faces.forget()
            time.sleep(2)


def _watch(boards, mode):
    import json
    import struct
    try:
        from core.devices import glass
        av_ids = {d for d, _ in glass.av_boards()}
    except Exception:
        av_ids = set()
    builders = [_AvBuilder(b) for b in boards if b[0] in av_ids]     # her voice and face as one stream (the pendant from 0.2.0)
    pushers = [_Pusher(b) for b in boards if b[0] not in av_ids]     # the old way: frames alone, paced by core
    playout = _Playout(pushers)
    clock = None
    grid = None
    last_kind = None
    try:
        from core.devices import glass
        talk_fps = max(1, min(TALK_FPS, min(glass.takes(b[0])["fps"] for b in boards)))   # the slowest board's beat, 1:1: no frame doubled or skipped
    except Exception:
        talk_fps = TALK_FPS
    last_word, checked = time.monotonic(), time.monotonic()
    boards_checked = time.monotonic()
    try:
        path, _ = faces.current()
        if path:                                 # the server renders nothing without a face: after its restart, her idle
            url, body, headers = faces.face_body(path)   # existed only once she spoke or someone pressed Watch (camera.py does this too)
            faces.http().post(f"{faces.base_url()}{url}", data=body, headers=headers, timeout=5)
    except Exception as e:
        logger.info(f"[flashhead] her face not sent before her stream ({e})")
    try:
        with faces.http().get(f"{faces.base_url()}/stream?audio={1 if builders else 0}&board={1 if builders else 0}", stream=True, timeout=(5, 30)) as r:   # board=1: the whole packet always, her voice tagged with its device
            _connected.set()
            raw = r.raw

            def read(n):
                b = b""
                while len(b) < n:
                    got = raw.read(n - len(b))
                    if not got:
                        raise EOFError("the face server closed the stream")
                    b += got
                return b

            logger.info(f"[flashhead] her face on {', '.join(b[0] for b in boards)} ({mode})")
            _again.clear()
            while not _stop.is_set() and not _again.is_set():
                now = time.monotonic()
                if mode == "talking" and now - last_word > HOLD_S and not _wake.is_set():
                    return
                if now - checked > 5:
                    checked = now
                    if _mode() != mode:
                        return
                if now - boards_checked > 30:
                    boards_checked = now
                    # The boards that take her as one stream may have changed since this stream was
                    # opened: a board rebooting as it opened was listed as none (the old frame path,
                    # no packets until the stream happened to restart: a blank tile for as long as
                    # that took, 2026-11-03). Then this stream ends and _run opens it again. Every
                    # 30 s: while a board is down, each look is a health call that can wait its timeout.
                    try:
                        if {d for d, _ in glass.av_boards()} != {b.id for b in builders}:
                            logger.info("[flashhead] her one-stream boards changed: the stream starts over")
                            return
                    except Exception:
                        pass
                head = json.loads(read(struct.unpack(">I", read(4))[0]))
                if head.get("kind") == "flush":
                    last_word = now - HOLD_S + 1
                    for b in builders:
                        b.flush()                    # she was stopped: what core holds for the board goes, the board drops its queue
                    continue
                body = read(head.get("audio", 0) + sum(head.get("frames", [])))   # always the whole packet: the stream stays in step
                if not head.get("frames"):
                    continue
                if head["kind"] == "speech":
                    last_word = now
                    _wake.clear()                    # the reply that woke us is here
                elif mode == "talking":
                    continue                     # between her lines the last frame holds
                fps = talk_fps if head["kind"] == "speech" or builders else IDLE_FPS   # a one-stream board idles at its beat: a region costs by motion, not by rate
                at = head["audio"]
                frames = []
                for n in head["frames"]:
                    frames.append(body[at:at + n])
                    at += n
                # One clock across packets: a packet's frames are due where the last packet left
                # off, not at its own arrival plus AHEAD_S. That restarted the clock every packet,
                # and the playout, done here in the reading thread, took longer than a packet: a
                # hitch every second and a lag that grew through a reply. The reader never sleeps
                # now; `playout` gives the frames when they are due (2026-11-02).
                # The clock is anchored ONCE per reply, at its first speech packet, and runs on: it was
                # restarted at every packet that came half a second late, and the face server's slices
                # come unevenly while it renders, so each hiccup pushed her face later while her voice
                # (a plain WAV on the board) kept real time: she talked on 1.5-2 s after the audio ended
                # (Krem, 2026-11-03). Now a late packet's frames are due in the past and the playout
                # drops them for the newest: her face skips to stay with her voice, and ends with it.
                if clock is None or head["kind"] != last_kind or now > clock + 3.0:
                    clock = now + AHEAD_S                       # a fresh start: a new reply, or idle after speech
                    grid = clock
                last_kind = head["kind"]
                # Frames picked by TIME on a grid of 1/fps, not every nth of the server's 25: 25/7
                # rounded to every 4th, 6.25 a second at a beat of 7 (2026-11-03). The grid runs
                # on across packets; each tick takes the frame nearest it.
                span = len(frames) / head["fps"]
                picked = []                                      # (ms into the slice, frame)
                while grid < clock + span:
                    i = min(len(frames) - 1, int((grid - clock) * head["fps"] + 0.5))
                    playout.put(grid, frames[i])
                    picked.append((int((grid - clock) * 1000), frames[i]))
                    grid += 1.0 / fps
                clock += span
                if builders:
                    head["ms"] = int(span * 1000)
                    audio = body[:head["audio"]] if head["kind"] == "speech" else b""   # her idle goes without its silence: the board paces it on the wall
                    for b in builders:
                        b.send(head, audio, picked)
    finally:
        _connected.clear()
        playout.close()
        for b in builders:
            b.close()
        for p in pushers:
            p.close()
        logger.info("[flashhead] her screens rest")
