"""Her screens: her face on every board whose picture door reads JPEG.

A frames-only viewer of the FlashHead server (?audio=0: it never makes the
server take her sound). `screens` setting: off, talking (from the start of a
reply until a few seconds after its last word), always (her idle loop too, at
a gentle rate). One pusher per board keeps only the newest frame, so a slow
board drops frames instead of falling behind. A board stays a speaker first:
nothing is sent to one that does not say `screen.jpeg` in its /health.
"""
import collections
import io
import logging
import threading
import time

from plugins.flashhead import faces

logger = logging.getLogger(__name__)

TALK_FPS, IDLE_FPS = 12, 12     # EXPERIMENT: idle at 12 for the size sweep (2026-11-02); the kept idle is 2
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
    pushers = [_Pusher(b) for b in boards]
    playout = _Playout(pushers)
    clock = None
    try:
        from core.devices import glass
        talk_fps = max(1, min(TALK_FPS, min(glass.takes(b[0])["fps"] for b in boards)))   # the slowest board's beat, 1:1: no frame doubled or skipped
    except Exception:
        talk_fps = TALK_FPS
    last_word, checked = time.monotonic(), time.monotonic()
    try:
        with faces.http().get(f"{faces.base_url()}/stream?audio=0", stream=True, timeout=(5, 30)) as r:
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
                head = json.loads(read(struct.unpack(">I", read(4))[0]))
                if head.get("kind") == "flush":
                    last_word = now - HOLD_S + 1
                    continue
                body = read(head.get("audio", 0) + sum(head.get("frames", [])))   # always the whole packet: the stream stays in step
                if not head.get("frames"):
                    continue
                if head["kind"] == "speech":
                    last_word = now
                    _wake.clear()                    # the reply that woke us is here
                elif mode == "talking":
                    continue                     # between her lines the last frame holds
                fps = talk_fps if head["kind"] == "speech" else IDLE_FPS
                step = max(1, round(head["fps"] / fps))
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
                if clock is None or now > clock + 0.5:
                    clock = now + AHEAD_S                       # a fresh start
                for i in range(0, len(frames), step):
                    playout.put(clock + i / head["fps"], frames[i])
                clock += len(frames) / head["fps"]
    finally:
        _connected.clear()
        playout.close()
        for p in pushers:
            p.close()
        logger.info("[flashhead] her screens rest")
