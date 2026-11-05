# FlashHead Viewer

Her live talking face in its own window, on any computer that can reach your Sapphire: a Raspberry Pi on a screen, a spare laptop. It plays her voice in time with her lips, or just her face with `--mute`.

Standalone: it needs none of Sapphire or FlashHead installed, only this folder.

## Setup

Linux (Raspberry Pi OS, Ubuntu):

```
sudo apt install libportaudio2          # sound, for sounddevice
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Windows: `python -m venv venv` then `venv\Scripts\pip install -r requirements.txt`.

PySide6 needs a 64-bit OS (a Pi 4 or 5 on 64-bit Raspberry Pi OS).

## The key

On your Sapphire: **Settings › System › API Keys**, make one for this viewer.

## Run

```
venv/bin/python flashhead-viewer.py --host 192.168.1.50 --key <API key> --fullscreen
```

| Option | |
|---|---|
| `--host` | Sapphire's address, or a full URL (`https://host:8073`, or `http://` if her SSL is off) |
| `--port` | her web port, default 8073 |
| `--key` | the API key |
| `--fullscreen` | start fullscreen |
| `--mute` | her face only: her voice stays where it plays now |
| `--device` | sound output, a number or part of a name |
| `--list-devices` | list sound outputs |

`FLASHHEAD_HOST`, `FLASHHEAD_KEY` and `FLASHHEAD_DEVICE` work in place of the options. The key in an environment variable stays out of `ps`.

Keys: **F** or double-click toggles fullscreen, **Esc** or **Q** quits. A click on her face (or **Space**) while she speaks stops her: here, at the server and at its source.

## Her voice

Without `--mute`, this viewer takes her voice: she speaks from here instead of the computer she runs on, with her lips in time. The newest viewer playing her voice takes it, so opening the browser's FlashHead panel later takes it back.

With `--mute`, her voice plays where it always does and this shows her face. The face runs a little behind her voice (her render time plus your network), so `--mute` suits a face across the room more than one beside the speaker.

## Her certificate

Sapphire's HTTPS certificate is her own, so nothing vouches for it. The viewer trusts the one it sees the first time and refuses any other after that, so your key never goes to a stranger. The pins live in `~/.config/flashhead-viewer/pins.json`. If you reinstall Sapphire, remove her line there.

## Start on boot (Pi)

`~/.config/systemd/user/flashhead-viewer.service`:

```
[Unit]
Description=FlashHead viewer

[Service]
Environment=FLASHHEAD_HOST=192.168.1.50
Environment=FLASHHEAD_KEY=<API key>
ExecStart=%h/flashhead/viewer/venv/bin/python %h/flashhead/viewer/flashhead-viewer.py --fullscreen
Restart=on-failure

[Install]
WantedBy=graphical-session.target
```

`systemctl --user enable --now flashhead-viewer`. With no desktop (Pi OS Lite), add `Environment=QT_QPA_PLATFORM=eglfs` to draw straight to the screen.
