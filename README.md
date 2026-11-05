# FlashHead Sapphire Plugin

by [Teri](https://github.com/MovLab2) and [Chris](https://github.com/ddxfish)

This is a plugin for [Sapphire AI](https://github.com/ddxfish/sapphire) that gives a personal avatar from any avatar image you want, lip-synced live to her voice by SoulX-FlashHead Lite on your GPU, in the chat sidebar.

A loader: this plugin ships only its own code. The engine and the weights are fetched on first start (below).

## Requirements

- **NVIDIA GPU, compute 8.0+** with **~6 GB of free VRAM**, or about 4 GB or 3 GB with the leaner **Video memory** presets (Settings › FlashHead), and a **driver 570 or newer** (CUDA 12.8). Tested on an RTX 5090: 6.7x real time at Full, 8.1x at Lean and 8.3x at Smallest, which render at 384 px. An RTX 5060 Ti renders about 1.4x at Full and keeps up better leaner; slower cards may not keep up live, and the panel says so after warm-up. **Windows:** a card that runs out of video memory pages to system RAM silently and her face crawls at a tenth of real time; the panel and the log say so at Ready, and a leaner preset fixes it. CPU, AMD (ROCm) and macOS are not supported.
- **~13 GB of disk**: the conda env (~5 GB) and the models (~7.7 GB).
- **Streaming TTS** (Kokoro or Piper).
- **Sapphire 2.15.3 or newer** (`min_core_version`).
- **conda**, as Sapphire's own install has: Sapphire builds this plugin's env itself when you enable it, after asking.
- **Linux and Windows** (`platforms`). Linux: tested. Windows: tested on a 5060 Ti; the env adds `triton-windows` for `torch.compile` (about 20% faster than without). If compiling can't start it loads uncompiled and says so in its log. Don't set `TORCH_COMPILE_DISABLE` or `TORCHDYNAMO_DISABLE`: they turn compiling off for good. On Windows keep Sapphire in a short folder path (or enable long paths): the download's work files sit deep.

A Sapphire with no GPU can use one on another computer: see **One GPU, several Sapphires** below. It needs none of the above but Sapphire itself.

The GPU is checked when the service starts, before the models download. The env itself is built before that check can run, so on a machine without a suitable GPU the ~5 GB env is still downloaded once.

## Install

1. **Settings › Plugins › Install from URL**: this repository's URL.
2. Enable **FlashHead** and say yes to **Build environment** (a few minutes, ~5 GB).
3. The service starts, checks the GPU, then downloads the engine and models (~7.7 GB). The sidebar panel shows progress; **↻ Retry** resumes a download that stopped.
4. Open the **FlashHead** sidebar panel and press **▶ Watch**. Talk to her.

Logs: `user/logs/flashhead-server.log`.

**Testing without a GPU** (a VM): Settings › FlashHead › Advanced › *Test the install without a GPU*. It fetches and imports everything, never renders, and reports in the panel whether every step worked.

**Uninstall** removes the plugin's code. What it fetched stays until you remove it: the env (`conda env remove -n sapphire-plugin-flashhead`), `user/models/flashhead/` (the models), and `user/flashhead/` (your uploaded faces and moods).

## One GPU, several Sapphires

One computer's FlashHead can give every Sapphire on your network her own live face: each is its own session (her face, her motion, her viewers), on one model.

1. **On the computer with the GPU**: Settings › FlashHead › Advanced › *Share this GPU with other computers*, save. Under *Computers that may use this GPU*, add one by name: it shows a pairing string once.
2. **On the other Sapphire**: enable FlashHead and choose **Client only** (or turn it on in Settings › FlashHead › Advanced). Paste the pairing string into *Pairing string from the GPU computer*, save.

**No Sapphire on the GPU computer?** Copy this folder there and run `./flashhead-server.sh` (Linux) or `flashhead-server.bat` (Windows, from an Anaconda Prompt, in a short folder path). It builds its own conda env (`flashhead-server`, the packages `plugin.json` lists), keeps everything in `./user/` beside it (inside a Sapphire install: that Sapphire's `user/`, the plugin's own folders, so nothing downloads twice), and serves only the shared door. `add <name>` prints a computer's pairing string, `remove <name>` shuts it out, `list` shows who is let in. `PORT` changes the door's port; `FLASHHEAD_ENV` uses an env you already have.

**Check a server from any computer** (after `flashhead-server` has made its env there): `./flashhead-test.sh "<pairing string>" [face image]` or `flashhead-test.bat` (it asks for the string). It checks the certificate against the pin, that a wrong key is refused, the server's health, and if it can render, her idle and a few seconds of sound mouthed back, saving one frame to the temp folder (`flashhead-test-frame.jpg`). On a computer with no NVIDIA GPU the first three pass and it says why it can't render.

The other computer's port (5018 by default) must be reachable on your network. Traffic is TLS, pinned to that computer's own certificate (no certificate authority, so a changed address still works and a different server never does), with a key per computer; removing a computer shuts it out. Her voice goes there and comes back with her face, which is what keeps her lips in time. Her words always go first: when two talk at the same moment, both keep up on a fast card; on a slower one (a 5060 Ti) the second plays her voice without a face until the first is done.

## Her face as a webcam

Her live face as a camera, and her voice as a mic, that a call or stream can pick (Zoom, Meet, Discord, OBS). It works on the computer running Sapphire, client mode included.

**Windows**

1. Install **OBS Studio 26 or newer** (obsproject.com). Her camera is OBS's virtual camera; OBS doesn't need to be running. Keep OBS's own *Start Virtual Camera* off, since only one program can drive it.
2. For her voice: install **VB-Cable** (free, vb-audio.com/Cable), run its setup as administrator, then reboot.
3. Settings › Plugins › FlashHead card: **Install** the webcam packages (pyvirtualcam).
4. Settings › FlashHead › **Webcam**: turn on *Her face as a webcam*, and *Her voice as a mic* if you want her voice too. Leave *Camera device* and *Mic device* empty. Save.
5. In the call, pick **OBS Virtual Camera** as the camera and **CABLE Output** as the mic.

**Linux**

1. Install v4l2loopback (`sudo apt install v4l2loopback-dkms` on Ubuntu) and load it:
   `sudo modprobe v4l2loopback exclusive_caps=1 card_label="Sapphire Cam"`
   To load it at every boot, put `v4l2loopback` in `/etc/modules-load.d/v4l2loopback.conf` and the options in `/etc/modprobe.d/v4l2loopback.conf` (`options v4l2loopback exclusive_caps=1 card_label="Sapphire Cam"`).
2. For her voice: PipeWire (Ubuntu's default) with `pw-cat`. Nothing else to install.
3. Settings › Plugins › FlashHead card: **Install** the webcam packages (pyvirtualcam).
4. Settings › FlashHead › **Webcam**: turn on *Her face as a webcam* and, if you want, *Her voice as a mic*. *Camera device* empty takes the first free loopback device; name one (`/dev/video3`) if you have several. Save.
5. In the call, pick **Sapphire Cam** as the camera and **Sapphire Voice** as the mic.

While the webcam is on, her face renders all the time: cheap with the calm loop, but with *Live* the GPU stays busy. With *Her voice as a mic* on, her voice goes only to the call; turn on *Hear her here too* to also hear her on this computer's speakers. If something's missing, the Webcam tab says what to install.

## Her face on another screen

`viewer/` is a standalone window for her face and voice on any computer that reaches your Sapphire (a Raspberry Pi on a screen): an API key, her address, `--fullscreen`, `--mute`. Its own `requirements.txt`, nothing of Sapphire's. See `viewer/README.md`.

## How fast is this GPU?

`./flashhead-bench.sh` or `flashhead-bench.bat` (in the env that renders; inside Sapphire set `FLASHHEAD_ENV=sapphire-plugin-flashhead`) loads the model the way the server does, once per variant, and times real chunks: 0.96 s of speech in, 24 frames out. Variants: the three Video memory presets `full`, `lean`, `smallest` (the default run), `eager`, `compiled`, `sdpa`, `autotune`, and the pieces `res384`, `int8`, `int4`, `vae8`, `decode25`. `--face` and `--audio` give a picture and a recording for a strip of frames per variant (quality is for eyes, not timers); `--long 300` lets her run on for 5 minutes and reports whether her face drifts. Every variant reports the video memory it took. Results land in `user/models/flashhead/bench/`. It never downloads; run the server once first.

RTX 5090, 1.5.0: Full 6.7x real time (0.14 s per chunk) holding 6.1 GB, Lean 8.1x and 3.65 GB, Smallest 8.3x and 2.97 GB, all holding her face for 5 minutes. Tried and dead: 256 px melts to a grey mask within 3 minutes, 2 or 3 sampling steps smear while she talks, CUDA graphs drift.

## How it flows

```
her streamed voice ─tts_chunk_audio─► server.py (own conda env, GPU)
                                        0.96 s of audio -> 24 frames
   sidebar / app page ◄── /api/plugin/flashhead/stream (sound + frames)
   her boards        ◄── screens.py (frames only) ── /screen/picture?format=jpeg
```

- **Watch** in the sidebar or the Chat flyout's FlashHead page. Collapsing or leaving stops it (**■ Stop watching**). **⚙** opens her settings.
- While she speaks a **■** sits on her face and the line under it says so: **tap her face and she stops**, at once, here and everywhere her words were going. The rest of that reply is dropped and the LLM keeps writing. The chat's own ⏹ does the same while her turn runs, but the face server holds the rest of a reply after the turn has ended, and only a tap reaches that.
- **Settings › FlashHead**: **Video memory** (below), what she does between her words, her screens, whose face (empty = the persona of the chat she speaks in), a **Faces** tab (upload a face, back to her avatar, moods) and **Advanced** (port, the models folder and where it is now, the no-GPU install test).
- **Video memory**: how much of the GPU her face takes. *Full* (512 px, 16-bit weights, about 6 GB) is her sharpest. *Lean* (384 px, 8-bit weights, about 4 GB) is a touch softer, half the memory, and faster. *Smallest* (384 px, 4-bit weights, about 3 GB) fits a card with 4 GB free. The leaner presets shrink the same models on the CPU as they load (nothing downloads again, and the shrink takes seconds; the first start of a leaner preset compiles for a minute or two, later starts are as quick as Full). Every start measures this card: the panel shows the speed and GB after "ready", and the line under the setting shows what each preset you have tried took and ran at here, so you can compare before you choose. fp16 sums and a bf16 audio encoder are on in every preset (free speed, measured safe; an overflow falls back to bf16 on its own).
- **Between her words**: *Calm loop* (default) renders a few seconds of silence once per face, keeps the stretch closest to her real face, and replays it forward then back: the GPU works only while she talks. *Live* renders her as she sits, never repeating, and her next words flow on from there; the GPU works whenever someone watches.
- Her voice is taken from this computer only on a lane that plays here and while a viewer plays sound. A board's voice is never taken; its face mouths along. If rendering keeps failing, the panel says so and her voice plays as usual.
- **Her screens** (setting): Sapphire ESP32 satellites with a screen (the Waveshare board on satellite firmware 0.7.7+) show her face while she talks, or always. A board stays a speaker first.
- **Moods**: `user/flashhead/moods/<emotion>.png`, matched to her active emotion pieces at the start of each reply. Add them with **+ Mood** on the Faces tab.
- **Stop** (the chat's ⏹, a board's button, a tap on her face) flushes the face with her voice.
- The service listens on 127.0.0.1, answers Sapphire's own calls only, and animates faces only from inside Sapphire's folder. With sharing on it also opens the TLS door above; nothing comes in through it without a key. Faces other computers send are kept in `user/models/flashhead/faces/` (copies, never backed up).

## What gets fetched, and their licenses

Nothing below is in this plugin; each comes from its source the first time it is needed.

| What | From | License |
|---|---|---|
| Engine `flash_head/` at commit `9bc03de0` | github.com/Soul-AILab/SoulX-FlashHead | Apache-2.0 |
| FlashHead 1.3B Lite DiT weights | huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B | Apache-2.0 |
| LTX-Video 0.9 VAE (`VAE_LTX`, used by Lite) | same repo | **OpenRAIL-M** ([terms](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.license.txt)): its Attachment A use restrictions bind everyone running Lite, among them no impersonating a real person without their consent |
| wav2vec2-base-960h | huggingface.co/facebook/wav2vec2-base-960h | Apache-2.0 |
| Python env: transformers, diffusers, accelerate, einops, pyloudnorm, loguru, pyyaml, pillow, huggingface_hub, cryptography, and what they pull in (numpy, requests) | PyPI | Apache-2.0 / MIT / BSD / MIT-CMU |
| torch, torchvision (cu128) | download.pytorch.org | BSD-3; they pull NVIDIA's CUDA runtime wheels (NVIDIA license) |
| av | PyPI | BSD-3 (PyAV itself); its PyPI wheels bundle a GPL build of FFmpeg (with x264/x265). Fetched by pip, never shipped here |
| triton-windows (Windows only) | PyPI | MIT |

The engine imports xfuser (multi-GPU) and mediapipe (face crop) at module top; one GPU and no crop never call them, so they are not installed and the server stands in for them (loudly, if anything ever calls one).

The engine and models land in `user/models/flashhead/`, which Sapphire's backups always skip (rebuildable: they download again). Her faces and moods live in `user/flashhead/` and are backed up. The models folder can be moved in Settings › FlashHead › Advanced.

## Credits

SoulX-FlashHead by Soul-AILab. The streaming approach was mapped from MovLab2's teri-realtime-avatar-5060ti.

## License

This plugin's own code: MIT (`LICENSE`). What it fetches keeps its own licenses (table above).
