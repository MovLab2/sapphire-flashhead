"""FlashHead routes — status, the live stream door, faces and moods, the webcam."""
import asyncio
import io
import logging

import httpx

from plugins.flashhead import faces

logger = logging.getLogger(__name__)

MAX_FACE = 8 * 1024 * 1024


async def status(settings=None, **_):
    out = {}
    try:
        kw = await asyncio.to_thread(faces.httpx_kwargs, settings)
        async with httpx.AsyncClient(timeout=2, **kw) as c:
            out.update((await c.get(f"{faces.base_url(settings)}/health")).json())
    except Exception as e:
        out.update(status="offline", error=(f"The FlashHead on the other computer is not answering: {faces.unreachable(e)}"
                                            if faces.worker(settings) else
                                            "Service not running — build the env or toggle the plugin"))
    path, mood = faces.current(settings)
    source = None if not path else "upload" if path == faces.FACE else "mood" if mood else "avatar"
    out.update(face_source=source, mood=mood, moods=faces.moods(),
               face_changed=int(path.stat().st_mtime) if path else 0)    # when, never where: the panels' still follows a new face
    for k in ("want", "face"):                  # the server's copies carry file paths: never past this door
        out.pop(k, None)
    return out


async def stream(settings=None, query=None, **_):
    from fastapi.responses import JSONResponse, StreamingResponse
    base = faces.base_url(settings)
    path, _ = faces.current(settings)
    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=30), **await asyncio.to_thread(faces.httpx_kwargs, settings))
    except Exception as e:
        return JSONResponse({"error": f"FlashHead service unreachable: {e}"}, status_code=503)
    try:
        if path:
            url, body, headers = faces.face_body(path, settings)
            await client.post(f"{base}{url}", content=body, headers=headers)
        try:                                # the browser's audio latency, for the server's log line
            lat = f"?lat={float((query or {}).get('lat') or 0):.3f}"
        except (TypeError, ValueError):
            lat = ""
        if (query or {}).get("audio") == "0":   # a face only: her voice stays where it plays now
            lat = f"{lat or '?'}{'&' if lat else ''}audio=0"
        resp = await client.send(client.build_request("GET", f"{base}/stream{lat}"), stream=True)
    except Exception as e:
        await client.aclose()
        return JSONResponse({"error": f"FlashHead service unreachable: {e}"}, status_code=503)

    async def body():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(body(), media_type="application/octet-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def viewer(request=None, settings=None, **_):
    """The browser's numbers for a reply it played, to the server's log beside its own."""
    try:
        body = await request.body()
        kw = await asyncio.to_thread(faces.httpx_kwargs, settings)
        async with httpx.AsyncClient(timeout=2, **kw) as c:
            await c.post(f"{faces.base_url(settings)}/viewer", content=body[:2048])
    except Exception as e:
        logger.debug(f"[flashhead] viewer numbers not logged: {e}")
    return {"ok": True}


async def stop(settings=None, **_):
    """Someone tapped her face: the server drops the rest of her reply, and its flush
    packet cuts what every viewer scheduled, at once. The browser stops the source
    itself (/api/tts/stop with the stream id each packet carries), or the next
    sentence would start her again; after a turn has ended only this reaches her."""
    try:
        kw = await asyncio.to_thread(faces.httpx_kwargs, settings)
        async with httpx.AsyncClient(timeout=2, **kw) as c:
            await c.post(f"{faces.base_url(settings)}/flush?by=viewer")
    except Exception as e:
        return {"error": f"not stopped: {faces.unreachable(e)}"}, 503
    return {"ok": True}


def get_face(settings=None, query=None, **_):
    """The base face, or ?mood=<name> for a mood thumbnail."""
    from fastapi.responses import FileResponse, JSONResponse
    mood = faces.slug((query or {}).get("mood"))
    path = faces.MOODS / f"{mood}.png" if mood else faces.base_face(settings)
    if not path or not path.exists():
        return JSONResponse({"error": "no face"}, status_code=404)
    return FileResponse(str(path), headers={"Cache-Control": "no-store"})


async def _save_image(request, dest):
    from PIL import Image, ImageOps
    form = await request.form()
    file = form.get("file")
    if not file:
        return {"error": "No file uploaded"}, 400
    raw = await file.read()
    if len(raw) > MAX_FACE:
        return {"error": "Image too large (8 MB max)"}, 400
    try:
        img = ImageOps.fit(Image.open(io.BytesIO(raw)).convert("RGB"), (512, 512))
    except Exception:
        return {"error": "Not an image"}, 400
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    return {"ok": True}


async def _show_now(settings):
    """A face changed in Settings: the server takes it for whoever is watching now."""
    path, _ = faces.current(settings)
    if not path:
        return
    try:
        url, body, headers = faces.face_body(path, settings)
        async with httpx.AsyncClient(timeout=3, **await asyncio.to_thread(faces.httpx_kwargs, settings)) as c:
            await c.post(f"{faces.base_url(settings)}{url}", content=body, headers=headers)
    except Exception:
        pass                                    # not running: it is picked up when a viewer opens the stream


async def upload_face(request=None, settings=None, **_):
    out = await _save_image(request, faces.FACE)
    await _show_now(settings)
    return out


async def reset_face(settings=None, **_):
    faces.FACE.unlink(missing_ok=True)
    await _show_now(settings)
    return {"ok": True}


async def upload_mood(name=None, request=None, **_):
    mood = faces.slug(name)
    if not mood:
        return {"error": "Mood needs a name (letters, numbers, - or _)"}, 400
    return await _save_image(request, faces.MOODS / f"{mood}.png")


def delete_mood(name=None, **_):
    mood = faces.slug(name)
    if mood:
        (faces.MOODS / f"{mood}.png").unlink(missing_ok=True)
    return {"ok": True}


async def restart(**_):
    """Cycle the service so it picks up saved settings."""
    from core.plugin_loader import plugin_loader
    await asyncio.to_thread(plugin_loader.restart_services, "flashhead")
    return {"ok": True}


def models(settings=None, **_):
    """Where her models live and what is there, for the settings page (a back-up, or a folder you already have)."""
    folder = faces.models_dir(settings)
    parts = []
    for name in ("SoulX-FlashHead-1_3B", "wav2vec2-base-960h"):
        d = folder / name
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.is_dir() else 0
        parts.append({"name": name, "present": d.is_dir() and size > 0, "gb": round(size / 1e9, 2)})
    return {"folder": str(folder), "custom": bool(str(settings.get("models_dir") or "").strip()) if settings else False,
            "parts": parts}


def camera_status(**_):
    """Her webcam: off | on | error, its device and mic, why not in words."""
    from plugins.flashhead import camera
    from core import extras
    return {**camera.status(), "installed": extras.installed("plugin:flashhead")}


def camera_retry(**_):
    """After an install, or a camera freed elsewhere: try again now."""
    from plugins.flashhead import camera
    camera.on_settings_saved(None)
    return {"ok": True}
