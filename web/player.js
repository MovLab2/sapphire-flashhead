// FlashHead viewer — one module behind the sidebar and the app page.
// The stream is packets of (header, PCM, JPEG frames) for the same 0.96 s;
// each is decoded whole, then its voice and frames are placed together on
// WebAudio's clock, read where the ear is: the mouth can't drift from the voice.
// While she speaks a ■ sits on her face: a tap stops her, here at once, at the
// server (the rest of the reply) and at its source (the pump, by stream id).
export const BASE = '/api/plugin/flashhead';
export const csrf = () => document.querySelector('meta[name="csrf-token"]')?.content || '';

function createPlayer(canvas, onTalk) {
    const g = canvas.getContext('2d');
    let ctx = null, abort = null, raf = 0, next = 0, frames = [], sources = [], gen = 0;
    let chain = Promise.resolve(), voiced = false, voiceEnd = 0, lastRaf = 0, clock = 0;
    let talkEnd = 0, talking = false, sid = '';     // her words being heard (voiced or mouthed here), and whose stream
    const fresh = () => ({ packets: 0, gaps: 0, gap_s: 0, wait: 0, wait_max: 0, late: 0, stalls: 0, paused: 0 });
    let st = fresh();                   // this reply's numbers: one line in her server log when she goes quiet

    // The moment now being heard: the device's own timestamp (WASAPI, USB,
    // HDMI, Bluetooth, every buffer between), else the latencies it reports.
    // Her mouth follows the ear, not the scheduler.
    function heard() {
        const ts = ctx.getOutputTimestamp?.();
        if (ts && ts.contextTime > 0) {
            const t = ts.contextTime + Math.max(0, performance.now() - ts.performanceTime) / 1000;
            if (t <= ctx.currentTime && t > ctx.currentTime - 1) { clock = 1; return t; }
        }
        clock = 0;
        return ctx.currentTime - (ctx.baseLatency || 0) - (ctx.outputLatency || 0);
    }

    function draw(ms) {
        raf = requestAnimationFrame(draw);
        if (!ctx) return;
        if (voiced && lastRaf && ms - lastRaf > 250 && ms - lastRaf < 5000) st.stalls++;   // the page froze mid-reply
        lastRaf = ms;
        const now = heard();
        if ((now < talkEnd) !== talking) onTalk(talking = now < talkEnd);
        let show = null;
        while (frames.length && frames[0].t <= now) {
            if (show) { show.bmp.close(); if (now - show.t > 0.1) st.late++; }    // never shown: its moment passed
            show = frames.shift();
        }
        if (show) {
            const px = Math.round(canvas.clientWidth * (window.devicePixelRatio || 1));
            if (px && canvas.width !== px) { canvas.width = canvas.height = px; g.imageSmoothingQuality = 'high'; }
            g.drawImage(show.bmp, 0, 0, canvas.width, canvas.height);
            show.bmp.close();
        }
        if (st.packets && ctx.currentTime > voiceEnd + 1.5) report();
    }

    function report() {
        const out = { ...st, wait_ms: st.wait / st.packets, out: Math.max(0, ctx.currentTime - heard()), clock, sr: ctx.sampleRate };
        delete out.wait;
        st = fresh();
        voiced = false;
        fetch(`${BASE}/viewer`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrf() },
                                  body: JSON.stringify(out) }).catch(() => {});
    }

    function flush() {                  // she was stopped: cut what's scheduled
        gen++;                          // packets still decoding belong to what was cut
        const now = heard();
        sources.forEach(s => { try { s.stop(); } catch {} });
        sources = [];
        frames = frames.filter(f => (f.t <= now) || (f.bmp.close(), false));
        next = 0;
        voiced = false;
        talkEnd = 0;
    }

    // A packet's frames are decoded BEFORE its voice is scheduled, and both
    // are placed together, in arrival order: a busy CPU then delays her
    // whole (a pause), it never lets her voice run on while her mouth waits.
    function schedule(head, bytes, off) {
        const g0 = gen, got = performance.now();
        const audio = head.audio ? bytes.slice(off, off + head.audio) : null;
        let p = off + head.audio;
        const hidden = document.hidden;     // a background tab: her voice plays on, frames nobody can see are not decoded
        const decoded = Promise.all(hidden ? [] : head.frames.map(len =>
            createImageBitmap(new Blob([bytes.subarray(p, p += len)], { type: 'image/jpeg' })).catch(() => null)));
        chain = chain.then(() => decoded).then(bmps => {
            if (!ctx || g0 !== gen) return bmps.forEach(b => b?.close());     // stopped or flushed while it decoded
            place(head, audio, bmps, performance.now() - got);
        }).catch(() => {});
    }

    function place(head, audio, bmps, waited) {
        const dur = head.frames.length ? head.frames.length / head.fps : head.audio / 2 / head.sr;   // her voice without a face still takes its time
        const t = Math.max(next, ctx.currentTime + 0.15);
        if (audio) {
            if (voiced && t > next + 0.02) { st.gaps++; st.gap_s += t - next; }     // she was mid-reply and this came late
            st.packets++; st.wait += waited; st.wait_max = Math.max(st.wait_max, waited);
            voiced = true;
            voiceEnd = t + dur;
            const pcm = new Int16Array(audio.buffer);
            const buf = ctx.createBuffer(1, pcm.length, head.sr);
            const ch = buf.getChannelData(0);
            for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
            const src = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(ctx.destination);
            src.start(t);
            sources.push(src);
            src.onended = () => { sources = sources.filter(s => s !== src); };
        } else voiced = false;
        next = t + dur;
        if (head.kind === 'speech') { talkEnd = t + dur; sid = head.sid || ''; }
        bmps.forEach((bmp, i) => bmp && frames.push({ t: t + i / head.fps, bmp }));
    }

    async function start() {
        ctx = new AudioContext();
        ctx.onstatechange = () => { if (ctx && ctx.state !== 'running') st.paused++; };   // Windows: a device change or sleep stops the clock
        next = 0;
        abort = new AbortController();
        raf = requestAnimationFrame(draw);
        const lat = (ctx.baseLatency || 0) + (ctx.outputLatency || 0);     // for her log: Bluetooth shows here
        const r = await fetch(`${BASE}/stream?lat=${lat.toFixed(3)}`, { signal: abort.signal });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
        const reader = r.body.getReader();
        let buf = new Uint8Array(0);
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            const joined = new Uint8Array(buf.length + value.length);
            joined.set(buf); joined.set(value, buf.length);
            buf = joined;
            while (buf.length >= 4) {
                const hl = new DataView(buf.buffer, buf.byteOffset).getUint32(0);
                if (buf.length < 4 + hl) break;
                const head = JSON.parse(new TextDecoder().decode(buf.subarray(4, 4 + hl)));
                const media = head.frames ? head.audio + head.frames.reduce((a, b) => a + b, 0) : 0;
                const total = 4 + hl + media;
                if (buf.length < total) break;
                if (head.kind === 'flush') flush();
                else if (head.frames) schedule(head, buf, 4 + hl);
                buf = buf.slice(total);
            }
        }
    }

    function stop() {
        if (ctx && st.packets) report();
        gen++;                              // a packet still decoding never lands in the next Watch
        abort?.abort();
        cancelAnimationFrame(raf);
        talkEnd = 0;
        if (talking) onTalk(talking = false);
        frames.forEach(f => f.bmp.close());
        frames = [];
        sources = [];
        ctx?.close();
        ctx = null;
    }

    // Her face was tapped: her voice stops here now, then at the server (the rest of
    // the reply, every viewer) and at its source (that pump, so the next sentence
    // never starts her again). After a turn has ended only the server's cut is left.
    function hush() {
        const id = sid;
        flush();
        fetch(`${BASE}/stop`, { method: 'POST', headers: { 'X-CSRF-Token': csrf() } }).catch(() => {});
        import('/static/audio.js').then(a => { a.stop(true); if (id) a.stopLocalTts(id); }).catch(() => {});
    }

    return { start, stop, hush };
}

// Her settings tab (web/index.js): from Settings already, switch tabs; else enter Settings on it.
async function openSettings() {
    const { switchView, getCurrentView } = await import('/static/core/router.js');
    if (getCurrentView() === 'settings') {
        document.getElementById('view-settings')
            ?.dispatchEvent(new CustomEvent('settings-navigate', { detail: { tab: 'flashhead' }, bubbles: true }));
        return;
    }
    window._settingsTab = 'flashhead';
    switchView('settings');
}

// The app page: type words (or hold 🎤 and say them: they land in the box,
// never sent), and she speaks them with no LLM between (/api/tts/stream). Her
// face takes them while someone watches, so Speak starts Watch first.
function mountSay(root, watchFirst) {
    const words = root.querySelector('.fh-words'), mic = root.querySelector('.fh-mic'), speak = root.querySelector('.fh-speak');
    const audio = () => import('/static/audio.js');
    const add = text => {
        words.value = words.value.trim() ? `${words.value.trimEnd()} ${text}` : text;
        words.focus();
    };
    let held = false;                   // a release before the mic opened still stops it
    const lit = on => { mic.style.outline = on ? '2px solid var(--accent-red,#e55)' : ''; };
    mic.onpointerdown = async e => {
        e.preventDefault();
        held = true;
        lit(true);
        const a = await audio();
        await a.handlePress(mic);
        if (!held) a.handleRelease(mic, add);
    };
    mic.onpointerup = mic.onpointerleave = mic.onpointercancel = async () => {
        if (!held) return;
        held = false;
        lit(false);
        (await audio()).handleRelease(mic, add);
    };
    async function say() {
        const text = words.value.trim();
        if (!text || speak.disabled) return;
        speak.disabled = true;
        try {
            watchFirst();
            await new Promise(r => setTimeout(r, 400));     // her face is listening before her first words reach it
            const a = await audio();
            if (!await a.playTextStreaming(text)) await a.playText(text);
        } catch (e) {
            root.querySelector('.fh-status').textContent = `couldn't speak: ${e.message}`;
        } finally { speak.disabled = false; }
    }
    speak.onclick = say;
    words.onkeydown = e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); say(); } };
}

export function mount(root, { big = false } = {}) {
    const size = big ? 'min(80vh, 90vw)' : '100%';
    root.innerHTML = `
        <div class="fh-wrap" style="display:flex;flex-direction:column;gap:var(--space-sm,8px);align-items:center;${big ? 'padding:24px;' : ''}">
            <div style="position:relative;width:${size};aspect-ratio:1;background:#000;border-radius:var(--radius-md,8px);overflow:hidden;">
                <img class="fh-still" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:.5;">
                <canvas class="fh-canvas" width="512" height="512" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
                <div class="fh-hush" hidden style="position:absolute;top:6%;right:6%;width:1.6em;height:1.6em;border-radius:4px;background:rgba(0,0,0,.55);color:#fff;font-size:${big ? '20px' : '12px'};line-height:1.6em;text-align:center;pointer-events:none;">■</div>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:center;">
                <button class="fh-watch btn">▶ Watch</button>
                <button class="fh-retry btn" hidden title="Start the service again: a failed download resumes where it stopped">↻ Retry</button>
                <button class="fh-settings btn" title="Faces, moods, idle and screens">⚙</button>
            </div>
            <div class="fh-status" style="font-size:var(--font-sm,12px);color:var(--text-muted);text-align:center;"></div>
            ${big ? `<div style="width:${size};display:flex;gap:6px;align-items:stretch;">
                <textarea class="fh-words" rows="2" placeholder="Words for her to say (Ctrl+Enter)"
                    style="flex:1;resize:vertical;padding:6px 8px;border-radius:var(--radius-md,8px);"></textarea>
                <button class="fh-mic btn" title="Hold to talk: your words land in the box">🎤</button>
                <button class="fh-speak btn">Speak</button>
            </div>` : ''}
        </div>`;
    const $ = s => root.querySelector(s);
    const canvas = $('.fh-canvas'), still = $('.fh-still'), btn = $('.fh-watch'), status = $('.fh-status');
    let watching = false, poll = 0, faceKey = '', line = '', talking = false;
    const show = () => { status.textContent = talking ? 'speaking · tap her face to stop her' : line; };
    const player = createPlayer(canvas, t => {           // her words start, or end, being heard here
        talking = t;
        $('.fh-hush').hidden = !t;
        canvas.style.cursor = t ? 'pointer' : '';
        canvas.title = t ? 'Tap her face to stop her' : '';
        show();
    });
    canvas.onclick = () => { if (talking) player.hush(); };

    async function tick() {
        if (document.hidden || !root.isConnected || !canvas.offsetParent) return;   // nobody sees it: no polling
        try {
            const s = await (await fetch(`${BASE}/status`)).json();
            const mood = s.mood ? ` · mood: ${s.mood}` : '';
            const stuck = ['error', 'unsupported', 'offline'].includes(s.status);
            $('.fh-retry').hidden = !stuck;
            line = stuck ? (s.error || s.status)       // the reason, in words, alone
                : !s.face_source ? 'No face yet: give her persona an avatar, or upload one in ⚙ › Faces'
                : s.note || s.warning
                || `${s.status} · face: ${s.face_source}${mood} · viewers: ${s.viewers ?? 0}`
                   + (s.gpu?.speed ? ` · ${s.gpu.speed}x real time, ${s.gpu.took_gb} GB` : '');
            show();
            const key = `${s.face_source}|${s.mood}|${s.face_changed || ''}`;
            if (key !== faceKey) {                     // a face changed in Settings: the still follows
                faceKey = key;
                still.src = `${BASE}/face?t=${Date.now()}`;
            }
        } catch { line = 'status unavailable'; show(); }
    }
    tick();
    poll = setInterval(tick, 3000);

    async function watch() {
        watching = true;
        btn.textContent = '■ Stop watching';
        still.style.opacity = '0';
        try { await player.start(); }
        catch (e) { if (e.name !== 'AbortError') status.textContent = `stream error: ${e.message}`; }
        unwatch();
    }
    function unwatch() {
        if (!watching) return;
        watching = false;
        player.stop();
        btn.textContent = '▶ Watch';
        still.style.opacity = '.5';
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    }

    btn.onclick = () => (watching ? unwatch() : watch());
    $('.fh-settings').onclick = openSettings;
    $('.fh-retry').onclick = async () => {
        $('.fh-retry').hidden = true;
        status.textContent = 'starting again…';
        await fetch(`${BASE}/restart`, { method: 'POST', headers: { 'X-CSRF-Token': csrf() } }).catch(() => {});
        tick();
    };

    if (big) mountSay(root, () => { if (!watching) watch(); });

    // Only while watching: a collapsed accordion or scrolled-away face stops the GPU.
    const seen = new IntersectionObserver(([e]) => { if (!e.isIntersecting) unwatch(); });
    seen.observe(canvas);

    return () => { unwatch(); clearInterval(poll); seen.disconnect(); };
}
