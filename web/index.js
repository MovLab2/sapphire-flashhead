// FlashHead settings: the manifest's knobs (core's form renderer) and a Faces
// tab for what a form can't hold: her face and her mood faces, which save on
// their own buttons. The sidebar's ⚙ opens this tab.
import { registerPluginSettings } from '/static/shared/plugin-registry.js';
import pluginsAPI from '/static/shared/plugins-api.js';
import { renderSettingsForm, readSettingsForm } from '/static/shared/plugin-settings-renderer.js';
import { BASE, csrf } from './player.js';

async function schema() {
    const list = await pluginsAPI.listPlugins();
    return (list.plugins || []).find(p => p.name === 'flashhead')?.settings_schema || [];
}

function upload(url, file) {
    const fd = new FormData();
    fd.append('file', file);
    return fetch(url, { method: 'POST', body: fd, headers: { 'X-CSRF-Token': csrf() } });
}

function mountFaces(el) {
    el.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:14px;">
            <div>
                <div style="font-weight:600;margin-bottom:4px;">Her face</div>
                <div style="color:var(--text-muted);font-size:var(--font-sm,12px);margin-bottom:8px;">
                    Her persona avatar, unless you upload one here. Square works best; it is cropped to 512 px.</div>
                <div style="display:flex;gap:12px;align-items:center;">
                    <img class="fh-base" style="width:96px;height:96px;border-radius:8px;object-fit:cover;background:#000;">
                    <div style="display:flex;flex-direction:column;gap:6px;">
                        <button class="btn fh-upload">Upload face</button>
                        <button class="btn fh-reset">Use her avatar</button>
                    </div>
                </div>
            </div>
            <div>
                <div style="font-weight:600;margin-bottom:4px;">Moods</div>
                <div style="color:var(--text-muted);font-size:var(--font-sm,12px);margin-bottom:8px;">
                    A face shown while one of her emotion pieces is active. Name it after the emotion (happy, sad, excited).</div>
                <div class="fh-moods" style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end;"></div>
            </div>
            <div class="fh-note" style="color:var(--text-muted);font-size:var(--font-sm,12px);"></div>
            <input class="fh-file" type="file" accept="image/*" hidden>
            <input class="fh-mood-file" type="file" accept="image/*" hidden>
        </div>`;
    const $ = s => el.querySelector(s);
    const note = t => { $('.fh-note').textContent = t; };
    let pending = '';

    async function refresh() {
        $('.fh-base').src = `${BASE}/face?t=${Date.now()}`;
        const s = await (await fetch(`${BASE}/status`)).json().catch(() => ({}));
        const box = $('.fh-moods');
        box.innerHTML = '';
        for (const m of s.moods || []) {
            const chip = document.createElement('div');
            chip.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:3px;font-size:12px;';
            chip.innerHTML = `<img src="${BASE}/face?mood=${encodeURIComponent(m)}&t=${Date.now()}"
                    style="width:64px;height:64px;border-radius:6px;object-fit:cover;border:2px solid ${m === s.mood ? 'var(--accent-blue,#4a9eff)' : 'transparent'};">
                <span>${m} <a href="#" title="Remove" style="color:var(--text-muted)">×</a></span>`;
            chip.querySelector('a').onclick = async e => {
                e.preventDefault();
                await fetch(`${BASE}/moods/${encodeURIComponent(m)}`, { method: 'DELETE', headers: { 'X-CSRF-Token': csrf() } });
                note(`mood "${m}" removed`);
                refresh();
            };
            box.appendChild(chip);
        }
        const add = document.createElement('button');
        add.className = 'btn';
        add.textContent = '+ Mood';
        add.onclick = () => {
            pending = (prompt('Emotion name (matches her emotion piece, e.g. happy):') || '').trim();
            if (pending) $('.fh-mood-file').click();
        };
        box.appendChild(add);
    }

    $('.fh-upload').onclick = () => $('.fh-file').click();
    $('.fh-file').onchange = async e => {
        const f = e.target.files[0];
        e.target.value = '';
        if (!f) return;
        const r = await upload(`${BASE}/face`, f);
        note(r.ok ? 'face uploaded: she wears it now' : `upload failed: ${(await r.json().catch(() => ({}))).error || r.status}`);
        refresh();
    };
    $('.fh-reset').onclick = async () => {
        await fetch(`${BASE}/face`, { method: 'DELETE', headers: { 'X-CSRF-Token': csrf() } });
        note('back to her avatar');
        refresh();
    };
    $('.fh-mood-file').onchange = async e => {
        const f = e.target.files[0];
        e.target.value = '';
        if (!f || !pending) return;
        const r = await upload(`${BASE}/moods/${encodeURIComponent(pending)}`, f);
        note(r.ok ? `mood "${pending}" saved` : `mood upload failed: ${(await r.json().catch(() => ({}))).error || r.status}`);
        refresh();
    };
    refresh();
}

// Under the Video memory setting: this card, and what each preset tried on it took
// and ran at (the server measures both at every start: its warm-up is the benchmark).
// Polled, so a saved preset shows "warming up" and then its numbers without a reload.
const PRESETS = [['full', 'Full'], ['lean', 'Lean'], ['smallest', 'Smallest']];

function mountMeasure(el) {
    el.innerHTML = `<div class="fh-measure" style="font-size:var(--font-sm,12px);color:var(--text-muted);margin-top:2px;"></div>`;
    const line = el.querySelector('.fh-measure');
    async function tick() {
        if (!el.isConnected) return clearInterval(poll);
        try {
            const s = await (await fetch(`${BASE}/status`)).json();
            const g = s.gpu, m = s.measured || {};
            if (!g) line.textContent = s.status === 'offline' ? (s.error || 'service not running') : (s.status || '');
            else if (s.status !== 'ready') line.textContent = `${g.name}: ${s.status}…`;
            else line.textContent = `${g.name}, ${g.total_gb} GB · ` + PRESETS.map(([k, name]) =>
                m[k] ? `${name}: ${m[k].took_gb} GB, ${m[k].speed}x real time` : `${name}: not tried yet`).join(' · ');
        } catch { line.textContent = ''; }
    }
    const poll = setInterval(tick, 3000);
    tick();
}

// Advanced: where her models live (to back up or move), and with sharing on,
// which computers may use this GPU (core's worker-door panel).
async function mountAdvanced(el, share) {
    await mountModels(el);
    if (!share) return;
    const box = document.createElement('div');
    box.style.cssText = 'margin-top:14px;';
    el.appendChild(box);
    const { mountWorkerDoor } = await import('/static/shared/worker-door.js');
    mountWorkerDoor(box, 'flashhead', { title: 'Computers that may use this GPU' });
}

async function mountModels(el) {
    el.innerHTML = `<div style="margin-top:6px;">
        <div style="font-weight:600;margin-bottom:4px;">Where her models live now</div>
        <code class="fh-folder" style="display:block;word-break:break-all;padding:6px 8px;border-radius:6px;background:var(--bg-tertiary,rgba(127,127,127,.12));"></code>
        <div class="fh-parts" style="color:var(--text-muted);font-size:var(--font-sm,12px);margin-top:6px;"></div>
        <div class="fh-gpu" style="color:var(--text-muted);font-size:var(--font-sm,12px);margin-top:6px;"></div></div>`;
    try {
        const m = await (await fetch(`${BASE}/models`)).json();
        el.querySelector('.fh-folder').textContent = m.folder;
        el.querySelector('.fh-parts').textContent = m.parts
            .map(p => `${p.name}: ${p.present ? `${p.gb} GB` : 'not there yet (downloads on start)'}`).join(' · ');
    } catch { el.querySelector('.fh-folder').textContent = 'unknown'; }
    try {                                   // what her face takes of this card, as the server measured it at start
        const g = (await (await fetch(`${BASE}/status`)).json()).gpu;
        if (g) el.querySelector('.fh-gpu').textContent = `This GPU: ${g.name}, ${g.total_gb} GB`
            + (g.took_gb != null ? `; her face takes ${g.took_gb} GB (Video memory: ${g.preset}), ${g.speed}x real time`
                                 : ` (${g.free_gb} GB free when the service started)`);
    } catch {}
}

// Webcam: what her camera is doing now, in words, and the packages it needs.
function mountWebcam(el) {
    el.innerHTML = `<div style="margin-top:6px;display:flex;flex-direction:column;gap:8px;">
        <div class="fh-cam" style="font-size:var(--font-sm,12px);"></div>
        <div style="display:flex;gap:6px;">
            <button class="btn fh-cam-install" hidden>Install webcam packages</button>
            <button class="btn fh-cam-retry" hidden>↻ Try again</button>
        </div></div>`;
    const $ = s => el.querySelector(s);
    const retry = () => fetch(`${BASE}/camera`, { method: 'POST', headers: { 'X-CSRF-Token': csrf() } });
    async function tick() {
        if (!el.isConnected) return clearInterval(poll);
        try {
            const c = await (await fetch(`${BASE}/camera`)).json();
            $('.fh-cam').textContent = c.status === 'on'
                ? `On: camera ${c.device}${c.mic ? `, her voice on ${c.mic}` : ', no voice'}`
                : c.status === 'error' ? c.error : 'Off';
            $('.fh-cam-install').hidden = c.installed;
            $('.fh-cam-retry').hidden = !(c.installed && c.status === 'error');
        } catch { $('.fh-cam').textContent = 'status unavailable'; }
    }
    $('.fh-cam-install').onclick = async () => {
        const { ensureExtra } = await import('/static/shared/extras.js');
        if (await ensureExtra('plugin:flashhead')) { await retry(); tick(); }
    };
    $('.fh-cam-retry').onclick = async () => { await retry(); setTimeout(tick, 1500); };
    const poll = setInterval(tick, 3000);
    tick();
}

const RESTARTS = ['memory', 'port', 'models_dir', 'install_test', 'share', 'share_port'];      // read once at service start: a change restarts it

export default {
    name: 'flashhead',

    init() {
        let fields = [], saved = {};
        registerPluginSettings({
            id: 'flashhead',
            name: 'FlashHead',
            icon: '🗣️',
            helpText: 'Her live face: idle, screens, faces and moods, webcam',
            async load() {
                fields = await schema();
                saved = await pluginsAPI.getSettings('flashhead');
                return saved;
            },
            render: (container, settings) =>
                renderSettingsForm(container, fields, settings,
                    { slots: [{ tab: 'General', mount: mountMeasure },
                             { tab: 'Faces', mount: mountFaces }, { tab: 'Webcam', mount: mountWebcam },
                             { tab: 'Advanced', mount: el => mountAdvanced(el, saved.share === true) }] }),
            async save(s) {
                const out = await pluginsAPI.saveSettings('flashhead', s);
                const restart = RESTARTS.some(k => String(s[k] ?? '') !== String(saved[k] ?? ''));
                saved = { ...saved, ...s };
                if (restart) await fetch(`${BASE}/restart`, { method: 'POST', headers: { 'X-CSRF-Token': csrf() } });
                return out;
            },
            getSettings: container => readSettingsForm(container, fields),
            reset: () => pluginsAPI.resetSettings('flashhead'),
        });
    },

    destroy() {},
};
