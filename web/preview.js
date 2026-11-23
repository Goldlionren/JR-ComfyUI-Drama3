import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// A user gesture opens the audio device. Chunks only arrive for the client
// that queued a generation with live preview enabled.
let context;
let active;
let cursor = 0;
const scheduled = new Set();

function stopPlayback() {
    for (const source of scheduled) {
        try { source.stop(); } catch { /* Already played to completion. */ }
    }
    scheduled.clear();
    active = undefined;
    cursor = 0;
}

app.registerExtension({
    name: "jr.drama3.preview",
    nodeCreated(node) {
        if (!["JRDrama3Generate", "JRDrama3LiveText", "JRDrama3Realtime"].includes(node.comfyClass)) return;
        node.addWidget("button", "启用本机流式试听", null, async () => {
            context ??= new AudioContext();
            await context.resume();
            const live = node.widgets.find(w => w.name === "live");
            if (live) live.value = true;
            node.setDirtyCanvas(true);
        }, { serialize: false });
        node.addWidget("button", "停止试听", null, () => {
            stopPlayback();
            context?.suspend();
        }, { serialize: false });
    },
    setup() {
        api.addEventListener("jr.drama3.audio", ({ detail }) => {
            if (!context || context.state !== "running" || detail.end) return;
            if (detail.id !== active) {
                stopPlayback();
                active = detail.id;
            }
            const raw = atob(detail.pcm);
            if (!raw.length) return;
            const buffer = context.createBuffer(1, raw.length / 2, detail.rate);
            const channel = buffer.getChannelData(0);
            for (let i = 0; i < channel.length; ++i) {
                const sample = raw.charCodeAt(i * 2) | raw.charCodeAt(i * 2 + 1) << 8;
                channel[i] = (sample >= 32768 ? sample - 65536 : sample) / 32768;
            }
            const source = context.createBufferSource();
            source.buffer = buffer;
            source.connect(context.destination);
            scheduled.add(source);
            source.onended = () => scheduled.delete(source);
            // Leave startup headroom once and join subsequent buffers sample-to-sample.
            const now = context.currentTime;
            if (!cursor || cursor <= now) cursor = now + 0.05;
            source.start(cursor);
            cursor += buffer.duration;
        });
        api.addEventListener("execution_interrupted", stopPlayback);
        api.addEventListener("execution_error", stopPlayback);
    },
});
