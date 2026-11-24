// Run against an isolated ComfyUI with the quantized service already warm.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

async function main() {
    const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_BIN});
    try {
        const origin = "http://127.0.0.1:" + (process.env.JR_TEST_PORT || "8298");
        const page = await browser.newPage();
        const workflowFile = process.env.JR_TEST_WORKFLOW || path.join(__dirname, "../example_workflows/07-中英实时语音.json");
        const workflow = JSON.parse(fs.readFileSync(workflowFile, "utf8"));
        await page.goto(origin);
        await page.waitForFunction(() => window.LiteGraph?.registered_node_types?.JRDrama3Realtime);
        const job = await page.evaluate(async ({graph, model}) => {
            const {app} = await import("/scripts/app.js");
            const {api} = await import("/scripts/api.js");
            await app.loadGraphData(graph);
            const node = app.graph._nodes.find(n => n.type === "JRDrama3Realtime");
            if (model) {
                const selector = node.widgets.find(w => w.name === "model");
                if (!selector?.options.values.includes(model)) throw Error("Missing model choice: " + model);
                selector.value = model;
            }
            await node.widgets.find(w => w.name === "启用本机流式试听").callback();
            const {output} = await app.graphToPrompt();
            const inputs = output[node.id].inputs;
            if (model && inputs.model !== model) throw Error("Selected model missing from queued prompt");
            if (inputs.seed !== 37003 || !inputs.live) throw Error("Workflow widget mismatch");
            if (graph.nodes.some(n => n.type === "LoadAudio") &&
                (!inputs.reference_audio || !inputs.reference_text || inputs.instructions !== "whisper")) {
                throw Error("Voice-control workflow lost its audio, transcript or direction");
            }
            const start = AudioBufferSourceNode.prototype.start;
            window.playback = {started: performance.now(), schedules: [], ended: false};
            AudioBufferSourceNode.prototype.start = function(when, ...args) {
                window.playback.schedules.push({when, duration: this.buffer.duration,
                    rate: this.buffer.sampleRate, state:this.context.state,
                    wall: (performance.now() - window.playback.started)/1000,
                    lead: when - this.context.currentTime});
                return start.call(this, when, ...args);
            };
            api.addEventListener("jr.drama3.audio", e => { if (e.detail.end) window.playback.ended = true; });
            const res = await api.fetchApi("/prompt", {method:"POST", headers:{"Content-Type":"application/json"},
                body:JSON.stringify({prompt:output,client_id:api.clientId})});
            if (!res.ok) throw Error(await res.text());
            return (await res.json()).prompt_id;
        }, {graph:workflow, model:process.env.JR_TEST_MODEL});
        await page.waitForFunction(() => window.playback?.ended, null, {timeout:120000});
        const packets = await page.evaluate(() => window.playback.schedules);
        assert.ok(packets.length > 1 && packets.every(p => p.rate === 44100 && p.state === "running"));
        let maxGap = 0;
        for (let i = 1; i < packets.length; i++) {
            maxGap = Math.max(maxGap, packets[i].when - packets[i-1].when - packets[i-1].duration);
        }
        const result = {job, scheduled_buffers:packets.length, first_scheduled_s:packets[0].wall,
            estimated_first_play_s:packets[0].wall+packets[0].lead, max_scheduled_gap_s:maxGap,
            duration_s:packets.reduce((s,p)=>s+p.duration,0)};
        if (process.env.JR_UI_RESULT) fs.writeFileSync(process.env.JR_UI_RESULT, JSON.stringify(result, null, 2));
        console.log(result);
        assert.ok(maxGap <= .001, "Browser scheduled a gap in playback");
    } finally {
        await browser.close();
    }
}
main().catch(e=>{console.error(e);process.exitCode=1;});
