import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "JR.Drama3.EditRegion",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "JRDrama3EditRegion") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const canvas = document.createElement("canvas");
            canvas.width = 720;
            canvas.height = 84;
            canvas.style.cssText = "width:100%;height:48px;touch-action:none;cursor:crosshair";
            canvas.setAttribute("aria-label", "拖动选择补录范围，单击插词。也可用起止输入框。");
            const widget = name => this.widgets.find(item => item.name === name);
            let data, anchor;
            const factor = () => ({seconds: 44100, samples: 1, codec_frames: 2048,
                video_frames: 44100 / widget("video_fps").value})[widget("units").value];
            const draw = () => {
                const c = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
                c.fillStyle = "#152028"; c.fillRect(0, 0, w, h);
                if (!data) {
                    c.fillStyle = "#c7dad6"; c.font = "21px system-ui";
                    c.fillText("运行节点载入波形", 12, h * 0.63);
                    return;
                }
                if (factor()) {
                    const x = key => w * widget(key).value * factor() / data.samples;
                    c.fillStyle = "#3b655a";
                    c.fillRect(x("start"), 0, Math.max(2, x("end") - x("start")), h);
                }
                c.strokeStyle = "#b7e8d7"; c.beginPath();
                for (let i = 0; i < data.min.length; i++) {
                    const x = w * i * data.stride / data.samples;
                    c.moveTo(x, (1-data.max[i])*h/2);
                    c.lineTo(x, Math.max((1-data.max[i])*h/2+1, (1-data.min[i])*h/2));
                }
                c.stroke();
            };
            const position = e => {
                const box = canvas.getBoundingClientRect();
                const sample = Math.max(0, Math.min(1, (e.clientX-box.left)/box.width)) * data.samples;
                if (widget("units").value === "seconds") return Math.round(sample)/44100;
                return Math.min(Math.round(sample/factor()), Math.floor(data.samples/factor()));
            };
            const select = e => {
                if (anchor === undefined) return;
                const next = position(e);
                widget("start").value = Math.min(anchor, next);
                widget("end").value = Math.max(anchor, next);
                this.graph?.change(); this.setDirtyCanvas(true, false); draw();
            };
            canvas.onpointerdown = e => {
                if (!data || !factor()) return;
                canvas.setPointerCapture(e.pointerId);
                anchor = position(e); select(e); e.stopPropagation();
            };
            canvas.onpointermove = select;
            canvas.onpointerup = e => { select(e); anchor = undefined; };
            canvas.onpointercancel = () => { anchor = undefined; };
            const dom = this.addDOMWidget("waveform", "custom", canvas, {serialize:false});
            dom.computeSize = width => [width,48];
            this.jrEditRegion = incoming => { data = incoming; draw(); };
            for (const key of ["units","start","end","video_fps"]) {
                const w = widget(key), callback = w.callback;
                w.callback = function (...values) { const r = callback?.apply(this,values); draw(); return r; };
            }
            draw();
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (payload) {
            executed?.call(this,payload);
            if (payload.region_wave?.[0]) this.jrEditRegion?.(payload.region_wave[0]);
        };
    }
});
