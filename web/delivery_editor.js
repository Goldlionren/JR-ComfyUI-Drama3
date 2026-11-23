import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const css = `
  :host { display:block; color:#e5e8ed; font:12px/1.4 system-ui,sans-serif; }
  * { box-sizing:border-box; }
  .panel { background:#1c2129; border:1px solid #46505e; border-radius:6px; padding:9px; display:grid; gap:8px; }
  .bar, .form, .actions { display:flex; flex-wrap:wrap; gap:6px; align-items:end; }
  button, input, textarea, select { font:inherit; color:inherit; background:#252d37; border:1px solid #586474; border-radius:4px; padding:4px 6px; }
  button { cursor:pointer; } button:hover { background:#354250; }
  button:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible { outline:2px solid #95dbcf; }
  button:disabled,input:disabled,textarea:disabled { opacity:.48; cursor:default; }
  select { flex:1; min-width:105px; width:0; }
  .tracks { max-height:150px; overflow:auto; border-block:1px solid #394450; }
  .lane { display:grid; grid-template-columns:70px 1fr; min-height:31px; border-bottom:1px solid #354150; }
  .name { align-self:center; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding-right:5px; color:#aebccb; }
  .strip { position:relative; background:repeating-linear-gradient(90deg,#313b47 0 1px,transparent 1px 10%); }
  .clip { position:absolute; height:23px; min-width:5px; padding:0 3px; background:#345453; color:#d9efea;
          border:1px solid #587874; overflow:hidden; white-space:nowrap; text-align:left; }
  .clip.sel { border:2px solid #95dbcf; background:#426866; }
  .wave { width:100%; height:42px; border-radius:3px; background:#141920; }
  audio { width:100%; height:30px; }
  label { display:grid; gap:2px; color:#c0cad4; }
  label input { width:81px; } label.role input { width:102px; } label.text { width:100%; }
  textarea { resize:vertical; min-height:28px; height:34px; }
  .msg { color:#a9c9c6; min-height:15px; }
`;

function timelineEditor(node) {
    const state = node.widgets.find(w => w.name === "edits_json");
    if (!state) return;
    const host = document.createElement("div");
    const shadow = host.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = css;
    shadow.append(style);
    function make(tag, parent, label = "") {
        const e = document.createElement(tag);
        if (label) e.textContent = label;
        parent.append(e);
        return e;
    }
    const panel = make("div", shadow);
    panel.className = "panel";
    const bar = make("div", panel);
    bar.className = "bar";
    const select = make("select", bar);
    select.setAttribute("aria-label", "选择台词");
    const tracks = make("div", panel);
    tracks.className = "tracks";
    const wave = make("canvas", panel);
    wave.className = "wave";
    wave.width = 800;
    wave.height = 52;
    wave.setAttribute("aria-label", "原录音波形");
    const player = make("audio", panel);
    player.controls = true;
    player.preload = "none";
    const form = make("div", panel);
    form.className = "form";
    const fields = {};
    function field(key, label, kind="number") {
        const group = make("label", form, label);
        const input = make(kind === "textarea" ? "textarea" : "input", group);
        if (kind !== "textarea") input.type = kind;
        if (kind === "number") {
            input.min = "0";
            input.step = "0.001";
        }
        group.className = key;
        fields[key] = input;
    }
    field("role", "角色", "text");
    field("start_ms", "时间线入点 s");
    field("end_ms", "时间线出点 s");
    field("in_ms", "原音入点 s");
    field("out_ms", "原音出点 s");
    field("text", "字幕台词（沿用原录音）", "textarea");
    const actions = make("div", panel);
    actions.className = "actions";
    const msg = make("div", panel);
    msg.className = "msg";
    msg.setAttribute("aria-live", "polite");
    const button = (parent, label, fn) => {
        const b = make("button", parent, label);
        b.type = "button";
        b.onclick = fn;
        return b;
    };
    let incoming = [], rows = [], chosen = "", page = 0, undo = [], redo = [];
    const pageSize = 32;
    function patches() {
        const p = JSON.parse(state.value || "{}");
        if (!p || typeof p !== "object" || Array.isArray(p)) throw Error("编辑记录需要 JSON 对象。");
        return p;
    }
    function setEdits(p, history=true) {
        const previous = state.value || "{}";
        if (history) {
            undo.push(previous);
            if (undo.length > 60) undo.shift();
            redo = [];
        }
        state.value = JSON.stringify(p);
        state.callback?.(state.value);
        node.graph?.change?.();
        node.setDirtyCanvas?.(true, true);
        draw();
    }
    const find = () => rows.find(r => r.id === chosen);
    function writePatch(changes) {
        const p = patches(), row = find();
        if (!row) return;
        // JSON cue IDs can be arbitrary labels, including JavaScript property names.
        Object.defineProperty(p, chosen, {value:{...(Object.hasOwn(p, chosen) ? p[chosen] : {}), ...changes, source:row.key},
            enumerable:true, writable:true, configurable:true});
        setEdits(p);
    }
    const back = button(bar, "‹", () => { page--; chosen=""; draw(); });
    const next = button(bar, "›", () => { page++; chosen=""; draw(); });
    back.setAttribute("aria-label", "上一组台词");
    next.setAttribute("aria-label", "下一组台词");
    const apply = button(actions, "应用修改", () => {
        const row = find();
        if (!row || row.locked) return;
        const changes = {role:fields.role.value.trim(), text:fields.text.value.trim()};
        for (const name of ["start_ms","end_ms","in_ms","out_ms"]) {
            const value = fields[name].value;
            changes[name] = value === "" ? (name === "in_ms" ? 0 : null) : Math.round(Number(value) * 1000);
            if (changes[name] !== null && (!Number.isSafeInteger(changes[name]) || changes[name] < 0)) {
                msg.textContent = "请填写有效的秒数。";
                return;
            }
        }
        if (!changes.text || (changes.start_ms === null) !== (changes.end_ms === null)
            || (changes.end_ms !== null && changes.end_ms <= changes.start_ms)
            || (changes.out_ms ?? row.raw_ms) > row.raw_ms
            || changes.in_ms >= (changes.out_ms ?? row.raw_ms)) {
            msg.textContent = "台词不能为空。时间线须同时留空或填写有效区间，原音裁切须在录音范围内。";
            return;
        }
        writePatch(changes);
        msg.textContent = "变更已写入工作流。再次运行以更新混音与交付。";
    });
    const lock = button(actions, "锁定", () => {
        const row = find();
        if (row) writePatch({locked:!row.locked});
        msg.textContent = "再次运行以应用录音锁定状态。";
    });
    const previousAction = button(actions, "撤销", () => {
        if (undo.length) {
            redo.push(state.value || "{}");
            setEdits(JSON.parse(undo.pop()), false);
        }
    });
    const nextAction = button(actions, "重做", () => {
        if (redo.length) {
            undo.push(state.value || "{}");
            setEdits(JSON.parse(redo.pop()), false);
        }
    });
    const clear = button(actions, "清除编辑", () => setEdits({}));
    function choose(id) {
        player.pause();
        chosen = id;
        draw();
    }
    select.onchange = () => choose(select.value);
    player.addEventListener("play", () => {
        const row = find();
        if (row && (player.currentTime < row.in_ms / 1000 || player.currentTime >= (row.out_ms ?? row.raw_ms)/1000)) {
            player.currentTime = row.in_ms / 1000;
        }
    });
    player.addEventListener("timeupdate", () => {
        const row = find();
        if (row && !player.paused && row.out_ms !== null && player.currentTime >= row.out_ms/1000) player.pause();
    });
    function drawTracks(current) {
        tracks.replaceChildren();
        const a = Math.min(...current.map(r=>r.placed_start_ms));
        const b = Math.max(a+1, ...current.map(r=>r.placed_end_ms));
        const lanes = new Map();
        for (const row of [...current].sort((x,y)=>x.placed_start_ms-y.placed_start_ms)) {
            const role = row.role || "未分配";
            if (!lanes.has(role)) {
                const holder = make("div",tracks);
                holder.className = "lane";
                const label = make("span",holder,role);
                label.title = role;
                label.className = "name";
                const strip = make("div",holder);
                strip.className = "strip";
                lanes.set(role,{strip,ends:[]});
            }
            const {strip,ends} = lanes.get(role);
            let level = ends.findIndex(end => end <= row.placed_start_ms);
            if (level < 0) level = ends.length;
            ends[level] = row.placed_end_ms;
            strip.style.minHeight = (ends.length * 26 + 3)+"px";
            const box = button(strip,row.id+" "+row.text,()=>choose(row.id));
            box.className = "clip" + (row.id === chosen ? " sel" : "");
            box.style.left = (row.placed_start_ms-a)*100/(b-a)+"%";
            box.style.width = (row.placed_end_ms-row.placed_start_ms)*100/(b-a)+"%";
            box.style.top = (level*26+2)+"px";
            box.title = role+" · "+row.text;
        }
    }
    function draw() {
        let p;
        try { p = patches(); } catch (err) { msg.textContent = err.message; return; }
        let cursor = 0;
        rows = incoming.map(row => {
            const edit = Object.hasOwn(p,row.id) ? p[row.id] : {};
            const result = {...row, ...(row.base || {}), ...edit};
            result.placed_start_ms = result.start_ms ?? cursor;
            const audioEnd = result.placed_start_ms + (result.out_ms ?? result.raw_ms) - result.in_ms;
            result.placed_end_ms = result.end_ms ?? audioEnd;
            cursor = Math.max(cursor, audioEnd, result.placed_end_ms);
            return result;
        });
        const pages = Math.ceil(rows.length/pageSize);
        page = Math.max(0,Math.min(page,pages-1));
        const current = rows.slice(page*pageSize,(page+1)*pageSize);
        if (!current.some(row=>row.id === chosen)) chosen = current[0]?.id || "";
        select.replaceChildren();
        for (const r of current) {
            const option = make("option",select,r.id+" · "+(r.role || "未分配")+" · "+r.text.slice(0,34));
            option.value = r.id;
        }
        select.value = chosen;
        back.disabled = page === 0;
        next.disabled = page+1 >= pages;
        previousAction.disabled = !undo.length;
        nextAction.disabled = !redo.length;
        clear.disabled = !Object.keys(p).length;
        const row = find();
        apply.disabled = !row || row.locked;
        lock.disabled = !row;
        lock.textContent = row?.locked ? "解锁录音" : "锁定录音";
        for (const element of Object.values(fields)) element.disabled = apply.disabled;
        tracks.replaceChildren();
        if (!row) { msg.textContent = "连接已收集或恢复的时间线，运行一次以载入录音。"; return; }
        drawTracks(current);
        for (const [key,input] of Object.entries(fields)) {
            input.value = row[key] == null ? "" : key.endsWith("_ms") ? row[key]/1000 : row[key];
        }
        const ctx = wave.getContext("2d"), peaks = row.peaks || [];
        ctx.clearRect(0,0,wave.width,wave.height);
        ctx.fillStyle = "#87c5bc";
        peaks.forEach((v,i)=>{
            const h = Math.min(1,v)*wave.height;
            ctx.fillRect(i*wave.width/peaks.length,(wave.height-h)/2,Math.ceil(wave.width/peaks.length)-1,Math.max(1,h));
        });
        const query = new URLSearchParams({type:"temp", subfolder:"jr_drama3_audition", filename:row.preview});
        const url = api.apiURL("/view?" + query);
        if (player.getAttribute("src") !== url) player.setAttribute("src",url);
        msg.textContent = row.raw_ms/1000+" s 原录音 · 第 "+(page+1)+"/"+pages+" 组";
    }
    panel.addEventListener("pointerdown",e=>e.stopPropagation());
    panel.addEventListener("keydown",e=>e.stopPropagation());
    const display = node.addDOMWidget("jr_timeline_view","JR_TIMELINE_EDITOR",host,{serialize:false,hideOnZoom:false,
        getMinHeight:()=>405,getHeight:()=>405});
    display.computeSize = width => [Math.max(435,width || 435),405];
    node.setSize?.([Math.max(node.size[0],520),Math.max(node.size[1],500)]);
    node.jrDeliveryUpdate = (data) => {
        incoming = Array.isArray(data) ? data : [];
        draw();
    };
    draw();
}

app.registerExtension({
    name:"JR.Drama3.DeliveryEditor",
    async beforeRegisterNodeDef(type, data) {
        if (data.name !== "JRDrama3TimelineEditor") return;
        const original = type.prototype.onNodeCreated;
        type.prototype.onNodeCreated = function(...args) {
            const result = original?.apply(this,args);
            timelineEditor(this);
            return result;
        };
        const executed = type.prototype.onExecuted;
        type.prototype.onExecuted = function(output,...args) {
            const result = executed?.call(this,output,...args);
            this.jrDeliveryUpdate?.(output?.jr_drama3_timeline);
            return result;
        };
    }
});
