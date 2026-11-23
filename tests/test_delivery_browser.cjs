// Run against an isolated ComfyUI instance with this extension installed.
// The test creates a short local input fixture and its own browser session.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const origin = "http://127.0.0.1:" + (process.env.JR_TEST_PORT || "8297");

function wav() {
    const samples = 8000, rate = 16000;
    const b = Buffer.alloc(44 + samples * 2);
    b.write("RIFF"); b.writeUInt32LE(b.length-8,4); b.write("WAVEfmt ",8);
    b.writeUInt32LE(16,16); b.writeUInt16LE(1,20); b.writeUInt16LE(1,22);
    b.writeUInt32LE(rate,24); b.writeUInt32LE(2*rate,28); b.writeUInt16LE(2,32);
    b.writeUInt16LE(16,34); b.write("data",36); b.writeUInt32LE(2*samples,40);
    for (let i=0;i<samples;i++) b.writeInt16LE(Math.round(10000*Math.sin(2*Math.PI*420*i/rate)),44+i*2);
    return b;
}

async function main() {
    const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_BIN});
    try {
        const ctx = await browser.newContext({viewport:{width:1450,height:1000}, locale:"en-US"});
        const page = await ctx.newPage();
        const errors = [];
        page.on("pageerror", e=>errors.push(String(e)));
        const audio = await ctx.request.post(origin+"/upload/image", {multipart:{
            image:{ name:"jr-delivery-ui-smoke.wav",mimeType:"audio/wav",buffer:wav() }, overwrite:"true"
        }});
        assert.equal(audio.status(),200,await audio.text());
        await page.goto(origin);
        await page.waitForFunction(() => window.LiteGraph?.registered_node_types?.JRDrama3TimelineEditor, {timeout:30000});
        const examples = path.resolve(__dirname,"../example_workflows");
        for (const filename of fs.readdirSync(examples).filter(x => /^0[1-5]-.*\.json$/.test(x)).sort()) {
            const doc = JSON.parse(fs.readFileSync(path.join(examples,filename),"utf8").replace(/^\uFEFF/,""));
            const values = await page.evaluate(async doc => {
                const { app } = await import("/scripts/app.js");
                await app.loadGraphData(doc);
                const { output } = await app.graphToPrompt();
                return {output, widgets:app.graph._nodes.map(n=>[n.id,n.widgets?.map(w=>[w.name,w.value])])};
            }, doc);
            console.log(filename,JSON.stringify(values.output));
            for (const n of doc.nodes) {
                assert.ok(values.output[n.id],n.type+" missing from prompt");
                const promptInput = values.output[n.id].inputs;
                for (const input of n.inputs || []) {
                    if(input.link == null) continue;
                    const link = doc.links.find(l=>l[0]===input.link);
                    assert.deepEqual(promptInput[input.name],[String(link[1]),link[2]],filename+" / "+input.name);
                }
                const ws = values.widgets.find(([id])=>id===n.id)?.[1] || [];
                for (const [index, widget] of ws.entries()) {
                    const [name, val] = widget;
                    if (index >= (n.widgets_values || []).length || !(name in promptInput) || Array.isArray(promptInput[name])) continue;
                    assert.equal(val,n.widgets_values[index],filename+" widget "+name);
                    assert.equal(promptInput[name],val,filename+" serialized "+name);
                }
            }
        }
        await page.evaluate(async () => {
            const { app } = await import("/scripts/app.js");
            const { api } = await import("/scripts/api.js");
            app.graph.clear();
            const editor = window.LiteGraph.createNode("JRDrama3TimelineEditor");
            app.graph.add(editor);
            editor.pos = [20,20];
            app.canvas.centerOnNode?.(editor);
            window.jrTest = {
                editor, api, app,
                async render() {
                    const graph = {
                        "10":{class_type:"LoadAudio",inputs:{audio:"jr-delivery-ui-smoke.wav"}},
                        "11":{class_type:"JRDrama3ReadSRT",inputs:{srt:"1\n00:00:00,000 --> 00:00:01,000\n【甲】Hello",input_file:""}},
                        "12":{class_type:"JRDrama3PutTake",inputs:{cues:["11",0],audio:["10",0],index:1,settings_json:"{}"}},
                        [String(editor.id)]:{class_type:"JRDrama3TimelineEditor",inputs:{
                            timeline:["12",0],edits_json:editor.widgets.find(w=>w.name==="edits_json").value
                        }}
                    };
                    const response = await api.fetchApi("/prompt",{method:"POST",headers:{"Content-Type":"application/json"},
                        body:JSON.stringify({prompt:graph,client_id:api.clientId})});
                    const result = await response.json();
                    if (!response.ok) throw Error(JSON.stringify(result));
                    return result.prompt_id;
                }
            };
        });
        async function queue() {
            const id = await page.evaluate(()=>window.jrTest.render());
            let entry;
            for(let i=0;i<100;i++) {
                const history = await (await ctx.request.get(origin+"/history/"+id)).json();
                entry = history[id];
                if(entry) break;
                await new Promise(resolve=>setTimeout(resolve,100));
            }
            assert.equal(entry?.status.status_str,"success",JSON.stringify(entry?.status));
            const idOnGraph = await page.evaluate(()=>String(window.jrTest.editor.id));
            const result = entry.outputs[idOnGraph];
            // Background tabs may miss the WebSocket event; dispatch the exact
            // returned node payload through ComfyUI's normal frontend hook.
            await page.evaluate(output=>window.jrTest.editor.onExecuted(output),result);
            return result.jr_drama3_timeline[0];
        }
        const first = await queue();
        assert.equal(first.role,"甲");
        const role = page.getByLabel("角色",{exact:true});
        await role.fill("乙");
        await page.getByLabel("原音入点 s").fill("0.1");
        await page.getByLabel("原音出点 s").fill("0.4");
        const button = title=>page.locator("button").filter({hasText:new RegExp("^"+title+"$")});
        await button("应用修改").click();
        await button("锁定录音").click();
        const revised = await queue();
        assert.equal(revised.role,"乙");
        assert.equal(revised.locked,true);
        assert.equal(revised.in_ms,100);
        assert.equal(revised.out_ms,400);
        const preview = await page.locator("audio").last().getAttribute("src");
        assert.equal((await ctx.request.get(new URL(preview,origin).href)).status(),200);
        const serialized = await page.evaluate(()=>JSON.stringify(window.jrTest.app.graph.serialize()));
        assert.ok(serialized.includes("out_ms"),"Time edits must serialize with the graph");
        if(process.env.JR_UI_SCREENSHOT) await button("应用修改").locator("xpath=../..").screenshot({path:process.env.JR_UI_SCREENSHOT});
        await button("撤销").click();
        assert.equal(await button("锁定录音").count(),1);
        assert.deepEqual(errors.filter(e=>/jr_drama3|delivery_editor/i.test(e)),[]);
        console.log("ComfyUI canvas editor, real execution, audible preview, scoped trim, lock, saved graph and undo passed.");
    } finally {
        await browser.close();
    }
}

main().catch(error=>{ console.error(error);process.exitCode=1; });
