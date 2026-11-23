import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const sessions = new Map();

app.registerExtension({
    name: "jr.drama3.liveText",
    nodeCreated(node) {
        if (node.comfyClass !== "JRDrama3LiveText") return;
        const field = document.createElement("textarea");
        field.placeholder = "运行后在这里输入下一段台词";
        field.style.cssText = "width:100%;height:100%;box-sizing:border-box;";
        field.setAttribute("aria-label", "下一段台词");
        node.addDOMWidget("next_line", "JRDrama3LiveInput", field, { serialize: false });
        async function send(action) {
            const session = sessions.get(String(node.id));
            if (!session) { field.placeholder = "请先将这个节点加入队列"; return; }
            const text = field.value.trim();
            if (action === "append" && !text) return;
            try {
                const response = await api.fetchApi("/jr-drama3/canvas/text", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session, action, text }),
                });
                if (!response.ok) throw Error(await response.text());
                field.value = "";
                field.placeholder = action === "finish" ? "正在生成最后一段" : "已收到，可继续输入下一段";
            } catch (error) { field.placeholder = String(error); }
        }
        node.addWidget("button", "送入下一段", null, () => send("append"), {serialize:false});
        node.addWidget("button", "结束输入", null, () => send("finish"), {serialize:false});
    },
    setup() {
        api.addEventListener("jr.drama3.text", ({detail}) => {
            if (detail.active) sessions.set(detail.node, detail.session);
            else sessions.delete(detail.node);
        });
        api.addEventListener("reconnecting", () => sessions.clear());
    },
});
