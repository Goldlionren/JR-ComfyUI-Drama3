"""ComfyUI V3 authoring and casting nodes for Drama 3."""

from __future__ import annotations

import json

from comfy_api.latest import io, ui

from .jr_script_core import (as_json, build_script, check_markup, checked_cast, chunk_script, line_prefix,
                             normalize_numbers, preview, pronunciations, suggest)
from .jr_script_library import auto_cast, find_references, get_reference, prepare_reference, save_reference, voice_db


CATEGORY = "audio/JR Drama 3/台本与选角"
CAST = io.Custom("DRAMA3_CAST")
NAME = "角色名称"


def text_input(key, default="", tip=""):
    return io.String.Input(key, default=default, multiline=True, tooltip=tip)


def text_output(key):
    return io.String.Output(id=key, display_name=key)


def db_signature():
    db = voice_db()
    try:
        st = db.stat()
    except FileNotFoundError:
        return None
    return (str(db), st.st_size, st.st_mtime_ns, st.st_ctime_ns)


class JRDrama3Reference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 角色与声音参考", category=CATEGORY,
                         description="可链式连接最多 16 位角色。同名同编号可添加多份参考录音。无参考时可填写声音描述。",
                         inputs=[io.String.Input("name", default="Lin", tooltip=NAME),
                                 io.Int.Input("speaker", default=0, min=0, max=15),
                                 text_input("transcript", tip="填写所选音频区间的逐字稿，可接本地 ASR 输出。"),
                                 text_input("description", tip="角色小传、声音描述或选角关键词。"),
                                 io.String.Input("language", default=""), io.Audio.Input("audio", optional=True),
                                 CAST.Input("cast", optional=True)],
                         outputs=[CAST.Output(display_name="cast")])

    @classmethod
    def execute(cls, name, speaker, transcript, description, language, audio=None, cast=None):
        item = dict(name=name, speaker=speaker, transcript=transcript.strip(), audio=audio,
                    description=description.strip(), language=language.strip())
        return io.NodeOutput(checked_cast([*(cast or []), item]))


class JRDrama3TrimReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 参考音裁剪", category=CATEGORY,
                         inputs=[io.Audio.Input("audio"),
                                 io.Float.Input("start_seconds", default=0., min=0., max=86400., step=.01),
                                 io.Float.Input("end_seconds", default=0., min=0., max=86400., step=.01,
                                                tooltip="0 表示到音频结尾。"),
                                 io.Boolean.Input("trim_edges", default=True),
                                 io.Float.Input("silence_db", default=-50., min=-100., max=0., step=1.),
                                 io.Boolean.Input("normalize_peak", default=False, tooltip="调整到 −1 dBFS 峰值。")],
                         outputs=[io.Audio.Output(), text_output("selection_json")])

    @classmethod
    def execute(cls, audio, start_seconds, end_seconds, trim_edges, silence_db, normalize_peak):
        clip, info = prepare_reference(audio, start_seconds, end_seconds, trim_edges, silence_db, normalize_peak)
        return io.NodeOutput(clip, as_json(info))


class JRDrama3SaveVoice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 保存音色版本", category=CATEGORY,
                         is_output_node=True, not_idempotent=True,
                         description="将参考录音、逐字稿及元数据作为可按 ID 取回的版本保存在本机 ComfyUI 用户目录。",
                         inputs=[io.Audio.Input("audio"), io.String.Input("name", default="Lin"),
                                 text_input("transcript"), text_input("description"), io.String.Input("language", default=""),
                                 io.String.Input("tags", default="")], outputs=[text_output("voice_id")])

    @classmethod
    def execute(cls, audio, name, transcript, description, language, tags):
        ref_id = save_reference(voice_db(), name, language, description, tags, transcript, audio)
        return io.NodeOutput(ref_id, ui=ui.PreviewText(f"{name} · {ref_id}"))


class JRDrama3FindVoice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 检索与载入音色", category=CATEGORY,
                         description="按名称、描述、标签或完整版本 ID 查找。空查询按保存时间排列，rank=1 选择首项。",
                         inputs=[io.String.Input("query", default=""), io.String.Input("language", default=""),
                                 io.Int.Input("rank", default=1, min=1, max=100000)],
                         outputs=[io.Audio.Output(), text_output("transcript"), text_output("voice_id"),
                                  text_output("matches_json")])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return db_signature()

    @classmethod
    def execute(cls, query, language, rank):
        rows = find_references(voice_db(), query, language)
        if not 1 <= rank <= len(rows):
            raise ValueError(f"找到 {len(rows)} 个候选。请修改检索词、语言或序号。")
        row = rows[rank - 1]
        audio, _ = get_reference(voice_db(), row["voice_id"])
        return io.NodeOutput(audio, row["transcript"], row["voice_id"], as_json(rows))


class JRDrama3AutoCast(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 按描述选角", category=CATEGORY,
                         description="从本机已保存的参考录音中匹配角色描述和语言；保留手工绑定，也可按音色版本 ID 覆盖。",
                         inputs=[CAST.Input("cast"), text_input("overrides", "{}", '示例 {"Lin":"完整音色 ID"}')],
                         outputs=[CAST.Output(display_name="cast"), text_output("selections_json")])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return db_signature()

    @classmethod
    def execute(cls, cast, overrides):
        assigned, info = auto_cast(voice_db(), cast, json.loads(overrides or "{}"))
        return io.NodeOutput(assigned, as_json(info))


class JRDrama3Script(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 多角色台本与场景", category=CATEGORY,
                         inputs=[text_input("script", "Lin：准备好了吗？\nBo: Ready.", "角色名：台词，每行一轮；也可直接输入 speaker 标签。"),
                                 text_input("scene", tip="场景描述，用于整场生成。角色小传不会混入台词。"),
                                 text_input("biographies", "{}", '例如 {"Lin":"沉稳的女导演","Bo":"轻快的青年男声"}'),
                                 CAST.Input("cast", optional=True)],
                         outputs=[text_output("text"), CAST.Output(display_name="cast"), text_output("scene_json")])

    @classmethod
    def execute(cls, script, scene, biographies, cast=None):
        text, roles, scenes = build_script(script, cast, scene, biographies)
        return io.NodeOutput(text, roles, as_json(scenes))


class JRDrama3Direct(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 导演与标记校验", category=CATEGORY,
                         description="在每轮开头加可选的导演提示，保留台词中的转折、点标记和范围标记。检查闭合及角色编号。",
                         inputs=[text_input("text"), text_input("direction", tip="可用中文或英语。为空时仅校验并透传。")],
                         outputs=[text_output("text"), text_output("preview")])

    @classmethod
    def execute(cls, text, direction):
        check_markup(text)
        direction = direction.strip()
        if any(c in direction for c in "[]<>"):
            raise ValueError("导演意见请使用自然语言；范围标记可直接写在台词中。")
        lines = []
        for line in text.splitlines():
            if direction and line.strip():
                head, body = line_prefix(line)
                line = f"{head}[{direction}] {body}"
            lines.append(line)
        directed = "\n".join(lines)
        return io.NodeOutput(directed, preview(text, directed))


class JRDrama3Pronunciation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 中英发音词典", category=CATEGORY,
                         description="词典随工作流保存。alias 指定读法，pinyin 和 arpabet 生成音素标记，"
                                     "english/chinese 切换局部语言（实验）。可选择数字与 ISO 日期的朗读语言。",
                         inputs=[text_input("text"),
                                 text_input("dictionary", "{}", '例如 {"重庆":"pinyin:chong2 qing4","Neil":"arpabet:N IY1 L","AI":"alias:A I"}'),
                                 io.Combo.Input("number_language", options=["off", "zh", "en"])],
                         outputs=[text_output("model_text"), text_output("original_text"), text_output("preview")])

    @classmethod
    def execute(cls, text, dictionary, number_language):
        check_markup(text)
        result = normalize_numbers(pronunciations(text, dictionary), number_language)
        check_markup(result)
        return io.NodeOutput(result, text, preview(text, result))


class JRDrama3SuggestCues(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 导演标记建议", category=CATEGORY,
                         description="离线规则按关键词、问号和感叹号提出建议。预览差异后打开 apply，或把修改稿填入 edited。",
                         inputs=[text_input("text"), text_input("rules", '{"悄悄":"轻声，靠近对方","secret":"under your breath"}'),
                                 io.Boolean.Input("apply", default=False), text_input("edited")],
                         outputs=[text_output("text"), text_output("suggestion"), text_output("diff")])

    @classmethod
    def execute(cls, text, rules, apply, edited):
        check_markup(text)
        proposed = edited if edited.strip() else suggest(text, rules)
        check_markup(proposed)
        diff = preview(text, proposed)
        return io.NodeOutput(proposed if apply else text, proposed, diff, ui=ui.PreviewText(diff))


class JRDrama3Chunks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 长文分块", category=CATEGORY,
                         description="按台词换行和 UTF-8 字节分块，为跨块的范围标记补足开闭，并保留当前角色。"
                                     "chunks 为可连接生成节点的 STRING 列表；需要连续上下文时使用长文渲染节点。",
                         inputs=[text_input("text"), io.Int.Input("max_bytes", default=300, min=32, max=32768)],
                         outputs=[io.String.Output(id="chunks", display_name="chunks", is_output_list=True),
                                  text_output("chunks_json"), text_output("warnings")])

    @classmethod
    def execute(cls, text, max_bytes):
        chunks, warnings = chunk_script(text, max_bytes)
        return io.NodeOutput(chunks, as_json(chunks), "\n".join(warnings))


NODES = [JRDrama3Reference, JRDrama3TrimReference, JRDrama3SaveVoice, JRDrama3FindVoice,
         JRDrama3AutoCast, JRDrama3Script, JRDrama3Direct, JRDrama3Pronunciation,
         JRDrama3SuggestCues, JRDrama3Chunks]
