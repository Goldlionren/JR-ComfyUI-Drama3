"""Script, pronunciation and cast preparation, independent of the inference runtime."""

from __future__ import annotations

import calendar
import datetime
import difflib
import json
import re
import unicodedata


SPEAKER = re.compile(r"<\|speaker:(1[0-5]|[0-9])\|>")
SCOPE = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9_-]*)>")
PHONEME = r"<\|phoneme_start\|>[^<>\n]+<\|phoneme_end\|>"
MARKS = re.compile(PHONEME + r"|<\|[^\n<>]*\|>|</?[A-Za-z][A-Za-z0-9_-]*>|\[[^\[\]\n]+\]")
NAMED = re.compile(r"^([^:\n：<>\[\]]{1,60}?)\s*[:：]\s*(.+)$")
SPACE = re.compile(r"\s+")
# CMUdict permits an optional stress digit on vowels only.
CMU_PHONE = re.compile(r"(?:AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)[012]?|"
                       r"B|CH|D|DH|F|G|HH|JH|K|L|M|N|NG|P|R|S|SH|T|TH|V|W|Y|Z|ZH")


def as_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def normalize_name(name):
    return unicodedata.normalize("NFKC", name).strip().casefold()


def line_prefix(line):
    speaker = SPEAKER.match(line)
    if speaker:
        return line[:speaker.end()] + " ", line[speaker.end():].lstrip()
    named = NAMED.match(line)
    if named and normalize_name(named[1]) not in {"direction", "line", "http", "https"}:
        return line[:named.start(2)], named[2]
    return "", line


def check_markup(text):
    stack = []
    end = 0
    for match in MARKS.finditer(text):
        _check_plain(text[end:match.start()])
        token = match.group()
        scoped = SCOPE.fullmatch(token)
        if scoped:
            closing, name = scoped.groups()
            if closing:
                if not stack or stack.pop() != name:
                    raise ValueError(f"Unexpected closing tag {token} at position {match.start() + 1}")
            else:
                stack.append(name)
        elif token.startswith("<|") and not re.fullmatch(PHONEME, token):
            speaker = SPEAKER.fullmatch(token)
            if not speaker:
                raise ValueError(f"Unknown control or speaker outside 0–15: {token}")
        end = match.end()
    _check_plain(text[end:])
    if stack:
        raise ValueError(f"Missing closing tag </{stack[-1]}>")


def _check_plain(text):
    if re.search(r"[\[\]]|<\||</?[A-Za-z][A-Za-z0-9_-]*(?:\s|$)", text):
        raise ValueError(f"Unclosed or malformed cue near {text[:60]!r}")


def checked_cast(cast):
    output = []
    known_names = {}
    known_ids = {}
    for item in cast or []:
        if not isinstance(item, dict):
            raise ValueError("A cast entry must be a role object")
        idx, name = item.get("speaker"), item.get("name")
        if type(idx) is not int or not 0 <= idx <= 15 or not isinstance(name, str) or not name.strip():
            raise ValueError("Each cast member needs a name and a speaker ID from 0 to 15")
        key = normalize_name(name)
        if known_names.setdefault(key, idx) != idx or known_ids.setdefault(idx, key) != key:
            raise ValueError(f"Conflicting speaker binding for {name!r} and ID {idx}")
        if item.get("audio") is not None and not str(item.get("transcript") or "").strip():
            raise ValueError(f"Supply the exact transcript for {name}'s reference")
        output.append({**item, "speaker": idx, "name": name.strip(), "audio": item.get("audio"),
                       "transcript": item.get("transcript") or ""})
    return output


def build_script(script, cast=None, scene="", biographies="{}"):
    """Reserve explicit IDs before assigning names. Untagged continuations keep their speaker."""
    cast = checked_cast(cast)
    bios = json.loads(biographies or "{}")
    if not isinstance(bios, dict) or not all(isinstance(k, str) and k.strip() and isinstance(v, str)
                                              for k, v in bios.items()):
        raise ValueError('Biographies must be a JSON object, for example {"Lin":"quiet student"}')
    by_id = {item["speaker"]: item["name"] for item in cast}
    by_name = {normalize_name(name): i for i, name in by_id.items()}
    script = script.strip()
    check_markup(script)
    if not script:
        raise ValueError("The script is empty")
    numeric = re.compile(r"^\s*(\d{1,3})\s*[:：](?!\d)", re.M)
    for match in [*SPEAKER.finditer(script), *numeric.finditer(script)]:
        i = int(match[1])
        if i > 15:
            raise ValueError(f"Speaker ID {i} is outside 0–15")
        by_id.setdefault(i, f"Speaker {i}")
        by_name.setdefault(normalize_name(by_id[i]), i)

    def assign(name):
        key = normalize_name(name)
        if key in by_name:
            return by_name[key]
        ids = [i for i in range(16) if i not in by_id]
        if not ids:
            raise ValueError("A scene supports up to 16 named speakers")
        i = ids[0]
        by_name[key], by_id[i] = i, name.strip()
        return i

    for name in bios:
        assign(name)
    lines = []
    current = None
    for row in script.splitlines():
        # A role switch can also occur mid-line in pasted Fish scripts.
        for line in re.split(r"(?=<\|speaker:\d+\|>)", row.strip()):
            line = line.strip()
            if not line:
                continue
            speaker = SPEAKER.match(line)
            role = NAMED.match(line)
            if speaker:
                current, body = int(speaker[1]), line[speaker.end():].strip()
            elif numeric.match(line) and role:
                current, body = int(role[1]), role[2].strip()
            elif role and normalize_name(role[1]) not in {"direction", "line", "http", "https"} and not role[1].strip().isdigit():
                current, body = assign(role[1]), role[2].strip()
            else:
                if current is None:
                    current = min(by_id) if by_id else assign("Narrator")
                body = line
            if body:
                lines.append({"speaker": current, "name": by_id[current], "text": body})
    if not lines:
        raise ValueError("The script contains no dialogue")
    descriptions = {normalize_name(k): v for k, v in bios.items()}
    entries = []
    for i, name in sorted(by_id.items()):
        existing = [r for r in cast if r["speaker"] == i]
        for item in existing or [{"speaker": i, "name": name, "transcript": "", "audio": None}]:
            entries.append({**item, "description": descriptions.get(normalize_name(name), item.get("description", ""))})
    characters = {r["speaker"]: {k: r[k] for k in ("speaker", "name", "description")} for r in entries}
    return ("\n".join(f"<|speaker:{r['speaker']}|> {r['text']}" for r in lines), entries,
            {"scene": scene.strip(), "characters": list(characters.values()), "lines": lines})


def chunk_script(text, max_bytes, *, allow_empty=False):
    """Pack text units while closing and reopening scoped tags at chunk boundaries."""
    if max_bytes < 32:
        raise ValueError("The chunk budget must be at least 32 UTF-8 bytes")
    check_markup(text)
    chunks, warnings, stack = [], [], []
    active_speaker = ""
    parts, prefix, has_words = [], "", False

    def closing():
        return "".join(f"</{name}>" for name in reversed(stack))

    def reset():
        return active_speaker + "".join(f"<{name}>" for name in stack)

    def emit():
        nonlocal parts, prefix, has_words
        if has_words:
            result = (prefix + "".join(parts) + closing()).strip()
            chunks.append(result)
            if len(result.encode()) > max_bytes:
                warnings.append(f"Chunk {len(chunks)} exceeds {max_bytes} bytes to keep an indivisible cue, word or phoneme.")
        prefix, parts, has_words = reset(), [], False

    units = re.compile(MARKS.pattern + r"|\r?\n|[^\S\n]+|[A-Za-z0-9]+(?:['’.-][A-Za-z0-9]+)*|.", re.DOTALL)
    sentence_end = False
    for match in units.finditer(text):
        token = match.group()
        if SPEAKER.fullmatch(token):
            emit()
            active_speaker = token + " "
            prefix = reset()
            continue
        if token == "\n" or token == "\r\n":
            emit()
            continue
        scope = SCOPE.fullmatch(token)
        if scope and scope[1]:
            stack.pop()
            parts.append(token)
            continue
        if sentence_end and not token.isspace() and token not in '。！？!?.…”’」』':
            emit()
            sentence_end = False
        pending = prefix + "".join(parts) + token + closing()
        if scope:
            pending += f"</{scope[2]}>"
        if has_words and len(pending.encode()) > max_bytes:
            emit()
        if not has_words and token.isspace():
            continue
        parts.append(token)
        if scope:
            stack.append(scope[2])
        else:
            has_words = has_words or bool(token.strip())
            if token in {"。", "！", "？", "!", "?", "."}:
                sentence_end = True
    emit()
    if not chunks and not allow_empty:
        raise ValueError("The script contains no speakable chunks")
    return chunks, warnings


def apply_to_spoken(text, fn):
    """Leave control tokens and director annotations untouched during text normalization."""
    output, end = [], 0
    for m in MARKS.finditer(text):
        output.extend((fn(text[end:m.start()]), m.group()))
        end = m.end()
    output.append(fn(text[end:]))
    return "".join(output)


def preview(before, after):
    return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile="original",
                                          tofile="prepared", lineterm="")) or "(unchanged)"


def load_rules(raw):
    rows = json.loads(raw or "{}")
    if not isinstance(rows, dict):
        raise ValueError("Rules must be a JSON object keyed by the exact word or name")
    for k, v in rows.items():
        if not k.strip() or not isinstance(v, str) or not v.strip():
            raise ValueError("Every rule needs a nonempty source and replacement")
    return rows


def replace_rules(text, rules):
    if not rules:
        return text
    # Guard Latin word boundaries while permitting Chinese subphrases.
    patterns = []
    for k in sorted(rules, key=len, reverse=True):
        before = r"(?<![A-Za-z0-9_])" if re.match(r"[A-Za-z0-9_]", k[0]) else ""
        after = r"(?![A-Za-z0-9_])" if re.match(r"[A-Za-z0-9_]", k[-1]) else ""
        patterns.append(before + re.escape(k) + after)
    regex = re.compile("|".join(patterns))
    return apply_to_spoken(text, lambda p: regex.sub(lambda m: rules[m.group()], p))


def pronunciations(text, rules):
    result = {}
    for term, value in load_rules(rules).items():
        mode, sep, val = value.partition(":")
        if not sep or mode not in ("pinyin", "arpabet", "alias", "english", "chinese"):
            raise ValueError(f"Use pinyin:, arpabet:, alias:, english: or chinese: for {term!r}")
        val = val.strip()
        if not val or re.search(r"[\[\]<>]", val):
            raise ValueError(f"Invalid pronunciation or reading for {term!r}")
        if mode == "pinyin":
            # Fish's training helper uses pypinyin.to_tone3 with v_to_u=False.
            syllables = unicodedata.normalize("NFC", val).replace("ü", "v").split()
            if not all(re.fullmatch(r"[a-zê]+[1-5]", syllable) for syllable in syllables):
                raise ValueError(f"Use one lowercase pinyin syllable with a tone from 1–5 per character for {term!r}")
            result[term] = "".join(f"<|phoneme_start|>{s}<|phoneme_end|>" for s in syllables)
        elif mode == "arpabet":
            phones = val.split()
            if not all(CMU_PHONE.fullmatch(phone) for phone in phones):
                raise ValueError(f"Expected CMUdict phonemes for {term!r}; only vowels take stress digits 0–2")
            result[term] = f"<|phoneme_start|>{' '.join(phones)}<|phoneme_end|>"
        elif mode == "alias":
            result[term] = val
        else:
            result[term] = f"<{mode}>{val}</{mode}>"
    return replace_rules(text, result)


EN_SMALL = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
            "fifteen sixteen seventeen eighteen nineteen").split()
EN_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
ZH_DIGITS = "零一二三四五六七八九"


def read_number(value, lang):
    sign = ""
    if value.startswith("-"):
        sign, value = ("负" if lang == "zh" else "minus "), value[1:]
    whole, dot, fraction = value.partition(".")
    if len(whole) > 12 or (whole.startswith("0") and len(whole) > 1):
        digits = ZH_DIGITS if lang == "zh" else EN_SMALL
        reading = ("" if lang == "zh" else " ").join(digits[int(d)] for d in whole)
    elif lang == "en":
        def en(n):
            if n < 20:
                return EN_SMALL[n]
            if n < 100:
                return EN_TENS[n // 10] + ("-" + EN_SMALL[n % 10] if n % 10 else "")
            for base, unit in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
                if n >= base:
                    return en(n // base) + " " + unit + (" " + en(n % base) if n % base else "")
        reading = en(int(whole))
    else:
        def zh(n):
            if n < 10:
                return ZH_DIGITS[n]
            for base, unit in ((10**8, "亿"), (10000, "万"), (1000, "千"), (100, "百"), (10, "十")):
                if n >= base:
                    left, right = divmod(n, base)
                    return zh(left) + unit + (("零" if right * 10 < base else "") + zh(right) if right else "")
        reading = zh(int(whole))
        if reading.startswith("一十"):
            reading = reading[1:]
    if dot:
        reading += ("点" + "".join(ZH_DIGITS[int(d)] for d in fraction) if lang == "zh"
                    else " point " + " ".join(EN_SMALL[int(d)] for d in fraction))
    return sign + reading


def normalize_numbers(text, lang):
    if lang == "off":
        return text
    if lang not in ("zh", "en"):
        raise ValueError("Choose zh, en or off for the reading language")
    def change(m):
        value = m.group()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            try:
                d = datetime.date.fromisoformat(value)
            except ValueError:
                return value
            if lang == "zh":
                return "".join(ZH_DIGITS[int(c)] for c in value[:4]) + "年" + read_number(str(d.month), lang) + "月" + read_number(str(d.day), lang) + "日"
            day = read_number(str(d.day), lang)
            endings = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                       "eight": "eighth", "nine": "ninth", "twelve": "twelfth", "twenty": "twentieth",
                       "thirty": "thirtieth"}
            beginning, hyphen, last = day.rpartition("-")
            ordinal = beginning + hyphen + endings.get(last, last + "th")
            return f"{calendar.month_name[d.month]} {ordinal}, {read_number(str(d.year), lang)}"
        return read_number(value, lang)
    tokens = re.compile(r"(?<![A-Za-z0-9_.])(?:\d{4}-\d{2}-\d{2}|-?\d+(?:\.\d+)?)(?![A-Za-z0-9_]|\.\d)", re.ASCII)
    return apply_to_spoken(text, lambda part: tokens.sub(change, part))


def suggest(text, custom):
    cues = load_rules(custom)
    result = []
    for row in text.splitlines():
        head, body = line_prefix(row)
        if not body.strip() or body.startswith("["):
            result.append(row)
            continue
        cue = next((v for k, v in cues.items() if k.casefold() in body.casefold()), None)
        if cue is None:
            cue = ("questioning" if body.rstrip().endswith(("?", "？")) else
                   "with urgency" if body.rstrip().endswith(("!", "！")) else None)
        if cue:
            if any(c in cue for c in "[]<>\n"):
                raise ValueError("Suggestion rules take plain-language cues, without brackets")
            result.append(f"{head}[{cue}] {body}")
        else:
            result.append(row)
    return "\n".join(result)
