"""Subtitle and book readers for the local delivery nodes."""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET
from zipfile import ZipFile


def local_path(root, name):
    name = str(name).strip()
    if not name or ":" in name or PureWindowsPath(name).drive or PureWindowsPath(name).root:
        raise ValueError("Choose a path relative to the ComfyUI input or output folder.")
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("File is outside the selected ComfyUI folder.")
    return path


def cue(number, text, role="", chapter="", start_ms=None, end_ms=None):
    return dict(
        id=str(number),
        text=text.strip(),
        role=role.strip(),
        chapter=chapter.strip(),
        start_ms=start_ms,
        end_ms=end_ms,
    )


def validate(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("The cue sheet must contain at least one line.")
    seen = set()
    result = []
    for row in rows:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k, ""), str) for k in ("text", "role", "chapter")
        ):
            raise ValueError("A cue needs text, an ID, and optional role and chapter strings.")
        identifier = row.get("id")
        if not isinstance(identifier, (int, str)) or not str(identifier).strip() or str(identifier) in seen:
            raise ValueError(f"Missing or duplicate cue ID: {identifier}")
        a, b = row.get("start_ms"), row.get("end_ms")
        if (
            (a is None) != (b is None)
            or a is not None
            and (type(a) is not int or type(b) is not int or a < 0 or b <= a)
        ):
            raise ValueError(f"Invalid start/end in milliseconds for cue {identifier}.")
        if not row.get("text", "").strip():
            raise ValueError(f"Cue {identifier} has no text.")
        result.append(cue(identifier, row["text"], row.get("role", ""), row.get("chapter", ""), a, b))
        seen.add(str(identifier))
    return result


def role_line(text):
    match = re.match(r"^\s*(?:<v\s+([^>]+)>|【([^】\n]+)】|([^:\n：]{1,42})(?:：\s*|:\s+))(.*)$", text, re.S)
    if match:
        return next(s.strip() for s in match.groups()[:3] if s), match[4].removesuffix("</v>").strip()
    return "", text.strip()


STAMP = r"(\d{1,}:\d{2}:\d{2}[,.]\d{3})"
RANGE = re.compile(rf"^\s*{STAMP}\s*-->\s*{STAMP}(?:\s+.*)?$")


def time_ms(stamp):
    hours, minutes, seconds, millis = map(int, re.split(r"[:,.]", stamp))
    if minutes > 59 or seconds > 59:
        raise ValueError(f"Invalid subtitle timestamp: {stamp}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def parse_srt(content):
    rows = []
    for i, block in enumerate(
        re.split(r"\n\s*\n", content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()), 1
    ):
        lines = block.splitlines()
        if not lines:
            continue
        identifier = lines.pop(0).strip() if "-->" not in lines[0] else str(i)
        times = RANGE.fullmatch(lines.pop(0)) if lines else None
        if times is None:
            raise ValueError(f"Invalid SRT timestamp for cue {identifier}.")
        role, text = role_line("\n".join(lines))
        rows.append(cue(identifier, text, role, start_ms=time_ms(times[1]), end_ms=time_ms(times[2])))
    return validate(rows)


def subtitle_stamp(ms):
    s, msec = divmod(int(ms), 1000)
    minute, sec = divmod(s, 60)
    hour, minute = divmod(minute, 60)
    return f"{hour:02}:{minute:02}:{sec:02},{msec:03}"


def as_srt(rows):
    return (
        "\n\n".join(
            f"{i}\n{subtitle_stamp(r['start_ms'])} --> {subtitle_stamp(r['end_ms'])}\n"
            + (f"【{r['role']}】" if r["role"] else "")
            + r["text"]
            for i, r in enumerate(rows, 1)
        )
        + "\n"
    )


_BOOK_MAX = 128 * 1024 * 1024
_HEADING = re.compile(
    r"^(?:#{1,6}\s+.+|第[\d零〇一二三四五六七八九十百千万两]+[章节回卷部篇].*|(?:chapter|part|book)\s+[\w.-]+(?:\b.*|$))$",
    re.I,
)


def read_xml(data):
    if b"<!entity" in data.lower():
        raise ValueError("EPUB XML entity declarations are unsupported.")
    return ET.fromstring(data)


def epub_contents(path):
    with ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > _BOOK_MAX:
            raise ValueError("Expanded EPUB exceeds 128 MiB.")
        container = read_xml(z.read("META-INF/container.xml"))
        opf = container.find(".//{*}rootfile")
        if opf is None:
            raise ValueError("Missing EPUB package document.")
        package_name = opf.attrib["full-path"]
        package = read_xml(z.read(package_name))
        items = {it.get("id"): it for it in package.findall(".//{*}manifest/{*}item")}
        for ref in package.findall(".//{*}spine/{*}itemref"):
            if ref.get("linear") == "no":
                continue
            item = items.get(ref.get("idref"))
            if item is None:
                raise ValueError("EPUB spine refers to an unknown document.")
            if item.get("media-type") not in ("application/xhtml+xml", "text/html"):
                continue
            url = urlsplit(item.attrib["href"])
            if url.scheme or url.netloc:
                raise ValueError("Remote EPUB resources are unsupported.")
            name = posixpath.normpath(posixpath.join(posixpath.dirname(package_name), unquote(url.path)))
            if name.startswith(("../", "/")):
                raise ValueError("Invalid EPUB member path.")
            doc = read_xml(z.read(name))
            body = doc.find(".//{*}body")
            if body is None:
                continue
            heading = body.find(".//{*}h1")
            if heading is None:
                heading = body.find(".//{*}h2")
            title = "".join(heading.itertext()).strip() if heading is not None else PurePosixPath(name).stem

            def extract(el):
                tag = el.tag.rsplit("}", 1)[-1]
                if el is heading or tag in ("script", "style", "nav"):
                    return
                if tag in ("p", "h1", "h2", "h3", "li", "blockquote"):
                    yield "".join(el.itertext()).strip()
                else:
                    if el.text and el.text.strip():
                        yield el.text.strip()
                    for child in el:
                        yield from extract(child)
                        if child.tail and child.tail.strip():
                            yield child.tail.strip()

            yield title, list(extract(body))


def read_book(path):
    path = Path(path)
    if path.stat().st_size > _BOOK_MAX:
        raise ValueError("Book exceeds 128 MiB.")
    if path.suffix.lower() == ".epub":
        sections = epub_contents(path)
    elif path.suffix.lower() == ".txt":
        sections = []
        title, lines = path.stem, []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if _HEADING.fullmatch(line) and len(line) <= 150:
                if lines:
                    sections.append((title, lines))
                title, lines = line.lstrip("# ").strip(), []
            elif line:
                lines.append(line)
        sections.append((title, lines))
    else:
        raise ValueError("Choose an EPUB or UTF-8 TXT file.")
    rows = []
    for title, lines in sections:
        for line in lines:
            if line.strip():
                role, text = role_line(line)
                rows.append(cue(len(rows) + 1, text, role, title))
    return validate(rows)


def sheet_json(rows):
    return json.dumps(rows, indent=2, ensure_ascii=False)
