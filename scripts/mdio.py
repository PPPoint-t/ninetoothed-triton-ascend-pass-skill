"""Strict Chinese Markdown records and atomic local checkpoints."""

import contextlib
import fcntl
import hashlib
import html
import os
import tempfile
from pathlib import Path


class WorkflowError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def cell(value):
    return (
        html.escape(str(value), quote=False)
        .replace("|", "&#124;")
        .replace("\n", "<br>")
    )


def table(headers, rows):
    lines = ["| " + " | ".join(map(cell, headers)) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def read_table(path, headers):
    lines = Path(path).read_text().splitlines()
    found = []
    active = False
    for line in lines:
        if not line.startswith("| ") or not line.endswith(" |"):
            active = False
            continue
        parts = [
            html.unescape(x.strip().replace("<br>", "\n"))
            for x in line.strip("|").split("|")
        ]
        if parts == list(headers):
            if found or active:
                raise WorkflowError(f"重复表格：{path}")
            active = True
            continue
        if not active or all(x == "---" for x in parts):
            continue
        if len(parts) != len(headers):
            raise WorkflowError(f"表格列数错误：{path}")
        found.append(dict(zip(headers, parts)))
    if not found:
        raise WorkflowError(f"缺少表格或表格为空：{path}，列：{headers}")
    return found


def fields(path):
    result = {}
    for row in read_table(path, ("项目", "内容")):
        if row["项目"] in result:
            raise WorkflowError(f"重复字段：{path}：{row['项目']}")
        result[row["项目"]] = row["内容"]
    if result.get("记录格式版本") != "1":
        raise WorkflowError(f"不支持的记录格式：{path}")
    return result


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".写入-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_fields(path, title, values, body=""):
    values = {"记录格式版本": "1", **values}
    atomic_write(
        path, f"# {title}\n\n" + table(("项目", "内容"), values.items()) + "\n" + body
    )


def update_fields(path, title, values):
    existing = Path(path).read_text()
    parts = existing.split("\n## ", 1)
    body = "## " + parts[1] if len(parts) == 2 else ""
    write_fields(path, title, values, body)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkflowError(f"已有工作流占用锁：{path}") from exc
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
