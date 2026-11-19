"""Turn pytest JUnit XML or mypy output into GitHub Actions error annotations.

Annotations show inline on pull requests and can be read through the public checks API, so a failure can be
diagnosed without opening the job log.

    python scripts/ci_annotate.py junit report.xml
    python scripts/ci_annotate.py mypy mypy.txt
    python scripts/ci_annotate.py notices ci-notices.txt   (reports written by tests, as notices)
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# GitHub keeps at most 10 error annotations per step; one is reserved for the summary.
MAX_DETAILED = 9
MAX_CHARS = 6000
_MYPY = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+): error: (?P<message>.*)$")


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape(text).replace(":", "%3A").replace(",", "%2C")


def _annotate(
    message: str,
    *,
    title: str,
    file: str | None = None,
    line: int | None = None,
    level: str = "error",
    limit: int = MAX_CHARS,
) -> None:
    props = []
    if file:
        props.append(f"file={_escape_property(file)}")
        if line:
            props.append(f"line={line}")
    props.append(f"title={_escape_property(title)}")
    print(f"::{level} {','.join(props)}::{_escape(message[:limit])}")


def notices(path: Path) -> int:
    """Each block a test wrote (separated by a ===== line) becomes one notice; never fails the step."""
    if not path.exists():
        return 0
    blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("\n=====\n") if b.strip()]
    for block in blocks[:MAX_DETAILED]:
        title = block.splitlines()[0].lstrip("# ").strip() or "Report"
        _annotate(block, title=title[:120], level="notice", limit=60_000)
    return 0


def junit(path: Path) -> int:
    if not path.exists():
        _annotate(f"{path} was not written, so pytest did not finish.", title="pytest")
        return 1
    root = ET.parse(path).getroot()
    failures: list[tuple[str, str]] = []
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                body = (node.get("message") or "") + "\n" + (node.text or "")
                failures.append((name, body.strip()))
    if not failures:
        return 0
    names = "\n".join(name for name, _ in failures)
    _annotate(f"{len(failures)} failing test(s):\n{names}", title="pytest summary")
    for name, body in failures[:MAX_DETAILED]:
        # Keep the end of the traceback: that is where the assertion and its values are.
        _annotate(body[-MAX_CHARS:], title=name[-200:])
    return 1


def mypy(path: Path) -> int:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
    errors = [m for m in (_MYPY.match(line) for line in lines) if m]
    if not errors:
        return 0
    summary = "\n".join(lines[-40:])
    _annotate(f"{len(errors)} mypy error(s):\n{summary}", title="mypy summary")
    for match in errors[:MAX_DETAILED]:
        _annotate(match["message"], title="mypy", file=match["file"], line=int(match["line"]))
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in {"junit", "mypy", "notices"}:
        print(__doc__)
        return 2
    path = Path(argv[2])
    return {"junit": junit, "mypy": mypy, "notices": notices}[argv[1]](path)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
