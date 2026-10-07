"""Convert an EPUB book into one markdown document.

An EPUB is a zip archive of XHTML pages. Chapter titles are often images
inside the pages, so we take them from the table of contents (toc.ncx)
and put each chapter under a `# <chapter title>` heading.
"""

import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from markdownify import markdownify

NCX = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}

# Top-level TOC entries that carry no trading knowledge.
SKIP_TITLES = {
    "Cover",
    "Title Page",
    "Copyright",
    "Dedication",
    "Contents",
    "About the Author",
    "About the Contributors",
    "Acknowledgments",
    "Selected Bibliography",
    "Selected Resources",
    "Index",
}


def read_toc(book: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Top-level TOC entries as (title, path of the XHTML file inside the zip)."""
    ncx_path = next(name for name in book.namelist() if name.endswith(".ncx"))
    root = ET.fromstring(book.read(ncx_path))
    base = posixpath.dirname(ncx_path)

    entries = []
    for point in root.find("ncx:navMap", NCX).findall("ncx:navPoint", NCX):
        title = point.find("ncx:navLabel/ncx:text", NCX).text.strip()
        src = point.find("ncx:content", NCX).get("src").split("#")[0]
        entries.append((title, posixpath.join(base, src)))
    return entries


def chapter_markdown(html: str) -> str:
    """One chapter's XHTML as markdown, with headings pushed one level down."""
    body = re.search(r"<body[^>]*>(.*)</body>", html, re.DOTALL).group(1)
    md = markdownify(body, heading_style="ATX", strip=["img", "a"])
    # h1 -> ##, h2 -> ###: level 1 is reserved for the chapter title.
    md = re.sub(r"^(#{1,5}) ", r"#\1 ", md, flags=re.MULTILINE)
    # "## **INTRODUCTION**" -> "## INTRODUCTION"
    md = re.sub(r"^(#+ )\*\*(.*?)\*\*\s*$", r"\1\2", md, flags=re.MULTILINE)
    return md.strip()


def epub_to_markdown(path: Path) -> str:
    with zipfile.ZipFile(path) as book:
        parts = [
            f"# {title}\n\n{chapter_markdown(book.read(src).decode('utf-8'))}"
            for title, src in read_toc(book)
            if title not in SKIP_TITLES
        ]
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path("data/md") / f"{src.stem}.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(epub_to_markdown(src), encoding="utf-8")
    print(f"{src} -> {dst}")
