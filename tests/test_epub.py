import zipfile
from pathlib import Path

from marginalia.epub import chapter_markdown, epub_to_markdown

TOC = """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <navMap>
    <navPoint id="n1">
      <navLabel><text>Copyright</text></navLabel>
      <content src="xhtml/copyright.xhtml"/>
    </navPoint>
    <navPoint id="n2">
      <navLabel><text>1 Trends</text></navLabel>
      <content src="xhtml/ch1.xhtml"/>
      <navPoint id="n3">
        <navLabel><text>Trendlines</text></navLabel>
        <content src="xhtml/ch1.xhtml#s1"/>
      </navPoint>
    </navPoint>
  </navMap>
</ncx>"""

CHAPTER = """<html><head><title>ignored</title></head><body>
<p><img src="../image/ch1.jpg"/></p>
<h1><strong>TRENDLINES</strong></h1>
<p>A trendline connects <em>lows</em>.</p>
<h2>Drawing</h2>
<p>Use two points.</p>
</body></html>"""


def test_chapter_markdown_pushes_headings_down_and_drops_bold():
    md = chapter_markdown(CHAPTER)

    lines = md.splitlines()
    assert "## TRENDLINES" in lines
    assert "### Drawing" in lines
    assert "A trendline connects *lows*." in lines
    assert "ch1.jpg" not in md


def test_epub_takes_chapter_titles_from_toc_and_skips_front_matter(tmp_path: Path):
    epub = tmp_path / "book.epub"
    with zipfile.ZipFile(epub, "w") as book:
        book.writestr("OEBPS/toc.ncx", TOC)
        book.writestr(
            "OEBPS/xhtml/copyright.xhtml",
            "<html><body><p>All rights reserved.</p></body></html>",
        )
        book.writestr("OEBPS/xhtml/ch1.xhtml", CHAPTER)

    md = epub_to_markdown(epub)

    assert md.startswith("# 1 Trends\n\n## TRENDLINES")
    assert "All rights reserved" not in md
