from marginalia.models import Chunk


BOOK = """Preface text before any heading.

# Chapter 4: Trends

Intro to trends.

## Trendlines

A trendline connects lows.

## Channels

A channel is two parallel lines.

# Chapter 5: Reversals

## Head and Shoulders

The most famous pattern.
"""


def chunk_markdown(markdown: str, book_id: str) -> list[Chunk]:
    chunks = []
    heading_path = ()
    lines = []

    for line in markdown.splitlines():
        rest = line.lstrip("#")
        level = len(line) - len(rest)

        if 1 <= level <= 6 and rest.startswith(" "):
            # 1) сохранить накопленный раздел — со СТАРЫМ путём
            text = "\n".join(lines).strip()
            if text:
                position = len(chunks)
                chunks.append(Chunk(
                    id=f"{book_id}:{position}",
                    book_id=book_id,
                    heading_path=heading_path,
                    text=text,
                    position=position,
                ))
            # 2) начать новый раздел
            lines = []
            heading_path = heading_path[:level - 1] + (rest.strip(),)
        else:
            lines.append(line)

    # 3) после цикла: сохранить последний раздел
    text = "\n".join(lines).strip()
    if text:
        position = len(chunks)
        chunks.append(Chunk(
            id=f"{book_id}:{position}",
            book_id=book_id,
            heading_path=heading_path,
            text=text,
            position=position,
        ))

    return chunks

				
				
				







	
