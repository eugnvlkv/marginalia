from marginalia.models import Chunk

def split_text(text: str, max_char: int ) -> list[str]:
    pieces  = []    # закрытые коробки
    current = ""    # текущая коробка
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if len(paragraph) == 0:
            continue
        if len(current) == 0:
            current=paragraph
            continue
        elif max_char >= len(current) + 2 + len(paragraph):
            current += "\n\n" + paragraph
        else:
            pieces.append(current)
            current = ""
            current = paragraph
    if len(current) > 0:
        pieces.append(current)

    return pieces

# 1. идём по строкам;
# 2. это заголовок? (1–6 # и пробел после);
# 3. да → сохраняем lines (склеить → split_text → Chunk со старым путём), очищаем lines, обновляем путь ([:level-1] + название);
# 4. нет → строку кладём в lines;
# 5. после цикла → сохраняем то, что осталось.
def chunk_markdown(markdown: str, book_id: str, max_chars: int = 1500) -> list[Chunk]:
    chunks = []
    heading_path = ()
    lines = []

    for line in markdown.splitlines():
        rest = line.lstrip("#")
        level = len(line) - len(rest)

        if 1 <= level <= 6 and rest.startswith(" "):
            # 1) сохранить накопленный раздел — со СТАРЫМ путём
            for piece in split_text("\n".join(lines), max_chars):

                position = len(chunks)
                chunks.append(Chunk(
                    id=f"{book_id}:{position}",
                    book_id=book_id,
                    heading_path=heading_path,
                    text=piece,
                    position=position,
                ))
            # 2) начать новый раздел
            lines = []
            heading_path = heading_path[:level - 1] + (rest.strip(),)
        else:
            lines.append(line)

    # 3) после цикла: сохранить последний раздел
    for piece in split_text("\n".join(lines), max_chars):
        position = len(chunks)
        chunks.append(Chunk(
            id=f"{book_id}:{position}",
            book_id=book_id,
            heading_path=heading_path,
            text=piece,
            position=position,
        ))

    return chunks
