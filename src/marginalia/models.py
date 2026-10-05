from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    id: str
    book_id: str
    heading_path: tuple[str, ...]
    text: str
    position: int
