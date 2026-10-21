"""Split loaded pages into token-bounded passages with traceable source spans."""

from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import TYPE_CHECKING, Iterable, Literal

from .loaders import DocumentPage

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_path: str
    page_number: int | None
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    extraction_method: Literal["text", "ocr"]


def _preferred_end(
    text: str,
    lower_bound: int,
    hard_end: int,
    paragraph_ends: list[int],
    sentence_ends: list[int],
    word_ends: list[int],
) -> int:
    for boundaries in (paragraph_ends, sentence_ends):
        index = bisect_right(boundaries, hard_end) - 1
        if index >= 0 and boundaries[index] > lower_bound:
            return boundaries[index]

    if hard_end < len(text) and not text[hard_end].isspace():
        index = bisect_right(word_ends, hard_end) - 1
        if index >= 0 and word_ends[index] > lower_bound:
            return word_ends[index]
    return hard_end


def chunk_page(
    page: DocumentPage,
    tokenizer: "PreTrainedTokenizerBase",
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 40,
    max_length: int = 512,
) -> list[Chunk]:
    """Chunk one page using the same fast tokenizer as the embedding model.

    chunk_size and token_count include model special tokens. max_length must
    match the encoder's configured limit. Overlap is measured in source tokens
    and may shrink to keep each chunk advancing. Offsets are zero-based,
    end-exclusive character positions in page.text; chunk_index is per page.
    No model or tokenizer is downloaded or loaded by this module.
    """
    if any(type(value) is not int for value in (chunk_size, chunk_overlap, max_length)):
        raise ValueError("chunk_size, chunk_overlap, and max_length must be integers")
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required to preserve character offsets")

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    token_budget = chunk_size - special_tokens
    limit = min(max_length, tokenizer.model_max_length)
    if token_budget <= 0:
        raise ValueError("chunk_size must leave room for text after special tokens")
    if chunk_size > limit:
        raise ValueError(f"chunk_size must not exceed the encoder limit of {limit} tokens")
    if not 0 <= chunk_overlap < token_budget:
        raise ValueError("chunk_overlap must be nonnegative and smaller than the text token budget")

    text = page.text
    if not text.strip():
        return []

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        verbose=False,
    )
    offsets = [(start, end) for start, end in encoded["offset_mapping"] if end > start]
    if not offsets:
        raise ValueError(f"No tokenizable text found in {page.source_path}")

    token_ends = [end for _, end in offsets]
    paragraph_ends = [match.start() for match in re.finditer(r"\n[ \t\r]*\n", text)]
    sentence_ends = [match.start() for match in re.finditer(r"(?<=[.!?…])\s+", text)]
    word_ends = [match.start() for match in re.finditer(r"\s+", text)]
    chunks = []
    start_token = 0

    while start_token < len(offsets):
        end_token = min(start_token + token_budget, len(offsets))
        start_char = 0 if start_token == 0 else offsets[start_token][0]
        end_char = len(text) if end_token == len(offsets) else offsets[end_token - 1][1]
        if end_token < len(offsets):
            midpoint = start_token + (end_token - start_token) // 2
            end_char = _preferred_end(
                text, offsets[midpoint][0], end_char,
                paragraph_ends, sentence_ends, word_ends,
            )

        while start_char < end_char and text[start_char].isspace():
            start_char += 1
        while end_char > start_char and text[end_char - 1].isspace():
            end_char -= 1
        end_token = bisect_right(token_ends, end_char)
        if start_char == end_char:
            start_token = max(start_token + 1, end_token)
            continue

        chunk_text = text[start_char:end_char]
        token_count = len(tokenizer.encode(chunk_text, add_special_tokens=True, truncation=False))
        while token_count > chunk_size:
            previous_end = end_char
            while end_token > start_token and offsets[end_token - 1][1] >= previous_end:
                end_token -= 1
            if end_token <= start_token:
                raise ValueError("chunk_size is too small to fit the next text character")
            end_char = offsets[end_token - 1][1]
            while end_char > start_char and text[end_char - 1].isspace():
                end_char -= 1
            end_token = bisect_right(token_ends, end_char)
            chunk_text = text[start_char:end_char]
            token_count = len(tokenizer.encode(chunk_text, add_special_tokens=True, truncation=False))

        identity = json.dumps(
            [page.source_path, page.page_number, start_char, end_char, chunk_text, page.extraction_method],
            ensure_ascii=False,
        ).encode("utf-8")
        chunks.append(Chunk(
            chunk_id=sha256(identity).hexdigest(),
            text=chunk_text,
            source_path=page.source_path,
            page_number=page.page_number,
            chunk_index=len(chunks),
            start_char=start_char,
            end_char=end_char,
            token_count=token_count,
            extraction_method=page.extraction_method,
        ))

        if end_token == len(offsets):
            break
        next_token = max(start_token + 1, end_token - chunk_overlap)
        while next_token < end_token and offsets[next_token][0] <= start_char:
            next_token += 1
        start_token = next_token

    return chunks


def chunk_pages(
    pages: Iterable[DocumentPage],
    tokenizer: "PreTrainedTokenizerBase",
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 40,
    max_length: int = 512,
) -> list[Chunk]:
    """Chunk loaded pages independently, preserving their original page numbers."""
    chunks = []
    for page in pages:
        chunks.extend(chunk_page(
            page, tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_length=max_length,
        ))
    return chunks
