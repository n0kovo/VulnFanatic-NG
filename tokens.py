"""Token counting and a priority-ordered context budgeter.

Uses ``tiktoken`` when available for accurate counts (and accurate truncation);
otherwise falls back to a conservative character heuristic. The exact tokenizer
of a local model may differ, but this is only used as a size cap, so an estimate
is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

try:
    import tiktoken  # type: ignore

    _HAVE_TIKTOKEN = True
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore
    _HAVE_TIKTOKEN = False

# Average bytes/token heuristic used when tiktoken is unavailable.
_CHARS_PER_TOKEN = 4

_ENC_CACHE: dict = {}


def _get_encoding(encoding_name: str):
    if not _HAVE_TIKTOKEN:
        return None
    if encoding_name not in _ENC_CACHE:
        try:
            _ENC_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
        except Exception:
            try:
                _ENC_CACHE[encoding_name] = tiktoken.get_encoding("cl100k_base")
            except Exception:
                _ENC_CACHE[encoding_name] = None
    return _ENC_CACHE[encoding_name]


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    if not text:
        return 0
    enc = _get_encoding(encoding_name)
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def truncate_to_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Return ``text`` cut to at most ``max_tokens`` tokens."""
    if max_tokens <= 0:
        return ""
    enc = _get_encoding(encoding_name)
    if enc is not None:
        try:
            toks = enc.encode(text, disallowed_special=())
            if len(toks) <= max_tokens:
                return text
            return enc.decode(toks[:max_tokens])
        except Exception:
            pass
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


@dataclass
class Section:
    label: str
    text: str
    truncated: bool = False
    dropped: bool = False


@dataclass
class Budget:
    """Assemble labeled context sections under a token cap.

    Sections are added in priority order (most important first). Each section is
    added in full if it fits; otherwise, if ``truncatable`` it is cut to the
    remaining budget, and if not it is dropped. A trailing note records what was
    truncated/dropped so the model knows the context is partial.
    """

    max_tokens: int
    encoding_name: str = "cl100k_base"
    used: int = 0
    sections: List[Section] = field(default_factory=list)
    _overhead_per_section: int = 6  # header/newline tokens, approximate

    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used)

    def add(self, label: str, text: str, truncatable: bool = True) -> Section:
        text = text or ""
        cost = count_tokens(text, self.encoding_name) + self._overhead_per_section
        if self.used + cost <= self.max_tokens:
            sec = Section(label=label, text=text)
            self.used += cost
            self.sections.append(sec)
            return sec

        room = self.remaining() - self._overhead_per_section
        if truncatable and room > 16:
            cut = truncate_to_tokens(text, room, self.encoding_name)
            sec = Section(label=label, text=cut, truncated=True)
            self.used += count_tokens(cut, self.encoding_name) + self._overhead_per_section
            self.sections.append(sec)
            return sec

        sec = Section(label=label, text="", dropped=True)
        self.sections.append(sec)
        return sec

    def render(self) -> str:
        parts: List[str] = []
        notes: List[str] = []
        for sec in self.sections:
            if sec.dropped:
                notes.append(f"{sec.label} (omitted: context budget exhausted)")
                continue
            header = f"===== {sec.label} ====="
            body = sec.text
            if sec.truncated:
                header += " [truncated to fit budget]"
                notes.append(f"{sec.label} (truncated)")
            parts.append(f"{header}\n{body}")
        rendered = "\n\n".join(parts)
        if notes:
            rendered += "\n\n===== NOTE =====\nThe following context was reduced to fit the token budget: " + "; ".join(notes) + "."
        return rendered

    @property
    def truncated_labels(self) -> List[str]:
        return [s.label for s in self.sections if s.truncated or s.dropped]
