"""Verbose debug logging for VulnFanatic-NG scans (online and offline).

Off by default. Enable with the ``vulnfanatic.debugLogging`` setting to trace the
scan pipeline on the Binary Ninja console. With ``vulnfanatic.debugAnonymous`` also
on, every binary-identifying detail — symbol/variable names, addresses, the file
name, and decompiled code / prompts / context bodies — is redacted: names and
addresses become salted per-session hashes (consistent within a run so the flow is
still followable), and bodies are logged as sizes only. That lets a user share a log
without disclosing anything about the analyzed file.

Secrets (API keys) are NEVER logged in any mode.
"""

from __future__ import annotations

import functools
import hashlib
import os
import time
from typing import Any, Optional

_enabled = False
_anonymous = False
_salt = b""


def configure(enabled: bool, anonymous: bool) -> None:
    """Enable/disable debug logging and (re)seed the anonymization salt."""
    global _enabled, _anonymous, _salt
    _enabled = bool(enabled)
    _anonymous = bool(anonymous)
    # Fresh random salt per run: hashed names/addresses can't be reversed with a
    # dictionary of common symbols, and don't correlate across separately shared logs.
    _salt = os.urandom(16)


def configure_from_cfg(cfg) -> None:
    configure(getattr(cfg, "debug_logging", False),
              getattr(cfg, "debug_anonymous", False))


def enabled() -> bool:
    return _enabled


def anonymous() -> bool:
    return _anonymous


def _emit(line: str) -> None:
    try:
        import binaryninja
        binaryninja.log_info(line)
    except Exception:
        print(line)


def log(msg: str) -> None:
    """Emit a debug line to the console (only when debug logging is enabled)."""
    if _enabled:
        _emit(f"[VulnFanatic-NG][dbg] {msg}")


def _hash(value: Any) -> str:
    return hashlib.sha1(_salt + str(value).encode("utf-8", "replace")).hexdigest()[:8]


def sym(name: Any) -> str:
    """A symbol / function / variable name (redacted to a stable hash when anonymous)."""
    if name is None:
        return "?"
    return f"sym#{_hash(name)}" if _anonymous else str(name)


def addr(a: Any) -> str:
    """A code address (redacted to a stable hash when anonymous)."""
    if a is None:
        return "?"
    if _anonymous:
        return f"loc#{_hash(a)}"
    try:
        return f"{int(a):#x}"
    except Exception:
        return str(a)


def field(value: Any) -> str:
    """Binary-derived free text (titles, explanations, variable lists): kept when not
    anonymous, replaced with <redacted> when anonymous."""
    if value is None:
        return ""
    return "<redacted>" if _anonymous else str(value)


def size(value: Any) -> str:
    """Large content (code / prompt / context) — always logged as a size, never the
    content itself, in either mode."""
    try:
        n = len(value) if value is not None else 0
    except Exception:
        n = 0
    return f"<{n} chars>"


def path(p: Any) -> str:
    """A file path / binary name (replaced with <binary> when anonymous)."""
    if not p:
        return "?"
    return "<binary>" if _anonymous else os.path.basename(str(p))


def endpoint(url: Any) -> str:
    """An LLM endpoint URL. Not binary-related, but when anonymous the host is hashed
    so a private server name isn't disclosed (scheme + path are kept)."""
    if not url:
        return "?"
    if not _anonymous:
        return str(url)
    try:
        import urllib.parse
        p = urllib.parse.urlsplit(str(url))
        host = f"host#{_hash(p.netloc)}" if p.netloc else ""
        return urllib.parse.urlunsplit((p.scheme, host, p.path, "", ""))
    except Exception:
        return "<endpoint>"


def trace(label: Optional[str] = None):
    """Decorator: log entry/exit (+ duration) of a pipeline function. Arguments are
    NOT auto-logged (they may identify the binary); the body adds details via log()."""
    def deco(fn):
        name = label or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return fn(*args, **kwargs)
            log(f"-> {name}")
            t0 = time.monotonic()
            try:
                return fn(*args, **kwargs)
            finally:
                log(f"<- {name} ({(time.monotonic() - t0) * 1000:.0f} ms)")
        return wrapper
    return deco
