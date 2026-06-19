"""Prototypes of common dangerous functions + format-argument resolution.

Single source of truth shared by:
  * the offline heuristics — *which* positional argument is the printf-style
    format string (so a constant string literal there is recognised as a
    controlled format and the call is not flagged); and
  * the online context builder — showing the called function's declared
    prototype to the model.

The important subtlety this module encodes: fortified (``__*_chk``) and
bounds-checked Annex-K (``*_s``) variants have DIFFERENT prototypes from their
base function. glibc's ``__*_chk`` insert a ``flag`` argument (and, for functions
that write into a caller buffer, also a ``strlen``) *before* the format string,
so the format argument is at a shifted index. Assuming the base prototype is a
common source of false positives (e.g. ``__fprintf_chk(stream, 1, "literal", …)``
being mis-read as an uncontrolled format string).
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# 0-based index of the printf-style format-string argument, by function name.
_FORMAT_ARG_INDEX: Dict[str, int] = {
    # nothing before the format string
    "printf": 0, "vprintf": 0, "wprintf": 0, "vwprintf": 0, "printk": 0, "nslog": 0,
    "vscprintf": 0, "_vscprintf": 0, "_vscwprintf": 0,
    "warn": 0, "warnx": 0,
    "g_print": 0, "g_printerr": 0, "g_warning": 0, "g_critical": 0, "g_message": 0,
    # one leading arg (stream / fd / buffer / eval / priority) before the format
    "fprintf": 1, "sprintf": 1, "siprintf": 1, "dprintf": 1, "vfprintf": 1,
    "vsprintf": 1, "vdprintf": 1, "fwprintf": 1, "vfwprintf": 1, "syslog": 1,
    "vsyslog": 1, "wsprintf": 1, "wsprintfa": 1, "wsprintfw": 1, "asprintf": 1,
    "asiprintf": 1, "vasprintf": 1, "err": 1, "errx": 1,
    # buffer + size before the format
    "snprintf": 2, "_snprintf": 2, "sniprintf": 2, "vsnprintf": 2, "_vsnprintf": 2,
    "swprintf": 2, "vswprintf": 2, "_snwprintf": 2, "_vsnwprintf": 2,
    # fortified glibc variants — flag (and, for buffer writers, strlen) inserted
    # before the format argument:
    "__printf_chk": 1, "__wprintf_chk": 1, "__vprintf_chk": 1, "__vwprintf_chk": 1,
    "__fprintf_chk": 2, "__fwprintf_chk": 2, "__vfprintf_chk": 2, "__vfwprintf_chk": 2,
    "__dprintf_chk": 2, "__vdprintf_chk": 2, "__asprintf_chk": 2, "__vasprintf_chk": 2,
    "__syslog_chk": 2,
    "__sprintf_chk": 3, "__vsprintf_chk": 3,
    "__snprintf_chk": 4, "__vsnprintf_chk": 4, "__swprintf_chk": 4, "__vswprintf_chk": 4,
}

# Base functions that write into a caller-supplied buffer: their ``__*_chk``
# variant inserts TWO arguments (flag, strlen) before the format; others insert
# one (flag). Used to derive an index for any ``_chk`` not listed above.
_BUFFER_WRITERS = {
    "sprintf", "vsprintf", "snprintf", "vsnprintf", "swprintf", "vswprintf",
    "siprintf", "sniprintf", "_snprintf", "_vsnprintf", "_snwprintf", "_vsnwprintf",
}

_CHK_RE = re.compile(r"^_+(?P<base>.+)_chk$")


def format_arg_index(name: str) -> Optional[int]:
    """0-based index of the format-string argument for ``name``, or ``None`` if the
    function is not a printf-style formatter.

    Unlisted ``__<base>_chk`` names are derived from the base function's index plus
    the ``_chk`` argument offset, so the table need not enumerate every variant.
    """
    n = (name or "").lower()
    if n in _FORMAT_ARG_INDEX:
        return _FORMAT_ARG_INDEX[n]
    m = _CHK_RE.match(n)
    if m:
        base = m.group("base")
        base_idx = _FORMAT_ARG_INDEX.get(base)
        if base_idx is not None:
            return base_idx + (2 if base in _BUFFER_WRITERS else 1)
    return None


# Declared C prototypes for the most common dangerous functions. Used to show the
# model the exact signature; for anything not listed, the context builder prefers
# Binary Ninja's own type information. Variant prototypes are spelled out because
# their argument layout is exactly what trips up naive position-based analysis.
KNOWN_PROTOTYPES: Dict[str, str] = {
    # printf family
    "printf": "int printf(const char *format, ...)",
    "fprintf": "int fprintf(FILE *stream, const char *format, ...)",
    "sprintf": "int sprintf(char *str, const char *format, ...)",
    "snprintf": "int snprintf(char *str, size_t size, const char *format, ...)",
    "dprintf": "int dprintf(int fd, const char *format, ...)",
    "vprintf": "int vprintf(const char *format, va_list ap)",
    "vfprintf": "int vfprintf(FILE *stream, const char *format, va_list ap)",
    "vsprintf": "int vsprintf(char *str, const char *format, va_list ap)",
    "vsnprintf": "int vsnprintf(char *str, size_t size, const char *format, va_list ap)",
    "syslog": "void syslog(int priority, const char *format, ...)",
    "asprintf": "int asprintf(char **strp, const char *format, ...)",
    "__printf_chk": "int __printf_chk(int flag, const char *format, ...)",
    "__fprintf_chk": "int __fprintf_chk(FILE *stream, int flag, const char *format, ...)",
    "__sprintf_chk": "int __sprintf_chk(char *str, int flag, size_t strlen, const char *format, ...)",
    "__snprintf_chk": "int __snprintf_chk(char *str, size_t maxlen, int flag, size_t strlen, const char *format, ...)",
    "__vsnprintf_chk": "int __vsnprintf_chk(char *str, size_t maxlen, int flag, size_t strlen, const char *format, va_list ap)",
    "__syslog_chk": "void __syslog_chk(int priority, int flag, const char *format, ...)",
    # scanf family (format also at the front / after the source)
    "scanf": "int scanf(const char *format, ...)",
    "sscanf": "int sscanf(const char *str, const char *format, ...)",
    "fscanf": "int fscanf(FILE *stream, const char *format, ...)",
    # string copy / cat
    "strcpy": "char *strcpy(char *dst, const char *src)",
    "strncpy": "char *strncpy(char *dst, const char *src, size_t n)",
    "strcat": "char *strcat(char *dst, const char *src)",
    "strncat": "char *strncat(char *dst, const char *src, size_t n)",
    "strlcpy": "size_t strlcpy(char *dst, const char *src, size_t size)",
    "strlcat": "size_t strlcat(char *dst, const char *src, size_t size)",
    "stpcpy": "char *stpcpy(char *dst, const char *src)",
    "strcpy_s": "errno_t strcpy_s(char *dst, rsize_t dstsz, const char *src)",
    "strncpy_s": "errno_t strncpy_s(char *dst, rsize_t dstsz, const char *src, rsize_t count)",
    "strcat_s": "errno_t strcat_s(char *dst, rsize_t dstsz, const char *src)",
    "strncat_s": "errno_t strncat_s(char *dst, rsize_t dstsz, const char *src, rsize_t count)",
    # memory copy / set
    "memcpy": "void *memcpy(void *dst, const void *src, size_t n)",
    "memmove": "void *memmove(void *dst, const void *src, size_t n)",
    "memset": "void *memset(void *s, int c, size_t n)",
    "__memcpy_chk": "void *__memcpy_chk(void *dst, const void *src, size_t n, size_t dstlen)",
    "__memmove_chk": "void *__memmove_chk(void *dst, const void *src, size_t n, size_t dstlen)",
    "__memset_chk": "void *__memset_chk(void *s, int c, size_t n, size_t dstlen)",
    "memcpy_s": "errno_t memcpy_s(void *dst, rsize_t dstsz, const void *src, rsize_t n)",
    # input read
    "read": "ssize_t read(int fd, void *buf, size_t count)",
    "recv": "ssize_t recv(int sockfd, void *buf, size_t len, int flags)",
    "recvfrom": "ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags, struct sockaddr *src_addr, socklen_t *addrlen)",
    "fread": "size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)",
    "fgets": "char *fgets(char *s, int size, FILE *stream)",
    "gets": "char *gets(char *s)",
    # allocation
    "malloc": "void *malloc(size_t size)",
    "calloc": "void *calloc(size_t nmemb, size_t size)",
    "realloc": "void *realloc(void *ptr, size_t size)",
    "alloca": "void *alloca(size_t size)",
    # command execution
    "system": "int system(const char *command)",
    "popen": "FILE *popen(const char *command, const char *type)",
    "execl": "int execl(const char *path, const char *arg, ...)",
    "execlp": "int execlp(const char *file, const char *arg, ...)",
    "execve": "int execve(const char *pathname, char *const argv[], char *const envp[])",
    # comparison
    "memcmp": "int memcmp(const void *s1, const void *s2, size_t n)",
    "strncmp": "int strncmp(const char *s1, const char *s2, size_t n)",
    "strcmp": "int strcmp(const char *s1, const char *s2)",
}


def lookup_prototype(name: str) -> Optional[str]:
    """Declared prototype string for ``name`` from the built-in table, or ``None``."""
    return KNOWN_PROTOTYPES.get((name or "").lower())
