"""Minimal multi-provider LLM client (stdlib only).

Supports four backends, selected by ``provider``:

* ``openai``   — OpenAI-compatible Chat/Completions (local llama.cpp/ollama/vLLM,
  OpenAI, AWS Bedrock's OpenAI-compatible endpoint, …): ``POST <base>/chat/completions``.
* ``anthropic`` — Claude Messages API: ``POST <base>/v1/messages`` with ``x-api-key``
  and ``anthropic-version`` headers and a top-level ``system`` field.
* ``google``   — Gemini generateContent: ``POST <base>/v1beta/models/<model>:generateContent?key=…``.
* ``azure``    — Azure OpenAI: ``POST <base>/openai/deployments/<model>/chat/completions?api-version=…``
  with an ``api-key`` header (``model`` is the deployment name).

Every backend is reached over ``urllib`` so there is no third-party dependency —
the plugin loads inside Binary Ninja's embedded Python with nothing to install.
The verdict parser is deliberately lenient because models often wrap JSON in
prose or code fences and may ignore JSON-output hints.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Connection-level failures that are worth one retry (server dropped the socket,
# reset the connection, or sent a short/incomplete body).
_TRANSIENT_ERRORS = (
    urllib.error.URLError,
    socket.timeout,
    TimeoutError,
    ConnectionError,
    http.client.IncompleteRead,
    http.client.HTTPException,
)
_MAX_ATTEMPTS = 2
_RETRY_BACKOFF_SEC = 1.0

VALID_SEVERITIES = ("critical", "high", "medium", "low", "info")
VALID_CONFIDENCE = ("high", "medium", "low")


def _log(msg: str) -> None:
    try:
        import binaryninja
        binaryninja.log_info(f"[VulnFanatic-NG] {msg}")
    except Exception:
        print(f"[VulnFanatic-NG] {msg}")


def _dbg_log(msg: str) -> None:
    # Lazy import to avoid any import-order coupling; no-op unless debug is enabled.
    try:
        from . import debuglog
        debuglog.log(msg)
    except Exception:
        pass


def _dbg_endpoint(url: str) -> str:
    try:
        from . import debuglog
        return debuglog.endpoint(url)
    except Exception:
        return url


class LLMError(Exception):
    """Base error for LLM interaction."""


class LLMConnectionError(LLMError):
    """Network/HTTP/timeout failure — the scan can skip this site and continue."""


class LLMResponseError(LLMError):
    """The server responded but the content could not be understood.

    ``raw`` carries the model's actual (unparseable) text when available — e.g. a
    reasoning model that replied in prose, or a response truncated mid-thought — so
    callers can surface it for manual review instead of discarding it."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw or ""


def build_endpoint(base_url: str, mode: str = "chat") -> str:
    """Resolve the request URL for ``mode`` ('chat' or 'completions').

    Accepts a base that already ends in either completions path and normalizes
    it to the requested one.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise LLMConnectionError("Empty API base URL.")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    elif base.endswith("/completions"):
        base = base[: -len("/completions")]
    suffix = "/completions" if mode == "completions" else "/chat/completions"
    return base + suffix


# Substrings in a server error body that mean "this model has no chat template",
# i.e. the /chat/completions endpoint can't be used — fall back to /completions.
_NO_CHAT_TEMPLATE_MARKERS = ("chat_template", "chat template")

def build_ssl_context(verify: bool = True, ca_bundle: str = ""):
    """SSL context for HTTPS requests.

    Binary Ninja's embedded Python (and python.org Python on macOS) often ships
    without a usable CA store, producing 'CERTIFICATE_VERIFY_FAILED'. Resolution
    order when verifying: an explicit ``ca_bundle`` (file or dir) → the ``certifi``
    bundle if importable → the system default. ``verify=False`` disables checks
    entirely (only for trusted/internal endpoints).
    """
    if not verify:
        return ssl._create_unverified_context()
    ca_bundle = (ca_bundle or "").strip()
    if ca_bundle:
        try:
            if os.path.isdir(ca_bundle):
                return ssl.create_default_context(capath=ca_bundle)
            return ssl.create_default_context(cafile=ca_bundle)
        except Exception:
            pass  # bad path — fall through to certifi/default
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


# Public default endpoints used when the base URL is blank.
_ANTHROPIC_DEFAULT_BASE = "https://api.anthropic.com"
_GOOGLE_DEFAULT_BASE = "https://generativelanguage.googleapis.com"
# Anthropic Messages API version header (stable; not the model version).
_ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class LLMClient:
    base_url: str
    api_key: str = ""
    model: str = ""
    temperature: float = 0.0
    max_response_tokens: int = 4096
    timeout_sec: int = 120
    send_json_response_format: bool = True
    api_mode: str = "chat"  # "chat" or "completions" (openai provider only)
    provider: str = "openai"  # openai | anthropic | google | azure
    azure_api_version: str = "2024-10-21"
    tls_verify: bool = True
    ca_bundle_path: str = ""
    reasoning_effort: str = "high"  # off | low | medium | high

    def __post_init__(self):
        # Set when a chat request fails because the model has no chat template;
        # subsequent requests in this client use the completions endpoint.
        self._force_completions = False
        # Set when a request is rejected for sending reasoning/thinking params;
        # subsequent requests in this client omit them.
        self._drop_reasoning = False
        self.provider = (self.provider or "openai").strip().lower()
        try:
            self._ssl_ctx = build_ssl_context(self.tls_verify, self.ca_bundle_path)
        except Exception:
            self._ssl_ctx = None  # let urllib use its default context

    @classmethod
    def from_config(cls, cfg) -> "LLMClient":
        return cls(
            base_url=cfg.api_base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_response_tokens=cfg.max_response_tokens,
            timeout_sec=cfg.request_timeout_sec,
            send_json_response_format=cfg.send_json_response_format,
            api_mode=getattr(cfg, "api_mode", "chat") or "chat",
            provider=getattr(cfg, "api_provider", "openai") or "openai",
            azure_api_version=getattr(cfg, "azure_api_version", "2024-10-21")
            or "2024-10-21",
            tls_verify=getattr(cfg, "tls_verify", True),
            ca_bundle_path=getattr(cfg, "ca_bundle_path", "") or "",
            reasoning_effort=getattr(cfg, "reasoning_effort", "high") or "high",
        )

    @classmethod
    def validator_from_config(cls, cfg) -> "LLMClient":
        """Build the SEPARATE client used for the validation pass. Uses the
        validator_* provider/base/key/model and inherits everything else (TLS, Azure
        API version, reasoning effort, max tokens, timeout, ...) from the analyst
        settings. Falls back to the analyst API key when no validator key is set."""
        return cls(
            base_url=getattr(cfg, "validator_base_url", "") or "",
            api_key=(getattr(cfg, "validator_api_key", "") or cfg.api_key),
            model=getattr(cfg, "validator_model", "") or "",
            temperature=cfg.temperature,
            max_response_tokens=cfg.max_response_tokens,
            timeout_sec=cfg.request_timeout_sec,
            send_json_response_format=cfg.send_json_response_format,
            api_mode=getattr(cfg, "api_mode", "chat") or "chat",
            provider=getattr(cfg, "validator_provider", "openai") or "openai",
            azure_api_version=getattr(cfg, "azure_api_version", "2024-10-21")
            or "2024-10-21",
            tls_verify=getattr(cfg, "tls_verify", True),
            ca_bundle_path=getattr(cfg, "ca_bundle_path", "") or "",
            reasoning_effort=getattr(cfg, "reasoning_effort", "high") or "high",
        )

    def _reasoning_on(self) -> bool:
        """True if the configured reasoning/thinking effort should be requested."""
        return (self.reasoning_effort or "").strip().lower() not in ("", "off", "none") \
            and not self._drop_reasoning

    # Request building ------------------------------------------------------
    def _effective_mode(self) -> str:
        if self.api_mode == "completions" or self._force_completions:
            return "completions"
        return "chat"

    @staticmethod
    def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Flatten chat messages into a single prompt for the completions API."""
        parts = []
        for m in messages:
            content = (m.get("content") or "").strip()
            if not content:
                continue
            role = m.get("role", "user")
            if role == "system":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        # Nudge the model to begin its (JSON) answer.
        return "\n\n".join(parts) + "\n\nResponse:\n"

    def _build_body(self, messages: List[Dict[str, str]], mode: str) -> dict:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_response_tokens,
            "stream": False,
        }
        if mode == "completions":
            body["prompt"] = self.messages_to_prompt(messages)
        else:
            body["messages"] = messages
            if self.send_json_response_format:
                body["response_format"] = {"type": "json_object"}
            # OpenAI reasoning models / Azure reasoning deployments.
            if self._reasoning_on():
                body["reasoning_effort"] = self.reasoning_effort
        return body

    def _headers(self) -> Dict[str, str]:
        """Auth + content headers for the selected provider."""
        headers = {"Content-Type": "application/json"}
        provider = self.provider
        if provider == "azure":
            if self.api_key:
                headers["api-key"] = self.api_key
        elif provider == "anthropic":
            headers["anthropic-version"] = _ANTHROPIC_VERSION
            if self.api_key:
                headers["x-api-key"] = self.api_key
        elif provider == "google":
            pass  # the API key goes in the query string, not a header
        else:  # openai (and OpenAI-compatible servers)
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # HTTP ------------------------------------------------------------------
    def _post(self, url: str, body: dict, headers: Dict[str, str]) -> str:
        data = json.dumps(body).encode("utf-8")
        _dbg_log(f"llm POST {_dbg_endpoint(url)} provider={self.provider} "
                 f"model={self.model} max_tokens={self.max_response_tokens} "
                 f"req_bytes={len(data)}")
        # Pass the SSL context only for HTTPS so plain-http (local) requests and
        # the test mocks are unaffected.
        open_kwargs = {"timeout": self.timeout_sec}
        if url.lower().startswith("https") and getattr(self, "_ssl_ctx", None) is not None:
            open_kwargs["context"] = self._ssl_ctx

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_ATTEMPTS):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, **open_kwargs) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                    _dbg_log(f"llm response: ok ({len(raw)} bytes)")
                    return raw
            except urllib.error.HTTPError as exc:
                # A definitive HTTP error response — do not retry.
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                raise LLMConnectionError(
                    f"HTTP {exc.code} from {url}: {detail or exc.reason}"
                ) from exc
            except _TRANSIENT_ERRORS as exc:
                # A TLS certificate-verification failure will never succeed on
                # retry — fail fast with actionable guidance instead.
                reason = getattr(exc, "reason", None)
                if isinstance(reason, ssl.SSLError) or \
                        "CERTIFICATE_VERIFY_FAILED" in str(exc).upper():
                    raise LLMConnectionError(
                        f"TLS certificate verification failed for {url}: {exc}. Binary "
                        "Ninja's Python could not find a CA bundle to verify the server "
                        "certificate (the certificate itself is fine). Fix with ONE of: "
                        "(1) install certifi into Binary Ninja's Python — `pip install "
                        "certifi`; (2) set 'vulnfanatic.caBundlePath' to a CA bundle "
                        "file (e.g. the output of `python3 -m certifi`, or /etc/ssl/"
                        "cert.pem); or (3) for a trusted endpoint only, turn OFF "
                        "'vulnfanatic.tlsVerify'."
                    ) from exc
                # The server dropped/reset the connection or sent a short body
                # (e.g. IncompleteRead) — often the model server crashed, ran out
                # of memory, or timed out mid-generation. Retry once.
                last_exc = exc
                _dbg_log(f"llm transient error ({type(exc).__name__}: {exc}); "
                         f"attempt {attempt + 1}/{_MAX_ATTEMPTS}")
                if attempt + 1 < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF_SEC)
                    continue
                raise LLMConnectionError(
                    f"Could not complete request to {url} after {_MAX_ATTEMPTS} "
                    f"attempt(s): {type(exc).__name__}: {exc}"
                ) from exc
        # Unreachable, but keeps type checkers happy.
        raise LLMConnectionError(f"Request to {url} failed: {last_exc}")

    def _parse_content(self, raw: str, mode: str) -> str:
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise LLMResponseError(f"Non-JSON response body: {raw[:200]}") from exc
        try:
            choice = payload["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"Unexpected response shape: {json.dumps(payload)[:300]}"
            ) from exc
        if not isinstance(choice, dict):
            raise LLMResponseError(
                f"Unexpected response shape: {json.dumps(payload)[:300]}")
        if mode == "completions":
            return choice.get("text") or _extract_message_text(choice.get("message"))
        # Chat: tolerate a missing/empty content field (some reasoning models —
        # gpt-oss/o1 — return only a hidden reasoning channel, or an empty message
        # when response_format suppresses the final channel). Return "" rather than
        # raising so the empty response flows into the JSON-salvage retry in analyze().
        return _extract_message_text(choice.get("message"))

    def complete(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to the configured provider; return the raw text content."""
        try:
            return self._dispatch(messages)
        except LLMConnectionError as exc:
            # If the server rejected our reasoning/thinking parameters, drop them
            # and retry once (then for the rest of this scan).
            if self._reasoning_on() and _is_reasoning_rejection(exc):
                self._drop_reasoning = True
                _log("server rejected reasoning/thinking parameters; retrying without "
                     "them for the rest of this scan")
                return self._dispatch(messages)
            raise

    def _dispatch(self, messages: List[Dict[str, str]]) -> str:
        if self.provider == "anthropic":
            return self._complete_anthropic(messages)
        if self.provider == "google":
            return self._complete_google(messages)
        if self.provider == "azure":
            return self._complete_azure(messages)
        return self._complete_openai(messages)

    # OpenAI-compatible -----------------------------------------------------
    def _complete_openai(self, messages: List[Dict[str, str]]) -> str:
        mode = self._effective_mode()
        url = build_endpoint(self.base_url, mode)
        try:
            raw = self._post(url, self._build_body(messages, mode), self._headers())
        except LLMConnectionError as exc:
            # Auto-fallback: a chat request rejected because the model lacks a chat
            # template can usually be served by the plain completions endpoint.
            if mode == "chat" and not self._force_completions and \
                    any(mark in str(exc).lower() for mark in _NO_CHAT_TEMPLATE_MARKERS):
                self._force_completions = True
                _log("model has no chat template; falling back to the /completions "
                     "endpoint for the rest of this scan")
                mode = "completions"
                url = build_endpoint(self.base_url, mode)
                raw = self._post(url, self._build_body(messages, mode), self._headers())
            else:
                raise
        return self._parse_content(raw, mode)

    # Azure OpenAI ----------------------------------------------------------
    def _azure_url(self) -> str:
        base = (self.base_url or "").strip().rstrip("/")
        if not base:
            raise LLMConnectionError(
                "Azure provider needs a Base URL (https://<resource>.openai.azure.com)."
            )
        if base.endswith("/openai"):
            base = base[: -len("/openai")]
        ver = (self.azure_api_version or "2024-10-21").strip()
        return (f"{base}/openai/deployments/{self.model}/chat/completions"
                f"?api-version={urllib.parse.quote(ver)}")

    def _complete_azure(self, messages: List[Dict[str, str]]) -> str:
        # Azure speaks the OpenAI chat shape; only the URL and auth header differ.
        raw = self._post(self._azure_url(), self._build_body(messages, "chat"),
                         self._headers())
        return self._parse_content(raw, "chat")

    # Anthropic Claude ------------------------------------------------------
    def _anthropic_url(self) -> str:
        base = (self.base_url or "").strip().rstrip("/") or _ANTHROPIC_DEFAULT_BASE
        if base.endswith("/v1/messages"):
            return base
        if base.endswith("/v1"):
            return base + "/messages"
        return base + "/v1/messages"

    def _complete_anthropic(self, messages: List[Dict[str, str]]) -> str:
        system, convo = _split_system(messages)
        if not convo:  # no user turn — fold the system text into a user message
            convo = [{"role": "user", "content": system}]
            system = ""
        body = {
            "model": self.model,
            "max_tokens": self.max_response_tokens,
            "messages": convo,
        }
        if system:
            body["system"] = system
        # NOTE: temperature/top_p are intentionally omitted — current Claude models
        # (Opus 4.7+/Fable) reject them, and the default sampling suits this task.
        if self._reasoning_on():
            # Adaptive thinking + effort (GA on Opus 4.6+/Sonnet 4.6/Fable; older
            # models that reject these trigger the strip-and-retry fallback).
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": self.reasoning_effort}
        raw = self._post(self._anthropic_url(), body, self._headers())
        return self._parse_anthropic(raw)

    def _parse_anthropic(self, raw: str) -> str:
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise LLMResponseError(f"Non-JSON response body: {raw[:200]}") from exc
        try:
            blocks = payload["content"]
            text = "".join(
                b.get("text", "") for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"Unexpected Anthropic response shape: {json.dumps(payload)[:300]}"
            ) from exc
        if not text:
            raise LLMResponseError(
                f"Empty Anthropic response: {json.dumps(payload)[:300]}"
            )
        return text

    # Google Gemini ---------------------------------------------------------
    def _google_url(self) -> str:
        base = (self.base_url or "").strip().rstrip("/") or _GOOGLE_DEFAULT_BASE
        url = f"{base}/v1beta/models/{self.model}:generateContent"
        if self.api_key:
            url += f"?key={urllib.parse.quote(self.api_key)}"
        return url

    def _complete_google(self, messages: List[Dict[str, str]]) -> str:
        system, convo = _split_system(messages)
        contents = [
            {"role": ("model" if m["role"] == "assistant" else "user"),
             "parts": [{"text": m["content"]}]}
            for m in convo
        ]
        if not contents:
            contents = [{"role": "user", "parts": [{"text": system}]}]
            system = ""
        gen: Dict = {
            "maxOutputTokens": self.max_response_tokens,
            "temperature": self.temperature,
        }
        if self.send_json_response_format:
            gen["responseMimeType"] = "application/json"
        if self._reasoning_on():
            # Gemini 2.5: -1 = dynamic thinking (model decides depth). Models that
            # do not support thinkingConfig trigger the strip-and-retry fallback.
            gen["thinkingConfig"] = {"thinkingBudget": -1}
        body: Dict = {"contents": contents, "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        raw = self._post(self._google_url(), body, {"Content-Type": "application/json"})
        return self._parse_google(raw)

    def _parse_google(self, raw: str) -> str:
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise LLMResponseError(f"Non-JSON response body: {raw[:200]}") from exc
        try:
            parts = payload["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"Unexpected Google response shape: {json.dumps(payload)[:300]}"
            ) from exc
        if not text:
            raise LLMResponseError(
                f"Empty Google response: {json.dumps(payload)[:300]}"
            )
        return text

    def analyze(self, system_prompt: str, user_prompt: str) -> Dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        content = self.complete(messages)
        try:
            return parse_verdict(content)
        except LLMResponseError:
            # Many local models occasionally emit a stray token (e.g. Gemma's
            # '<unused..>') or prose instead of JSON. Try once more, re-asking for
            # JSON only with the structured-output format disabled, before giving up.
            salvaged = self._salvage_unparseable(messages, content)
            if salvaged is not None:
                return salvaged
            raise

    def _salvage_unparseable(self, messages: List[Dict[str, str]],
                             bad_content: str) -> Optional[Dict]:
        """One corrective retry when a response has no parseable JSON. Appends a
        'JSON only' instruction to the existing user turn (kept as a single user turn
        so provider role-alternation rules are respected) and disables
        ``response_format`` (some servers degrade with it on). Returns the parsed
        verdict, or None if the retry also fails."""
        if not messages or messages[-1].get("role") != "user":
            return None
        snippet = (bad_content or "").strip()[:80]
        instruction = (
            "\n\n[RETRY — your previous reply was not valid JSON"
            + (f" (it was: {snippet!r})" if snippet else "")
            + "]. Reply with ONLY a single minified JSON object for the verdict "
            "described above: no prose, no markdown, no code fences, no special tokens."
        )
        retry_messages = messages[:-1] + [
            {"role": "user", "content": messages[-1]["content"] + instruction}
        ]
        prev_fmt = self.send_json_response_format
        self.send_json_response_format = False
        _dbg_log("llm: unparseable response; one JSON-only salvage retry")
        try:
            content = self.complete(retry_messages)
        except LLMError:
            self.send_json_response_format = prev_fmt
            return None
        try:
            verdict = parse_verdict(content)
        except LLMResponseError:
            self.send_json_response_format = prev_fmt
            return None
        # The retry succeeded with structured output OFF. Keep it off for the rest of
        # this scan so we don't pay the doomed first attempt on every request (gpt-oss
        # and similar reasoning models frequently return an empty final channel when
        # response_format=json_object is set).
        if prev_fmt:
            _log("the model returned valid JSON only with response_format disabled; "
                 "leaving it off for the rest of this scan")
        return verdict


# Markers in an error body that mean the server rejected our reasoning/thinking
# parameters (so we strip them and retry). Each names a parameter we actually send.
_REASONING_REJECT_MARKERS = (
    "reasoning_effort", "output_config", "thinkingconfig", "thinking_budget",
    "thinkingbudget", "thinking", "reasoning",
)


def _is_reasoning_rejection(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(m in s for m in _REASONING_REJECT_MARKERS)


def _split_system(messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """Split OpenAI-style messages into (system_text, conversation_turns).

    Anthropic and Gemini take the system prompt as a separate top-level field and
    only accept user/assistant ('model') turns in the message list.
    """
    system_parts: List[str] = []
    convo: List[Dict[str, str]] = []
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        role = m.get("role", "user")
        if role == "system":
            system_parts.append(content)
        else:
            convo.append({
                "role": "assistant" if role == "assistant" else "user",
                "content": content,
            })
    return "\n\n".join(system_parts), convo


def _extract_message_text(message) -> str:
    """Text from an OpenAI-style chat message, tolerant of reasoning-model shapes.

    Handles: a plain ``content`` string; ``content`` returned as a list of parts
    (``[{type, text}, ...]``); and reasoning models (gpt-oss / o1-style) that leave
    ``content`` empty and place the text in a ``reasoning_content`` / ``reasoning``
    channel. Returns ``""`` when the message carries no text at all (e.g. the server
    returned ``{"role": "assistant"}`` only) so the caller can retry rather than crash.
    """
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    if not content:
        content = message.get("reasoning_content") or message.get("reasoning") or ""
    return content or ""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    # 1) direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    # 2) fenced ```json ... ```
    m = _FENCE_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    # 3) first balanced {...} span
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except ValueError:
                            break
        start = text.find("{", start + 1)
    return None


def _coerce_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in ("true", "yes", "y", "1", "vulnerable", "vuln")
    return False


def _coerce_text(val) -> str:
    """Normalize a possibly-structured field (scratchpad/validation) to text."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    try:
        return json.dumps(val, indent=2)
    except Exception:
        return str(val)


def _coerce_choice(val, choices, default: str) -> str:
    if isinstance(val, str):
        v = val.strip().lower()
        if v in choices:
            return v
    return default


# A verdict whose explanation concludes the code is SAFE while is_vulnerable=true
# is a self-contradiction (common with local models). We override it to false.
_SAFE_CONCLUSION_PHRASES = (
    "not a vulnerability", "not vulnerable", "no vulnerability",
    "not a security issue", "not a security concern", "not exploitable",
    "is not exploitable", "not an issue", "is not an issue", "this is safe",
    "is safe because", "false positive", "not a real vulnerability", "is benign",
    "poses no risk", "no security impact", "not a concern", "not a bug",
    "no overflow", "cannot overflow", "cannot be overflowed", "is bounded",
    "fixed and known", "constant and known",
)
# If the explanation also clearly AFFIRMS a real issue, it is ambiguous (e.g.
# "the format is constant — not a format bug — but the size overflows"); keep it.
_AFFIRM_VULN_PHRASES = (
    "is a vulnerability", "is vulnerable", "this is a vulnerability",
    "can overflow", "will overflow", "can be overflowed", "is exploitable",
    "attacker can control", "allows an attacker to", "remote code execution",
    "arbitrary write", "out-of-bounds write", "buffer overflow occurs",
)


def _explanation_concludes_safe(verdict: Dict) -> bool:
    text = (str(verdict.get("explanation", "")) + " "
            + str(verdict.get("title", ""))).lower()
    if not any(p in text for p in _SAFE_CONCLUSION_PHRASES):
        return False
    if any(p in text for p in _AFFIRM_VULN_PHRASES):
        return False
    return True


def parse_verdict(content: str) -> Dict:
    """Normalize a model response into the verdict contract.

    Always returns a dict with at least ``is_vulnerable`` (bool). Raises
    ``LLMResponseError`` only if no JSON object can be recovered at all.
    """
    obj = _extract_json_object(content or "")
    if obj is None:
        stripped = (content or "").strip()
        hint = ""
        if not stripped:
            hint = (" — the model returned an EMPTY message (no content). For reasoning "
                    "models such as gpt-oss / o1 served over an OpenAI-compatible API "
                    "this usually means the answer went only to a hidden reasoning "
                    "channel, or that 'response_format=json_object' suppressed the final "
                    "answer channel. Try turning OFF vulnfanatic.sendJsonResponseFormat, "
                    "and/or lowering vulnfanatic.reasoningEffort")
        elif stripped.startswith("{") and not stripped.endswith("}"):
            hint = (" — the response looks truncated (it starts with '{' but does "
                    "not close). If raising vulnfanatic.maxResponseTokens does not help, the "
                    "prompt + response likely exceed the model's context window: lower "
                    "vulnfanatic.maxContextTokens, set vulnfanatic.modelContextWindow to your "
                    "server's window, and/or raise the server's context length "
                    "(ollama num_ctx, llama.cpp -c, vLLM --max-model-len)")
        raise LLMResponseError(
            f"No JSON object found in model response{hint}: {stripped[:200]!r}",
            raw=content or "",
        )

    verdict = {
        "is_vulnerable": _coerce_bool(obj.get("is_vulnerable", False)),
        "confidence": _coerce_choice(obj.get("confidence"), VALID_CONFIDENCE, "low"),
        "severity": _coerce_choice(obj.get("severity"), VALID_SEVERITIES, "info"),
        "cwe": str(obj.get("cwe", "") or "").strip(),
        "title": str(obj.get("title", "") or "").strip(),
        "explanation": str(obj.get("explanation", "") or "").strip(),
        "tainted_input": str(obj.get("tainted_input", "") or "").strip(),
        "recommendation": str(obj.get("recommendation", "") or "").strip(),
        "scratchpad": _coerce_text(obj.get("scratchpad")),
        "validation": _coerce_text(obj.get("validation")),
    }

    # Guard against self-contradiction: explanation says safe but flag is true.
    if verdict["is_vulnerable"] and _explanation_concludes_safe(verdict):
        verdict["is_vulnerable"] = False

    return verdict


_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def meets_min_confidence(confidence: str, minimum: str) -> bool:
    """True if ``confidence`` is at least as strong as ``minimum``."""
    c = _CONFIDENCE_ORDER.get((confidence or "").lower(), 99)
    m = _CONFIDENCE_ORDER.get((minimum or "low").lower(), 2)
    return c <= m
