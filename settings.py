"""Binary Ninja Settings registration and typed configuration access for VulnFanatic-NG.

All tunables live under the ``vulnfanatic.*`` settings group so they are editable from
Binary Ninja's *Settings* dialog (search for "vulnfanatic"). ``get_config`` reads the
registered values (optionally scoped to a BinaryView) and returns a plain
``Config`` dataclass so the rest of the plugin never touches the Settings API
directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

try:
    from binaryninja import Settings
except Exception:  # pragma: no cover - allows importing headless/for tests
    Settings = None  # type: ignore


# Group + key constants ------------------------------------------------------

GROUP = "vulnfanatic"

K_API_PROVIDER = "vulnfanatic.apiProvider"
K_API_BASE_URL = "vulnfanatic.apiBaseUrl"
K_API_KEY = "vulnfanatic.apiKey"
K_MODEL = "vulnfanatic.model"
K_API_MODE = "vulnfanatic.apiMode"
K_AZURE_API_VERSION = "vulnfanatic.azureApiVersion"
K_TLS_VERIFY = "vulnfanatic.tlsVerify"
K_CA_BUNDLE_PATH = "vulnfanatic.caBundlePath"
K_MAX_CONTEXT_TOKENS = "vulnfanatic.maxContextTokens"
K_MAX_RESPONSE_TOKENS = "vulnfanatic.maxResponseTokens"
K_STORED_CONTEXT_CHARS = "vulnfanatic.storedContextChars"
K_TEMPERATURE = "vulnfanatic.temperature"
K_REASONING_EFFORT = "vulnfanatic.reasoningEffort"
K_REQUEST_TIMEOUT = "vulnfanatic.requestTimeoutSec"
K_CALL_PATH_MAX_DEPTH = "vulnfanatic.callPathMaxDepth"
K_CALL_PATH_MAX_PATHS = "vulnfanatic.callPathMaxPaths"
K_PHASE2_REQUIRE_SYMBOLS = "vulnfanatic.phase2RequireSymbols"
K_PHASE2_FORCE_ENABLE = "vulnfanatic.phase2ForceEnable"
K_TOKENIZER_ENCODING = "vulnfanatic.tokenizerEncoding"
K_RUN_PHASE1 = "vulnfanatic.runPhase1"
K_RUN_PHASE2 = "vulnfanatic.runPhase2"
K_RUN_PHASE3 = "vulnfanatic.runPhase3"
K_RULES_PHASE1_PATH = "vulnfanatic.rulesPhase1Path"
K_RULES_PHASE2_PATH = "vulnfanatic.rulesPhase2Path"
K_RULES_PHASE3_PATH = "vulnfanatic.rulesPhase3Path"
K_SEND_JSON_RESPONSE_FORMAT = "vulnfanatic.sendJsonResponseFormat"
K_CALL_PATH_INCLUDE_BODIES = "vulnfanatic.callPathIncludeBodies"
K_CALL_PATH_MAX_BODIES = "vulnfanatic.callPathMaxBodies"
K_INCLUDE_CALL_PATH_SIBLINGS = "vulnfanatic.includeCallPathSiblings"
K_CALL_PATH_SIBLING_MAX_BODIES = "vulnfanatic.callPathSiblingMaxBodies"
K_INCLUDE_DATA_TYPES = "vulnfanatic.includeDataTypes"
K_MAX_TYPE_DEFS = "vulnfanatic.maxTypeDefs"
K_INCLUDE_VARIABLE_DATAFLOW = "vulnfanatic.includeVariableDataflow"
K_DATAFLOW_MAX_FUNCTIONS = "vulnfanatic.dataflowMaxFunctions"
K_INCLUDE_STACK_LAYOUT = "vulnfanatic.includeStackLayout"
K_SCAN_INDIRECT_CALLS = "vulnfanatic.scanIndirectCalls"
K_VALIDATION_PASS = "vulnfanatic.validationPass"
K_VALIDATOR_PROVIDER = "vulnfanatic.validatorProvider"
K_VALIDATOR_BASE_URL = "vulnfanatic.validatorBaseUrl"
K_VALIDATOR_API_KEY = "vulnfanatic.validatorApiKey"
K_VALIDATOR_MODEL = "vulnfanatic.validatorModel"
K_VERDICT_REASONING = "vulnfanatic.verdictReasoning"
K_MIN_CONFIDENCE = "vulnfanatic.minConfidence"
K_FLAG_UNPARSEABLE = "vulnfanatic.flagUnparseableResponses"
K_SKIP_CONSTANT_ARG_CALLS = "vulnfanatic.skipConstantArgCalls"
K_OFFLINE_BUILD_CONTEXT = "vulnfanatic.offlineBuildContext"
K_MODEL_CONTEXT_WINDOW = "vulnfanatic.modelContextWindow"
K_DEBUG_LOGGING = "vulnfanatic.debugLogging"
K_DEBUG_ANONYMOUS = "vulnfanatic.debugAnonymous"

# Environment fallbacks for the API key (never logged).
ENV_API_KEYS = ("VULNFANATIC_API_KEY", "OPENAI_API_KEY")

# Tokens reserved for chat formatting/overhead when fitting prompt + response
# into the model's context window.
_CONTEXT_OVERHEAD_TOKENS = 768


def clamp_context_tokens(max_context: int, max_response: int, window: int) -> int:
    """Cap the context budget so prompt + response fit the model's context window.

    With ``window <= 0`` (unknown) the configured ``max_context`` is used as-is.
    """
    if window and window > 0:
        room = window - max_response - _CONTEXT_OVERHEAD_TOKENS
        return max(1000, min(max_context, room))
    return max_context


@dataclass
class Config:
    """Snapshot of the plugin configuration at scan time."""

    api_provider: str = "openai"  # openai | anthropic | google | azure
    api_base_url: str = "http://localhost:8080/v1"
    api_key: str = ""
    model: str = ""
    api_mode: str = "chat"
    azure_api_version: str = "2024-10-21"
    tls_verify: bool = True
    ca_bundle_path: str = ""
    max_context_tokens: int = 100000
    max_response_tokens: int = 8192
    model_context_window: int = 0
    stored_context_chars: int = 0  # 0 = store the full context, no truncation
    temperature: float = 0.0
    reasoning_effort: str = "high"  # off | low | medium | high
    request_timeout_sec: int = 120
    call_path_max_depth: int = 8
    call_path_max_paths: int = 5
    call_path_include_bodies: bool = True
    call_path_max_bodies: int = 12
    include_call_path_siblings: bool = True
    call_path_sibling_max_bodies: int = 12
    include_data_types: bool = True
    max_type_defs: int = 24
    include_variable_dataflow: bool = True
    dataflow_max_functions: int = 8
    include_stack_layout: bool = True
    scan_indirect_calls: bool = True
    validation_pass: bool = False
    # Optional SEPARATE model for the validation pass (a second opinion is far more
    # valuable from an independent model than from the same one re-checking itself).
    # When validator_model is blank the validation pass reuses the analyst model.
    # Everything except provider/base URL/key/model is inherited from the analyst
    # connection settings (TLS, Azure API version, reasoning effort, max tokens, ...).
    validator_provider: str = "openai"  # openai | anthropic | google | azure
    validator_base_url: str = ""
    validator_api_key: str = ""
    validator_model: str = ""
    verdict_reasoning: str = "concise"  # concise | full | none
    min_confidence: str = "low"
    flag_unparseable: bool = True
    skip_constant_arg_calls: bool = False
    offline_build_context: bool = True
    debug_logging: bool = False
    debug_anonymous: bool = False
    phase2_require_symbols: bool = True
    phase2_force_enable: bool = False
    tokenizer_encoding: str = "cl100k_base"
    run_phase1: bool = True
    run_phase2: bool = True
    run_phase3: bool = False
    rules_phase1_path: str = ""
    rules_phase2_path: str = ""
    rules_phase3_path: str = ""
    send_json_response_format: bool = True

    @property
    def is_test_mode(self) -> bool:
        """TEST mode: dump prompts to /tmp instead of calling the LLM."""
        return self.api_base_url.strip().upper() == "TEST"

    @property
    def validator_active(self) -> bool:
        """True when a separate validator model is configured (used by the validation
        pass instead of the analyst model)."""
        return bool((self.validator_model or "").strip())

    def validation_error(self) -> str:
        """Return a human-readable problem with the config, or "" if usable."""
        provider = (self.api_provider or "openai").strip().lower()
        # anthropic/google have built-in public default endpoints, so a blank
        # base URL is fine for them; openai/azure must point at a server.
        if not self.api_base_url.strip() and provider in ("openai", "azure"):
            return "No API base URL configured (set 'vulnfanatic.apiBaseUrl' in Settings)."
        if not self.is_test_mode and not self.model.strip():
            return "No model configured (set 'vulnfanatic.model' in Settings)."
        if not self.run_phase1 and not self.run_phase2 and not self.run_phase3:
            return "All phases are disabled; nothing to scan."
        return ""


# Registration ---------------------------------------------------------------

_REGISTERED = False


def _setting(schema: dict) -> str:
    return json.dumps(schema)


def register_settings() -> None:
    """Register the ``vulnfanatic.*`` settings group. Safe to call multiple times."""
    global _REGISTERED
    if _REGISTERED or Settings is None:
        return

    settings = Settings()
    settings.register_group(GROUP, "VulnFanatic-NG")

    defaults = Config()

    settings.register_setting(
        K_API_PROVIDER,
        _setting({
            "title": "API Provider",
            "type": "string",
            "default": defaults.api_provider,
            "enum": ["openai", "anthropic", "google", "azure"],
            "enumDescriptions": [
                "OpenAI-compatible Chat/Completions API (local llama.cpp/ollama/vLLM, "
                "OpenAI, AWS Bedrock's OpenAI-compatible endpoint, etc.).",
                "Anthropic Claude Messages API (POST <base>/v1/messages; base "
                "defaults to https://api.anthropic.com).",
                "Google Gemini generateContent API (base defaults to "
                "https://generativelanguage.googleapis.com).",
                "Azure OpenAI (set Base URL to your resource endpoint "
                "https://<resource>.openai.azure.com and Model to the deployment name).",
            ],
            "description": "Which LLM backend to call. All providers are reached over "
            "the Python standard library (no extra packages to install). The Model, "
            "API Key and Base URL fields apply to the selected provider.",
        }),
    )
    settings.register_setting(
        K_API_BASE_URL,
        _setting({
            "title": "API Base URL",
            "type": "string",
            "default": defaults.api_base_url,
            "description": "Endpoint base for the selected API Provider. openai: an "
            "OpenAI-compatible server (the plugin POSTs to '<base>/chat/completions'); "
            "anthropic: blank uses https://api.anthropic.com; google: blank uses "
            "https://generativelanguage.googleapis.com; azure: your resource endpoint "
            "https://<resource>.openai.azure.com. Set to the literal value 'TEST' to "
            "enable test mode: the LLM is never called, every candidate is flagged, "
            "and the full prompt + context for each is written under /tmp/vulnfanatic_ng.",
        }),
    )
    settings.register_setting(
        K_API_KEY,
        _setting({
            "title": "API Key",
            "type": "string",
            "default": defaults.api_key,
            "description": "Bearer token for the API. May be left blank for local "
            "servers that do not require auth. Stored in plaintext in Binary "
            "Ninja's settings; the environment variables VULNFANATIC_API_KEY or "
            "OPENAI_API_KEY override this value when set.",
        }),
    )
    settings.register_setting(
        K_MODEL,
        _setting({
            # Titled "API Language Model" (not just "Model") so Binary Ninja's
            # alphabetical-by-title ordering places it in the connection cluster,
            # right below "API Base URL"/"API Key" rather than down at the "M"s.
            "title": "API Language Model",
            "type": "string",
            "default": defaults.model,
            "description": "Model identifier passed to the API — for azure, the "
            "deployment name; otherwise the model name your server/provider exposes "
            "(e.g. a local model name, or claude-opus-4-8, gemini-2.0-flash, …). "
            "Required before scanning.",
        }),
    )
    settings.register_setting(
        K_API_MODE,
        _setting({
            "title": "API Mode",
            "type": "string",
            "default": defaults.api_mode,
            "enum": ["chat", "completions"],
            "enumDescriptions": [
                "Use POST <base>/chat/completions with a system+user message list "
                "(default; needs a model that has a chat template).",
                "Use POST <base>/completions with a single flattened prompt — for "
                "base/instruct models served WITHOUT a chat template.",
            ],
            "description": "Which OpenAI-compatible endpoint to call. If a chat "
            "request fails because the model has no chat template, VulnFanatic-NG "
            "automatically falls back to 'completions' for the rest of the scan. "
            "Only applies when API Provider is 'openai'.",
        }),
    )
    settings.register_setting(
        K_AZURE_API_VERSION,
        _setting({
            "title": "Azure API Version",
            "type": "string",
            "default": defaults.azure_api_version,
            "description": "api-version query parameter for Azure OpenAI requests "
            "(e.g. 2024-10-21). Only used when API Provider is 'azure'.",
        }),
    )
    settings.register_setting(
        K_TLS_VERIFY,
        _setting({
            "title": "TLS Certificate Verification",
            "type": "boolean",
            "default": defaults.tls_verify,
            "description": "Verify the server's TLS certificate for HTTPS endpoints "
            "(recommended). If you get 'CERTIFICATE_VERIFY_FAILED', prefer fixing the "
            "CA bundle (install certifi into Binary Ninja's Python, or set the TLS CA "
            "Bundle Path below) over disabling this. Turn OFF only for a trusted "
            "internal endpoint or a self-signed local server.",
        }),
    )
    settings.register_setting(
        K_CA_BUNDLE_PATH,
        _setting({
            "title": "TLS CA Bundle Path",
            "type": "string",
            "default": defaults.ca_bundle_path,
            "description": "Path to a CA certificate bundle file (or directory) used "
            "to verify HTTPS endpoints. Blank = use certifi if installed, otherwise "
            "the system default. Set this if Binary Ninja's Python lacks a usable CA "
            "store (common cause of CERTIFICATE_VERIFY_FAILED). Example: the path "
            "printed by `python3 -m certifi`, or /etc/ssl/cert.pem.",
        }),
    )
    settings.register_setting(
        K_MAX_CONTEXT_TOKENS,
        _setting({
            "title": "Max Context Tokens",
            "type": "number",
            "default": defaults.max_context_tokens,
            "minValue": 1000,
            "maxValue": 2000000,
            "description": "Upper bound on the number of tokens of generated context "
            "sent to the model per call site. This PLUS the response must fit the "
            "model's context window — if responses come back truncated, lower this "
            "or set the Model Context Window below.",
        }),
    )
    settings.register_setting(
        K_MODEL_CONTEXT_WINDOW,
        _setting({
            "title": "Model Context Window",
            "type": "number",
            "default": defaults.model_context_window,
            "minValue": 0,
            "maxValue": 2000000,
            "description": "The model server's total context window in tokens (e.g. "
            "ollama num_ctx, llama.cpp -c, vLLM --max-model-len). When set above 0, "
            "VulnFanatic-NG automatically caps the context it sends so that prompt + "
            "response fit the window (preventing truncated responses). 0 = unknown "
            "(no automatic cap). Set this to your server's configured window.",
        }),
    )
    settings.register_setting(
        K_MAX_RESPONSE_TOKENS,
        _setting({
            "title": "Max Response Tokens",
            "type": "number",
            "default": defaults.max_response_tokens,
            "minValue": 64,
            "maxValue": 65536,
            "description": "max_tokens requested from the model for each verdict. The "
            "verdict includes a detailed scratchpad, so this must be large enough for "
            "the full response. If you see 'No JSON object found' / truncated-response "
            "errors, increase this value.",
        }),
    )
    settings.register_setting(
        K_STORED_CONTEXT_CHARS,
        _setting({
            "title": "Stored Context Characters",
            "type": "number",
            "default": defaults.stored_context_chars,
            "minValue": 0,
            "maxValue": 100000000,
            "description": "How many characters of the per-finding context to keep on "
            "the finding (shown in the detail pane and used for the fine-tuning "
            "export). 0 = store the FULL context (no truncation; best for training "
            "data but grows the BNDB). Set a positive cap (e.g. 4000) to limit "
            "database size.",
        }),
    )
    settings.register_setting(
        K_TEMPERATURE,
        _setting({
            "title": "Temperature",
            "type": "number",
            "default": defaults.temperature,
            "minValue": 0.0,
            "maxValue": 2.0,
            "description": "Sampling temperature for the model. Ignored by some "
            "providers (e.g. Anthropic does not accept it for current Claude models).",
        }),
    )
    settings.register_setting(
        K_REASONING_EFFORT,
        _setting({
            "title": "Reasoning Effort / Thinking",
            "type": "string",
            "default": defaults.reasoning_effort,
            "enum": ["off", "low", "medium", "high"],
            "enumDescriptions": [
                "Do not request any reasoning/thinking (send no extra parameters).",
                "Request minimal reasoning effort.",
                "Request moderate reasoning effort.",
                "Request high reasoning effort / enable thinking (recommended).",
            ],
            "description": "Ask the model to think before answering, where the "
            "provider supports it. Mapped per provider: openai/azure -> "
            "'reasoning_effort'; anthropic -> adaptive thinking + 'output_config.effort'; "
            "google -> dynamic 'thinkingConfig'. If a model/server rejects these, "
            "VulnFanatic-NG automatically retries without them for the rest of the scan. "
            "Note: for Anthropic/Gemini, thinking tokens count toward the response "
            "budget — raise Max Response Tokens if verdicts come back truncated.",
        }),
    )
    settings.register_setting(
        K_REQUEST_TIMEOUT,
        _setting({
            "title": "Request Timeout (seconds)",
            "type": "number",
            "default": defaults.request_timeout_sec,
            "minValue": 5,
            "maxValue": 3600,
            "description": "Per-request HTTP timeout.",
        }),
    )
    settings.register_setting(
        K_CALL_PATH_MAX_DEPTH,
        _setting({
            "title": "Call Path Max Depth",
            "type": "number",
            "default": defaults.call_path_max_depth,
            "minValue": 1,
            "maxValue": 64,
            "description": "How far back to walk callers when building call paths to "
            "a dangerous call site.",
        }),
    )
    settings.register_setting(
        K_CALL_PATH_MAX_PATHS,
        _setting({
            "title": "Call Path Max Paths",
            "type": "number",
            "default": defaults.call_path_max_paths,
            "minValue": 1,
            "maxValue": 64,
            "description": "Maximum number of distinct call paths included in context.",
        }),
    )
    settings.register_setting(
        K_CALL_PATH_INCLUDE_BODIES,
        _setting({
            "title": "Include Call-Path Function Bodies",
            "type": "boolean",
            "default": defaults.call_path_include_bodies,
            "description": "Include the decompiled body of every function along the "
            "call path(s) to a dangerous call (Phase 1), with the call site and the "
            "conditions guarding each hop. Lets the model reason about the checks "
            "that gate reaching the dangerous code. Disable to send only the call "
            "path as a list of names.",
        }),
    )
    settings.register_setting(
        K_CALL_PATH_MAX_BODIES,
        _setting({
            "title": "Call-Path Max Function Bodies",
            "type": "number",
            "default": defaults.call_path_max_bodies,
            "minValue": 0,
            "maxValue": 256,
            "description": "Maximum number of distinct call-path function bodies to "
            "include (nearest the dangerous call are kept first). The token budget "
            "may trim further.",
        }),
    )
    settings.register_setting(
        K_INCLUDE_CALL_PATH_SIBLINGS,
        _setting({
            "title": "Include Call-Path Sibling Functions",
            "type": "boolean",
            "default": defaults.include_call_path_siblings,
            "description": "Also include the bodies of OTHER functions called by the "
            "functions on the call path (e.g. if MAIN->ABCD->strcpy, also include "
            "functions MAIN and ABCD call elsewhere), because they may contain the "
            "bounds/validation checks that prevent the value from reaching the "
            "dangerous call. Filled only while context budget allows.",
        }),
    )
    settings.register_setting(
        K_CALL_PATH_SIBLING_MAX_BODIES,
        _setting({
            "title": "Call-Path Sibling Max Function Bodies",
            "type": "number",
            "default": defaults.call_path_sibling_max_bodies,
            "minValue": 0,
            "maxValue": 256,
            "description": "Maximum number of distinct sibling/callee function bodies "
            "to include from functions on the call path. The token budget may trim "
            "further.",
        }),
    )
    settings.register_setting(
        K_INCLUDE_DATA_TYPES,
        _setting({
            "title": "Include Data Type Definitions",
            "type": "boolean",
            "default": defaults.include_data_types,
            "description": "Include the definitions of struct/union/enum types "
            "referenced by the functions and variables in the context, so the model "
            "can reason about field layouts and sizes.",
        }),
    )
    settings.register_setting(
        K_MAX_TYPE_DEFS,
        _setting({
            "title": "Max Data Type Definitions",
            "type": "number",
            "default": defaults.max_type_defs,
            "minValue": 0,
            "maxValue": 512,
            "description": "Maximum number of type definitions included per context.",
        }),
    )
    settings.register_setting(
        K_INCLUDE_VARIABLE_DATAFLOW,
        _setting({
            "title": "Include Variable Data-Flow (Phase 1)",
            "type": "boolean",
            "default": defaults.include_variable_dataflow,
            "description": "Trace the dangerous call's argument variables back through "
            "their definitions and uses, include a per-argument def/use summary, and "
            "pull in the bodies of functions that produce or consume those variables.",
        }),
    )
    settings.register_setting(
        K_DATAFLOW_MAX_FUNCTIONS,
        _setting({
            "title": "Data-Flow Max Function Bodies",
            "type": "number",
            "default": defaults.dataflow_max_functions,
            "minValue": 0,
            "maxValue": 128,
            "description": "Maximum number of producer/consumer function bodies to "
            "include from the argument data-flow. The token budget may trim further.",
        }),
    )
    settings.register_setting(
        K_INCLUDE_STACK_LAYOUT,
        _setting({
            "title": "Include Stack Frame Layout (Phase 1)",
            "type": "boolean",
            "default": defaults.include_stack_layout,
            "description": "Include the calling function's stack-variable layout "
            "(offsets and byte sizes) when it contains a fixed-size buffer, so the "
            "model can reason about stack overflows reaching adjacent variables or "
            "the saved return address.",
        }),
    )
    settings.register_setting(
        K_SCAN_INDIRECT_CALLS,
        _setting({
            "title": "Scan Indirect Calls (Phase 1)",
            "type": "boolean",
            "default": defaults.scan_indirect_calls,
            "description": "Also sweep every function's decompiled calls and match "
            "dangerous calls dispatched through a function pointer / vtable that "
            "Binary Ninja resolved (in addition to direct calls to named symbols and "
            "calls routed through forwarding thunks). Improves recall on indirectly "
            "dispatched code; turn off for a faster scan on very large binaries.",
        }),
    )
    settings.register_setting(
        K_VALIDATION_PASS,
        _setting({
            "title": "Validation Pass",
            "type": "boolean",
            "default": defaults.validation_pass,
            "description": "After the first analysis flags an issue, run a second LLM "
            "pass that double-checks the claim against the same context (verifying "
            "the scratchpad snippets and re-tracing the flow) before reporting it. "
            "Doubles the number of LLM calls for flagged candidates and can reduce "
            "false positives, but may also drop borderline real issues. Off by "
            "default; enable it if you are getting too many false positives.",
        }),
    )
    settings.register_setting(
        K_VALIDATOR_MODEL,
        _setting({
            "title": "Validator Model (separate second opinion)",
            "type": "string",
            "default": defaults.validator_model,
            "description": "Optional. When the Validation Pass is on, run the second "
            "(double-check) pass on THIS model instead of the analyst model. A second "
            "opinion is far more valuable from an INDEPENDENT model — it shares fewer "
            "blind spots and is less likely to rubber-stamp the first verdict. Tip: use "
            "a fast model for analysis and a stronger one here (a cascade), since the "
            "validator only runs on flagged candidates. Leave BLANK to validate with "
            "the analyst model. The validator should be at least as capable as the "
            "analyst — a weaker validator mostly adds false rejections.",
        }),
    )
    settings.register_setting(
        K_VALIDATOR_PROVIDER,
        _setting({
            "title": "Validator Provider",
            "type": "string",
            "default": defaults.validator_provider,
            "enum": ["openai", "anthropic", "google", "azure"],
            "description": "Backend for the Validator Model (only used when a Validator "
            "Model is set). Same provider choices as the analyst.",
        }),
    )
    settings.register_setting(
        K_VALIDATOR_BASE_URL,
        _setting({
            "title": "Validator Base URL",
            "type": "string",
            "default": defaults.validator_base_url,
            "description": "Endpoint base for the Validator Provider (only used when a "
            "Validator Model is set). Required for openai/azure; blank uses the public "
            "default for anthropic/google. All other connection settings (TLS, Azure "
            "API version, reasoning effort, max tokens, timeout) are inherited from the "
            "analyst settings.",
        }),
    )
    settings.register_setting(
        K_VALIDATOR_API_KEY,
        _setting({
            "title": "Validator API Key",
            "type": "string",
            "default": defaults.validator_api_key,
            "description": "API key for the Validator Provider (only used when a "
            "Validator Model is set). Leave blank to reuse the analyst's API Key. "
            "Stored in plaintext in Binary Ninja's settings.",
        }),
    )
    settings.register_setting(
        K_VERDICT_REASONING,
        _setting({
            "title": "Verdict Reasoning Detail",
            "type": "string",
            "default": defaults.verdict_reasoning,
            "enum": ["concise", "full", "none"],
            "enumDescriptions": [
                "Brief 1-3 sentence rationale, no verbatim code — much faster, keeps "
                "most of the accuracy (recommended).",
                "Detailed scratchpad with verbatim code snippets — most auditable but "
                "slowest (the model writes long reasoning into every response).",
                "Verdict only, no written rationale — fastest; best paired with a "
                "reasoning-capable backend (set Reasoning Effort) whose internal "
                "thinking does the work. On a plain local model this loses accuracy.",
            ],
            "description": "How much written reasoning the model emits per verdict. "
            "Most of the per-call latency is the reasoning text, so 'concise' is the "
            "default speed/accuracy balance. With 'concise' you can also lower Max "
            "Response Tokens. The reasoning is still stored on each finding (used by "
            "the fine-tuning export), so 'none' leaves that field empty.",
        }),
    )
    settings.register_setting(
        K_MIN_CONFIDENCE,
        _setting({
            "title": "Minimum Confidence to Report",
            "type": "string",
            "default": defaults.min_confidence,
            "enum": ["low", "medium", "high"],
            "enumDescriptions": [
                "Report every confirmed issue (low confidence and up).",
                "Report only medium- and high-confidence issues.",
                "Report only high-confidence issues (strictest).",
            ],
            "description": "Drop confirmed issues whose final confidence is below "
            "this threshold.",
        }),
    )
    settings.register_setting(
        K_FLAG_UNPARSEABLE,
        _setting({
            "title": "Report Unparseable Responses as Low-Confidence Leads",
            "type": "boolean",
            "default": defaults.flag_unparseable,
            "description": "When the model fails to return a parseable JSON verdict for "
            "a candidate (e.g. it replied in prose, ran out of output budget mid-"
            "reasoning, or returned an empty message — common with GPT-OSS / o1-style "
            "models), still report that candidate as a LOW-confidence 'Unscored' lead "
            "for manual review, keeping the model's partial output, instead of dropping "
            "it. Recall-preserving; turn off to drop such sites (they then appear only "
            "as analysis errors / debug ERROR rows).",
        }),
    )
    settings.register_setting(
        K_SKIP_CONSTANT_ARG_CALLS,
        _setting({
            "title": "Skip Constant-Argument Calls",
            "type": "boolean",
            "default": defaults.skip_constant_arg_calls,
            "description": "For overflow-class rules, skip (without an LLM call) any "
            "dangerous call whose arguments are all compile-time constants. Reduces "
            "false positives and LLM calls, but skips a few sites entirely. Off by "
            "default so nothing is skipped; enable it to cut noise on large binaries.",
        }),
    )
    settings.register_setting(
        K_OFFLINE_BUILD_CONTEXT,
        _setting({
            "title": "Offline Scan: Build Full Context",
            "type": "boolean",
            "default": defaults.offline_build_context,
            "description": "During an offline (no-LLM) scan, also build the full "
            "interprocedural context (as for an online scan) for each FLAGGED finding "
            "and store it, so triaged offline results can be exported as fine-tuning "
            "data. Adds work per flagged site; turn off for maximum offline speed.",
        }),
    )
    settings.register_setting(
        K_DEBUG_LOGGING,
        _setting({
            "title": "Debug Logging",
            "type": "boolean",
            "default": defaults.debug_logging,
            "description": "Verbose step-by-step logging of the scan pipeline (online "
            "and offline) to the Binary Ninja console — every stage, decision and LLM "
            "request. Off by default; enable it to diagnose behavior or to capture a "
            "log to share. API keys are never logged.",
        }),
    )
    settings.register_setting(
        K_DEBUG_ANONYMOUS,
        _setting({
            "title": "Debug Logging: Anonymous",
            "type": "boolean",
            "default": defaults.debug_anonymous,
            "description": "When Debug Logging is on, REDACT everything that could "
            "identify the analyzed binary: symbol/variable names and addresses become "
            "per-run salted hashes (still consistent within a run so the flow is "
            "followable), the file name is hidden, and decompiled code / prompts / "
            "context are logged as sizes only. Turn this on to share a debug log "
            "without disclosing anything about your file.",
        }),
    )
    settings.register_setting(
        K_PHASE2_REQUIRE_SYMBOLS,
        _setting({
            "title": "Phase 2 Requires Symbols",
            "type": "boolean",
            "default": defaults.phase2_require_symbols,
            "description": "Only run Phase 2 when the binary appears to have real "
            "symbols / variable names.",
        }),
    )
    settings.register_setting(
        K_PHASE2_FORCE_ENABLE,
        _setting({
            "title": "Phase 2 Force Enable",
            "type": "boolean",
            "default": defaults.phase2_force_enable,
            "description": "Run Phase 2 even when the symbol heuristic fails. Useful "
            "if the heuristic misjudges a stripped-looking binary that actually has "
            "meaningful names.",
        }),
    )
    settings.register_setting(
        K_TOKENIZER_ENCODING,
        _setting({
            "title": "Tokenizer Encoding",
            "type": "string",
            "default": defaults.tokenizer_encoding,
            "description": "tiktoken encoding name used to estimate token counts. "
            "Ignored (falls back to a character heuristic) if tiktoken is not "
            "installed.",
        }),
    )
    settings.register_setting(
        K_RUN_PHASE1,
        _setting({
            "title": "Run Phase 1 (dangerous calls)",
            "type": "boolean",
            "default": defaults.run_phase1,
            "description": "Enable the dangerous-function call-site scan.",
        }),
    )
    settings.register_setting(
        K_RUN_PHASE2,
        _setting({
            "title": "Run Phase 2 (security-sensitive code)",
            "type": "boolean",
            "default": defaults.run_phase2,
            "description": "Enable the security-sensitive code scan (symbol-gated).",
        }),
    )
    settings.register_setting(
        K_RUN_PHASE3,
        _setting({
            "title": "Run Phase 3 (HW-attack hardening)",
            "type": "boolean",
            "default": defaults.run_phase3,
            "description": "Enable the Phase 3 firmware hardware-attack hardening audit "
            "(fault-injection / side-channel controls). ONLINE only (needs the LLM); "
            "symbol-gated; OFF by default. Usually toggled per-scan via the checkbox "
            "on the New Scan tab rather than here.",
        }),
    )
    settings.register_setting(
        K_RULES_PHASE1_PATH,
        _setting({
            "title": "Phase 1 Rules Path",
            "type": "string",
            "default": defaults.rules_phase1_path,
            "description": "Override path to a phase1_rules.json file. Blank uses the "
            "bundled defaults.",
        }),
    )
    settings.register_setting(
        K_RULES_PHASE2_PATH,
        _setting({
            "title": "Phase 2 Rules Path",
            "type": "string",
            "default": defaults.rules_phase2_path,
            "description": "Override path to a phase2_rules.json file. Blank uses the "
            "bundled defaults.",
        }),
    )
    settings.register_setting(
        K_RULES_PHASE3_PATH,
        _setting({
            "title": "Phase 3 Rules Path",
            "type": "string",
            "default": defaults.rules_phase3_path,
            "description": "Override path to a phase3_rules.json file (hardware-attack "
            "hardening controls). Blank uses the bundled defaults.",
        }),
    )
    settings.register_setting(
        K_SEND_JSON_RESPONSE_FORMAT,
        _setting({
            "title": "Request JSON response_format",
            "type": "boolean",
            "default": defaults.send_json_response_format,
            "description": "Send response_format={type:json_object} in requests. Turn "
            "off if your local server rejects unknown parameters.",
        }),
    )

    _REGISTERED = True


# Reading --------------------------------------------------------------------

def _resolve_api_key(settings_value: str) -> str:
    for env_name in ENV_API_KEYS:
        env_val = os.environ.get(env_name)
        if env_val:
            return env_val
    return settings_value


def get_config(bv=None) -> Config:
    """Read the current configuration fresh from Binary Ninja's Settings.

    Read at the active (user/global) scope — the value the Settings dialog edits —
    so a change made before a scan always applies. We deliberately do NOT scope the
    read to ``bv``: a binary-view-scoped read resolves through the resource scope and,
    if that lookup is unavailable or errors, silently falls back to the registered
    DEFAULT — which made dialog changes appear to be ignored. The ``bv`` parameter is
    kept for call-site compatibility but the plugin does not use per-binary overrides.
    """
    if Settings is None:
        # Headless / test fallback: defaults + env API key.
        cfg = Config()
        cfg.api_key = _resolve_api_key(cfg.api_key)
        return cfg

    register_settings()
    settings = Settings()

    def s(key: str, default: str) -> str:
        try:
            return settings.get_string(key)
        except Exception:
            return default

    def i(key: str, default: int) -> int:
        try:
            return int(settings.get_integer(key))
        except Exception:
            return default

    def d(key: str, default: float) -> float:
        try:
            return float(settings.get_double(key))
        except Exception:
            return default

    def b(key: str, default: bool) -> bool:
        try:
            return settings.get_bool(key)
        except Exception:
            return default

    defaults = Config()
    cfg = Config(
        api_provider=s(K_API_PROVIDER, defaults.api_provider).strip().lower() or "openai",
        api_base_url=s(K_API_BASE_URL, defaults.api_base_url).strip(),
        api_key=_resolve_api_key(s(K_API_KEY, defaults.api_key)),
        model=s(K_MODEL, defaults.model).strip(),
        api_mode=s(K_API_MODE, defaults.api_mode).strip().lower() or "chat",
        azure_api_version=s(K_AZURE_API_VERSION, defaults.azure_api_version).strip()
        or defaults.azure_api_version,
        tls_verify=b(K_TLS_VERIFY, defaults.tls_verify),
        ca_bundle_path=s(K_CA_BUNDLE_PATH, defaults.ca_bundle_path).strip(),
        max_context_tokens=i(K_MAX_CONTEXT_TOKENS, defaults.max_context_tokens),
        max_response_tokens=i(K_MAX_RESPONSE_TOKENS, defaults.max_response_tokens),
        stored_context_chars=i(K_STORED_CONTEXT_CHARS, defaults.stored_context_chars),
        model_context_window=i(K_MODEL_CONTEXT_WINDOW, defaults.model_context_window),
        temperature=d(K_TEMPERATURE, defaults.temperature),
        reasoning_effort=s(K_REASONING_EFFORT, defaults.reasoning_effort).strip().lower()
        or defaults.reasoning_effort,
        request_timeout_sec=i(K_REQUEST_TIMEOUT, defaults.request_timeout_sec),
        call_path_max_depth=i(K_CALL_PATH_MAX_DEPTH, defaults.call_path_max_depth),
        call_path_max_paths=i(K_CALL_PATH_MAX_PATHS, defaults.call_path_max_paths),
        call_path_include_bodies=b(K_CALL_PATH_INCLUDE_BODIES, defaults.call_path_include_bodies),
        call_path_max_bodies=i(K_CALL_PATH_MAX_BODIES, defaults.call_path_max_bodies),
        include_call_path_siblings=b(K_INCLUDE_CALL_PATH_SIBLINGS, defaults.include_call_path_siblings),
        call_path_sibling_max_bodies=i(K_CALL_PATH_SIBLING_MAX_BODIES, defaults.call_path_sibling_max_bodies),
        include_data_types=b(K_INCLUDE_DATA_TYPES, defaults.include_data_types),
        max_type_defs=i(K_MAX_TYPE_DEFS, defaults.max_type_defs),
        include_variable_dataflow=b(K_INCLUDE_VARIABLE_DATAFLOW, defaults.include_variable_dataflow),
        dataflow_max_functions=i(K_DATAFLOW_MAX_FUNCTIONS, defaults.dataflow_max_functions),
        include_stack_layout=b(K_INCLUDE_STACK_LAYOUT, defaults.include_stack_layout),
        scan_indirect_calls=b(K_SCAN_INDIRECT_CALLS, defaults.scan_indirect_calls),
        validation_pass=b(K_VALIDATION_PASS, defaults.validation_pass),
        validator_provider=s(K_VALIDATOR_PROVIDER, defaults.validator_provider).strip().lower()
        or defaults.validator_provider,
        validator_base_url=s(K_VALIDATOR_BASE_URL, defaults.validator_base_url).strip(),
        validator_api_key=s(K_VALIDATOR_API_KEY, defaults.validator_api_key),
        validator_model=s(K_VALIDATOR_MODEL, defaults.validator_model).strip(),
        verdict_reasoning=s(K_VERDICT_REASONING, defaults.verdict_reasoning).strip().lower()
        or defaults.verdict_reasoning,
        min_confidence=s(K_MIN_CONFIDENCE, defaults.min_confidence).strip().lower(),
        flag_unparseable=b(K_FLAG_UNPARSEABLE, defaults.flag_unparseable),
        skip_constant_arg_calls=b(K_SKIP_CONSTANT_ARG_CALLS, defaults.skip_constant_arg_calls),
        offline_build_context=b(K_OFFLINE_BUILD_CONTEXT, defaults.offline_build_context),
        debug_logging=b(K_DEBUG_LOGGING, defaults.debug_logging),
        debug_anonymous=b(K_DEBUG_ANONYMOUS, defaults.debug_anonymous),
        phase2_require_symbols=b(K_PHASE2_REQUIRE_SYMBOLS, defaults.phase2_require_symbols),
        phase2_force_enable=b(K_PHASE2_FORCE_ENABLE, defaults.phase2_force_enable),
        tokenizer_encoding=s(K_TOKENIZER_ENCODING, defaults.tokenizer_encoding).strip(),
        run_phase1=b(K_RUN_PHASE1, defaults.run_phase1),
        run_phase2=b(K_RUN_PHASE2, defaults.run_phase2),
        run_phase3=b(K_RUN_PHASE3, defaults.run_phase3),
        rules_phase1_path=s(K_RULES_PHASE1_PATH, defaults.rules_phase1_path).strip(),
        rules_phase2_path=s(K_RULES_PHASE2_PATH, defaults.rules_phase2_path).strip(),
        rules_phase3_path=s(K_RULES_PHASE3_PATH, defaults.rules_phase3_path).strip(),
        send_json_response_format=b(K_SEND_JSON_RESPONSE_FORMAT, defaults.send_json_response_format),
    )
    # Keep prompt + response within the model's context window when it is known.
    cfg.max_context_tokens = clamp_context_tokens(
        cfg.max_context_tokens, cfg.max_response_tokens, cfg.model_context_window
    )
    return cfg
