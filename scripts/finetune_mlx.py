#!/usr/bin/env python3
"""Fine-tune an LLM on VulnFanatic-NG triaged exports using MLX (Apple Silicon).

VulnFanatic-NG's **Export triaged (fine-tuning)…** button writes one OpenAI
chat-format JSONL file per scan/binary, where each line is::

    {"messages": [{"role": "system", ...},
                  {"role": "user", ...},
                  {"role": "assistant", ...}]}

The assistant turn is the *human-corrected* verdict (a False Positive teaches
``is_vulnerable=false`` with your reason; a Real Issue reinforces a true verdict),
so the data trains the model toward your judgements on your binaries.

Point this script at a **folder** containing those exports (collected from many
binaries). It merges + validates them, de-duplicates, makes a train/valid split
in the layout ``mlx_lm.lora`` expects, then runs a LoRA fine-tune — optionally
fusing the adapters into a standalone model you can serve back to the plugin.

This is a standalone developer tool — it is NOT imported by the Binary Ninja
plugin and has no effect at scan time.

Requirements (Apple Silicon / macOS):

    pip install mlx-lm

Examples:

    # LoRA fine-tune a local 4-bit model on every *.jsonl under ./exports
    python scripts/finetune_mlx.py ./exports \\
        --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \\
        --adapter-path ./vf-adapters --iters 800

    # Prepare the data and print the command without training
    python scripts/finetune_mlx.py ./exports --model <base> --dry-run

    # Train, then fuse the adapters into a standalone model
    python scripts/finetune_mlx.py ./exports --model <base> \\
        --fuse --fused-path ./vf-qwen-coder-vuln

Anything after a literal ``--`` is forwarded verbatim to ``mlx_lm lora`` (e.g.
``-- --grad-checkpoint --val-batches 25``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_VALID_ROLES = {"system", "user", "assistant"}


# --- data preparation (pure stdlib; unit-tested without MLX) ----------------

def _is_within(path: Path, parent: str) -> bool:
    """True if ``path`` is inside directory ``parent`` (both resolved)."""
    if not parent:
        return False
    try:
        path.resolve().relative_to(Path(parent).resolve())
        return True
    except (ValueError, OSError):
        return False


def iter_jsonl_files(data_dir: str, exclude_dir: str = "") -> List[Path]:
    """Return every *.jsonl file under ``data_dir`` (recursive, sorted).

    Files inside ``exclude_dir`` are skipped — that is where this script writes the
    prepared train/valid split, so it never feeds its own output back in as input.
    Any other filename (including a user export named ``test.jsonl``) is included.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")
    files = [
        p for p in sorted(root.rglob("*.jsonl"))
        if p.is_file() and not _is_within(p, exclude_dir)
    ]
    return files


def validate_example(obj) -> str:
    """Return "" if ``obj`` is a usable chat example, else a reason string."""
    if not isinstance(obj, dict):
        return "not a JSON object"
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return "missing non-empty 'messages' list"
    has_user = False
    for m in msgs:
        if not isinstance(m, dict):
            return "message is not an object"
        role = m.get("role")
        if role not in _VALID_ROLES:
            return f"invalid role {role!r}"
        if not isinstance(m.get("content", ""), str):
            return "non-string message content"
        if role == "user":
            has_user = True
    if msgs[-1].get("role") != "assistant" or not (msgs[-1].get("content") or "").strip():
        return "last message must be a non-empty assistant turn (the training target)"
    if not has_user:
        return "no user message"
    return ""


def load_examples(files: List[Path]) -> Tuple[List[str], Dict[str, int]]:
    """Read+validate every line of every file. Returns (normalized JSON lines,
    stats), de-duplicating identical examples across binaries."""
    seen = set()
    examples: List[str] = []
    stats = {"files": len(files), "total": 0, "kept": 0, "skipped": 0, "duplicates": 0}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:  # unreadable file — note and continue
            print(f"  ! could not read {path}: {exc}", file=sys.stderr)
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                obj = json.loads(line)
            except ValueError:
                stats["skipped"] += 1
                print(f"  ! {path}:{lineno}: invalid JSON — skipped", file=sys.stderr)
                continue
            reason = validate_example(obj)
            if reason:
                stats["skipped"] += 1
                print(f"  ! {path}:{lineno}: {reason} — skipped", file=sys.stderr)
                continue
            norm = json.dumps(obj, ensure_ascii=False, sort_keys=True)
            if norm in seen:
                stats["duplicates"] += 1
                continue
            seen.add(norm)
            # Re-dump WITHOUT sort_keys so the on-disk order stays human-readable.
            examples.append(json.dumps(obj, ensure_ascii=False))
            stats["kept"] += 1
    return examples, stats


def split_and_write(examples: List[str], out_dir: str, valid_split: float,
                    seed: int) -> Tuple[int, int]:
    """Shuffle (seeded), split into train/valid, and write train.jsonl/valid.jsonl
    into ``out_dir`` in the layout ``mlx_lm.lora --data`` expects."""
    if len(examples) < 2:
        raise ValueError(
            f"Need at least 2 valid examples to make a train/valid split; got "
            f"{len(examples)}. Triage and export more findings.")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_valid = max(1, min(len(shuffled) - 1, round(len(shuffled) * valid_split)))
    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "valid.jsonl").write_text("\n".join(valid) + "\n", encoding="utf-8")
    return len(train), len(valid)


def effective_batch_size(requested: int, n_train: int, n_valid: int) -> int:
    """MLX requires batch_size <= the size of BOTH the train and valid splits.
    Clamp to fit a tiny split (floor 1) so smoke tests on little data still run."""
    return max(1, min(requested, n_train, n_valid))


# Auto-fit bounds for --max-seq-length when the user does not set it explicitly.
# Long sequences dominate training memory, so a large model near this cap can run a
# smaller Mac out of memory (the trainer is then killed by SIGBUS / exit -10). If
# that happens, restrict it manually with --max-seq-length (and/or -- --grad-checkpoint).
_AUTO_SEQ_FLOOR = 4096
_AUTO_SEQ_CAP = 16384
_AUTO_SEQ_ROUND = 2048


def fit_seq_length(longest_tokens: int, floor: int = _AUTO_SEQ_FLOOR,
                   cap: int = _AUTO_SEQ_CAP, round_to: int = _AUTO_SEQ_ROUND) -> int:
    """Pick a max sequence length that fits the longest example: round the token
    estimate up to a multiple of ``round_to`` and clamp to ``[floor, cap]``.

    VulnFanatic-NG contexts are large and variable, so fitting to the data avoids
    silent truncation while the cap keeps memory bounded."""
    import math
    rounded = int(math.ceil(max(1, longest_tokens) / round_to) * round_to)
    return max(floor, min(cap, rounded))


def longest_example_chars(examples: List[str]) -> int:
    """Length, in characters, of the longest example's concatenated content —
    a rough proxy for sequence length so we can warn about truncation."""
    longest = 0
    for line in examples:
        try:
            msgs = json.loads(line)["messages"]
            n = sum(len(m.get("content", "")) for m in msgs)
        except Exception:
            n = len(line)
        longest = max(longest, n)
    return longest


# --- command construction ---------------------------------------------------

def build_lora_cmd(args, prepared_dir: str, extra: List[str]) -> List[str]:
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--train",
        "--data", prepared_dir,
        "--adapter-path", args.adapter_path,
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.num_layers),
        "--learning-rate", str(args.learning_rate),
        "--max-seq-length", str(args.max_seq_length),
        "--fine-tune-type", args.fine_tune_type,
        "--seed", str(args.seed),
    ]
    return cmd + list(extra)


def build_fuse_cmd(args) -> List[str]:
    return [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", args.model,
        "--adapter-path", args.adapter_path,
        "--save-path", args.fused_path,
    ]


# --- CLI --------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="finetune_mlx.py",
        description="Fine-tune an LLM on VulnFanatic-NG triaged exports with MLX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("data_dir",
                   help="Folder containing the exported *.jsonl files (recursed).")
    p.add_argument("--model", required=True,
                   help="Base model: a local path or an MLX/HF repo id "
                        "(e.g. mlx-community/Qwen2.5-Coder-7B-Instruct-4bit).")
    p.add_argument("--adapter-path", default="./vulnfanatic-adapters",
                   help="Where LoRA adapters are written.")
    p.add_argument("--prepared-data", default="",
                   help="Where to write the train/valid split. "
                        "Default: '<adapter-path>/_data'.")
    p.add_argument("--valid-split", type=float, default=0.1,
                   help="Fraction of examples held out for validation.")
    p.add_argument("--iters", type=int, default=600, help="Training iterations.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=16,
                   help="Number of layers to apply LoRA to (-1 = all).")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-seq-length", type=int, default=0,
                   help="Max sequence length. 0 (default) = auto-fit to the longest "
                        "example, rounded up and capped at "
                        f"{_AUTO_SEQ_CAP}. Set a positive value to force it (higher = "
                        "more memory). Examples longer than this are truncated by MLX.")
    p.add_argument("--fine-tune-type", default="lora",
                   choices=["lora", "dora", "full"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fuse", action="store_true",
                   help="After training, fuse adapters into a standalone model.")
    p.add_argument("--fused-path", default="./vulnfanatic-fused",
                   help="Where the fused model is written (with --fuse).")
    p.add_argument("--dry-run", action="store_true",
                   help="Prepare the data and print the commands, but do not run them.")
    # Anything after `--` is forwarded verbatim to `mlx_lm lora`.
    args, extra = p.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    # Catch the common mistake of copying the docs' optional-argument brackets
    # (e.g. `[--fuse --fused-path ./out]`) — those `[ ]` are notation, not literal.
    bracketed = [t for t in extra if "[" in t or "]" in t]
    if bracketed:
        p.error(
            f"unrecognized argument(s) {bracketed}: these look like the docs' "
            f"optional-argument brackets. Remove the '[' and ']' — e.g. write "
            f"'--fuse --fused-path ./out', not '[--fuse --fused-path ./out]'.")
    return args, extra


def _report_failure(stage: str, rc: int, args) -> None:
    """Explain a subprocess failure — decode a signal kill (negative rc) into the
    likely cause, which for training is almost always out of memory."""
    if rc < 0:
        try:
            import signal
            signame = signal.Signals(-rc).name
        except Exception:
            signame = f"signal {-rc}"
        print(
            f"ERROR: {stage} was killed by {signame} (exit {rc}). During MLX "
            f"training this almost always means it ran OUT OF MEMORY: the model plus "
            f"a {args.max_seq_length}-token sequence does not fit this machine. "
            f"Try, in order: lower --max-seq-length (e.g. 4096 or 2048); add gradient "
            f"checkpointing with `-- --grad-checkpoint`; use a smaller / more-quantized "
            f"model; or lower vulnfanatic.storedContextChars before exporting so "
            f"examples are shorter.", file=sys.stderr)
    else:
        print(f"ERROR: {stage} failed (exit {rc}). See the mlx_lm output above.",
              file=sys.stderr)


def main(argv=None) -> int:
    args, extra = _parse_args(sys.argv[1:] if argv is None else argv)

    prepared = args.prepared_data or os.path.join(args.adapter_path, "_data")

    print(f"[1/4] Collecting training data from {args.data_dir} …")
    files = iter_jsonl_files(args.data_dir, exclude_dir=prepared)
    if not files:
        print(f"ERROR: no *.jsonl files found under {args.data_dir} (files must end "
              f"in .jsonl).", file=sys.stderr)
        return 2
    print(f"      found {len(files)} file(s)")

    examples, stats = load_examples(files)
    print(f"      {stats['kept']} examples kept "
          f"({stats['duplicates']} duplicate(s), {stats['skipped']} skipped, "
          f"{stats['total']} total lines)")
    if stats["kept"] < 2:
        print("ERROR: need at least 2 valid examples to fine-tune.", file=sys.stderr)
        return 2

    # Resolve the sequence length. Default (0) = auto-fit to the longest example;
    # a positive --max-seq-length is honored as-is.
    approx_tokens = longest_example_chars(examples) // 4  # rough chars→tokens
    if args.max_seq_length and args.max_seq_length > 0:
        if approx_tokens > args.max_seq_length:
            print(f"      WARNING: longest example ≈{approx_tokens} tokens > "
                  f"--max-seq-length {args.max_seq_length}; it will be truncated. "
                  f"Raise --max-seq-length (more memory) or lower "
                  f"vulnfanatic.storedContextChars before exporting.", file=sys.stderr)
    else:
        args.max_seq_length = fit_seq_length(approx_tokens)
        print(f"      auto max-seq-length={args.max_seq_length} "
              f"(longest example ≈{approx_tokens} tokens; pass --max-seq-length "
              f"to override)")
        if approx_tokens > args.max_seq_length:  # hit the auto cap
            print(f"      WARNING: longest example ≈{approx_tokens} tokens exceeds the "
                  f"auto cap {_AUTO_SEQ_CAP}; it will be truncated. Pass a higher "
                  f"--max-seq-length (more memory), or lower "
                  f"vulnfanatic.storedContextChars before exporting.", file=sys.stderr)

    print(f"[2/4] Writing train/valid split to {prepared} (valid_split="
          f"{args.valid_split}) …")
    n_train, n_valid = split_and_write(examples, prepared, args.valid_split, args.seed)
    print(f"      train={n_train}  valid={n_valid}")

    eff_batch = effective_batch_size(args.batch_size, n_train, n_valid)
    if eff_batch < args.batch_size:
        print(f"      WARNING: reducing --batch-size {args.batch_size} -> {eff_batch} "
              f"— MLX needs batch_size <= each split (train={n_train}, valid={n_valid}). "
              f"This lets a smoke test run, but {stats['kept']} examples is far too few "
              f"for a real fine-tune; export triaged findings from many more binaries.",
              file=sys.stderr)
        args.batch_size = eff_batch

    lora_cmd = build_lora_cmd(args, prepared, extra)
    fuse_cmd = build_fuse_cmd(args) if args.fuse else None

    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        print("      NOTE: MLX targets Apple Silicon (macOS arm64); this host is "
              f"{platform.system()}/{platform.machine()}.", file=sys.stderr)

    print(f"[3/4] LoRA fine-tune:\n      {' '.join(lora_cmd)}")
    if fuse_cmd:
        print(f"[4/4] Fuse:\n      {' '.join(fuse_cmd)}")

    if args.dry_run:
        print("\n--dry-run: not executing. Remove --dry-run to train.")
        return 0

    if importlib.util.find_spec("mlx_lm") is None:
        print("ERROR: mlx-lm is not installed in this Python. Install it with "
              "`pip install mlx-lm` (Apple Silicon required).", file=sys.stderr)
        return 3

    print("\n--- running mlx_lm lora ---")
    rc = subprocess.run(lora_cmd).returncode
    if rc != 0:
        _report_failure("training", rc, args)
        return rc
    print(f"Adapters written to {args.adapter_path}")

    if fuse_cmd:
        print("\n--- running mlx_lm fuse ---")
        rc = subprocess.run(fuse_cmd).returncode
        if rc != 0:
            _report_failure("fuse", rc, args)
            return rc
        print(f"Fused model written to {args.fused_path}")

    print("\nDone. Serve the result with an OpenAI-compatible server "
          "(e.g. `mlx_lm.server --model <path>`) and point "
          "vulnfanatic.apiBaseUrl at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
