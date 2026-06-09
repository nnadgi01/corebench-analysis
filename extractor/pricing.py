"""Per-million-token prices for the models in the CORE-Bench runs.

Used by the Codex parser, since codex_exec.log does not carry a cost field.
Other scaffolds report cost directly in their logs.

Prices in USD per 1M tokens. Source: public OpenAI / Anthropic price pages.
These are reference prices at the time of the runs; if the paper needs
exact billed cost, the team should swap in invoiced rates.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


# OpenAI GPT-5 family. Shared rate card across the GPT-5.x line for
# input/cached/output tiers; reasoning effort doesn't change pricing.
GPT_5 = Price(input_per_1m=1.25, cached_input_per_1m=0.125, output_per_1m=10.00)

# Anthropic. Opus 4.5 launched 2025-11-24 with a 67% cut to $5/$25 per 1M
# (down from $15/$75 on Opus 4.0/4.1); Opus 4.6 kept the same rate. Cache
# read is the cheap tier; cache write is priced higher than uncached input
# but isn't separately tracked here — fine for the CoreAgent runs in this
# dataset, which made no cache reads/writes (verified from UPLOAD.json).
CLAUDE_OPUS_4_X = Price(input_per_1m=5.00, cached_input_per_1m=0.50, output_per_1m=25.00)
CLAUDE_HAIKU_4_5 = Price(input_per_1m=1.00, cached_input_per_1m=0.10, output_per_1m=5.00)


def lookup(model: str) -> Price | None:
    m = model.lower().lstrip("/")
    # Strip provider prefixes used in metadata.json (e.g. "anthropic/claude-opus-4-5").
    if "/" in m:
        m = m.split("/", 1)[1]
    if m.startswith("gpt-5"):
        return GPT_5
    if m.startswith("claude-opus-4"):
        return CLAUDE_OPUS_4_X
    if m.startswith("claude-haiku-4"):
        return CLAUDE_HAIKU_4_5
    return None


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    cached_input_tokens: int = 0,
    output_tokens: int = 0,
) -> float | None:
    """Compute USD cost given token breakdown. Returns None if model unknown.

    `input_tokens` here is the *uncached* portion. Codex reports
    `input_tokens` (total) and `cached_input_tokens` separately, so the
    caller should pass `input_tokens - cached_input_tokens` here.
    """
    p = lookup(model)
    if p is None:
        return None
    return (
        input_tokens * p.input_per_1m
        + cached_input_tokens * p.cached_input_per_1m
        + output_tokens * p.output_per_1m
    ) / 1_000_000
