"""
Run rubric v2 locally using Docent SDK's build_judge.

Usage:
    export OPENAI_API_KEY=sk-...
    conda run -n core-bench python run_rubric_v2_local.py [--max-runs N] [--model MODEL] [--resume]

Reads traces from data/raw_focus.json, runs the judge on each, saves results
to data/rubric_v2_results.json incrementally.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from docent._llm_util.llm_svc import BaseLLMService
from docent._llm_util.providers.preference_types import ModelOption
from docent.data_models.agent_run import AgentRun
from docent.data_models.transcript import Transcript
from docent.judges.impl import build_judge
from docent.judges.types import OutputParsingMode, Rubric, ResultType

SCRIPT_DIR = Path(__file__).parent
FOCUS_PATH = SCRIPT_DIR / "data" / "raw_focus.json"
RUBRIC_TEXT_PATH = SCRIPT_DIR / "rubric_v2_instructions.md"
SCHEMA_PATH = SCRIPT_DIR / "rubric_v2_schema.json"
OUTPUT_PATH = SCRIPT_DIR / "data" / "rubric_v2_results.json"


def load_rubric_text() -> str:
    return RUBRIC_TEXT_PATH.read_text()


def load_output_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def load_focus_runs() -> list[dict]:
    with open(FOCUS_PATH) as f:
        return json.load(f)


def build_agent_run(row: dict) -> AgentRun:
    return AgentRun(
        id=row["agent_run_id"],
        name=row.get("agent_run_name", ""),
        metadata_json=row.get("agent_run_metadata", {}),
        transcripts=[
            Transcript(
                messages=row["messages"],
                metadata=row.get("transcript_metadata", {}),
            )
        ],
    )


def load_existing_results() -> dict[str, dict]:
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
        return {r["agent_run_id"]: r for r in results}
    return {}


def save_results(results: list[dict]):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)


async def run_judge_on_batch(
    judge,
    rows: list[dict],
    existing: dict[str, dict],
    concurrency: int = 3,
):
    results = list(existing.values())
    semaphore = asyncio.Semaphore(concurrency)
    completed = len(existing)
    total = len(rows) + completed

    async def evaluate_one(row: dict):
        nonlocal completed
        agent_run_id = row["agent_run_id"]
        meta = row.get("agent_run_metadata", {})
        capsule_id = meta.get("capsule_id", "unknown")
        model = meta.get("model", "unknown")
        scaffold = meta.get("scaffold", "unknown")
        accuracy = meta.get("scores", {}).get("accuracy", None)

        async with semaphore:
            try:
                agent_run = build_agent_run(row)
                result = await judge(agent_run)

                if result.result_type == ResultType.DIRECT_RESULT:
                    entry = {
                        "agent_run_id": agent_run_id,
                        "capsule_id": capsule_id,
                        "model": model,
                        "scaffold": scaffold,
                        "accuracy": accuracy,
                        "result_type": "DIRECT_RESULT",
                        "output": result.output,
                    }
                else:
                    entry = {
                        "agent_run_id": agent_run_id,
                        "capsule_id": capsule_id,
                        "model": model,
                        "scaffold": scaffold,
                        "accuracy": accuracy,
                        "result_type": "FAILURE",
                        "error": str(result.result_metadata),
                    }
            except Exception as e:
                entry = {
                    "agent_run_id": agent_run_id,
                    "capsule_id": capsule_id,
                    "model": model,
                    "scaffold": scaffold,
                    "accuracy": accuracy,
                    "result_type": "ERROR",
                    "error": str(e),
                }

            results.append(entry)
            completed += 1

            out = entry.get("output", {})
            label = out.get("label", entry.get("result_type"))
            src = out.get("answer_source", "")
            obs = out.get("primary_obstacle", "")
            res = out.get("obstacle_resolution", "")
            print(
                f"[{completed}/{total}] {capsule_id} | {model[:20]} | "
                f"{label} | {src} | {obs} | {res}",
                flush=True,
            )

            if completed % 5 == 0:
                save_results(results)

    tasks = [evaluate_one(row) for row in rows]
    await asyncio.gather(*tasks)

    save_results(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run rubric v2 judge locally")
    parser.add_argument("--max-runs", type=int, default=None, help="Max runs to evaluate")
    parser.add_argument("--model", type=str, default="gpt-5.5", help="Judge model name")
    parser.add_argument("--provider", type=str, default="openai", help="Model provider")
    parser.add_argument("--reasoning-effort", type=str, default="high", help="Reasoning effort level")
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel judge calls")
    parser.add_argument("--resume", action="store_true", help="Skip already-evaluated runs")
    parser.add_argument("--failures-only", action="store_true", help="Only evaluate failing runs (accuracy=0)")
    args = parser.parse_args()

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    print("Loading rubric and schema...")
    rubric_text = load_rubric_text()
    output_schema = load_output_schema()

    model_opts = {
        "provider": args.provider,
        "model_name": args.model,
    }
    if args.reasoning_effort:
        model_opts["reasoning_effort"] = args.reasoning_effort

    rubric = Rubric(
        rubric_text=rubric_text,
        output_schema=output_schema,
        judge_model=ModelOption(**model_opts),
        output_parsing_mode=OutputParsingMode.CONSTRAINED_DECODING,
    )

    effort_str = f" (reasoning_effort={args.reasoning_effort})" if args.reasoning_effort else ""
    print(f"Building judge with {args.provider}/{args.model}{effort_str}...")
    llm_svc = BaseLLMService()
    judge = build_judge(rubric, llm_svc)

    print("Loading focus runs...")
    rows = load_focus_runs()
    print(f"  Total runs: {len(rows)}")

    if args.failures_only:
        rows = [r for r in rows if r.get("agent_run_metadata", {}).get("scores", {}).get("accuracy", 1) == 0]
        print(f"  Filtered to failures only: {len(rows)}")

    existing = {}
    if args.resume:
        existing = load_existing_results()
        print(f"  Already evaluated: {len(existing)}")
        rows = [r for r in rows if r["agent_run_id"] not in existing]
        print(f"  Remaining: {len(rows)}")

    if args.max_runs and len(rows) > args.max_runs:
        rows = rows[: args.max_runs]
        print(f"  Capped to: {len(rows)}")

    if not rows:
        print("Nothing to evaluate.")
        return

    print(f"\nStarting evaluation ({len(rows)} runs, concurrency={args.concurrency})...\n")
    start = time.time()
    results = asyncio.run(run_judge_on_batch(judge, rows, existing, args.concurrency))
    elapsed = time.time() - start

    successes = sum(1 for r in results if r.get("result_type") == "DIRECT_RESULT")
    failures = sum(1 for r in results if r.get("result_type") != "DIRECT_RESULT")
    print(f"\nDone in {elapsed:.1f}s. Results: {successes} evaluated, {failures} errors.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
