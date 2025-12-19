#!/usr/bin/env python3
"""
Helper script to run the BioML TriMul task on Modal directly, using the
task definition in `bioml/trimul/task.yml`.

NEW:
  - Accepts a JSONL file of generations and runs all submissions asynchronously.
  - Aggregates:
      * % failed invocations
      * mean/median/min/max of overall leaderboard score (microseconds) per JSONL entry

Assumes each JSONL line contains a field with the submission code:
  - tries: "content", then "submission", then "code"
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Optional

from libkernelbot.consts import ModalGPU, SubmissionMode
from libkernelbot.launchers import ModalLauncher
from libkernelbot.report import RunProgressReporter
from libkernelbot.run_eval import FullResult
from libkernelbot.submission import compute_score
from libkernelbot.task import LeaderboardTask, build_task_config, make_task_definition


PROJECT_ROOT = Path(__file__).resolve().parent
TRIMUL_TASK_YAML = PROJECT_ROOT / "bioml" / "trimul" / "task.yml"


class SimpleReporter(RunProgressReporter):
    """Minimal reporter that prints to console."""

    async def _update_message(self):
        print(f"[{self.title}]")
        for line in self.lines:
            print(f"  {line}")

    async def display_report(self, title: str, report):
        print(f"\n=== {title} ===")
        print(f"Report has {len(report.data)} items")


def load_trimul_task() -> LeaderboardTask:
    """Load the TriMul LeaderboardTask from its YAML definition."""
    if not TRIMUL_TASK_YAML.exists():
        raise FileNotFoundError(
            f"Could not find TriMul task definition at {TRIMUL_TASK_YAML}. "
            "Run this script from the project root."
        )
    definition = make_task_definition(TRIMUL_TASK_YAML)
    return definition.task


async def run_trimul_on_modal(
    submission_code: str,
    gpu_type: str = "T4",
    mode: str = "test",
) -> tuple[FullResult, LeaderboardTask]:
    """
    Run a TriMul submission on Modal using the official task definition.

    Args:
        submission_code: Contents of the user's `submission.py`
        gpu_type: One of ModalGPU names (T4, L4, A100, H100, B200, L4x4)
        mode: One of: test, benchmark, leaderboard, profile, private
    """
    task = load_trimul_task()

    try:
        mode_enum = SubmissionMode(mode)
    except ValueError as e:
        valid = ", ".join(m.value for m in SubmissionMode)
        raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid}") from e

    config = build_task_config(
        task=task,
        submission_content=submission_code,
        arch=None,
        mode=mode_enum,
    )

    launcher = ModalLauncher(add_include_dirs=[])
    gpu_enum = ModalGPU[gpu_type.upper()]

    reporter = SimpleReporter(f"TriMul on {gpu_enum.name} (Modal)")
    print(f"Submitting TriMul task to Modal on {gpu_enum.name} with mode='{mode_enum.value}'...")

    result = await launcher.run_submission(config, gpu_enum, reporter)
    return result, task


def print_benchmark_details(result: FullResult):
    """Print per-benchmark statistics for a leaderboard run if available."""
    if "leaderboard" not in result.runs:
        return

    run_res = result.runs["leaderboard"].run
    if not run_res or not run_res.result:
        return

    data = run_res.result
    if "benchmark-count" not in data:
        return

    num_benchmarks = int(data["benchmark-count"])
    print(f"\nLeaderboard benchmarks: {num_benchmarks}")
    for i in range(num_benchmarks):
        prefix = f"benchmark.{i}."
        mean_ns = float(data.get(prefix + "mean", 0.0))
        std_ns = float(data.get(prefix + "std", 0.0))
        best_ns = float(data.get(prefix + "best", 0.0))
        worst_ns = float(data.get(prefix + "worst", 0.0))

        mean_s = mean_ns / 1e9 if mean_ns else 0.0
        std_s = std_ns / 1e9 if std_ns else 0.0
        best_s = best_ns / 1e9 if best_ns else 0.0
        worst_s = worst_ns / 1e9 if worst_ns else 0.0

        print(f"  Benchmark {i}:")
        print(f"    mean:   {mean_s:.6f} s")
        print(f"    std:    {std_s:.6f} s")
        print(f"    best:   {best_s:.6f} s")
        print(f"    worst:  {worst_s:.6f} s")


def print_result(result: FullResult, task: LeaderboardTask | None = None):
    """Pretty print a FullResult and optionally the leaderboard score."""
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"Success: {result.success}")

    if not result.success:
        print("\nSystem Info:")
        if result.system.gpu:
            print(f"  GPU: {result.system.gpu}")
        if result.system.cpu:
            print(f"  CPU: {result.system.cpu}")
        print(f"  Requeues: {result.system.requeues}")
        print(f"Error: {result.error}")
        return

    print("\nSystem Info:")
    print(f"  GPU: {result.system.gpu}")
    print(f"  CPU: {result.system.cpu}")
    print(f"  Requeues: {result.system.requeues}")
    print(f"  Torch: {result.system.torch}")
    print(f"  Runtime: {result.system.runtime}")

    print(f"\nRuns: {len(result.runs)}")
    for run_name, run_result in result.runs.items():
        print(f"\n  Run: {run_name}")
        print(f"    Start: {run_result.start}")
        print(f"    End:   {run_result.end}")

        if run_result.compilation:
            comp = run_result.compilation
            print("    Compilation:")
            print(f"      Success:  {comp.success}")
            if not comp.success:
                print(f"      ExitCode: {comp.exit_code}")
                print(f"      Stderr:   {comp.stderr}...")

        if run_result.run:
            run = run_result.run
            print("    Execution:")
            print(f"      Success:   {run.success}")
            print(f"      Passed:    {run.passed}")
            print(f"      Duration:  {run.duration:.2f}s")
            print(f"      Exit Code: {run.exit_code}")

            if run.stdout:
                print(f"      Stdout:\n{run.stdout[:500]}{'...' if len(run.stdout) > 500 else ''}")
            if run.stderr:
                print(f"      Stderr:\n{run.stderr}{'...' if len(run.stderr) > 500 else ''}")

    if task is not None and "leaderboard" in result.runs:
        print_benchmark_details(result)
        try:
            score_seconds = compute_score(result, task, submission_id=-1)
            score_us = score_seconds * 1_000_000
            print(f"\nOverall leaderboard score (microseconds, {task.ranking_by.value}): {score_us:.3f} us")
        except Exception as e:
            print(f"\nCould not compute leaderboard score: {e}")


# ---------------- NEW: JSONL runner + stats ----------------

def _extract_submission_code(d: dict) -> str:
    # for k in ("content", "submission", "code"):
    for k in ("postprocessed_content",):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    raise ValueError("Could not find submission code in JSONL line (expected one of: content/submission/code).")


async def run_jsonl_on_modal(
    jsonl_path: Path,
    gpu_type: str,
    mode: str,
    concurrency: int,
    print_each: bool,
):
    lines = jsonl_path.read_text().splitlines()
    items = [json.loads(line) for line in lines if line.strip()]

    sem = asyncio.Semaphore(concurrency)

    async def _run_one(idx: int, item: dict):
        submission_code = _extract_submission_code(item)
        async with sem:
            try:
                result, task = await run_trimul_on_modal(
                    submission_code=submission_code,
                    gpu_type=gpu_type,
                    mode=mode,
                )
            except Exception as e:
                return {"idx": idx, "ok": False, "error": f"{type(e).__name__}: {e}", "score_us": None}

        print_result(result, task)
        score_us = None
        ok = bool(result.success)
        err = None

        if ok:
            try:
                score_seconds = compute_score(result, task, submission_id=-1)
                score_us = score_seconds * 1_000_000
            except Exception as e:
                ok = False
                err = f"compute_score failed: {type(e).__name__}: {e}"

        if not ok and err is None:
            err = str(result.error)

        if print_each:
            print("\n" + "-" * 60)
            print(f"[{idx}] ok={ok} score_us={score_us}")
            if err:
                print(f"[{idx}] error={err}")

        return {"idx": idx, "ok": ok, "error": err, "score_us": score_us}

    results = await asyncio.gather(*[_run_one(i, item) for i, item in enumerate(items)])

    total = len(results)
    n_fail = sum(1 for r in results if not r["ok"])
    fail_pct = 100.0 * n_fail / total if total else 0.0

    scores = [r["score_us"] for r in results if r["ok"] and r["score_us"] is not None]

    print("\n" + "=" * 60)
    print("AGGREGATE STATS")
    print("=" * 60)
    print(f"Total invocations: {total}")
    print(f"Failed: {n_fail} ({fail_pct:.2f}%)")
    print(f"Succeeded w/ score: {len(scores)}")

    if scores:
        print(f"mean score_us:   {statistics.mean(scores):.3f}")
        print(f"median score_us: {statistics.median(scores):.3f}")
        print(f"min score_us:    {min(scores):.3f}")
        print(f"max score_us:    {max(scores):.3f}")
    else:
        print("No successful scores to summarize.")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BioML TriMul submission(s) on Modal using the official task definition.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--submission",
        "-s",
        help="Path to your TriMul submission.py file.",
    )
    group.add_argument(
        "--jsonl",
        "-j",
        help="Path to JSONL file containing many generated submissions (field: content/submission/code).",
    )

    parser.add_argument(
        "--gpu",
        "-g",
        default="T4",
        choices=[g.name for g in ModalGPU],
        help="Modal GPU type to use (default: T4).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="test",
        choices=[m.value for m in SubmissionMode],
        help="Submission mode (default: test).",
    )

    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=8,
        help="Max number of concurrent Modal invocations for JSONL mode (default: 8).",
    )
    parser.add_argument(
        "--print-each",
        action="store_true",
        help="Print per-entry success/score/error while running JSONL mode.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # if args.submission:
    #     submission_path = Path(args.submission)
    #     if not submission_path.exists():
    #         raise FileNotFoundError(f"Submission file not found: {submission_path}")
    #     submission_code = submission_path.read_text()

    #     # import pdb; pdb.set_trace()

    #     print(submission_code)

    #     result, task = await run_trimul_on_modal(
    #         submission_code=submission_code,
    #         gpu_type=args.gpu,
    #         mode=args.mode,
    #     )
    #     print_result(result, task)
    #     return

    # JSONL mode
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    await run_jsonl_on_modal(
        jsonl_path=jsonl_path,
        gpu_type=args.gpu,
        mode=args.mode,
        concurrency=args.concurrency,
        print_each=args.print_each,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")