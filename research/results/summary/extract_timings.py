"""
Agent timing extractor for Omni-Shield eval results.

Reads eval_full.json and computes per-agent timing statistics from the
`agent_timings` field on each per_doc entry.

NOTE: The current eval_full.json was produced by eval.py which records only
total request latency per document, not per-agent breakdowns.  To populate
agent_timings, the backend's /redact/image response must include a
`agent_timings` dict (e.g. {"LayoutAgent": 1.2, "OCRAgent": 0.4, ...}).
If that field is absent from the response, this script writes an empty
result and explains how to enable it.

Usage:
    cd ~/omni-shield/research/results/summary
    python extract_timings.py [--eval ../../raw/eval_full.json]
                              [--output ./agent_timings.json]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

AGENT_ORDER = [
    "RouterAgent",
    "LayoutAgent",
    "OCRAgent",
    "TextPIIAgent",
    "ContextAgent",
    "BBRefinerAgent",
    "CriticAgent",
    "RedactionAgent",
]


def stats(values: list) -> dict:
    if not values:
        return {"mean_s": None, "min_s": None, "max_s": None, "n": 0}
    return {
        "mean_s": round(sum(values) / len(values), 3),
        "min_s":  round(min(values), 3),
        "max_s":  round(max(values), 3),
        "n":      len(values),
    }


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",   default=str(here / "../raw/eval_full.json"))
    parser.add_argument("--output", default=str(here / "agent_timings.json"))
    args = parser.parse_args()

    data    = json.loads(Path(args.eval).read_text())
    per_doc = data.get("per_doc", [])

    # Collect per-agent timing lists
    agent_times   = defaultdict(list)
    total_times   = []
    docs_with_timings = 0

    for doc in per_doc:
        timings = doc.get("agent_timings")
        if not timings:
            continue
        docs_with_timings += 1
        doc_total = 0.0
        for agent, t in timings.items():
            agent_times[agent].append(float(t))
            doc_total += float(t)
        total_times.append(doc_total)

    out_path = Path(args.output)

    if docs_with_timings == 0:
        print("=" * 60)
        print("No agent_timings data found in eval_full.json.")
        print()
        print("The eval script records total HTTP latency per document,")
        print("but does not currently break this down by agent.")
        print()
        print("To enable per-agent timing:")
        print("  1. In omni_shield_agents.py, record wall-clock time for")
        print("     each agent call and include a 'agent_timings' dict in")
        print("     the JSON response from /redact/image.")
        print("  2. In eval.py call_api(), extract and store")
        print("     resp.get('agent_timings', {}) alongside per-doc metrics.")
        print("  3. Re-run eval.py to regenerate eval_full.json.")
        print("=" * 60)
        print()
        print("Falling back to total latency statistics across all docs ...")

        # Fallback: use total request latency from eval results
        latencies = [
            doc["latency_s"]
            for doc in per_doc
            if "latency_s" in doc
        ]
        fallback = {
            "_note": (
                "agent_timings not available in eval_full.json. "
                "Values below are total per-request latency (all agents combined). "
                "See extract_timings.py for how to enable per-agent breakdown."
            ),
            "total_pipeline": stats(latencies),
            "per_agent": {
                agent: {"mean_s": None, "min_s": None, "max_s": None, "n": 0}
                for agent in AGENT_ORDER
            },
        }
        out_path.write_text(json.dumps(fallback, indent=2))
        print(f"\nTotal latency stats (n={len(latencies)} docs):")
        s = stats(latencies)
        print(f"  mean={s['mean_s']}s  min={s['min_s']}s  max={s['max_s']}s")
        print(f"\nOutput → {out_path}")
        return

    # Build output with agents in canonical order
    result = {}
    for agent in AGENT_ORDER:
        if agent in agent_times:
            result[agent] = stats(agent_times[agent])
    # Any extra agents not in canonical order
    for agent in sorted(agent_times):
        if agent not in result:
            result[agent] = stats(agent_times[agent])

    result["total_mean_s"] = round(
        sum(v["mean_s"] for v in result.values() if isinstance(v, dict) and v["mean_s"])
        , 3)

    out_path.write_text(json.dumps(result, indent=2))

    print(f"Agent timing summary ({docs_with_timings} docs):")
    print(f"{'Agent':<18} {'mean':>8} {'min':>8} {'max':>8}  {'n':>5}")
    print("-" * 50)
    for agent in AGENT_ORDER:
        if agent in result:
            s = result[agent]
            print(f"{agent:<18} {s['mean_s']:>7.3f}s {s['min_s']:>7.3f}s "
                  f"{s['max_s']:>7.3f}s {s['n']:>6}")
    print("-" * 50)
    print(f"{'Total (sum of means)':<18} {result['total_mean_s']:>7.3f}s")
    print(f"\nOutput → {out_path}")


if __name__ == "__main__":
    main()
