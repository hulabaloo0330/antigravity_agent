"""Unified CLI entrypoint (`python -m src.cli <subcommand>`).

Supported subcommands:
  - `audit-data`: Scientific data provenance, rawness, connectome polarity & vault-lock check (`M1`)
  - `score`: Standalone read-only scorer (`--pred <path> [--ref-pred <path>] --split dev|confirmatory`)
  - `verify-invariance`: End-to-end future-frame poison, OOF label tamper & self/bilateral exclusion audit (`--model <name|all>`)
  - `yemini-audit`: Reproduce 21-worm Yemini 10s-mean scalar audit (`results/yemini_historical_audit.json`)
  - `run-m2-baselines`: Execute all 6 baseline families across all 3 immobilized Dev splits (`Round 0`)
  - `load-split`: Verify `load_dataset` governance & vault read-lock (`--dataset <id> --split <dev|vault>`)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from evaluators.scorer import main as scorer_main
from models.m2_pipeline import (
    run_m2_baselines_suite,
    run_yemini_historical_audit,
    verify_model_invariance,
)
from src.audit_cli import main as audit_main
from src.data_adapters import DATA_ROOT_DEFAULT, EXP_ROOT_DEFAULT, load_dataset


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 1

    sub = args[0]
    rest = args[1:]

    if sub == "audit-data":
        if "--check-all" not in rest:
            rest = ["--check-all"] + rest
        return audit_main(rest)

    if sub == "score":
        return scorer_main(rest)

    if sub == "verify-invariance":
        parser = argparse.ArgumentParser(prog="python -m src.cli verify-invariance")
        parser.add_argument("--model", required=True, help="Baseline model ID or 'all'")
        parser.add_argument("--dataset", default="yemini2021", help="Dev dataset ID for live invariance probe")
        parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
        parser.add_argument("--exp-root", type=Path, default=EXP_ROOT_DEFAULT)
        pargs = parser.parse_args(rest)
        res = verify_model_invariance(
            model_name=pargs.model,
            dataset_id=pargs.dataset,
            data_root=pargs.data_root,
            exp_root=pargs.exp_root,
        )
        print(json.dumps(res, indent=2))
        return 0 if res.get("passed", False) else 1

    if sub == "yemini-audit":
        parser = argparse.ArgumentParser(prog="python -m src.cli yemini-audit")
        parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
        parser.add_argument("--exp-root", type=Path, default=EXP_ROOT_DEFAULT)
        pargs = parser.parse_args(rest)
        res = run_yemini_historical_audit(data_root=pargs.data_root, exp_root=pargs.exp_root)
        print(json.dumps({k: v for k, v in res.items() if k not in ("causal_10335_reproduction", "strict_fold_internal_eligibility_reproduction")}, indent=2))
        return 0

    if sub == "run-m2-baselines":
        parser = argparse.ArgumentParser(prog="python -m src.cli run-m2-baselines")
        parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
        parser.add_argument("--exp-root", type=Path, default=EXP_ROOT_DEFAULT)
        parser.add_argument("--resume", action="store_true", help="Reuse existing predictions_dev_*.npz runs on disk")
        pargs = parser.parse_args(rest)
        res = run_m2_baselines_suite(data_root=pargs.data_root, exp_root=pargs.exp_root, resume=pargs.resume)
        print(
            json.dumps(
                {
                    "reference_baseline": res["reference_baseline"],
                    "strongest_overall_m2_baseline": res["strongest_overall_m2_baseline"],
                    "total_suite_wall_clock_sec": res["total_suite_wall_clock_sec"],
                    "peak_rss_mb": res["peak_rss_mb"],
                },
                indent=2,
            )
        )
        return 0

    if sub == "load-split":
        parser = argparse.ArgumentParser(prog="python -m src.cli load-split")
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--split", required=True)
        parser.add_argument("--allow-vault-unseal", action="store_true")
        parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
        parser.add_argument("--exp-root", type=Path, default=EXP_ROOT_DEFAULT)
        pargs = parser.parse_args(rest)
        try:
            recs = load_dataset(
                dataset_id=pargs.dataset,
                split=pargs.split,
                allow_vault_unseal=pargs.allow_vault_unseal,
                data_root=pargs.data_root,
                exp_root=pargs.exp_root,
                require_final_manifest_for_vault=(pargs.split != "dev"),
                caller_reason="cli.load-split",
            )
            print(json.dumps({"dataset": pargs.dataset, "split": pargs.split, "n_records": len(recs)}))
            return 0
        except PermissionError as exc:
            print(json.dumps({"error": "PermissionError", "message": str(exc)}), file=sys.stderr)
            return 2

    return audit_main(args)


if __name__ == "__main__":
    sys.exit(main())
