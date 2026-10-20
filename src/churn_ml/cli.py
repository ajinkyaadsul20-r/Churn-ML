"""Command line interface:  python -m churn_ml.cli {generate-data,train,predict}"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config
from .data import generate_churn_data
from .predict import ChurnPredictor
from .train import run_training


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(prog="churn-ml", description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate-data", help="Write the synthetic dataset to the configured path")
    sub.add_parser("train", help="Train, evaluate and save artifacts")
    pr = sub.add_parser("predict", help="Score customers from a JSON file (list of records)")
    pr.add_argument("input_json")
    pr.add_argument("--artifacts", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "generate-data":
        out = Path(cfg["data"]["raw_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        generate_churn_data(cfg["data"]["n_samples"], cfg["seed"]).to_csv(out, index=False)
        print(f"Wrote {out}")
    elif args.cmd == "train":
        res = run_training(cfg)
        print(json.dumps(res["metrics"], indent=2))
    elif args.cmd == "predict":
        records = json.loads(Path(args.input_json).read_text())
        predictor = ChurnPredictor(args.artifacts or cfg["artifacts_dir"])
        print(json.dumps(predictor.predict(records), indent=2))


if __name__ == "__main__":
    main()
