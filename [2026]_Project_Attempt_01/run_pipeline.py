"""
End-to-end pipeline: data -> features -> model -> simulation -> this month's buy list.

Usage
  python run_pipeline.py --source download   # fetch fresh data (internet needed), cache it
  python run_pipeline.py --source file       # reuse cached data (fast, reproducible)

Cron example (1st weekday of each month, 08:00):
  0 8 1-3 * 1-5  cd /path/to/repo && python run_pipeline.py --source download >> outputs/cron.log 2>&1
"""
import argparse
import os

import pandas as pd

import config as cfg
from src.data_loader import load_data
from src.features import build_monthly_table
from src.modeling import (add_split, feature_importance, fit_final_and_predict,
                          test_report, tune_models)
from src.simulation import compare_strategies, latest_buy_list


def main(source: str):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    pd.set_option("display.width", 160)

    # 1) Data --------------------------------------------------------------
    prices, macro = load_data(source, cfg)

    # 2) Features ----------------------------------------------------------
    monthly, fs = build_monthly_table(prices, macro, cfg)
    monthly = add_split(monthly, cfg.TRAIN_END, cfg.VALID_END)
    features = fs["FEATURES"]
    print(f"\nMonthly table: {len(monthly):,} rows, {len(features)} features")
    print(monthly.groupby("split")["ym"].agg(["min", "max", "count"]))

    # 3) Model tuning on VALIDATION, final fit on TRAIN+VALIDATION -----------
    tuning = tune_models(monthly, features, cfg)
    print("\nTop configurations (validation):\n", tuning.head(5).to_string(index=False))
    best = tuning.dropna(subset=["val_precision"]).iloc[0]
    model, monthly = fit_final_and_predict(monthly, features, cfg, best["model"], int(best["max_depth"]))
    print(f"\nFinal model: {best['model']} (max_depth={int(best['max_depth'])})")
    print("\nTEST precision by threshold:\n", test_report(monthly).round(3).to_string(index=False))
    print("\nTop features:\n", feature_importance(model, features).round(3).to_string())

    # 4) Simulation on the TEST period (out-of-sample) ------------------------
    summary, curves = compare_strategies(monthly, cfg, period_mask=monthly["split"] == "test")
    print("\nBuy-only monthly investing, TEST period:\n", summary.round(4).to_string(index=False))

    # 5) This month's purchases --------------------------------------------
    buy = latest_buy_list(monthly, cfg)
    print("\n", buy.round(4).to_string())

    # Save outputs (small files only)
    tuning.to_csv(f"{cfg.OUTPUT_DIR}/tuning.csv", index=False)
    summary.to_csv(f"{cfg.OUTPUT_DIR}/strategy_summary.csv", index=False)
    curves.to_csv(f"{cfg.OUTPUT_DIR}/value_curves.csv")
    buy.to_csv(f"{cfg.OUTPUT_DIR}/latest_buy_list.csv")
    return monthly, summary, curves, buy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buy-only ML-tilted monthly investing pipeline")
    parser.add_argument("--source", choices=["download", "file"], default="file")
    main(parser.parse_args().source)
