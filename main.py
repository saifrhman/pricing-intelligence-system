"""CLI entry point for pricing intelligence workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import format_markdown_report, run_pipeline, save_markdown_report
from src.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pricing intelligence pipeline.")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", type=str, default=None, help="yfinance interval, e.g. 1d")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--disable-sentiment",
        action="store_true",
        help="Disable optional sentiment module",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="Use fallback rule-based sentiment instead of transformer model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    ticker = args.ticker or config["project"]["default_ticker"]
    start_date = args.start_date or config["data"]["start_date"]
    end_date = args.end_date or config["data"]["end_date"]
    interval = args.interval or config["data"]["interval"]

    include_sentiment = bool(config["modeling"].get("include_sentiment", True))
    if args.disable_sentiment:
        include_sentiment = False

    use_transformer_sentiment = bool(config["modeling"].get("use_transformer_sentiment", True))
    if args.no_transformer:
        use_transformer_sentiment = False

    results = run_pipeline(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        include_sentiment=include_sentiment,
        use_transformer_sentiment=use_transformer_sentiment,
        raw_data_dir=config["data"]["raw_data_dir"],
        processed_data_dir=config["data"]["processed_data_dir"],
        outputs_dir=config["outputs"]["base_dir"],
    )

    markdown_report = format_markdown_report(results)
    report_path = save_markdown_report(
        markdown_text=markdown_report,
        ticker=ticker,
        output_dir=str(Path(config["outputs"]["base_dir"]) / "reports"),
    )

    decision = results["decision"]
    print("\n=== Pricing Intelligence Decision Summary ===")
    print(f"Ticker: {decision.ticker}")
    print(f"Predicted next-day return: {decision.latest_predicted_return:.4f}")
    print(f"Direction: {decision.direction}")
    print(f"Risk level: {decision.risk_level}")
    print(f"Anomaly status: {decision.anomaly_status}")
    print(f"Sentiment summary: {decision.sentiment_summary}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
