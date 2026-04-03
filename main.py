"""CLI entry point for pricing intelligence workflow."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from src.pipeline import format_markdown_report, run_pipeline, save_markdown_report
from src.utils import load_yaml_config


logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--allow-cache",
        action="store_true",
        help="Allow cached raw market data fallback when fresh download fails.",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Allow demo fallbacks (synthetic market data and mock sentiment headlines).",
    )
    parser.add_argument(
        "--sentiment-headlines-file",
        type=str,
        default=None,
        help="Optional text file containing one headline per line for sentiment analysis.",
    )
    parser.add_argument(
        "--ingestion-timeout-seconds",
        type=int,
        default=30,
        help="Per-request timeout for market data download.",
    )
    parser.add_argument(
        "--ingestion-max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for market data download.",
    )
    return parser.parse_args()


def _load_headlines_from_file(path: str | None) -> List[str] | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    headlines = [line.strip() for line in text.splitlines() if line.strip()]
    return headlines or None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

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

    manual_headlines = _load_headlines_from_file(args.sentiment_headlines_file)

    logger.info(
        "Starting run: ticker=%s start=%s end=%s allow_cache=%s demo_mode=%s",
        ticker,
        start_date,
        end_date,
        args.allow_cache,
        args.demo_mode,
    )

    try:
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
            manual_headlines=manual_headlines,
            allow_cache_fallback=args.allow_cache,
            demo_mode=args.demo_mode,
            ingestion_timeout_seconds=args.ingestion_timeout_seconds,
            ingestion_max_retries=args.ingestion_max_retries,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        print(f"ERROR: Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

    markdown_report = format_markdown_report(results)
    report_path = save_markdown_report(
        markdown_text=markdown_report,
        ticker=ticker,
        output_dir=str(Path(config["outputs"]["base_dir"]) / "reports"),
    )

    decision = results["decision"]
    provenance = results.get("provenance", {})
    ingestion = provenance.get("ingestion", {})
    print("\n=== Pricing Intelligence Decision Summary ===")
    print(f"Ticker: {decision.ticker}")
    print(f"Predicted next-day return: {decision.latest_predicted_return:.4f}")
    print(f"Direction: {decision.direction}")
    print(f"Risk level: {decision.risk_level}")
    print(f"Anomaly status: {decision.anomaly_status}")
    print(f"Sentiment summary: {decision.sentiment_summary}")
    print(
        "Data source: "
        f"{ingestion.get('source_type', 'unknown')} "
        f"(status={ingestion.get('status', 'unknown')})"
    )

    warnings = provenance.get("warnings", [])
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
