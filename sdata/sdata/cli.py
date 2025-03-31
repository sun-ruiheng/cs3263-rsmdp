# sdata/cli.py

import pandas as pd
import requests
import argparse
import os

from datetime import datetime, timedelta


def eod():
    # Default dates: one month ago to seven days before that
    today = datetime.utcnow().date()
    one_month_ago = today - timedelta(days=30)

    default_to_str = today.strftime("%Y%m%d")
    default_from_str = one_month_ago.strftime("%Y%m%d")

    parser = argparse.ArgumentParser(
        description="Fetch end-of-day (EOD) stock data and save as CSV.")
    parser.add_argument("symbol", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "-f", "--date_from",
        default=default_from_str,
        help=f"Start date in YYYYMMDD format (default: {default_from_str})"
    )
    parser.add_argument(
        "-t", "--date_to",
        default=default_to_str,
        help=f"End date in YYYYMMDD format (default: {default_to_str})"
    )
    args = parser.parse_args()

    # Convert to ISO for API
    try:
        date_from_iso = datetime.strptime(
            args.date_from, "%Y%m%d").date().isoformat()
        date_to_iso = datetime.strptime(
            args.date_to, "%Y%m%d").date().isoformat()
    except ValueError:
        raise ValueError("Dates must be in YYYYMMDD format")

    url = (
        f"https://api.stockdata.org/v1/data/eod?"
        f"symbols={args.symbol}&"
        f"date_from={date_from_iso}&"
        f"date_to={date_to_iso}&"
        f"api_token=alowStTWZUYV0oYJXnKE0fGalx7X4b7U6Mkn8q1U"
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch EOD data: {response.status_code} - {response.text}")

    data = response.json()
    rows = []
    for entry in data.get("data", []):
        date_time = datetime.fromisoformat(
            entry["date"].replace("Z", "+00:00"))
        rows.append({
            "date": date_time.date(),
            "open": entry["open"],
            "high": entry["high"],
            "low": entry["low"],
            "close": entry["close"],
            "volume": entry["volume"],
            "symbol": args.symbol.upper()
        })

    if not rows:
        print("No EOD data returned for the specified range.")
        return

    output_dir = "output/eod"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir, f"{args.date_from}_{args.date_to}_{args.symbol.upper()}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"EOD CSV file saved to {csv_path}")


def intraday():
    # Default dates: today and 7 days ago
    today = datetime.utcnow().date()
    seven_days_ago = today - timedelta(days=7)

    default_to_str = today.strftime("%Y%m%d")
    default_from_str = seven_days_ago.strftime("%Y%m%d")

    parser = argparse.ArgumentParser(
        description="Fetch intraday stock data and save as CSV.")
    parser.add_argument("symbol", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "-f", "--date_from",
        default=default_from_str,
        help=f"Start date in YYYYMMDD format (default: {default_from_str})"
    )
    parser.add_argument(
        "-t", "--date_to",
        default=default_to_str,
        help=f"End date in YYYYMMDD format (default: {default_to_str})"
    )
    args = parser.parse_args()

    # Validate date format
    try:
        date_from_iso = datetime.strptime(
            args.date_from, "%Y%m%d").date().isoformat()
        date_to_iso = datetime.strptime(
            args.date_to, "%Y%m%d").date().isoformat()
    except ValueError:
        raise ValueError("Dates must be in YYYYMMDD format")

    # API request
    url = (
        f"https://api.stockdata.org/v1/data/intraday?"
        f"symbols={args.symbol}&"
        f"date_from={date_from_iso}&"
        f"date_to={date_to_iso}&"
        f"api_token=alowStTWZUYV0oYJXnKE0fGalx7X4b7U6Mkn8q1U"
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch data: {response.status_code} - {response.text}")

    data = response.json()
    rows = []
    for entry in data.get("data", []):
        date_time = datetime.fromisoformat(
            entry["date"].replace("Z", "+00:00"))
        rows.append({
            "date": date_time.date(),
            "minute": date_time.strftime("%H:%M"),
            "label": date_time.strftime("%I:%M %p"),
            "high": entry["data"]["high"],
            "low": entry["data"]["low"],
            "open": entry["data"]["open"],
            "close": entry["data"]["close"],
            "average": (entry["data"]["high"] + entry["data"]["low"]) / 2,
            "volume": entry["data"]["volume"],
            "symbol": entry["ticker"]
        })

    if not rows:
        print("No data returned for the specified range.")
        return

    # Build output path
    output_dir = "output/intraday"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir, f"{args.date_from}_{args.date_to}_{args.symbol.upper()}.csv")

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV file saved to {csv_path}")
