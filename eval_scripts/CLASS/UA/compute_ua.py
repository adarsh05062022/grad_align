import pandas as pd
import json
import argparse
import os


def compute_ua(csv_path):

    df = pd.read_csv(csv_path)

    classes = [
        "tench",
        "English springer",
        "cassette player",
        "chain saw",
        "church",
        "French horn",
        "garbage truck",
        "gas pump",
        "golf ball",
        "parachute",
    ]

    results = {}
    ua_list = []

    for i, cls in enumerate(classes):

        subset = df[df["case_number"] == i]

        total = len(subset)

        if total == 0:
            ua = 0
        else:
            failures = (
                subset["category_top1"].str.lower() == cls.lower()
            ).sum()

            ua = (1 - failures / total) * 100

        ua = round(ua, 2)

        results[cls] = ua
        ua_list.append(ua)

    results["Average"] = round(sum(ua_list) / len(ua_list), 2)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", required=False, default="cls9.csv", help="Path to the CSV file containing classification results")
    parser.add_argument("--output_json", default="ua_results.json")

    args = parser.parse_args()

    # Compute UA
    ua_results = compute_ua(args.csv_path)

    # Get experiment name from CSV
    csv_name = os.path.basename(args.csv_path).replace(".csv", "")

    new_entry = {csv_name: ua_results}

    # Load existing JSON if present
    if os.path.exists(args.output_json):

        try:
            with open(args.output_json, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}

    else:
        existing = {}

    # Append new results
    existing.update(new_entry)

    # Save updated JSON
    with open(args.output_json, "w") as f:
        json.dump(existing, f, indent=4)

    # Print results
    print("\nUpdated UA Results:\n")
    print(json.dumps(existing, indent=4))