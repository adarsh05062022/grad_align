import pandas as pd
import json
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

    topk_cols = [
        "category_top1",
        "category_top2",
        "category_top3",
        "category_top4",
        "category_top5",
    ]

    results = {}
    ua_list = []

    for i, cls in enumerate(classes):

        subset = df[df["case_number"] == i]
        total = len(subset)

        if total == 0:
            ua = 0
        else:
            failures = subset[topk_cols].fillna("").apply(
                lambda row: any(cls.lower() in str(x).lower() for x in row),
                axis=1
            ).sum()

            ua = (1 - failures / total) * 100

        ua = round(ua, 2)

        results[cls] = ua
        ua_list.append(ua)

    results["Average"] = round(sum(ua_list) / len(ua_list), 2)

    return results


if __name__ == "__main__":

    # List of CSV files
    csv_files = [
        "/storage/s25017/MUNBa/SD/eval_scripts/CLASS/UA/abalation3/diffusers-cls_0-MUNBa-method_full-lr_1e-05_E5_U963_masked_nash_topk10_single_mask.pt_classification.csv",
        "/storage/s25017/MUNBa/SD/eval_scripts/CLASS/UA/abalation3/diffusers-cls_3-MUNBa-method_full-lr_1e-05_E5_U858_masked_nash_topk10_single_mask.pt_classification.csv",
        "/storage/s25017/MUNBa/SD/eval_scripts/CLASS/UA/abalation3/diffusers-cls_5-MUNBa-method_full-lr_1e-05_E5_U956_masked_nash_topk10_single_mask.pt_classification.csv",
        "/storage/s25017/MUNBa/SD/eval_scripts/CLASS/UA/abalation3/diffusers-cls_7-MUNBa-method_full-lr_1e-05_E5_U931_masked_nash_topk10_single_mask.pt_classification.csv",
        "/storage/s25017/MUNBa/SD/eval_scripts/CLASS/UA/abalation3/diffusers-cls_9-MUNBa-method_full-lr_1e-05_E5_U960_masked_nash_topk10_single_mask.pt_classification.csv"
    ]

    output_json = "abalation_3_ua_results.json"

    # Load existing JSON if present
    if os.path.exists(output_json):
        try:
            with open(output_json, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    # Process each CSV
    for csv_file in csv_files:

        ua_results = compute_ua(csv_file)

        csv_name = os.path.basename(csv_file).replace(".csv", "")

        existing[csv_name] = ua_results

        print(f"\nProcessed {csv_file}")

    # Save updated JSON
    with open(output_json, "w") as f:
        json.dump(existing, f, indent=4)

    print("\nUpdated UA Results:\n")
    print(json.dumps(existing, indent=4))