"""Aggregate RoleKV subset summaries across modes."""

import argparse
import csv
import json
import os


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--modes", default="baseline,rolekv,random,inverted")
    parser.add_argument("--output_prefix", default="comparison")
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    rows = []
    task_names = set()
    summaries = {}

    for mode in modes:
        path = os.path.join(args.root, mode, "summary.json")
        if not os.path.exists(path):
            rows.append({"mode": mode, "status": "missing"})
            continue
        summary = load_json(path)
        summaries[mode] = summary
        for task in summary.get("tasks", {}):
            task_names.add(task)

    task_names = sorted(task_names)
    baseline = summaries.get("baseline", {})
    baseline_overall = baseline.get("overall")

    for mode in modes:
        if mode not in summaries:
            continue
        summary = summaries[mode]
        row = {
            "mode": mode,
            "status": "done",
            "n": summary.get("n"),
            "overall": summary.get("overall"),
            "delta_vs_baseline": (
                None
                if baseline_overall is None or summary.get("overall") is None
                else summary.get("overall") - baseline_overall
            ),
        }
        for task in task_names:
            row[f"{task}:n"] = summary.get("tasks", {}).get(task, {}).get("n")
            row[f"{task}:acc"] = summary.get("tasks", {}).get(task, {}).get("acc")
        rows.append(row)

    output_json = os.path.join(args.root, f"{args.output_prefix}.json")
    output_csv = os.path.join(args.root, f"{args.output_prefix}.csv")
    with open(output_json, "w") as f:
        json.dump({"modes": modes, "rows": rows}, f, indent=2)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"json": output_json, "csv": output_csv, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
