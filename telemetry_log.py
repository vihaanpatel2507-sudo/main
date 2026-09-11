import csv


def load_entries(csv_path):
    entries = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                {
                    "timestamp": row["timestamp"],
                    "seconds": float(row["seconds"]),
                    "text": row["event"],
                }
            )
    return entries


def entries_near(entries, seconds, window=5.0):
    return [e for e in entries if abs(e["seconds"] - seconds) <= window]


def entries_matching(entries, keyword):
    keyword = keyword.lower()
    return [e for e in entries if keyword in e["text"].lower()]
