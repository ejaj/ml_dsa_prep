import csv
from collections import defaultdict

def compute_total_volume(csv_path):
    totals = defaultdict(int)
    try:
        with open(csv_path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                trader = row.get('trader')
                try:
                    volume = int(row.get('volume', 0))
                except ValueError:
                    continue
                if trader:
                    totals[trader] += volume
    except FileExistsError:
        print("file not found")
    return dict(totals)