import re

log_lines = [
    "2025-07-14 10:20:45,123 INFO Starting service...",
    "2025-07-14 10:20:46,234 WARNING Low memory detected",
    "2025-07-14 10:20:47,345 ERROR Failed to connect to database",
]

log_pattern = re.compile(r'^(?P<date>\d{4}-\d{2}-\d{2})\s(?P<time>[\d:,]+)\s(?P<level>[A-Z]+)\s(?P<message>.+)$')

for line in log_lines:
    match = log_pattern.match(line)
    if match:
        print(f"Date: {match.group('date')}")
        print(f"Time: {match.group('time')}")
        print(f"Level: {match.group('level')}")
        print(f"Message: {match.group('message')}")
        print("-" * 40)