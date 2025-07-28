import subprocess
import re
from pathlib import Path

fake_output = """
test_login PASSED
test_logout FAILED
test_register PASSED
test_profile SKIPPED
"""

log_file = Path("test_log.txt")
log_file.write_text(fake_output)
print(f"[+] Log saved to {log_file.resolve()}")

log_contents = log_file.read_text()

pattern = re.compile(r'^(test_\w+)\s+(PASSED|FAILED|SKIPPED)$', re.MULTILINE)

results = pattern.findall(log_contents)

print("\n[+] Parsed Test Results:")

for test_name, status in results:
    print(f" - {test_name}: {status}")

import logging
logging.basicConfig(level=logging.INFO)
logging.info("Login test started")