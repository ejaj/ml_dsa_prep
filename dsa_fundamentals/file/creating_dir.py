import subprocess
import os
from pathlib import Path

project_dir = Path("example_project")
project_dir.mkdir(exist_ok=True)

print(f"Created directory: {project_dir.resolve()}")

file_path = project_dir / "hello.text"
file_path.write_text("Hello from Python!\nThis file was created using pathlib.")
print(f"Created file: {file_path.resolve()}")

if os.name == "nt":
    subprocess.run(["run"], shell=True, cwd=project_dir)
else:
    subprocess.run(["ls", "-l"], cwd=project_dir)

os.remove(file_path)
os.remove(project_dir)
print("\nCleaned up: deleted file and directory.")
