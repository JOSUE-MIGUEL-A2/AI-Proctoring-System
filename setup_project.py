# setup_project.py — run once to create directory structure
import os

directories = [
    "src/core",
    "src/utils",
    "assets/sounds",
    "assets/snapshots",
    "logs",
    "config",
]

files = {
    "src/__init__.py": "",
    "src/core/__init__.py": "",
    "src/utils/__init__.py": "",
    "config/settings.py": "# Thresholds and config loaded here\n",
    "main.py": "# Entry point\n",
    "requirements.txt": (
        "opencv-python==4.9.0.80\n"
        "ultralytics\n"
        "numpy\n"
        "playsound==1.3.0\n"
        "requests\n"
        "Pillow\n"
    ),
}

for d in directories:
    os.makedirs(d, exist_ok=True)
    print(f"Created: {d}/")

for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)
    print(f"Created: {path}")

print("\nProject scaffold complete!")