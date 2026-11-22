# EchoFlow-Q/check_env.py
import os
import sys
import platform
import subprocess
from pathlib import Path

def get_ram_gb():
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                stderr=subprocess.DEVNULL
            ).decode().splitlines()
            for line in out:
                if line.strip().isdigit():
                    return round(int(line.strip()) / (1024 ** 3), 2)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 2)
        elif platform.system() == "Darwin":
            bytes_ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            return round(bytes_ram / (1024 ** 3), 2)
    except Exception:
        pass
    return "Unknown"

def check_nvidia_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return out if out else "No NVIDIA GPU detected"
    except Exception:
        return "No NVIDIA GPU detected (or nvidia-smi not in PATH)"

def check_pytorch():
    try:
        import torch
        return f"Installed (v{torch.__version__}) | CUDA available: {torch.cuda.is_available()}"
    except ImportError:
        return "Not installed in current environment"

def create_project_structure():
    dirs = [
        "data/raw/indicvoices",
        "data/raw/indic_diarbench",
        "data/processed",
        "data/enrollment",
        "data/metadata",
        "src/data",
        "src/preprocessing",
        "src/features",
        "src/models",
        "src/diarization",
        "src/language",
        "src/evaluation",
        "app/backend",
        "app/frontend",
        "notebooks",
        "results/figures",
        "results/metrics",
        "results/saved_models",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        # Add __init__.py to make src packages importable
        if d.startswith("src/"):
            (Path(d) / "__init__.py").touch(exist_ok=True)
    Path("src/__init__.py").touch(exist_ok=True)

    # Create baseline .gitignore and .env template
    gitignore_content = (
        "venv/\n.venv/\n__pycache__/\n*.pyc\n.env\n"
        "data/raw/\ndata/processed/\nresults/saved_models/\n.DS_Store\n"
    )
    if not Path(".gitignore").exists():
        Path(".gitignore").write_text(gitignore_content)

    if not Path(".env").exists():
        Path(".env").write_text(
            "# Keep API keys and Hugging Face tokens here. Never commit this file.\n"
            "HF_TOKEN=\n"
            "GNANI_API_KEY=\n"
            "GNANI_API_ENDPOINT=\n"
        )

if __name__ == "__main__":
    print("=== ECHOWFLOW-Q SYSTEM DIAGNOSTIC ===")
    print(f"OS           : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python       : {sys.version.split()[0]} ({sys.executable})")
    print(f"System RAM   : {get_ram_gb()} GB")
    print(f"NVIDIA GPU   : {check_nvidia_gpu()}")
    print(f"PyTorch      : {check_pytorch()}")
    print("\n=== CREATING PROJECT STRUCTURE ===")
    create_project_structure()
    print("All folders, __init__.py files, .env, and .gitignore created successfully.")