import os
import subprocess
import sys


def run_command(command):
    subprocess.run(command, shell=True, check=True)


def install_glassbox():
    print("Détection du matériel en cours...")

    # Par defaut, installer version CPU
    index_url = "https://download.pytorch.org/whl/cpu"
    extra_packages = []

    # Detection NVIDIA
    try:
        nvidia_check = subprocess.run(
            "nvidia-smi", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        if nvidia_check.returncode == 0:
            print("GPU NVIDIA détecté (CUDA)")
            index_url = "https://download.pytorch.org/whl/cu121"

    except FileNotFoundError:
        pass

    # Detection Intel
    try:
        intel_check = subprocess.check_output(
            "wmic path win32_VideoController get name", shell=True
        ).decode()

        if "Intel" in intel_check and "Arc" in intel_check or "Iris" in intel_check:
            print("GPU Intel detecte (XPU)")
            index_url = "https://download.pytorch.org/whl/xpu"
            extra_packages = ["intel-extension-for-pytorch"]

    except:
        pass

    print(f"Installation de PyTorch depuis : {index_url}")


install_glassbox()
