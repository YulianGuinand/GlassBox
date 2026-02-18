import subprocess
import sys
import os

def run_command(command):
    print(f"--- Execution : {command} ---")
    subprocess.run(command, shell=True, check=True)

def get_gpu_name():
    """Detecte le nom de la carte graphique."""
    try:
        if sys.platform == "win32":
            cmd = "powershell -command \"Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name\""
            return subprocess.check_output(cmd, shell=True).decode(errors='ignore').upper()
        else:
            return subprocess.check_output("lspci | grep -i vga", shell=True).decode(errors='ignore').upper()
    except:
        return ""

def install():
    print("Analyse du materiel pour optimisation GlassBox...")
    gpu_name = get_gpu_name()
    
    # Configuration par defaut (CPU)
    index_url = "https://download.pytorch.org/whl/cpu"
    target = "CPU (Par defaut)"

    # Verification NVIDIA (CUDA)
    try:
        nvidia_check = subprocess.run("nvidia-smi", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if nvidia_check.returncode == 0 or "NVIDIA" in gpu_name:
            index_url = "https://download.pytorch.org/whl/cu121"
            target = "NVIDIA (CUDA)"
    except:
        pass

    # Verification INTEL (XPU) - Priorite si pas de NVIDIA
    if "NVIDIA" not in target:
        if "INTEL" in gpu_name and ("ARC" in gpu_name or "IRIS" in gpu_name or "GRAPHICS" in gpu_name):
            index_url = "https://download.pytorch.org/whl/xpu"
            target = "INTEL (XPU)"

    print(f"Materiel cible detecte : {target}")
    print(f"Utilisation de l'index : {index_url}")

    # Installation de PyTorch optimise
    run_command(f"uv pip install torch torchvision torchaudio --index-url {index_url}")

    # Installation des extensions specifiques si Intel
    if "INTEL" in target:
        run_command(f"uv pip install intel-extension-for-pytorch --extra-index-url {index_url}")

    # Installation des dependances communes (Identiques pour tous)
    print("plus Installation des bibliotheques standards...")
    run_command("uv pip install pandas scikit-learn streamlit plotly sympy==1.13.1")

    print(f"\nInstallation terminee ! GlassBox est optimise pour : {target}")

if __name__ == "__main__":
    install()