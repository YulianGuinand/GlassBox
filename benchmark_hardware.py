import torch
import time
import sys

def run_benchmark():
    print("--- Diagnostic Materiel GlassBox ---")
    
    # 1. Detection du Backend
    device = torch.device("cpu")
    backend_name = "CPU"
    
    if torch.xpu.is_available():
        device = torch.device("xpu")
        backend_name = f"Intel XPU ({torch.xpu.get_device_name(0)})"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        backend_name = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    
    print(f"Cible detectee : {backend_name}")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"PyTorch version : {torch.__version__}")

    # 2. Test de calcul intensif (Multiplication matricielle)
    size = 4000
    print(f"Lancement du benchmark (Matrice {size}x{size})...")
    
    try:
        # Initialisation des tenseurs sur le peripherique
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Prechauffage (Warm-up)
        _ = torch.matmul(a, b)
        if device.type == 'xpu': torch.xpu.synchronize()
        if device.type == 'cuda': torch.cuda.synchronize()

        # Mesure du temps
        start_time = time.perf_counter()
        
        # Operation repetee pour stabiliser la mesure
        for _ in range(10):
            c = torch.matmul(a, b)
            
        # Synchronisation forcee pour obtenir le temps reel de calcul GPU
        if device.type == 'xpu': torch.xpu.synchronize()
        if device.type == 'cuda': torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        duration = (end_time - start_time) / 10
        print(f"Calcul reussi en : {duration:.4f} secondes par iteration.")
        print(f"Statut : L'installation est OPTIMISEE pour {device.type.upper()}.")

    except Exception as e:
        print(f"Erreur lors du calcul : {e}")

if __name__ == "__main__":
    run_benchmark()