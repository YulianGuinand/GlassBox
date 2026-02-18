import torch
from pathlib import Path

# --- PROJECT PATHS ---
SRC_ROOT = Path(__file__).parent
PROJECT_ROOT = SRC_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# --- HARDWARE ---
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# --- DEFAULTS ---
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 10
DEFAULT_TEST_SIZE = 0.2
RANDOM_SEED = 42

# --- UI COLORS (From IA_README) ---
COLOR_DENSE = "#0047AB"  # Bleu Cobalt
COLOR_DROPOUT = "#E34234" # Rouge Vermillon
COLOR_ACTIVATION = "#50C878" # Vert Émeraude
COLOR_NORM = "#FFBF00" # Ambre
