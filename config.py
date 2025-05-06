"""
Moduł zawierający konfigurację globalną dla projektu.
"""

import torch
import random
import numpy as np

# Ustawienia sprzętowe
USE_GPU = torch.cuda.is_available()  # Automatyczne wykrywanie GPU
CPU_THREAD_COUNT = 4  # Liczba wątków CPU dla obliczeń
USE_FLOAT16 = False  # Czy używać niższej precyzji na GPU (szybsze, ale mniej dokładne)

# Parametry gry
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
BLOCK_SIZE = 40
GAME_SPEED = 20

# Parametry bufora doświadczeń
EXPERIENCE_BUFFER_SIZE = 50000  # Rozmiar bufora pamięci doświadczeń

# Parametry sieci neuronowej
HIDDEN_SIZE = 256  # Rozmiar warstwy ukrytej
LEARNING_RATE = 0.0003  # Współczynnik uczenia

# Parametry treningu
BATCH_SIZE = 128  # Rozmiar batcha
GAMMA = 0.98  # Współczynnik dyskontowania przyszłych nagród
EPSILON_START = 1.0  # Początkowa wartość współczynnika eksploracji
EPSILON_MIN = 0.01  # Minimalna wartość współczynnika eksploracji
EPSILON_DECAY = 0.995  # Współczynnik zmniejszania epsilon

# Ustawienie ziarna dla generatorów liczb losowych (spójność między uruchomieniami)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Ścieżki do katalogów
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
