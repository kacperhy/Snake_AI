"""
Moduł zawierający konfigurację globalną dla projektu.
"""

import torch
import random
import numpy as np

# Ustawienia sprzętowe
UŻYJ_GPU = torch.cuda.is_available()  # Automatyczne wykrywanie GPU
LICZBA_WĄTKÓW_CPU = 4  # Liczba wątków CPU dla obliczeń
UŻYJ_FLOAT16 = False  # Czy używać niższej precyzji na GPU (szybsze, ale mniej dokładne)

# Parametry gry
SZEROKOŚĆ_OKNA = 640
WYSOKOŚĆ_OKNA = 480
ROZMIAR_BLOKU = 40
PRĘDKOŚĆ_GRY = 20

# Parametry bufora doświadczeń
ROZMIAR_BUFORA_DOŚWIADCZEŃ = 50000  # Rozmiar bufora pamięci doświadczeń

# Parametry sieci neuronowej
ROZMIAR_UKRYTY = 256  # Rozmiar warstwy ukrytej
WSPÓŁCZYNNIK_UCZENIA = 0.0003  # Współczynnik uczenia

# Parametry treningu
ROZMIAR_PARTII = 128  # Rozmiar batcha
GAMMA = 0.98  # Współczynnik dyskontowania przyszłych nagród
EPSILON_START = 1.0  # Początkowa wartość współczynnika eksploracji
EPSILON_MIN = 0.01  # Minimalna wartość współczynnika eksploracji
SPADEK_EPSILON = 0.995  # Współczynnik zmniejszania epsilon

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
KATALOG_MODELI = 'models'
KATALOG_WYNIKÓW = 'results'
