"""
Moduł zawierający implementację modelu sieciowego dla agenta DQN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import USE_GPU, USE_FLOAT16


class QNetwork(nn.Module):
    """
    Implementacja sieci neuronowej dla Deep Q-Network.
    
    Atrybuty:
        rozmiar_wejścia (int): Rozmiar wektora wejściowego.
        rozmiar_ukryty (int): Liczba neuronów w warstwie ukrytej.
        rozmiar_wyjścia (int): Liczba możliwych akcji.
    """
    def __init__(self, rozmiar_wejścia, rozmiar_ukryty, rozmiar_wyjścia):
        super(QNetwork, self).__init__()
        # Głębsza sieć z większą liczbą warstw
        self.fc1 = nn.Linear(rozmiar_wejścia, rozmiar_ukryty)
        self.dropout1 = nn.Dropout(0.2)  # Dropout dla regularyzacji (zapobiega przeuczeniu)
        self.fc2 = nn.Linear(rozmiar_ukryty, rozmiar_ukryty)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(rozmiar_ukryty, rozmiar_ukryty // 2)
        self.fc4 = nn.Linear(rozmiar_ukryty // 2, rozmiar_wyjścia)
        
        # Inicjalizacja wag dla lepszej zbieżności
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        """
        Przepuszczenie danych przez sieć neuronową.
        
        Args:
            x (Tensor): Tensor wejściowy reprezentujący stan gry.
            
        Returns:
            Tensor: Tensor wyjściowy reprezentujący wartości Q dla każdej akcji.
        """
        # Dodanie wymiaru wsadu dla pojedynczego przykładu jeśli potrzeba
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Przepływ do przodu przez sieć z aktywacjami Leaky ReLU
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
