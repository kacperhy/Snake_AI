"""
Szybki test wpływu learning rate na wyniki agenta DQN.
Używa mniejszej liczby learning rates i krótszych sesji treningowych
dla szybkiego wstępnego badania.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from snake_game import UproszczonySnake
from agent import DQNAgent
from training import trenuj_hybrydowo
from config import (
    UŻYJ_GPU, SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU, ROZMIAR_UKRYTY
)

def szybki_test_lr(learning_rates=None, liczba_epizodów=50, katalog_wyjściowy=None):
    """
    Przeprowadza szybki test różnych wartości learning rate.
    
    Args:
        learning_rates (list): Lista wartości learning rate do przetestowania,
                              domyślnie [0.0001, 0.001, 0.01]
        liczba_epizodów (int): Liczba epizodów dla każdego testu
        katalog_wyjściowy (str): Katalog do zapisania wyników
    """
    if learning_rates is None:
        learning_rates = [0.0003, 0.003, 0.03]
    
    # Tworzenie katalogu wyjściowego, jeśli nie podano
    if katalog_wyjściowy is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        katalog_wyjściowy = f"wyniki/szybki_test_lr_{timestamp}"
    
    os.makedirs(katalog_wyjściowy, exist_ok=True)
    
    # Parametry gry
    parametry_gry = {
        'szerokość': SZEROKOŚĆ_OKNA,
        'wysokość': WYSOKOŚĆ_OKNA,
        'rozmiar_bloku': ROZMIAR_BLOKU
    }
    
    # Parametry agenta DQN
    rozmiar_stanu = 11  # Liczba cech w reprezentacji stanu
    rozmiar_akcji = 3   # Liczba możliwych akcji (prosto, w prawo, w lewo)
    liczba_równoległych = 4  # Mniejsza liczba równoległych gier dla szybszego testu
    
    # Przygotowanie list na wyniki
    wszystkie_wyniki = []
    wszystkie_historie_ep = []
    
    # Dla każdego learning rate
    for lr in tqdm(learning_rates, desc="Testowanie learning rates"):
        print(f"\nTrenuję agenta z learning rate = {lr}")
        
        # Tworzenie agenta z bieżącym learning rate
        agent = DQNAgent(rozmiar_stanu, rozmiar_akcji, ROZMIAR_UKRYTY, współczynnik_uczenia=lr)
        
        # Trening agenta
        wyniki, historia_ep = trenuj_hybrydowo(
            agent, 
            parametry_gry, 
            liczba_epizodów=liczba_epizodów, 
            aktualizacja_docelowa=10, 
            interwał_zapisu=liczba_epizodów,  # Bez zapisywania pośrednich modeli
            liczba_równoległych=liczba_równoległych
        )
        
        # Zapisywanie wyników
        wszystkie_wyniki.append(wyniki)
        wszystkie_historie_ep.append(historia_ep)
        
        # Obliczanie metryk
        max_wynik = max(wyniki)
        śr_wynik = np.mean(wyniki)
        śr_ostatnie_10 = np.mean(wyniki[-10:]) if len(wyniki) >= 10 else np.mean(wyniki)
        
        print(f"LR = {lr}: Max wynik = {max_wynik}, Średni wynik = {śr_wynik:.2f}, " +
             f"Średnia z ostatnich 10 = {śr_ostatnie_10:.2f}")
    
    # Tworzenie wykresu
    plt.figure(figsize=(14, 8))
    
    # Dla każdego learning rate i jego wyników
    for i, (lr, wyniki) in enumerate(zip(learning_rates, wszystkie_wyniki)):
        # Obliczenie średniej ruchomej
        rozmiar_okna = min(10, len(wyniki))
        ruchoma_średnia = [np.mean(wyniki[max(0, i-rozmiar_okna):i+1]) for i in range(len(wyniki))]
        
        plt.plot(ruchoma_średnia, label=f'LR = {lr}', linewidth=2)
    
    plt.xlabel('Epizod', fontsize=12)
    plt.ylabel('Średnia ruchoma wyników (okno 10 epizodów)', fontsize=12)
    plt.title('Porównanie wartości Learning Rate', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Zapisanie wykresu
    ścieżka_wykresu = os.path.join(katalog_wyjściowy, 'porównanie_lr.png')
    plt.savefig(ścieżka_wykresu, dpi=300)
    plt.close()
    
    print(f"\nTest zakończony. Wykres porównania learning rates zapisany w {ścieżka_wykresu}")
    
    # Tworzenie wykresu maksymalnych i średnich wyników
    plt.figure(figsize=(10, 6))
    
    # Obliczanie maksymalnych i średnich wyników dla każdego learning rate
    maksymalne_wyniki = [max(wyniki) for wyniki in wszystkie_wyniki]
    średnie_wyniki = [np.mean(wyniki) for wyniki in wszystkie_wyniki]
    średnie_końcowe = [np.mean(wyniki[-10:]) if len(wyniki) >= 10 else np.mean(wyniki) 
                      for wyniki in wszystkie_wyniki]
    
    # Wykres słupkowy
    x = np.arange(len(learning_rates))
    szerokość = 0.25
    
    plt.bar(x - szerokość, maksymalne_wyniki, szerokość, label='Maksymalny wynik')
    plt.bar(x, średnie_wyniki, szerokość, label='Średni wynik')
    plt.bar(x + szerokość, średnie_końcowe, szerokość, label='Średnia z ostatnich 10')
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Wynik')
    plt.title('Wpływ Learning Rate na wyniki agenta')
    plt.xticks(x, [str(lr) for lr in learning_rates])
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Zapisanie wykresu
    ścieżka_wykresu = os.path.join(katalog_wyjściowy, 'lr_wyniki_słupkowy.png')
    plt.savefig(ścieżka_wykresu, dpi=300)
    plt.close()
    
    print(f"Wykres słupkowy wyników zapisany w {ścieżka_wykresu}")
    
    # Podsumowanie wyników
    najlepszy_lr_index = np.argmax(maksymalne_wyniki)
    najlepszy_lr = learning_rates[najlepszy_lr_index]
    
    print("\n=== PODSUMOWANIE ===")
    print(f"Najlepszy learning rate (max wynik): {najlepszy_lr}")
    print(f"  Maksymalny wynik: {maksymalne_wyniki[najlepszy_lr_index]}")
    print(f"  Średni wynik: {średnie_wyniki[najlepszy_lr_index]:.2f}")
    print(f"  Średnia z ostatnich 10: {średnie_końcowe[najlepszy_lr_index]:.2f}")
    
    return learning_rates, wszystkie_wyniki, wszystkie_historie_ep

if __name__ == "__main__":
    print("Szybki test wpływu learning rate na wyniki agenta DQN")
    print("Ten skrypt przeprowadza krótkie testy dla kilku wartości learning rate,")
    print("co pozwala na szybką wstępną ocenę ich wpływu.")
    
    # Domyślne wartości learning rate
    domyślne_lr = [0.0001, 0.001, 0.01]
    
    # Możliwość podania własnych wartości learning rate
    własne_lr = input(f"Podaj wartości learning rate oddzielone przecinkami (domyślnie: {domyślne_lr}): ")
    if własne_lr.strip():
        try:
            learning_rates = [float(lr.strip()) for lr in własne_lr.split(',')]
        except ValueError:
            print("Nieprawidłowy format. Używam domyślnych wartości.")
            learning_rates = domyślne_lr
    else:
        learning_rates = domyślne_lr
    
    # Pytanie o liczbę epizodów
    liczba_epizodów = input("Podaj liczbę epizodów dla każdego testu (domyślnie: 50): ")
    if liczba_epizodów.strip() and liczba_epizodów.isdigit():
        liczba_epizodów = int(liczba_epizodów)
    else:
        liczba_epizodów = 50
    
    print(f"\nRozpoczynam testy dla learning rates: {learning_rates}")
    print(f"Liczba epizodów treningu: {liczba_epizodów}")
    
    szybki_test_lr(learning_rates, liczba_epizodów)
