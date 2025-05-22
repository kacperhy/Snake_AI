"""
Eksperyment badający wpływ learning rate na wyniki agenta DQN.
"""

import os
import numpy as np
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
from scipy import stats

from snake_game import UproszczonySnake
from agent import DQNAgent
from training import trenuj_hybrydowo
from config import (
    UŻYJ_GPU, SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU, ROZMIAR_UKRYTY
)

def ustaw_ziarno(ziarno):
    """Ustawia ziarno losowości dla powtarzalności eksperymentów."""
    random.seed(ziarno)
    np.random.seed(ziarno)
    torch.manual_seed(ziarno)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(ziarno)
        torch.cuda.manual_seed_all(ziarno)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def wczytaj_wyniki(katalog_wyjściowy):
    """Wczytuje istniejące wyniki eksperymentu, jeśli są dostępne."""
    plik_wyników = os.path.join(katalog_wyjściowy, 'wyniki_learning_rate.csv')
    if os.path.exists(plik_wyników):
        return pd.read_csv(plik_wyników)
    return pd.DataFrame(columns=['learning_rate', 'ziarno', 'max_wynik', 'śr_wynik', 'śr_ostatnie_100'])

def przeprowadź_eksperyment(learning_rates, katalog_wyjściowy, liczba_epizodów=300, 
                           ziarna=None, liczba_ziaren=5, liczba_równoległych=4):
    """
    Przeprowadza eksperyment badający wpływ learning rate na wyniki agenta.
    
    Args:
        learning_rates (list): Lista wartości learning rate do przetestowania
        katalog_wyjściowy (str): Katalog, w którym zostaną zapisane wyniki
        liczba_epizodów (int): Liczba epizodów treningu dla każdego agenta
        ziarna (list): Lista ziaren losowości do użycia
        liczba_ziaren (int): Liczba ziaren losowości, jeśli ziarna nie są podane
        liczba_równoległych (int): Liczba równoległych gier dla treningu hybrydowego
    
    Returns:
        pd.DataFrame: Ramka danych z wynikami
    """
    # Tworzenie katalogu wyjściowego, jeśli nie istnieje
    os.makedirs(katalog_wyjściowy, exist_ok=True)
    
    # Wczytywanie istniejących wyników, jeśli są dostępne
    ramka_wyników = wczytaj_wyniki(katalog_wyjściowy)
    
    if ziarna is None:
        # Generowanie losowych ziaren, jeśli nie są podane
        ziarna = [random.randint(1, 10000) for _ in range(liczba_ziaren)]
    
    # Parametry gry
    parametry_gry = {
        'szerokość': SZEROKOŚĆ_OKNA,
        'wysokość': WYSOKOŚĆ_OKNA,
        'rozmiar_bloku': ROZMIAR_BLOKU
    }
    
    # Parametry agenta DQN
    rozmiar_stanu = 11  # Liczba cech w reprezentacji stanu
    rozmiar_akcji = 3   # Liczba możliwych akcji (prosto, w prawo, w lewo)
    
    # Dla każdego learning rate
    for lr in tqdm(learning_rates, desc="Learning rates"):
        # Dla każdego ziarna
        for ziarno in tqdm(ziarna, desc=f"Ziarna (lr={lr})", leave=False):
            # Sprawdzenie, czy ta konfiguracja została już przetestowana
            if len(ramka_wyników[(ramka_wyników['learning_rate'] == lr) & 
                               (ramka_wyników['ziarno'] == ziarno)]) > 0:
                tqdm.write(f"Pomijam lr={lr}, ziarno={ziarno} - już przetestowane")
                continue
                
            tqdm.write(f"Trenuję agenta z lr={lr}, ziarno={ziarno}")
            
            # Ustawienie ziarna dla powtarzalności
            ustaw_ziarno(ziarno)
            
            # Utworzenie agenta z bieżącym learning rate
            agent = DQNAgent(rozmiar_stanu, rozmiar_akcji, ROZMIAR_UKRYTY, współczynnik_uczenia=lr)
            
            # Trening agenta
            wyniki, _ = trenuj_hybrydowo(
                agent, 
                parametry_gry, 
                liczba_epizodów=liczba_epizodów, 
                aktualizacja_docelowa=10, 
                interwał_zapisu=liczba_epizodów, # Zapisujemy tylko na końcu
                liczba_równoległych=liczba_równoległych
            )
            
            # Obliczanie metryk
            max_wynik = max(wyniki)
            śr_wynik = np.mean(wyniki)
            śr_ostatnie_100 = np.mean(wyniki[-100:]) if len(wyniki) >= 100 else np.mean(wyniki)
            
            # Zapisywanie wyników
            nowy_wiersz = pd.DataFrame([{
                'learning_rate': lr,
                'ziarno': ziarno,
                'max_wynik': max_wynik,
                'śr_wynik': śr_wynik,
                'śr_ostatnie_100': śr_ostatnie_100
            }])
            
            ramka_wyników = pd.concat([ramka_wyników, nowy_wiersz], ignore_index=True)
            
            # Zapisywanie wyników po każdym agencie
            ramka_wyników.to_csv(os.path.join(katalog_wyjściowy, 'wyniki_learning_rate.csv'), index=False)
            
            # Wyświetlanie pośrednich wyników
            tqdm.write(f"  Max wynik: {max_wynik}, Śr. wynik: {śr_wynik:.2f}, " +
                      f"Śr. ostatnie 100: {śr_ostatnie_100:.2f}")
            
            # Aktualizacja wykresów po każdym agencie
            rysuj_wykresy(ramka_wyników, katalog_wyjściowy=katalog_wyjściowy)
    
    return ramka_wyników

def rysuj_wykresy(ramka_wyników, katalog_wyjściowy='wyniki'):
    """
    Tworzy wykresy pokazujące zależność między learning rate a wynikami.
    
    Args:
        ramka_wyników (pd.DataFrame): Ramka danych z wynikami eksperymentu
        katalog_wyjściowy (str): Katalog, w którym zostaną zapisane wykresy
    """
    # Tworzenie katalogu wyjściowego, jeśli nie istnieje
    os.makedirs(katalog_wyjściowy, exist_ok=True)
    
    # Grupowanie wyników według learning rate
    zgrupowane = ramka_wyników.groupby('learning_rate')
    
    # Obliczanie statystyk
    statystyki = pd.DataFrame()
    metryki = ['max_wynik', 'śr_wynik', 'śr_ostatnie_100']
    
    for metryka in metryki:
        for stat in ['mean', 'std', 'min', 'max']:
            statystyki[f'{metryka}_{stat}'] = zgrupowane[metryka].agg(stat)
    
    # Wykres dla maksymalnego wyniku
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        statystyki.index, 
        statystyki['max_wynik_mean'], 
        yerr=statystyki['max_wynik_std'], 
        fmt='o-', 
        ecolor='lightgray', 
        capsize=5
    )
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Maksymalny wynik')
    plt.title('Wpływ Learning Rate na maksymalny wynik')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Dodanie pojedynczych punktów
    for lr in ramka_wyników['learning_rate'].unique():
        wyniki = ramka_wyników[ramka_wyników['learning_rate'] == lr]['max_wynik']
        plt.scatter([lr] * len(wyniki), wyniki, alpha=0.3, color='orange')
    
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, 'lr_max_wynik.png'))
    plt.close()
    
    # Wykres dla średniego wyniku
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        statystyki.index, 
        statystyki['śr_wynik_mean'], 
        yerr=statystyki['śr_wynik_std'], 
        fmt='o-', 
        ecolor='lightgray', 
        capsize=5
    )
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Średni wynik')
    plt.title('Wpływ Learning Rate na średni wynik')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Dodanie pojedynczych punktów
    for lr in ramka_wyników['learning_rate'].unique():
        wyniki = ramka_wyników[ramka_wyników['learning_rate'] == lr]['śr_wynik']
        plt.scatter([lr] * len(wyniki), wyniki, alpha=0.3, color='orange')
    
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, 'lr_śr_wynik.png'))
    plt.close()
    
    # Wykres połączony
    plt.figure(figsize=(12, 8))
    
    plt.errorbar(
        statystyki.index, 
        statystyki['max_wynik_mean'], 
        yerr=statystyki['max_wynik_std'], 
        fmt='o-', 
        label='Maksymalny wynik', 
        capsize=5
    )
    
    plt.errorbar(
        statystyki.index, 
        statystyki['śr_wynik_mean'], 
        yerr=statystyki['śr_wynik_std'], 
        fmt='s-', 
        label='Średni wynik', 
        capsize=5
    )
    
    plt.errorbar(
        statystyki.index, 
        statystyki['śr_ostatnie_100_mean'], 
        yerr=statystyki['śr_ostatnie_100_std'], 
        fmt='^-', 
        label='Średnia z ostatnich 100', 
        capsize=5
    )
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Wynik')
    plt.title('Wpływ Learning Rate na wyniki agenta')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, 'lr_wyniki_połączone.png'))
    plt.close()
    
    # Zapisanie statystyk do CSV
    statystyki.to_csv(os.path.join(katalog_wyjściowy, 'lr_statystyki.csv'))

def analizuj_wyniki(ramka_wyników, katalog_wyjściowy):
    """
    Przeprowadza szczegółową analizę wyników eksperymentu.
    
    Args:
        ramka_wyników (pd.DataFrame): Ramka danych z wynikami eksperymentu
        katalog_wyjściowy (str): Katalog, w którym zostaną zapisane wyniki analizy
    """
    # Grupowanie według learning rate
    zgrupowane = ramka_wyników.groupby('learning_rate')
    
    # Obliczanie statystyk dla każdej metryki
    metryki = ['max_wynik', 'śr_wynik', 'śr_ostatnie_100']
    statystyki_df = pd.DataFrame()
    
    for metryka in metryki:
        for stat in ['mean', 'std', 'min', 'max']:
            statystyki_df[f'{metryka}_{stat}'] = zgrupowane[metryka].agg(stat)
    
    # Wyznaczanie najlepszego learning rate dla każdej metryki
    najlepsze_lr = {}
    for metryka in metryki:
        najlepsze_lr[metryka] = statystyki_df[f'{metryka}_mean'].idxmax()
    
    print("Najlepsze wartości learning rate:")
    for metryka, lr in najlepsze_lr.items():
        print(f"  {metryka}: {lr}")
        print(f"    Średnia: {statystyki_df.loc[lr, f'{metryka}_mean']:.2f}")
        print(f"    Odchylenie std: {statystyki_df.loc[lr, f'{metryka}_std']:.2f}")
    
    # Tworzenie ulepszonego wykresu
    plt.figure(figsize=(14, 8))
    
    # Paleta kolorów
    kolory = sns.color_palette("husl", 3)
    
    # Wykres dla każdej metryki
    for i, metryka in enumerate(metryki):
        # Wykres średniej z przedziałami błędu
        plt.errorbar(
            statystyki_df.index, 
            statystyki_df[f'{metryka}_mean'], 
            yerr=statystyki_df[f'{metryka}_std'], 
            fmt='o-', 
            label=f'{metryka.replace("_", " ").title()}', 
            capsize=5,
            color=kolory[i],
            linewidth=2,
            markersize=8
        )
        
        # Dodanie pojedynczych punktów
        for lr in ramka_wyników['learning_rate'].unique():
            wartości = ramka_wyników[ramka_wyników['learning_rate'] == lr][metryka]
            plt.scatter([lr] * len(wartości), wartości, alpha=0.3, color=kolory[i])
        
        # Wyróżnienie najlepszego learning rate
        najlepsza_wartość = statystyki_df.loc[najlepsze_lr[metryka], f'{metryka}_mean']
        plt.scatter([najlepsze_lr[metryka]], [najlepsza_wartość], s=150, marker='*', color=kolory[i], 
                   edgecolor='black', linewidth=1.5)
    
    plt.xscale('log')
    plt.xlabel('Learning Rate', fontsize=14)
    plt.ylabel('Wynik', fontsize=14)
    plt.title('Wpływ Learning Rate na wyniki agenta', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Dodanie adnotacji tekstowych
    for metryka, lr in najlepsze_lr.items():
        najlepsza_wartość = statystyki_df.loc[lr, f'{metryka}_mean']
        plt.annotate(
            f"Najlepszy {metryka.split('_')[0]}: {lr}",
            xy=(lr, najlepsza_wartość),
            xytext=(0, 20),
            textcoords="offset points",
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, 'lr_analiza_ulepszona.png'), dpi=300)
    plt.close()
    
    # Wykresy pudełkowe dla każdego learning rate i metryki
    plt.figure(figsize=(16, 10))
    
    for i, metryka in enumerate(metryki):
        plt.subplot(1, 3, i+1)
        
        # Wykres pudełkowy z seaborn dla lepszego wyglądu
        sns.boxplot(
            x='learning_rate', 
            y=metryka, 
            data=ramka_wyników,
            palette='Blues'
        )
        
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Wynik', fontsize=12)
        plt.title(f'{metryka.replace("_", " ").title()}', fontsize=14)
        plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, 'lr_wykresy_pudełkowe.png'), dpi=300)
    plt.close()
    
    # Testy statystyczne
    # ANOVA dla sprawdzenia, czy różnice są istotne statystycznie
    wyniki_anova = {}
    for metryka in metryki:
        grupy = [ramka_wyników[ramka_wyników['learning_rate'] == lr][metryka] for lr in ramka_wyników['learning_rate'].unique()]
        f_val, p_val = stats.f_oneway(*grupy)
        wyniki_anova[metryka] = {
            'F-wartość': f_val,
            'p-wartość': p_val,
            'istotne': p_val < 0.05
        }
    
    # Wyświetlenie wyników ANOVA
    print("\nWyniki ANOVA (testowanie, czy różne learning rate mają istotnie różny wpływ):")
    for metryka, wynik in wyniki_anova.items():
        tekst_istotności = "ISTOTNE" if wynik['istotne'] else "NIE istotne"
        print(f"  {metryka}: F={wynik['F-wartość']:.2f}, p={wynik['p-wartość']:.4f} ({tekst_istotności})")
    
    # Podsumowanie rekomendacji
    print("\n=== PODSUMOWANIE REKOMENDACJI ===")
    for metryka in metryki:
        print(f"\nNajlepszy Learning Rate dla {metryka.replace('_', ' ').title()}: {najlepsze_lr[metryka]}")
        print(f"  Średni wynik: {statystyki_df.loc[najlepsze_lr[metryka], f'{metryka}_mean']:.2f}")
        print(f"  Odchylenie standardowe: {statystyki_df.loc[najlepsze_lr[metryka], f'{metryka}_std']:.2f}")
        
        if wyniki_anova[metryka]['istotne']:
            print(f"  Wpływ learning rate jest statystycznie istotny (p={wyniki_anova[metryka]['p-wartość']:.4f})")
        else:
            print(f"  Wpływ learning rate NIE jest statystycznie istotny (p={wyniki_anova[metryka]['p-wartość']:.4f})")
    
    # Zapisanie szczegółowych statystyk do CSV
    statystyki_df.to_csv(os.path.join(katalog_wyjściowy, 'lr_szczegółowe_statystyki.csv'))
    
    return statystyki_df, wyniki_anova

def main():
    """Główna funkcja eksperymentu."""
    # Definicja wartości learning rate do przetestowania (skala logarytmiczna)
    learning_rates = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
    
    # Definicja ziaren
    ziarna = [42, 100, 200, 300, 400]
    
    # Parametry eksperymentu
    liczba_epizodów = 300  # Liczba epizodów treningu dla każdego agenta
    liczba_równoległych = 8  # Liczba równoległych gier dla treningu hybrydowego
    
    # Tworzenie timestampu dla katalogu wyjściowego
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    katalog_wyjściowy = f"wyniki/eksperyment_learning_rate_{timestamp}"
    
    # Tworzenie katalogu wyjściowego
    os.makedirs(katalog_wyjściowy, exist_ok=True)
    
    # Zapisanie konfiguracji eksperymentu
    konfiguracja = {
        'learning_rates': learning_rates,
        'ziarna': ziarna,
        'liczba_epizodów': liczba_epizodów,
        'liczba_równoległych': liczba_równoległych,
        'timestamp': timestamp
    }
    
    with open(os.path.join(katalog_wyjściowy, 'konfiguracja_eksperymentu.json'), 'w') as f:
        json.dump(konfiguracja, f, indent=4)
    
    print(f"Rozpoczynam eksperyment z learning rates: {learning_rates}")
    print(f"Używam {len(ziarna)} ziaren losowości: {ziarna}")
    print(f"Trenuję każdego agenta przez {liczba_epizodów} epizodów")
    print(f"Używam {liczba_równoległych} równoległych gier")
    print(f"Wyniki zostaną zapisane w {katalog_wyjściowy}")
    
    # Przeprowadzenie eksperymentu
    ramka_wyników = przeprowadź_eksperyment(
        learning_rates=learning_rates,
        katalog_wyjściowy=katalog_wyjściowy,
        liczba_epizodów=liczba_epizodów,
        ziarna=ziarna,
        liczba_równoległych=liczba_równoległych
    )
    
    # Analiza wyników
    print("\nAnalizuję wyniki...")
    statystyki_df, wyniki_anova = analizuj_wyniki(ramka_wyników, katalog_wyjściowy=katalog_wyjściowy)
    
    print(f"\nEksperyment zakończony. Wyniki zapisane w {katalog_wyjściowy}")

if __name__ == "__main__":
    main()
