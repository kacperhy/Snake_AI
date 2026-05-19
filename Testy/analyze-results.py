"""
Narzędzie do analizy istniejących wyników eksperymentów z learning rate.
Pozwala na ponowną analizę i wizualizację danych bez potrzeby ponownego uruchamiania eksperymentów.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import argparse
from datetime import datetime
import json

def znajdź_katalogi_wyników(katalog_główny="wyniki"):
    """Znajduje wszystkie katalogi z wynikami eksperymentów learning rate."""
    katalogi = []
    
    if not os.path.exists(katalog_główny):
        return katalogi
    
    for katalog in os.listdir(katalog_główny):
        ścieżka = os.path.join(katalog_główny, katalog)
        if os.path.isdir(ścieżka) and "eksperyment_learning_rate" in katalog:
            # Sprawdzamy, czy zawiera plik z wynikami
            if os.path.exists(os.path.join(ścieżka, "wyniki_learning_rate.csv")):
                katalogi.append(ścieżka)
    
    return katalogi

def analizuj_wyniki(katalog_wyników, katalog_wyjściowy=None, prefix=""):
    """
    Analizuje wyniki eksperymentu z danego katalogu.
    
    Args:
        katalog_wyników (str): Katalog zawierający wyniki eksperymentu
        katalog_wyjściowy (str): Katalog do zapisania wyników analizy (domyślnie ten sam co wejściowy)
        prefix (str): Prefiks dla nazw plików wyjściowych
    
    Returns:
        tuple: Para (statystyki_df, wyniki_anova)
    """
    # Sprawdzenie, czy katalog istnieje
    if not os.path.exists(katalog_wyników):
        print(f"Błąd: Katalog {katalog_wyników} nie istnieje!")
        return None, None
    
    # Sprawdzenie, czy plik z wynikami istnieje
    plik_wyników = os.path.join(katalog_wyników, "wyniki_learning_rate.csv")
    if not os.path.exists(plik_wyników):
        print(f"Błąd: Plik {plik_wyników} nie istnieje!")
        return None, None
    
    # Ustawienie katalogu wyjściowego
    if katalog_wyjściowy is None:
        katalog_wyjściowy = katalog_wyników
    
    # Wczytanie wyników
    ramka_wyników = pd.read_csv(plik_wyników)
    
    # Sprawdzenie, czy dane są poprawne
    if 'learning_rate' not in ramka_wyników.columns:
        print(f"Błąd: Plik {plik_wyników} nie zawiera kolumny 'learning_rate'!")
        return None, None
    
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
    
    print(f"\nAnaliza wyników z katalogu: {katalog_wyników}")
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
    plt.savefig(os.path.join(katalog_wyjściowy, f'{prefix}lr_analiza_ulepszona.png'), dpi=300)
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
    plt.savefig(os.path.join(katalog_wyjściowy, f'{prefix}lr_wykresy_pudełkowe.png'), dpi=300)
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
    statystyki_df.to_csv(os.path.join(katalog_wyjściowy, f'{prefix}lr_szczegółowe_statystyki.csv'))
    
    # Wykres porównujący rozkłady wyników
    plt.figure(figsize=(16, 6))
    
    for metryka in metryki:
        plt.subplot(1, 3, metryki.index(metryka) + 1)
        sns.violinplot(
            x='learning_rate', 
            y=metryka, 
            data=ramka_wyników,
            palette='viridis'
        )
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Wynik', fontsize=12)
        plt.title(f'{metryka.replace("_", " ").title()} - Rozkłady', fontsize=14)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, f'{prefix}lr_rozkłady.png'), dpi=300)
    plt.close()
    
    # Rysowanie krzywych narastających dla najlepszego LR dla każdej metryki
    if 'ziarno' in ramka_wyników.columns:
        # Tworzenie trójwymiarowego wykresu z najlepszymi wartościami LR
        fig = plt.figure(figsize=(15, 10))
        
        for i, metryka in enumerate(metryki):
            lr = najlepsze_lr[metryka]
            plt.subplot(1, 3, i+1)
            
            dane_lr = ramka_wyników[ramka_wyników['learning_rate'] == lr]
            ziarna = dane_lr['ziarno'].unique()
            
            for z in ziarna:
                wynik = dane_lr[dane_lr['ziarno'] == z][metryka].values[0]
                plt.bar(z, wynik, alpha=0.7)
            
            plt.xlabel('Ziarno', fontsize=12)
            plt.ylabel(metryka.replace('_', ' ').title(), fontsize=12)
            plt.title(f'Najlepszy LR dla {metryka.replace("_", " ").title()}: {lr}', fontsize=14)
            plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(katalog_wyjściowy, f'{prefix}lr_najlepsze_ziarna.png'), dpi=300)
        plt.close()
    
    # Dodatkowa analiza - regresja wielomianowa dla przewidywania optymalnego LR
    plt.figure(figsize=(12, 8))
    
    for i, metryka in enumerate(metryki):
        lr_values = statystyki_df.index.to_numpy()
        wyniki = statystyki_df[f'{metryka}_mean'].to_numpy()
        
        # Konwersja do skali logarytmicznej
        log_lr = np.log10(lr_values)
        
        # Dopasowanie wielomianu stopnia 2
        coeffs = np.polyfit(log_lr, wyniki, 2)
        p = np.poly1d(coeffs)
        
        # Tworzenie gęstszej siatki punktów dla wykresu
        log_x_fine = np.linspace(min(log_lr), max(log_lr), 100)
        
        # Wykres punktów oryginalnych
        plt.scatter(lr_values, wyniki, label=f'{metryka.replace("_", " ").title()} - dane', 
                   color=kolory[i], s=80, alpha=0.7)
        
        # Wykres krzywej dopasowania
        plt.plot(10**log_x_fine, p(log_x_fine), linestyle='--', linewidth=2, 
                color=kolory[i], alpha=0.8)
        
        # Znalezienie optymalnej wartości LR z modelu
        optimal_log_lr = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else 0
        if min(log_lr) <= optimal_log_lr <= max(log_lr):
            optimal_lr = 10**optimal_log_lr
            optimal_value = p(optimal_log_lr)
            
            plt.scatter([optimal_lr], [optimal_value], marker='*', s=200, 
                       color=kolory[i], edgecolor='black', linewidth=1.5)
            
            plt.annotate(
                f"Predicted optimal: {optimal_lr:.6f}",
                xy=(optimal_lr, optimal_value),
                xytext=(0, 20),
                textcoords="offset points",
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    
    plt.xscale('log')
    plt.xlabel('Learning Rate', fontsize=14)
    plt.ylabel('Wynik', fontsize=14)
    plt.title('Modelowanie wpływu Learning Rate i przewidywanie optymalnych wartości', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, f'{prefix}lr_model_optymalny.png'), dpi=300)
    plt.close()
    
    return statystyki_df, wyniki_anova

def porównaj_eksperymenty(katalogi, katalog_wyjściowy=None):
    """
    Porównuje wyniki z wielu eksperymentów.
    
    Args:
        katalogi (list): Lista katalogów z wynikami eksperymentów
        katalog_wyjściowy (str): Katalog do zapisania wyników porównania (domyślnie pierwszy z listy)
    
    Returns:
        dict: Słownik z wynikami porównania
    """
    if not katalogi:
        print("Brak katalogów do porównania!")
        return None
    
    # Ustawienie katalogu wyjściowego
    if katalog_wyjściowy is None:
        katalog_wyjściowy = katalogi[0]
    
    # Sprawdzenie, czy katalog wyjściowy istnieje
    os.makedirs(katalog_wyjściowy, exist_ok=True)
    
    # Wczytanie wyników z każdego katalogu
    wszystkie_wyniki = []
    nazwy_katalogów = []
    
    for katalog in katalogi:
        plik_wyników = os.path.join(katalog, "wyniki_learning_rate.csv")
        if os.path.exists(plik_wyników):
            wyniki = pd.read_csv(plik_wyników)
            wszystkie_wyniki.append(wyniki)
            nazwy_katalogów.append(os.path.basename(katalog))
        else:
            print(f"Pominięto katalog {katalog} - brak pliku wyników.")
    
    if not wszystkie_wyniki:
        print("Brak danych do porównania!")
        return None
    
    # Porównanie najlepszych wartości learning rate
    najlepsze_lr = {}
    najlepsze_wyniki = {}
    metryki = ['max_wynik', 'śr_wynik', 'śr_ostatnie_100']
    
    for i, (wyniki, nazwa) in enumerate(zip(wszystkie_wyniki, nazwy_katalogów)):
        grupa = wyniki.groupby('learning_rate')
        statystyki = pd.DataFrame()
        
        for metryka in metryki:
            for stat in ['mean', 'std']:
                statystyki[f'{metryka}_{stat}'] = grupa[metryka].agg(stat)
        
        for metryka in metryki:
            if metryka not in najlepsze_lr:
                najlepsze_lr[metryka] = []
                najlepsze_wyniki[metryka] = []
            
            lr = statystyki[f'{metryka}_mean'].idxmax()
            wynik = statystyki.loc[lr, f'{metryka}_mean']
            
            najlepsze_lr[metryka].append(lr)
            najlepsze_wyniki[metryka].append(wynik)
    
    # Tworzenie wykresu porównawczego
    plt.figure(figsize=(15, 10))
    
    for i, metryka in enumerate(metryki):
        plt.subplot(1, 3, i+1)
        
        x = np.arange(len(nazwy_katalogów))
        plt.bar(x, najlepsze_wyniki[metryka], alpha=0.7)
        
        for j, (lr, wynik) in enumerate(zip(najlepsze_lr[metryka], najlepsze_wyniki[metryka])):
            plt.text(j, wynik + 0.5, f'LR: {lr}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(x, [f'Exp {i+1}' for i in range(len(nazwy_katalogów))], rotation=45)
        plt.xlabel('Eksperyment', fontsize=12)
        plt.ylabel(metryka.replace('_', ' ').title(), fontsize=12)
        plt.title(f'Najlepsze wyniki dla {metryka.replace("_", " ").title()}', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, 'porównanie_eksperymentów.png'), dpi=300)
    plt.close()
    
    # Zebranie wyników
    wyniki_porównania = {
        'katalogi': nazwy_katalogów,
        'najlepsze_lr': najlepsze_lr,
        'najlepsze_wyniki': najlepsze_wyniki
    }
    
    # Tworzenie tabeli porównawczej
    tabela = pd.DataFrame()
    
    for i, nazwa in enumerate(nazwy_katalogów):
        for metryka in metryki:
            kolumna = f'{nazwa}_{metryka}'
            tabela.loc['Najlepszy LR', kolumna] = najlepsze_lr[metryka][i]
            tabela.loc['Najlepszy wynik', kolumna] = najlepsze_wyniki[metryka][i]
    
    tabela.to_csv(os.path.join(katalog_wyjściowy, 'porównanie_tabela.csv'))
    
    # Wyświetlenie podsumowania
    print("\n=== PORÓWNANIE EKSPERYMENTÓW ===")
    for metryka in metryki:
        print(f"\nWyniki dla {metryka.replace('_', ' ').title()}:")
        for i, (nazwa, lr, wynik) in enumerate(zip(nazwy_katalogów, najlepsze_lr[metryka], najlepsze_wyniki[metryka])):
            print(f"  Eksperyment {i+1} ({nazwa}):")
            print(f"    Najlepszy LR: {lr}")
            print(f"    Wynik: {wynik:.2f}")
    
    return wyniki_porównania

def main():
    """Funkcja główna do analizy wyników eksperymentów."""
    parser = argparse.ArgumentParser(description='Narzędzie do analizy wyników eksperymentów learning rate')
    parser.add_argument('--katalog', type=str, default=None,
                        help='Katalog z wynikami eksperymentu do analizy')
    parser.add_argument('--wszystkie', action='store_true',
                        help='Analizuj wszystkie znalezione katalogi z eksperymentami')
    parser.add_argument('--porównaj', action='store_true',
                        help='Porównaj wyniki z wielu eksperymentów')
    parser.add_argument('--wyjście', type=str, default=None,
                        help='Katalog wyjściowy dla wyników analizy')
    
    args = parser.parse_args()
    
    # Znalezienie katalogów z wynikami
    if args.wszystkie or args.porównaj:
        katalogi = znajdź_katalogi_wyników()
        if not katalogi:
            print("Nie znaleziono katalogów z wynikami eksperymentów!")
            return
        
        print(f"Znaleziono {len(katalogi)} katalogów z wynikami:")
        for i, katalog in enumerate(katalogi):
            print(f"  {i+1}. {katalog}")
    
    # Analiza pojedynczego katalogu
    if args.katalog:
        print(f"Analizuję katalog: {args.katalog}")
        analizuj_wyniki(args.katalog, args.wyjście)
    
    # Analiza wszystkich katalogów
    elif args.wszystkie:
        for katalog in katalogi:
            print(f"\nAnalizuję katalog: {katalog}")
            analizuj_wyniki(katalog, args.wyjście)
    
    # Porównanie eksperymentów
    if args.porównaj:
        print("\nPorównuję wyniki eksperymentów:")
        porównaj_eksperymenty(katalogi, args.wyjście)

if __name__ == "__main__":
    main()
