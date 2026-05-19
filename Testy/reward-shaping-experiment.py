"""
Eksperyment badający wpływ różnych strategii nagród (reward shaping) na wyniki agenta DQN w grze Snake.
"""
import os
import numpy as np
import torch
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from datetime import datetime

# Wyciszenie Matplotlib, żeby nie próbował wyświetlać interfejsu
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Importy z naszego projektu
from agent import DQNAgent
from snake_game import UproszczonySnake, Kierunek
from config import (
    UŻYJ_GPU, SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU, ROZMIAR_UKRYTY, KATALOG_MODELI
)

print("Importy zakończone")

class SnakeCustomRewards(UproszczonySnake):
    """
    Zmodyfikowana wersja gry Snake z konfigurowalnymi nagrodami.
    """
    def __init__(self, szerokość=SZEROKOŚĆ_OKNA, wysokość=WYSOKOŚĆ_OKNA, rozmiar_bloku=ROZMIAR_BLOKU,
                 rewards=None):
        super().__init__(szerokość, wysokość, rozmiar_bloku)
        
        # Domyślne wartości nagród
        self.rewards = {
            'food_reward': 10.0,     # Nagroda za zjedzenie jedzenia
            'death_penalty': -10.0,  # Kara za śmierć
            'closer_reward': 0.1,    # Nagroda za zbliżenie się do jedzenia
            'farther_penalty': -0.1, # Kara za oddalenie się od jedzenia
            'time_penalty': 0.0      # Kara za każdy krok (domyślnie brak)
        }
        
        # Aktualizacja nagród, jeśli podano
        if rewards:
            self.rewards.update(rewards)

    def krok(self, akcja):
        """
        Wykonanie jednego kroku gry z niestandardowymi nagrodami.
        """
        self.iteracja_klatki += 1
        self.kroki_bez_jedzenia += 1
        
        # Aktualizacja kierunku na podstawie akcji
        clock_wise = [Kierunek.PRAWO, Kierunek.DÓŁ, Kierunek.LEWO, Kierunek.GÓRA]
        indeks = clock_wise.index(self.Kierunek)
        
        if akcja == 0:  # Prosto
            nowy_kier = clock_wise[indeks]
        elif akcja == 1:  # W prawo
            next_idx = (indeks + 1) % 4
            nowy_kier = clock_wise[next_idx]
        else:  # W lewo
            next_idx = (indeks - 1) % 4
            nowy_kier = clock_wise[next_idx]
            
        self.Kierunek = nowy_kier
        
        # Aktualizacja pozycji głowy
        x = self.snake[0][0]
        y = self.snake[0][1]
        if self.Kierunek == Kierunek.PRAWO:
            x += self.rozmiar_bloku
        elif self.Kierunek == Kierunek.LEWO:
            x -= self.rozmiar_bloku
        elif self.Kierunek == Kierunek.DÓŁ:
            y += self.rozmiar_bloku
        elif self.Kierunek == Kierunek.GÓRA:
            y -= self.rozmiar_bloku
            
        self.głowa = [x, y]
        self.snake.insert(0, self.głowa.copy())
        
        # Sprawdzenie, czy gra się zakończyła
        nagroda = self.rewards['time_penalty']  # Opcjonalna kara za każdy krok
        koniec_gry = False
        
        # Kolizja lub przekroczenie limitu ruchów bez jedzenia
        maks_kroków_bez_jedzenia = 100 * len(self.snake)
        if len(self.snake) > 10:
            maks_kroków_bez_jedzenia = 50 * len(self.snake)
            
        if self._czy_kolizja() or self.kroki_bez_jedzenia > maks_kroków_bez_jedzenia:
            koniec_gry = True
            nagroda += self.rewards['death_penalty']
            return self._pobierz_stan(), nagroda, koniec_gry, self.wynik
            
        # Zjedzenie jedzenia
        if self.głowa == self.jedzenie:
            self.wynik += 1
            nagroda += self.rewards['food_reward']
            self.kroki_bez_jedzenia = 0
            self._umieść_jedzenie()
        else:
            self.snake.pop()  # usunięcie ostatniego segmentu węża, jeśli nie zjadł jedzenia
            
            # Dodatkowe nagrody za zbliżanie się do jedzenia
            poprz_odl_do_jedzenia = abs(self.snake[1][0] - self.jedzenie[0]) + abs(self.snake[1][1] - self.jedzenie[1])
            obecna_odl_do_jedzenia = abs(self.głowa[0] - self.jedzenie[0]) + abs(self.głowa[1] - self.jedzenie[1])
            
            if obecna_odl_do_jedzenia < poprz_odl_do_jedzenia:
                nagroda += self.rewards['closer_reward']  # Nagroda za zbliżanie się do jedzenia
            elif obecna_odl_do_jedzenia > poprz_odl_do_jedzenia:
                nagroda += self.rewards['farther_penalty']  # Kara za oddalanie się od jedzenia
        
        # Zwrócenie nowego stanu, nagrody i informacji, czy gra się zakończyła
        return self._pobierz_stan(), nagroda, koniec_gry, self.wynik


def ustaw_ziarno(seed):
    """Ustawia ziarno losowości dla powtarzalności wyników."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def testuj_schemat_nagród(nazwa_schematu, schemat_nagród, liczba_epizodów=300, liczba_równoległych=16, ziarno=42):
    """
    Testuje pojedynczy schemat nagród i zwraca wyniki.
    
    Args:
        nazwa_schematu (str): Nazwa schematu nagród.
        schemat_nagród (dict): Słownik z wartościami nagród.
        liczba_epizodów (int): Liczba epizodów treningu.
        liczba_równoległych (int): Liczba równoległych gier.
        ziarno (int): Ziarno losowości.
        
    Returns:
        tuple: (historia_wyników, max_wynik, średni_wynik, końcowy_średni)
    """
    print(f"Testuję schemat '{nazwa_schematu}'...")
    
    # Ustawienie ziarna dla powtarzalności
    ustaw_ziarno(ziarno)
    
    # Inicjalizacja agenta
    rozmiar_stanu = 11  # Liczba cech w reprezentacji stanu
    rozmiar_akcji = 3   # Liczba możliwych akcji (prosto, w prawo, w lewo)
    agent = DQNAgent(rozmiar_stanu, rozmiar_akcji, ROZMIAR_UKRYTY)
    
    # Parametry gry
    parametry_gry = {
        'szerokość': SZEROKOŚĆ_OKNA,
        'wysokość': WYSOKOŚĆ_OKNA,
        'rozmiar_bloku': ROZMIAR_BLOKU
    }
    
    # Zbieranie wyników
    wszystkie_wyniki = []
    
    # Określić liczbę grup epizodów
    liczba_grup = (liczba_epizodów + liczba_równoległych - 1) // liczba_równoległych
    
    # Progress bar dla grup epizodów
    for i in tqdm(range(liczba_grup), desc=f"Trening {nazwa_schematu}"):
        # Określić liczbę gier w tej grupie
        liczba_gier = min(liczba_równoległych, liczba_epizodów - i * liczba_równoległych)
        wyniki_grupy = []
        
        # Uruchomienie kilku gier "równolegle"
        for _ in range(liczba_gier):
            # Utworzenie gry z niestandardowymi nagrodami
            gra = SnakeCustomRewards(**parametry_gry, rewards=schemat_nagród)
            stan = gra.reset()
            koniec = False
            wynik = 0
            
            # Rozegranie jednej gry
            while not koniec:
                # Wybór akcji przez agenta
                akcja = agent.pobierz_akcję(stan)
                
                # Wykonanie akcji w środowisku
                następny_stan, nagroda, koniec, wynik = gra.krok(akcja)
                
                # Zapamiętanie doświadczenia
                agent.zapamiętaj(stan, akcja, nagroda, następny_stan, koniec)
                
                # Uczenie agenta
                agent.ucz_się()
                
                # Przejście do następnego stanu
                stan = następny_stan
            
            # Zapisanie wyników tej gry
            wyniki_grupy.append(wynik)
            wszystkie_wyniki.append(wynik)
        
        # Wyświetlenie postępu co 5 grup
        if i % 5 == 0 or i == liczba_grup - 1:
            epizod = min((i + 1) * liczba_równoległych, liczba_epizodów)
            średnia_grupy = np.mean(wyniki_grupy)
            maks_grupy = max(wyniki_grupy) if wyniki_grupy else 0
            tqdm.write(f"Epizod {epizod}/{liczba_epizodów}, Średni wynik: {średnia_grupy:.2f}, Maks: {maks_grupy}")
    
    # Zapisanie modelu
    katalog_modeli = os.path.join("wyniki", "modele")
    os.makedirs(katalog_modeli, exist_ok=True)
    agent.save(os.path.join(katalog_modeli, f"model_{nazwa_schematu}.pth"))
    
    # Obliczenie statystyk
    max_wynik = max(wszystkie_wyniki)
    średni_wynik = np.mean(wszystkie_wyniki)
    końcowy_średni = np.mean(wszystkie_wyniki[-min(100, len(wszystkie_wyniki)):])
    
    print(f"Schemat '{nazwa_schematu}' zakończony. Max: {max_wynik}, Średnia: {średni_wynik:.2f}, Końcowa średnia: {końcowy_średni:.2f}")
    
    return wszystkie_wyniki, max_wynik, średni_wynik, końcowy_średni


def rysuj_wykresy(wyniki, katalog_wyjściowy):
    """
    Rysuje wykresy porównujące różne schematy nagród.
    
    Args:
        wyniki (dict): Słownik z wynikami dla różnych schematów.
        katalog_wyjściowy (str): Katalog do zapisania wykresów.
    """
    # Utworzenie katalogu na wykresy
    os.makedirs(katalog_wyjściowy, exist_ok=True)
    
    # Kolory dla schematów
    paleta = sns.color_palette("husl", len(wyniki))
    
    # 1. Wykres najwyższych wyników
    plt.figure(figsize=(12, 6))
    schematy = list(wyniki.keys())
    max_wyniki = [wyniki[s]['max'] for s in schematy]
    
    # Sortowanie według wartości
    sorted_indices = np.argsort(max_wyniki)[::-1]
    sorted_schematy = [schematy[i] for i in sorted_indices]
    sorted_max_wyniki = [max_wyniki[i] for i in sorted_indices]
    sorted_kolory = [paleta[schematy.index(s)] for s in sorted_schematy]
    
    bars = plt.bar(sorted_schematy, sorted_max_wyniki, color=sorted_kolory)
    
    # Dodanie wartości na szczycie słupków
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}', ha='center', va='bottom')
    
    plt.title('Maksymalny wynik dla różnych schematów nagród')
    plt.xlabel('Schemat nagród')
    plt.ylabel('Maksymalny wynik')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, 'maksymalne_wyniki.png'))
    plt.close()
    
    # 2. Wykres średnich wyników
    plt.figure(figsize=(12, 6))
    średnie_wyniki = [wyniki[s]['średnia'] for s in schematy]
    
    # Sortowanie według wartości
    sorted_indices = np.argsort(średnie_wyniki)[::-1]
    sorted_schematy = [schematy[i] for i in sorted_indices]
    sorted_średnie_wyniki = [średnie_wyniki[i] for i in sorted_indices]
    sorted_kolory = [paleta[schematy.index(s)] for s in sorted_schematy]
    
    bars = plt.bar(sorted_schematy, sorted_średnie_wyniki, color=sorted_kolory)
    
    # Dodanie wartości na szczycie słupków
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Średni wynik dla różnych schematów nagród')
    plt.xlabel('Schemat nagród')
    plt.ylabel('Średni wynik')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, 'średnie_wyniki.png'))
    plt.close()
    
    # 3. Wykres końcowych średnich wyników
    plt.figure(figsize=(12, 6))
    końcowe_średnie = [wyniki[s]['końcowa_średnia'] for s in schematy]
    
    # Sortowanie według wartości
    sorted_indices = np.argsort(końcowe_średnie)[::-1]
    sorted_schematy = [schematy[i] for i in sorted_indices]
    sorted_końcowe_średnie = [końcowe_średnie[i] for i in sorted_indices]
    sorted_kolory = [paleta[schematy.index(s)] for s in sorted_schematy]
    
    bars = plt.bar(sorted_schematy, sorted_końcowe_średnie, color=sorted_kolory)
    
    # Dodanie wartości na szczycie słupków
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Średni wynik pod koniec treningu dla różnych schematów nagród')
    plt.xlabel('Schemat nagród')
    plt.ylabel('Średni wynik z ostatnich 100 epizodów')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, 'końcowe_średnie.png'))
    plt.close()
    
    # 4. Wspólny wykres krzywych uczenia
    plt.figure(figsize=(14, 8))
    
    for i, schemat in enumerate(schematy):
        # Obliczenie średniej ruchomej
        historia = wyniki[schemat]['historia']
        okno = min(10, len(historia))
        średnia_ruchoma = [np.mean(historia[max(0, j-okno):j+1]) for j in range(len(historia))]
        
        plt.plot(średnia_ruchoma, label=schemat, color=paleta[i], linewidth=2)
    
    plt.title('Krzywe uczenia dla różnych schematów nagród')
    plt.xlabel('Epizod')
    plt.ylabel('Średni wynik (okno 10 epizodów)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyjściowy, 'krzywe_uczenia.png'))
    plt.close()
    
    # 5. Zapisanie wyników w tabeli CSV
    tabela = pd.DataFrame({
        'Schemat': schematy,
        'Maks. wynik': max_wyniki,
        'Średni wynik': średnie_wyniki,
        'Końcowa średnia': końcowe_średnie
    })
    tabela.to_csv(os.path.join(katalog_wyjściowy, 'wyniki.csv'), index=False)
    
    # 6. Sumaryczna tabela
    # Tworzymy ranking dla każdej metryki
    tabela['Ranking maks.'] = tabela['Maks. wynik'].rank(ascending=False)
    tabela['Ranking śr.'] = tabela['Średni wynik'].rank(ascending=False)
    tabela['Ranking końc.'] = tabela['Końcowa średnia'].rank(ascending=False)
    
    # Suma rankingów (niższa suma = lepszy schemat)
    tabela['Suma rankingów'] = tabela['Ranking maks.'] + tabela['Ranking śr.'] + tabela['Ranking końc.']
    
    # Sortujemy według sumy rankingów
    tabela = tabela.sort_values(by='Suma rankingów')
    
    # Zapisujemy tabelę z rankingami
    tabela.to_csv(os.path.join(katalog_wyjściowy, 'ranking.csv'), index=False)


def main():
    print("Rozpoczynam eksperyment z reward shaping...")
    
    # Definiowanie schematów nagród do przetestowania
    schematy_nagród = {
        'oryginalny': {
            'food_reward': 10.0,     # Nagroda za zjedzenie jedzenia
            'death_penalty': -10.0,  # Kara za śmierć
            'closer_reward': 0.1,    # Nagroda za zbliżenie się do jedzenia
            'farther_penalty': -0.1, # Kara za oddalenie się od jedzenia
            'time_penalty': 0.0      # Kara za każdy krok
        },
        'sparse': {
            'food_reward': 10.0,
            'death_penalty': -10.0,
            'closer_reward': 0.0,    # Brak nagrody za zbliżanie się
            'farther_penalty': 0.0,  # Brak kary za oddalanie się
            'time_penalty': 0.0
        },
        'dense': {
            'food_reward': 10.0,
            'death_penalty': -10.0,
            'closer_reward': 0.5,    # Większa nagroda za zbliżanie się
            'farther_penalty': -0.5, # Większa kara za oddalanie się
            'time_penalty': 0.0
        },
        'jedzenie': {
            'food_reward': 20.0,     # Zwiększona nagroda za jedzenie
            'death_penalty': -10.0,
            'closer_reward': 0.1,
            'farther_penalty': -0.1,
            'time_penalty': 0.0
        },
        'przetrwanie': {
            'food_reward': 10.0,
            'death_penalty': -20.0,  # Zwiększona kara za śmierć
            'closer_reward': 0.1,
            'farther_penalty': -0.1,
            'time_penalty': 0.0
        }
    }
    
    # Parametry eksperymentu
    liczba_epizodów = 300
    liczba_równoległych = 16
    
    # Utworzenie katalogu na wyniki
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    katalog_wyjściowy = os.path.join("wyniki", f"reward_shaping_{timestamp}")
    os.makedirs(katalog_wyjściowy, exist_ok=True)
    
    # Zapisanie konfiguracji eksperymentu
    config = {
        'schematy_nagród': schematy_nagród,
        'liczba_epizodów': liczba_epizodów,
        'liczba_równoległych': liczba_równoległych,
        'timestamp': timestamp
    }
    
    with open(os.path.join(katalog_wyjściowy, 'konfiguracja.json'), 'w') as f:
        import json
        json.dump(config, f, indent=4)
    
    print(f"Trening każdego schematu na {liczba_epizodów} epizodach, {liczba_równoległych} równoległych gier.")
    print(f"Wyniki będą zapisane w '{katalog_wyjściowy}'")
    
    # Uruchomienie eksperymentu dla każdego schematu nagród
    wyniki = {}
    
    for nazwa, schemat in schematy_nagród.items():
        print(f"\n==== Schemat: {nazwa} ====")
        historia, max_wynik, średni_wynik, końcowy_średni = testuj_schemat_nagród(
            nazwa_schematu=nazwa,
            schemat_nagród=schemat,
            liczba_epizodów=liczba_epizodów,
            liczba_równoległych=liczba_równoległych
        )
        
        # Zapisanie wyników
        wyniki[nazwa] = {
            'historia': historia,
            'max': max_wynik,
            'średnia': średni_wynik,
            'końcowa_średnia': końcowy_średni
        }
        
        # Zapisanie historii wyników dla tego schematu
        np.save(os.path.join(katalog_wyjściowy, f"historia_{nazwa}.npy"), np.array(historia))
    
    # Rysowanie wykresów porównawczych
    print("\nTworzenie wykresów...")
    rysuj_wykresy(wyniki, katalog_wyjściowy)
    
    print(f"\nEksperyment zakończony. Wyniki zapisane w '{katalog_wyjściowy}'")


if __name__ == "__main__":
    main()