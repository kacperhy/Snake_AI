"""
Moduł do badania optymalnej architektury sieci MLP dla agenta DQN.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from snake_game import UproszczonySnake
from agent import DQNAgent
from model import QNetwork  # Importujemy oryginalną sieć
from training import uruchom_epizod
from config import SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU, UŻYJ_GPU, ROZMIAR_UKRYTY

# Ustawiamy backend matplotlib na 'Agg', który nie wymaga GUI i jest bezpieczny w wielowątkowości
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Klasa do tworzenia własnych modeli z różnymi konfiguracjami warstw
class CustomQNetwork(nn.Module):
    """
    Elastyczna implementacja sieci neuronowej dla Deep Q-Network z różną liczbą warstw.
    
    Args:
        rozmiar_wejścia (int): Rozmiar wektora wejściowego.
        rozmiary_warstw (list): Lista określająca rozmiary kolejnych warstw ukrytych.
        rozmiar_wyjścia (int): Liczba możliwych akcji.
        dropout_rate (float): Współczynnik dropout (domyślnie 0.2).
    """
    def __init__(self, rozmiar_wejścia, rozmiary_warstw, rozmiar_wyjścia, dropout_rate=0.2):
        super(CustomQNetwork, self).__init__()
        
        # Przechowujemy rozmiary warstw
        self.rozmiary_warstw = rozmiary_warstw
        
        # Tworzymy warstwy w pętli na podstawie listy rozmiarów
        self.warstwy = nn.ModuleList()
        
        # Pierwsza warstwa (wejście -> pierwsza warstwa ukryta)
        self.warstwy.append(nn.Linear(rozmiar_wejścia, rozmiary_warstw[0]))
        
        # Pozostałe warstwy ukryte
        for i in range(len(rozmiary_warstw) - 1):
            self.warstwy.append(nn.Linear(rozmiary_warstw[i], rozmiary_warstw[i+1]))
        
        # Warstwa wyjściowa
        self.warstwa_wyjsciowa = nn.Linear(rozmiary_warstw[-1], rozmiar_wyjścia)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Inicjalizacja wag dla lepszej zbieżności
        for i, warstwa in enumerate(self.warstwy):
            if i < len(self.warstwy) - 1:
                torch.nn.init.kaiming_uniform_(warstwa.weight)
            else:
                torch.nn.init.xavier_uniform_(warstwa.weight)
        
        torch.nn.init.xavier_uniform_(self.warstwa_wyjsciowa.weight)
        
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
            
        # Przepływ do przodu przez warstwy ukryte
        for i, warstwa in enumerate(self.warstwy):
            x = F.leaky_relu(warstwa(x))
            x = self.dropout(x)
        
        # Warstwa wyjściowa (bez aktywacji)
        return self.warstwa_wyjsciowa(x)
    
    def __str__(self):
        """Zwraca opis architektury sieci."""
        return f"CustomQNetwork: {len(self.warstwy)} warstwy ukryte, rozmiary: {self.rozmiary_warstw}"


# Modyfikacja klasy DQNAgent, aby używała niestandardową sieć lub oryginalną sieć
class CustomDQNAgent(DQNAgent):
    """
    Agent DQN z możliwością definiowania własnej architektury sieci.
    """
    def __init__(self, rozmiar_stanu, rozmiar_akcji, architektura, współczynnik_uczenia=0.0003):
        # Inicjalizujemy część parametrów z klasy bazowej
        self.rozmiar_stanu = rozmiar_stanu
        self.rozmiar_akcji = rozmiar_akcji
        
        # Parametry uczenia
        self.gamma = 0.99  # Współczynnik dyskontowania przyszłych nagród
        self.epsilon = 1.0  # Współczynnik eksploracji
        self.epsilon_min = 0.1
        self.spadek_epsilon = 0.999
        self.rozmiar_partii = 256 if UŻYJ_GPU else 64
        self.częstotliwość_aktualizacji = 2
        self.wykonane_kroki = 0
        self.częstotliwość_aktualizacji_docelowej = 1000
        self.historia_epsilon = []  # Dodajemy historię epsilon
        
        # Urządzenie (CPU/GPU)
        self.device = torch.device("cuda" if UŻYJ_GPU else "cpu")
        self.typ_danych = torch.float32
        
        # Bufor doświadczeń
        from agent import BuforDoświadczeń
        self.pamięć = BuforDoświadczeń()
        
        # Model sieci neuronowej (policy network)
        if architektura == "oryginalna":
            # Używamy oryginalnej sieci z modelu.py
            self.model = QNetwork(rozmiar_stanu, ROZMIAR_UKRYTY, rozmiar_akcji).to(self.device)
            self.model_docelowy = QNetwork(rozmiar_stanu, ROZMIAR_UKRYTY, rozmiar_akcji).to(self.device)
        elif isinstance(architektura, list):
            # Używamy niestandardowej sieci
            self.model = CustomQNetwork(rozmiar_stanu, architektura, rozmiar_akcji).to(self.device)
            self.model_docelowy = CustomQNetwork(rozmiar_stanu, architektura, rozmiar_akcji).to(self.device)
        else:
            raise ValueError("Nieprawidłowy format architektury sieci")
            
        self.aktualizuj_model_docelowy()
        
        # Optymalizator
        if UŻYJ_GPU:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=współczynnik_uczenia, weight_decay=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=współczynnik_uczenia)
        
        # Funkcja straty
        self.criterion = nn.SmoothL1Loss()
        
        print(f"Używam urządzenia: {self.device}")
        if self.device.type == 'cuda':
            print(f"Model GPU: {torch.cuda.get_device_name(0)}")
        
        # Utworzenie katalogu na modele
        if not os.path.exists('models'):
            os.makedirs('models')
            print("Utworzono katalog 'models' do przechowywania modeli.")

    def pobierz_akcję(self, state):
        """
        Wybiera akcję zgodnie z polityką epsilon-greedy.
        """
        # Zapisujemy epsilon do historii przy każdym wywołaniu
        self.historia_epsilon.append(self.epsilon)
        
        # Reszta identyczna jak w oryginalnej implementacji
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.rozmiar_akcji)
        
        with torch.no_grad():
            tensor_stanu = torch.tensor(state, dtype=self.typ_danych).unsqueeze(0).to(self.device)
            wartości_q = self.model(tensor_stanu)
            return torch.argmax(wartości_q).item()


def policz_parametry_modelu(model):
    """
    Oblicza całkowitą liczbę parametrów w modelu.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rysuj_wykres_treningu(historia_wyników, historia_strat, nazwa, katalog_wyników):
    """
    Bezpieczne rysowanie i zapisywanie wykresu treningu.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(historia_wyników)
    plt.title(f'Historia wyników treningu - {nazwa}')
    plt.xlabel('Epizod')
    plt.ylabel('Wynik')
    
    plt.subplot(2, 1, 2)
    plt.plot(historia_strat)
    plt.title('Historia straty')
    plt.xlabel('Epizod')
    plt.ylabel('Wartość straty')
    
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyników, f"historia_treningu_{nazwa.replace(' ', '_')}.png"))
    plt.close('all')  # Zamykamy wszystkie wykresy, aby uniknąć wycieków pamięci


def rysuj_porównanie_architektur(df_wyniki, katalog_wyników):
    """
    Bezpieczne rysowanie i zapisywanie wykresu porównania architektur.
    """
    plt.figure(figsize=(15, 12))
    
    # Wykres średnich wyników
    plt.subplot(3, 2, 1)
    df_sorted = df_wyniki.sort_values('liczba_parametrów')
    plt.plot(df_sorted['liczba_parametrów'], df_sorted['średni_wynik'], 'o')
    
    # Dodajemy etykiety dla każdego punktu
    for i, row in df_sorted.iterrows():
        plt.annotate(row['nazwa_architektury'], 
                     (row['liczba_parametrów'], row['średni_wynik']),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
        
    plt.title('Średni wynik vs Liczba parametrów')
    plt.xlabel('Liczba parametrów modelu')
    plt.ylabel('Średni wynik')
    plt.grid(True)
    plt.xscale('log')  # Skala logarytmiczna dla lepszej wizualizacji
    
    # Wykres czasów treningu
    plt.subplot(3, 2, 2)
    df_sorted = df_wyniki.sort_values('liczba_parametrów')
    plt.plot(df_sorted['liczba_parametrów'], df_sorted['czas_treningu'], 'o')
    
    # Dodajemy etykiety dla każdego punktu
    for i, row in df_sorted.iterrows():
        plt.annotate(row['nazwa_architektury'], 
                     (row['liczba_parametrów'], row['czas_treningu']),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
    
    plt.title('Czas treningu vs Liczba parametrów')
    plt.xlabel('Liczba parametrów modelu')
    plt.ylabel('Czas [s]')
    plt.grid(True)
    plt.xscale('log')  # Skala logarytmiczna dla lepszej wizualizacji
    
    # Wykres rozmiaru modelu
    plt.subplot(3, 2, 3)
    df_sorted = df_wyniki.sort_values('liczba_parametrów')
    plt.plot(df_sorted['liczba_parametrów'], df_sorted['rozmiar_modelu'], 'o')
    
    # Dodajemy etykiety dla każdego punktu
    for i, row in df_sorted.iterrows():
        plt.annotate(row['nazwa_architektury'], 
                     (row['liczba_parametrów'], row['rozmiar_modelu']),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
    
    plt.title('Rozmiar modelu vs Liczba parametrów')
    plt.xlabel('Liczba parametrów modelu')
    plt.ylabel('Rozmiar [MB]')
    plt.grid(True)
    plt.xscale('log')  # Skala logarytmiczna dla lepszej wizualizacji
    
    # Wykres średniej straty
    plt.subplot(3, 2, 4)
    df_sorted = df_wyniki.sort_values('średnia_strata')
    bars = plt.bar(df_sorted['nazwa_architektury'], df_sorted['średnia_strata'])
    plt.title('Średnia strata dla różnych architektur')
    plt.xlabel('Architektura')
    plt.ylabel('Średnia strata')
    plt.xticks(rotation=45, ha='right')
    
    # Dokładniejszy wykres średnich wyników
    plt.subplot(3, 2, 5)
    df_sorted = df_wyniki.sort_values('średni_wynik', ascending=False)
    bars = plt.bar(df_sorted['nazwa_architektury'], df_sorted['średni_wynik'])
    plt.title('Ranking architektur wg średniego wyniku')
    plt.xlabel('Architektura')
    plt.ylabel('Średni wynik')
    plt.xticks(rotation=45, ha='right')
    
    # Wykres stosunku wydajności do liczby parametrów
    plt.subplot(3, 2, 6)
    df_wyniki['wydajność_na_parametr'] = df_wyniki['średni_wynik'] / df_wyniki['liczba_parametrów']
    df_sorted = df_wyniki.sort_values('wydajność_na_parametr', ascending=False)
    bars = plt.bar(df_sorted['nazwa_architektury'], df_sorted['wydajność_na_parametr'])
    plt.title('Wydajność na parametr (wyższy=lepszy)')
    plt.xlabel('Architektura')
    plt.ylabel('Średni wynik / liczba parametrów')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(katalog_wyników, 'porównanie_architektur.png'))
    plt.close('all')  # Zamykamy wszystkie wykresy


def badaj_architektury_sieci(architektury, liczba_epizodów=200, liczba_testów=10, katalog_wyników='wyniki_badania'):
    """
    Przeprowadza badanie wpływu różnych architektur sieci na wydajność modelu.
    
    Args:
        architektury (dict): Słownik opisujący różne architektury do przetestowania.
                            Klucz: nazwa architektury
                            Wartość: lista rozmiarów warstw ukrytych lub "oryginalna"
        liczba_epizodów (int): Liczba epizodów treningu dla każdego modelu.
        liczba_testów (int): Liczba testów dla każdego wytrenowanego modelu.
        katalog_wyników (str): Katalog, w którym zostaną zapisane wyniki.
    """
    # Przygotowanie katalogu na wyniki
    if not os.path.exists(katalog_wyników):
        os.makedirs(katalog_wyników)
    
    # Parametry dla inicjalizacji gry
    parametry_gry = {
        'szerokość': SZEROKOŚĆ_OKNA,
        'wysokość': WYSOKOŚĆ_OKNA,
        'rozmiar_bloku': ROZMIAR_BLOKU
    }
    
    # Przygotowanie struktur na wyniki
    rozmiar_stanu = 11  # Liczba cech w reprezentacji stanu
    rozmiar_akcji = 3   # Liczba możliwych akcji
    
    wyniki = {
        'nazwa_architektury': [],
        'liczba_warstw': [],
        'liczba_parametrów': [],
        'średni_wynik': [],
        'maks_wynik': [],
        'czas_treningu': [],
        'rozmiar_modelu': [],
        'konwergencja_epsilon': [],
        'średnia_strata': []
    }
    
    # Badanie dla każdej architektury
    for nazwa, architektura in architektury.items():
        print(f"\n=== Testowanie architektury: {nazwa} ===")
        if architektura == "oryginalna":
            print(f"Używam oryginalnej architektury z modelu.py")
            # Wyświetlenie struktury oryginalnej sieci
            oryginalna_siec = QNetwork(rozmiar_stanu, ROZMIAR_UKRYTY, rozmiar_akcji)
            print(f"Struktura oryginalnej sieci: wejście({rozmiar_stanu}) -> ukryta1({ROZMIAR_UKRYTY}) -> ukryta2({ROZMIAR_UKRYTY}) -> ukryta3({ROZMIAR_UKRYTY//2}) -> wyjście({rozmiar_akcji})")
        else:
            print(f"Warstwy ukryte: {architektura}")
        
        # Inicjalizacja agenta z odpowiednią siecią
        agent = CustomDQNAgent(rozmiar_stanu, rozmiar_akcji, architektura)
        
        # Obliczenie liczby parametrów
        liczba_parametrów = policz_parametry_modelu(agent.model)
        print(f"Liczba parametrów modelu: {liczba_parametrów}")
        
        # Pomiar czasu treningu
        czas_start = time.time()
        
        # Struktury do śledzenia postępu treningu
        historia_wyników = []
        historia_strat = []
        
        # Trening agenta
        with tqdm(total=liczba_epizodów, desc=f"Trening ({nazwa})") as pbar:
            for e in range(liczba_epizodów):
                # Zbieranie doświadczeń
                doświadczenia, _, wynik = uruchom_epizod(agent, parametry_gry)
                historia_wyników.append(wynik)
                
                # Dodawanie doświadczeń do pamięci
                for exp in doświadczenia:
                    agent.zapamiętaj(*exp)
                
                # Uczenie agenta
                łączna_strata = 0
                liczba_strat = 0
                iteracje_treningu = min(len(doświadczenia), 1000)
                
                for _ in range(iteracje_treningu):
                    strata = agent.ucz_się()
                    if strata is not None:
                        łączna_strata += strata
                        liczba_strat += 1
                
                # Zapisywanie średniej straty
                if liczba_strat > 0:
                    średnia_strata = łączna_strata / liczba_strat
                    historia_strat.append(średnia_strata)
                else:
                    historia_strat.append(0)
                
                # Aktualizacja paska postępu
                pbar.update(1)
                pbar.set_postfix({
                    'Wynik': f'{wynik}',
                    'Epsilon': f'{agent.epsilon:.4f}',
                    'Śr. strata': f'{średnia_strata:.4f}' if liczba_strat > 0 else 'N/A'
                })
        
        # Pomiar czasu treningu
        czas_treningu = time.time() - czas_start
        print(f"Czas treningu: {czas_treningu:.2f} sekund")
        
        # Zapisanie modelu
        model_path = os.path.join(katalog_wyników, f"model_{nazwa.replace(' ', '_')}.pth")
        agent.save(model_path)
        
        # Obliczenie rozmiaru modelu
        rozmiar_modelu_bytes = os.path.getsize(model_path)
        rozmiar_modelu_mb = rozmiar_modelu_bytes / (1024 * 1024)
        print(f"Rozmiar modelu: {rozmiar_modelu_mb:.2f} MB")
        
        # Testowanie wytrenowanego modelu
        wyniki_testów = []
        game = UproszczonySnake(**parametry_gry)
        
        for _ in tqdm(range(liczba_testów), desc="Testowanie"):
            stan = game.reset()
            zakończone = False
            wynik_gry = 0
            
            while not zakończone:
                # Agent wybiera akcję bez eksploracji (greedy policy)
                tensor_stanu = torch.tensor(stan, dtype=agent.typ_danych).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    wartości_q = agent.model(tensor_stanu)
                akcja = torch.argmax(wartości_q).item()
                
                # Wykonanie akcji
                stan, _, zakończone, wynik_gry = game.krok(akcja)
            
            wyniki_testów.append(wynik_gry)
        
        średni_wynik_testów = np.mean(wyniki_testów)
        maks_wynik_testów = np.max(wyniki_testów)
        print(f"Średni wynik z {liczba_testów} testów: {średni_wynik_testów:.2f}")
        print(f"Maksymalny wynik: {maks_wynik_testów}")
        
        # Analiza konwergencji epsilon
        epizod_konwergencji = next((i for i, e in enumerate(agent.historia_epsilon) if e <= agent.epsilon_min + 0.01), liczba_epizodów)
        
        # Zapisanie wyników dla tej architektury
        wyniki['nazwa_architektury'].append(nazwa)
        
        # Określenie liczby warstw dla różnych typów architektury
        if architektura == "oryginalna":
            liczba_warstw = 4  # Oryginalna sieć ma 4 warstwy
        else:
            liczba_warstw = len(architektura)
            
        wyniki['liczba_warstw'].append(liczba_warstw)
        wyniki['liczba_parametrów'].append(liczba_parametrów)
        wyniki['średni_wynik'].append(średni_wynik_testów)
        wyniki['maks_wynik'].append(maks_wynik_testów)
        wyniki['czas_treningu'].append(czas_treningu)
        wyniki['rozmiar_modelu'].append(rozmiar_modelu_mb)
        wyniki['konwergencja_epsilon'].append(epizod_konwergencji)
        wyniki['średnia_strata'].append(np.mean(historia_strat[-20:]))  # Średnia z ostatnich 20 epizodów
        
        # Zapisywanie wykresu historii treningu
        rysuj_wykres_treningu(historia_wyników, historia_strat, nazwa, katalog_wyników)
        
        # Zapisujemy częściowe wyniki po każdej architekturze
        df_wyniki_częściowe = pd.DataFrame(wyniki)
        df_wyniki_częściowe.to_csv(os.path.join(katalog_wyników, 'wyniki_częściowe.csv'), index=False)
    
    # Zapisanie wszystkich wyników do pliku CSV
    df_wyniki = pd.DataFrame(wyniki)
    df_wyniki.to_csv(os.path.join(katalog_wyników, 'wyniki_wszystkie.csv'), index=False)
    
    # Tworzenie wykresów porównawczych
    rysuj_porównanie_architektur(df_wyniki, katalog_wyników)
    
    # Wydrukowanie tabeli z wynikami
    print("\n=== Podsumowanie wyników badania ===")
    print(df_wyniki.to_string(index=False))
    
    # Określenie optymalnej architektury na podstawie wyników
    idx_optymalny = df_wyniki['średni_wynik'].idxmax()
    optymalna_architektura = df_wyniki.iloc[idx_optymalny]['nazwa_architektury']
    
    print(f"\nOptymalna architektura na podstawie średniego wyniku: {optymalna_architektura}")
    print(f"Średni wynik dla {optymalna_architektura}: {df_wyniki.iloc[idx_optymalny]['średni_wynik']:.2f}")
    print(f"Liczba parametrów: {df_wyniki.iloc[idx_optymalny]['liczba_parametrów']}")
    print(f"Czas treningu: {df_wyniki.iloc[idx_optymalny]['czas_treningu']:.2f} sekund")
    
    return df_wyniki


# Przykładowe użycie funkcji
if __name__ == "__main__":
    # Definiowanie różnych architektur do przetestowania
    architektury = {
        # Oryginalna sieć z modelu.py
        "Oryginalna sieć (QNetwork)": "oryginalna",
        
        # Proste sieci jednowarstwowe z różnymi rozmiarami
        "1 warstwa (32)": [32],
        "1 warstwa (64)": [64],
        "1 warstwa (128)": [128],
        "1 warstwa (256)": [256],
        
        # Sieci dwuwarstwowe o różnych rozmiarach
        "2 warstwy (64, 32)": [64, 32],
        "2 warstwy (128, 64)": [128, 64],
        
        # Głębsze sieci o strukturze podobnej do oryginalnej
        "3 warstwy (128, 128, 64)": [128, 128, 64],
    }
    
    # Przeprowadzenie badania
    wyniki = badaj_architektury_sieci(
        architektury,
        liczba_epizodów=5000,     # Liczba epizodów treningu dla każdego modelu
        liczba_testów=20,        # Liczba testów dla każdego wytrenowanego modelu
        katalog_wyników='wyniki_badania_architektury'
    )