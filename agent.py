"""
Moduł zawierający implementację agenta DQN oraz bufora doświadczeń.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import QNetwork
import config
from config import (
    ROZMIAR_BUFORA_DOŚWIADCZEŃ, ROZMIAR_UKRYTY, WSPÓŁCZYNNIK_UCZENIA,
    GAMMA, EPSILON_START, EPSILON_MIN, SPADEK_EPSILON, ROZMIAR_PARTII
)
# UWAGA: UŻYJ_GPU i UŻYJ_FLOAT16 czytamy jako config.UŻYJ_GPU w chwili
# tworzenia agenta. "from config import UŻYJ_GPU" kopiowałoby wartość przy
# imporcie, więc późniejsza zmiana (np. opcja "tryb CPU" w main.py) nie
# docierałaby do agenta.


def wybierz_urządzenie(urządzenie=None) -> torch.device:
    """
    Zwraca urządzenie obliczeniowe.

    Args:
        urządzenie: 'cuda', 'cpu', torch.device albo None.
                    None = według config.UŻYJ_GPU (odczyt w tej chwili).
    """
    if urządzenie is None:
        urządzenie = 'cuda' if config.UŻYJ_GPU else 'cpu'
    urządzenie = torch.device(urządzenie)
    if urządzenie.type == 'cuda' and not torch.cuda.is_available():
        print("UWAGA: CUDA niedostępna – używam procesora.")
        urządzenie = torch.device('cpu')
    return urządzenie


def typ_danych_dla(urządzenie: torch.device) -> torch.dtype:
    """float16 tylko na karcie i tylko gdy włączone w config."""
    if urządzenie.type == 'cuda' and config.UŻYJ_FLOAT16:
        return torch.float16
    return torch.float32


class BuforDoświadczeń:
    """
    Bufor cykliczny trzymany od razu na urządzeniu obliczeniowym.

    Zamiast kolejki krotek (deque) mamy 5 gotowych tensorów, np. przy
    pojemności 50 000 i stanie z 11 cech: `stany` to tensor 50 000 × 11.
    Losowanie partii to wtedy tylko wybór indeksów na karcie – dane nie są
    przy każdym kroku uczenia pakowane w numpy i kopiowane z RAM na kartę.

    Nowe doświadczenia czekają w małej liście w RAM i trafiają na urządzenie
    jednym kopiowaniem tuż przed losowaniem (zamiast osobno każdy ruch).

    Atrybuty:
        pojemność (int): Maksymalna liczba przechowywanych doświadczeń.
        device (torch.device): Urządzenie, na którym leżą dane.
        typ_danych (torch.dtype): Typ danych stanów i nagród.
    """

    def __init__(self, pojemność=ROZMIAR_BUFORA_DOŚWIADCZEŃ, rozmiar_stanu=11,
                 device=None, typ_danych=None):
        self.device = wybierz_urządzenie(device)
        self.typ_danych = typ_danych or typ_danych_dla(self.device)
        self.pojemność = pojemność
        self.rozmiar_stanu = rozmiar_stanu

        d, t = self.device, self.typ_danych
        self.stany = torch.zeros((pojemność, rozmiar_stanu), dtype=t, device=d)
        self.następne_stany = torch.zeros((pojemność, rozmiar_stanu), dtype=t, device=d)
        self.akcje = torch.zeros(pojemność, dtype=torch.long, device=d)
        self.nagrody = torch.zeros(pojemność, dtype=t, device=d)
        self.zakończone = torch.zeros(pojemność, dtype=torch.bool, device=d)

        self.pozycja = 0          # gdzie wpiszemy następne doświadczenie
        self.wypełnienie = 0      # ile miejsc jest już zajętych
        self._oczekujące = []     # nowe doświadczenia jeszcze w RAM

    def dodaj(self, state, action, reward, next_state, done):
        """Dodaje nowe doświadczenie (na urządzenie trafi przy losowaniu)."""
        self._oczekujące.append((state, action, reward, next_state, done))

    def _przenieś_oczekujące(self):
        """Kopiuje wszystkie oczekujące doświadczenia na urządzenie naraz."""
        if not self._oczekujące:
            return
        # Starsze niż ostatnie `pojemność` i tak zostałyby od razu nadpisane
        nowe = self._oczekujące[-self.pojemność:]
        self._oczekujące = []

        stany, akcje, nagrody, następne, zakończone = zip(*nowe)
        n = len(nowe)
        d, t = self.device, self.typ_danych
        idx = torch.arange(self.pozycja, self.pozycja + n, device=d) % self.pojemność

        self.stany[idx] = torch.as_tensor(np.array(stany), dtype=t).to(d)
        self.następne_stany[idx] = torch.as_tensor(np.array(następne), dtype=t).to(d)
        self.akcje[idx] = torch.as_tensor(akcje, dtype=torch.long).to(d)
        self.nagrody[idx] = torch.as_tensor(nagrody, dtype=t).to(d)
        self.zakończone[idx] = torch.as_tensor(zakończone, dtype=torch.bool).to(d)

        self.pozycja = (self.pozycja + n) % self.pojemność
        self.wypełnienie = min(self.wypełnienie + n, self.pojemność)

    def sample(self, rozmiar_partii):
        """
        Losuje partię doświadczeń (z powtórzeniami – standard w DQN).

        Returns:
            tuple tensorów (stany, akcje, nagrody, następne_stany, zakończone)
            już na urządzeniu, albo None gdy w buforze jest za mało danych.
        """
        self._przenieś_oczekujące()
        if self.wypełnienie < rozmiar_partii:
            return None
        idx = torch.randint(0, self.wypełnienie, (rozmiar_partii,), device=self.device)
        return (self.stany[idx], self.akcje[idx], self.nagrody[idx],
                self.następne_stany[idx], self.zakończone[idx])

    def przenieś_na(self, device, typ_danych):
        """Zwraca kopię bufora na innym urządzeniu (np. przy zmianie CPU↔GPU)."""
        self._przenieś_oczekujące()
        nowy = BuforDoświadczeń(self.pojemność, self.rozmiar_stanu, device, typ_danych)
        for nazwa in ('stany', 'następne_stany', 'nagrody'):
            getattr(nowy, nazwa).copy_(getattr(self, nazwa))
        nowy.akcje.copy_(self.akcje)
        nowy.zakończone.copy_(self.zakończone)
        nowy.pozycja, nowy.wypełnienie = self.pozycja, self.wypełnienie
        return nowy

    # Zgodność wstecz – wcześniejsza wersja pobierała partie w osobnym wątku.
    # Przy buforze na karcie losowanie jest natychmiastowe, więc to niepotrzebne.
    def pobierz_wstępnie_batch(self, rozmiar_partii):
        pass

    def pobierz_wstępnie_pobrany_batch(self):
        return None

    def __len__(self):
        return min(self.wypełnienie + len(self._oczekujące), self.pojemność)


class DQNAgent:
    """
    Agent używający algorytmu Deep Q-Network (w wariancie Double DQN).

    Atrybuty:
        rozmiar_stanu (int): Wymiar wektora stanu.
        rozmiar_akcji (int): Liczba możliwych akcji.
        device (torch.device): Gdzie liczona jest sieć (CPU lub karta).
        pamięć (BuforDoświadczeń): Bufor przechowujący doświadczenia.
        model (QNetwork): Główna sieć neuronowa.
        model_docelowy (QNetwork): Sieć docelowa do stabilizacji uczenia.
    """
    def __init__(self, rozmiar_stanu, rozmiar_akcji, rozmiar_ukryty=ROZMIAR_UKRYTY,
                 współczynnik_uczenia=WSPÓŁCZYNNIK_UCZENIA, urządzenie=None):
        """
        Args:
            urządzenie: 'cuda', 'cpu' albo None (= według config.UŻYJ_GPU).
        """
        self.rozmiar_stanu = rozmiar_stanu
        self.rozmiar_akcji = rozmiar_akcji

        # Parametry uczenia – TAKIE SAME na CPU i GPU, żeby porównanie
        # urządzeń mierzyło tylko szybkość, a nie inne ustawienia uczenia.
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.spadek_epsilon = SPADEK_EPSILON
        self.rozmiar_partii = ROZMIAR_PARTII
        self.częstotliwość_aktualizacji = 2          # co ile wywołań ucz_się uczyć
        self.wykonane_kroki = 0
        self.częstotliwość_aktualizacji_docelowej = 1000

        # Urządzenie (CPU/GPU)
        self.device = wybierz_urządzenie(urządzenie)
        self.typ_danych = typ_danych_dla(self.device)

        # Bufor doświadczeń – leży na tym samym urządzeniu co sieć
        self.pamięć = BuforDoświadczeń(rozmiar_stanu=rozmiar_stanu,
                                       device=self.device, typ_danych=self.typ_danych)

        # Sieci i optymalizator
        self._zbuduj_sieci(rozmiar_ukryty, współczynnik_uczenia)

        # Funkcja straty - Huber jest bardziej stabilna
        self.criterion = nn.SmoothL1Loss()

        print(f"Używam urządzenia: {self.device}")
        if self.device.type == 'cuda':
            print(f"Model GPU: {torch.cuda.get_device_name(0)}")
            if self.typ_danych == torch.float16:
                print("Używam precyzji float16 dla szybszego treningu")

        # Utworzenie katalogu na modele
        if not os.path.exists('models'):
            os.makedirs('models')
            print("Utworzono katalog 'models' do przechowywania modeli.")

    def _zbuduj_sieci(self, rozmiar_ukryty, współczynnik_uczenia):
        """Tworzy sieć główną, docelową i optymalizator na bieżącym urządzeniu."""
        self.rozmiar_ukryty = rozmiar_ukryty
        self.model = QNetwork(self.rozmiar_stanu, rozmiar_ukryty,
                              self.rozmiar_akcji).to(self.device, self.typ_danych)
        self.model_docelowy = QNetwork(self.rozmiar_stanu, rozmiar_ukryty,
                                       self.rozmiar_akcji).to(self.device, self.typ_danych)
        self.aktualizuj_model_docelowy()
        # Ten sam optymalizator na CPU i GPU (wcześniej CPU: Adam, GPU: AdamW)
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=współczynnik_uczenia, weight_decay=1e-5)

    def liczba_parametrów(self) -> int:
        """Liczba uczonych wag sieci głównej."""
        return sum(p.numel() for p in self.model.parameters())

    def aktualizuj_model_docelowy(self):
        """Aktualizacja wag modelu docelowego."""
        self.model_docelowy.load_state_dict(self.model.state_dict())
        self.model_docelowy.eval()  # sieć docelowa nigdy nie jest trenowana

    def zapamiętaj(self, state, action, reward, next_state, done):
        """Zapisuje doświadczenie w buforze pamięci."""
        self.pamięć.dodaj(state, action, reward, next_state, done)

    def pobierz_akcję(self, state):
        """
        Wybiera akcję zgodnie z polityką epsilon-greedy.

        Args:
            state (numpy.ndarray): Stan gry.

        Returns:
            int: Indeks wybranej akcji.
        """
        # Eksploracja - wybór losowej akcji
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.rozmiar_akcji)

        # Eksploatacja - wybór akcji o najwyższej wartości Q
        with torch.no_grad():
            tensor_stanu = torch.tensor(state, dtype=self.typ_danych).unsqueeze(0).to(self.device)
            wartości_q = self.model(tensor_stanu)
            return torch.argmax(wartości_q).item()

    def pobierz_akcje(self, stany):
        """
        Epsilon-greedy dla wielu gier naraz – jedno zapytanie do sieci
        zamiast osobnego dla każdej gry.

        Args:
            stany (numpy.ndarray): Tablica n × rozmiar_stanu.

        Returns:
            numpy.ndarray: n indeksów akcji.
        """
        n = len(stany)
        akcje = np.random.randint(self.rozmiar_akcji, size=n)
        chciwe = np.random.rand(n) > self.epsilon      # te gry pytają sieć
        if chciwe.any():
            with torch.no_grad():
                t = torch.as_tensor(stany[chciwe], dtype=self.typ_danych).to(self.device)
                akcje[chciwe] = self.model(t).argmax(1).cpu().numpy()
        return akcje

    def ucz_się(self):
        """
        Uczenie na podstawie próbki doświadczeń z pamięci.

        Returns:
            float or None: Wartość straty jeśli uczenie miało miejsce, None w przeciwnym razie.
        """
        self.wykonane_kroki += 1

        # Aktualizacja tylko co kilka kroków dla szybszego treningu
        if self.wykonane_kroki % self.częstotliwość_aktualizacji != 0:
            return None

        partia = self.pamięć.sample(self.rozmiar_partii)
        if partia is None:
            return None  # Za mało doświadczeń

        tensor_stanów, tensor_akcji, tensor_nagród, tensor_następnych_stanów, tensor_zakończonych = partia

        # Obliczanie przewidywanych wartości Q dla obecnych stanów
        obecne_wartości_q = self.model(tensor_stanów).gather(1, tensor_akcji.unsqueeze(1)).squeeze(1)

        # Obliczanie wartości docelowych z podwójnym DQN dla lepszej stabilności
        with torch.no_grad():
            # Podwójny DQN: wybieramy akcje za pomocą sieci głównej
            następne_akcje = self.model(tensor_następnych_stanów).max(1)[1].unsqueeze(1)
            # Ale wartości Q bierzemy z sieci docelowej
            następne_wartości_q = self.model_docelowy(tensor_następnych_stanów).gather(1, następne_akcje).squeeze(1)
            # Obliczamy docelowe wartości Q
            docelowe_wartości_q = tensor_nagród + self.gamma * następne_wartości_q * (~tensor_zakończonych)

        # Obliczanie straty i aktualizacja modelu
        strata = self.criterion(obecne_wartości_q, docelowe_wartości_q)

        # Optymalizacja
        self.optimizer.zero_grad()
        strata.backward()
        # Przycinanie gradientów dla stabilności
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Aktualizacja modelu docelowego co określoną liczbę kroków
        if self.wykonane_kroki % self.częstotliwość_aktualizacji_docelowej == 0:
            self.aktualizuj_model_docelowy()

        # Zmniejszanie wartości epsilon w czasie
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.spadek_epsilon

        return strata.item()

    def przenieś_na(self, urządzenie):
        """Przenosi sieci, optymalizator i bufor na inne urządzenie (CPU↔GPU)."""
        nowe = wybierz_urządzenie(urządzenie)
        if nowe == self.device:
            return
        print(f"Przenoszenie modelu z {self.device} na {nowe}...")
        stan_optymalizatora = self.optimizer.state_dict()
        lr = self.optimizer.param_groups[0]['lr']

        self.device, self.typ_danych = nowe, typ_danych_dla(nowe)
        self.model.to(nowe, self.typ_danych)
        self.model_docelowy.to(nowe, self.typ_danych)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.optimizer.load_state_dict(stan_optymalizatora)   # stan trafia na nowe urządzenie
        self.pamięć = self.pamięć.przenieś_na(nowe, self.typ_danych)

    def save(self, file_name='models/dqn_model.pth'):
        """
        Zapisuje model do pliku.

        Args:
            file_name (str): Ścieżka do pliku, w którym zostanie zapisany model.
        """
        stan_modelu = {
            'model_state_dict': self.model.state_dict(),
            'stan_optymalizatora': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'rozmiar_stanu': self.rozmiar_stanu,
            'rozmiar_akcji': self.rozmiar_akcji,
            'rozmiar_ukryty': self.rozmiar_ukryty,
            'wykonane_kroki': self.wykonane_kroki,
            'device_type': self.device.type,
            'UŻYJ_FLOAT16': self.typ_danych == torch.float16,
        }
        torch.save(stan_modelu, file_name)
        print(f"Model został zapisany do {file_name}")

    def wczytaj(self, file_name='models/dqn_model.pth'):
        """
        Wczytuje model z pliku.

        Jeśli plik ma inny rozmiar warstwy ukrytej niż obecny agent (np. po
        zmianie ROZMIAR_UKRYTY w config), sieci są przebudowywane pod plik –
        dzięki temu stare modele dalej się wczytują.

        Args:
            file_name (str): Ścieżka do pliku z zapisanym modelem.

        Returns:
            bool: True jeśli wczytanie się powiodło, False w przeciwnym razie.
        """
        try:
            punkt_kontrolny = torch.load(file_name, map_location=self.device)

            # Sprawdzenie zgodności modelu
            if punkt_kontrolny['rozmiar_stanu'] != self.rozmiar_stanu or punkt_kontrolny['rozmiar_akcji'] != self.rozmiar_akcji:
                print(f"UWAGA: Niezgodność wymiarów modelu! Oczekiwano: {self.rozmiar_stanu}x{self.rozmiar_akcji}, "
                      f"Wczytano: {punkt_kontrolny['rozmiar_stanu']}x{punkt_kontrolny['rozmiar_akcji']}")
                if input("Czy chcesz kontynuować wczytywanie? (t/n): ").lower() != 't':
                    return False

            # Rozmiar warstwy ukrytej – starsze pliki go nie zapisywały,
            # więc odczytujemy go z kształtu wag pierwszej warstwy.
            wagi = punkt_kontrolny['model_state_dict']
            rozmiar_z_pliku = punkt_kontrolny.get('rozmiar_ukryty', wagi['fc1.weight'].shape[0])
            if rozmiar_z_pliku != self.rozmiar_ukryty:
                print(f"Model ma warstwę ukrytą {rozmiar_z_pliku} (obecnie {self.rozmiar_ukryty}) "
                      f"– przebudowuję sieć pod plik.")
                self._zbuduj_sieci(rozmiar_z_pliku, self.optimizer.param_groups[0]['lr'])

            # Wczytywanie stanu modelu (typ float16/float32 dopasuje się sam)
            self.model.load_state_dict(wagi)
            try:
                self.optimizer.load_state_dict(punkt_kontrolny['stan_optymalizatora'])
            except (ValueError, KeyError) as e:
                print(f"Nie wczytano stanu optymalizatora ({e}) – startuje od nowa.")
            self.epsilon = punkt_kontrolny['epsilon']
            self.gamma = punkt_kontrolny['gamma']
            self.rozmiar_stanu = punkt_kontrolny['rozmiar_stanu']
            self.rozmiar_akcji = punkt_kontrolny['rozmiar_akcji']

            if 'wykonane_kroki' in punkt_kontrolny:
                self.wykonane_kroki = punkt_kontrolny['wykonane_kroki']

            wczytane_urządzenie = punkt_kontrolny.get('device_type', 'cpu')
            if wczytane_urządzenie != self.device.type:
                print(f"Informacja: model trenowano na {wczytane_urządzenie}, teraz działa na {self.device.type}.")

            # Aktualizacja modelu docelowego
            self.aktualizuj_model_docelowy()

            print(f"Model został wczytany z {file_name}")
            return True
        except FileNotFoundError:
            print(f"Nie znaleziono pliku {file_name}")
            return False
        except Exception as e:
            print(f"Wystąpił błąd podczas wczytywania modelu: {e}")
            return False

    def continue_training(self, train_function, game_params, liczba_epizodów=100, interwał_zapisu=10,
                          force_device=None, update_learning_params=False, **train_kwargs):
        """
        Kontynuuje trening istniejącego modelu z możliwością zmiany urządzenia obliczeniowego.

        Args:
            train_function: Funkcja trenująca (trenuj_hybrydowo lub trenuj_tylko_cpu)
            game_params: Parametry gry (albo obiekt gry dla trenuj_tylko_cpu)
            liczba_epizodów: Liczba epizodów do kontynuacji treningu
            interwał_zapisu: Co ile epizodów zapisywać model
            force_device: Jeśli podano, wymusza urządzenie ('cpu' lub 'cuda')
            update_learning_params: Czy aktualizować parametry uczenia (gamma, epsilon, itp.)
            **train_kwargs: Dodatkowe parametry dla funkcji trenującej

        Returns:
            tuple: Para (wyniki, historia epsilon) z kontynuowanego treningu
        """
        original_device = self.device
        if force_device:
            self.przenieś_na(force_device)

        # Aktualizacja parametrów uczenia, jeśli wymagane
        if update_learning_params:
            print("Aktualizuję parametry uczenia...")
            self.gamma = float(input(f"Podaj współczynnik dyskontowania (obecny: {self.gamma}): ") or self.gamma)
            self.epsilon = float(input(f"Podaj początkowy współczynnik eksploracji (obecny: {self.epsilon}): ") or self.epsilon)
            self.epsilon_min = float(input(f"Podaj minimalny współczynnik eksploracji (obecny: {self.epsilon_min}): ") or self.epsilon_min)
            self.spadek_epsilon = float(input(f"Podaj współczynnik zmniejszania eksploracji (obecny: {self.spadek_epsilon}): ") or self.spadek_epsilon)
            self.rozmiar_partii = int(input(f"Podaj rozmiar batcha (obecny: {self.rozmiar_partii}): ") or self.rozmiar_partii)

        print(f"Kontynuuję trening na urządzeniu: {self.device}")
        print(f"Gamma: {self.gamma}, Epsilon: {self.epsilon}, partia size: {self.rozmiar_partii}")
        print(f"Trening przez {liczba_epizodów} epizodów...")

        try:
            scores, eps_history = train_function(self, game_params,
                                                 liczba_epizodów=liczba_epizodów,
                                                 interwał_zapisu=interwał_zapisu,
                                                 **train_kwargs)
            print("Kontynuacja treningu zakończona. Model zapisany.")
            return scores, eps_history
        except Exception as e:
            print(f"Błąd podczas kontynuacji treningu: {e}")
            if self.device != original_device:
                print(f"Przywracam model na oryginalne urządzenie: {original_device}")
                self.przenieś_na(original_device)
            raise e