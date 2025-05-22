"""
Moduł zawierający implementację agenta DQN oraz bufora doświadczeń.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import threading
from model import QNetwork
from config import (
    UŻYJ_GPU, UŻYJ_FLOAT16, ROZMIAR_BUFORA_DOŚWIADCZEŃ,
    ROZMIAR_UKRYTY, WSPÓŁCZYNNIK_UCZENIA, GAMMA, EPSILON_START,
    EPSILON_MIN, SPADEK_EPSILON, ROZMIAR_PARTII
)

#minimalne wymagania dla GPU
class BuforDoświadczeń:
    """
    Bufor doświadczeń z optymalizacjami dla szybkiego przetwarzania.
    
    Atrybuty:
        pojemność (int): Maksymalna liczba przechowywanych doświadczeń.
        bufor (deque): Kolejka przechowująca doświadczenia.
        device (torch.device): Urządzenie do obliczeń (CPU/GPU).
        typ_danych (torch.typ_danych): Typ danych dla tensorów.
    """
    def __init__(self, pojemność=ROZMIAR_BUFORA_DOŚWIADCZEŃ):
        self.bufor = deque(maxlen=pojemność)
        self.device = torch.device("cuda" if UŻYJ_GPU else "cpu")
        self.typ_danych = torch.float16 if UŻYJ_FLOAT16 and UŻYJ_GPU else torch.float32
        self.wstępnie_pobrany_batch = None
        self.blokada_pobierania = threading.Lock()
        self.czy_pobieranie_wstępne = False
    
    def dodaj(self, state, action, reward, next_state, done):
        """Dodaje nowe doświadczenie do bufora."""
        self.bufor.append((state, action, reward, next_state, done))
    
    def sample(self, rozmiar_partii):
        """Pobiera losową próbkę z bufora i konwertuje na tensory."""
        if len(self.bufor) < rozmiar_partii:
            return None
            
        # Wybór losowej próbki
        partia = random.sample(self.bufor, rozmiar_partii)
        
        # Rozpakowanie danych
        stany, akcje, nagrody, następne_stany, zakończone = zip(*partia)
        
        # Konwersja na tensory (na CPU dla uniknięcia wąskich gardeł transferu)
        tensor_stanów = torch.tensor(np.array(stany), dtype=self.typ_danych).to(self.device)
        tensor_akcji = torch.tensor(akcje, dtype=torch.long).to(self.device)
        tensor_nagród = torch.tensor(nagrody, dtype=self.typ_danych).to(self.device)
        tensor_następnych_stanów = torch.tensor(np.array(następne_stany), dtype=self.typ_danych).to(self.device)
        tensor_zakończonych = torch.tensor(zakończone, dtype=torch.bool).to(self.device)
        
        return tensor_stanów, tensor_akcji, tensor_nagród, tensor_następnych_stanów, tensor_zakończonych
    
    def pobierz_wstępnie_batch(self, rozmiar_partii):
        """Wstępnie pobiera partia w osobnym wątku dla szybszego dostępu."""
        if not self.czy_pobieranie_wstępne and len(self.bufor) >= rozmiar_partii:
            self.czy_pobieranie_wstępne = True
            threading.Thread(target=self._wątek_wstępnego_pobierania, args=(rozmiar_partii,)).start()
    
    def _wątek_wstępnego_pobierania(self, rozmiar_partii):
        """Wątek pobierający partia."""
        partia = self.sample(rozmiar_partii)
        with self.blokada_pobierania:
            self.wstępnie_pobrany_batch = partia
            self.czy_pobieranie_wstępne = False
    
    def pobierz_wstępnie_pobrany_batch(self):
        """Pobiera wcześniej pobrany partia lub tworzy nowy jeśli nie ma."""
        with self.blokada_pobierania:
            partia = self.wstępnie_pobrany_batch
            self.wstępnie_pobrany_batch = None
        return partia
    
    def __len__(self):
        return len(self.bufor)


class DQNAgent:
    """
    Agent używający algorytmu Deep Q-Network z optymalizacjami.
    
    Atrybuty:
        rozmiar_stanu (int): Wymiar wektora stanu.
        rozmiar_akcji (int): Liczba możliwych akcji.
        pamięć (BuforDoświadczeń): Bufor przechowujący doświadczenia.
        model (QNetwork): Główna sieć neuronowa.
        model_docelowy (QNetwork): Sieć docelowa do stabilizacji uczenia.
    """
    def __init__(self, rozmiar_stanu, rozmiar_akcji, rozmiar_ukryty=256, współczynnik_uczenia=WSPÓŁCZYNNIK_UCZENIA):
        self.rozmiar_stanu = rozmiar_stanu
        self.rozmiar_akcji = rozmiar_akcji
        
        # Parametry uczenia
        self.gamma = GAMMA # Współczynnik dyskontowania przyszłych nagród
        self.epsilon = EPSILON_START  # Współczynnik eksploracji
        self.epsilon_min = EPSILON_MIN
        self.spadek_epsilon = SPADEK_EPSILON
        self.rozmiar_partii = ROZMIAR_PARTII if UŻYJ_GPU else 64  # Większy partia na GPU, mniejszy na CPU
        self.częstotliwość_aktualizacji = 2  # Co ile kroków aktualizować model
        self.wykonane_kroki = 0
        self.częstotliwość_aktualizacji_docelowej = 1000  # Co ile kroków aktualizować model docelowy
        
        # Urządzenie (CPU/GPU)
        self.device = torch.device("cuda" if UŻYJ_GPU else "cpu")
        self.typ_danych = torch.float16 if UŻYJ_FLOAT16 and UŻYJ_GPU else torch.float32
        
        # Bufor doświadczeń
        self.pamięć = BuforDoświadczeń()
        
        # Model sieci neuronowej (policy network)
        self.model = QNetwork(rozmiar_stanu, rozmiar_ukryty, rozmiar_akcji).to(self.device)
        if UŻYJ_FLOAT16 and UŻYJ_GPU:
            self.model = self.model.half()  # Konwersja do float16 dla GPU
        
        # Model sieci docelowej (target network)
        self.model_docelowy = QNetwork(rozmiar_stanu, rozmiar_ukryty, rozmiar_akcji).to(self.device)
        if UŻYJ_FLOAT16 and UŻYJ_GPU:
            self.model_docelowy = self.model_docelowy.half()
            
        self.aktualizuj_model_docelowy()
        
        # Optymalizator - dla CPU używamy Adam, dla GPU AdamW (lepszy dla GPU)
        if UŻYJ_GPU:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=współczynnik_uczenia, weight_decay=1e-5)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=współczynnik_uczenia)
        
        # Funkcja straty - Huber jest bardziej stabilna
        self.criterion = nn.SmoothL1Loss()
        
        print(f"Używam urządzenia: {self.device}")
        if self.device.type == 'cuda':
            print(f"Model GPU: {torch.cuda.get_device_name(0)}")
            if UŻYJ_FLOAT16:
                print("Używam precyzji float16 dla szybszego treningu")
        
        # Utworzenie katalogu na modele
        if not os.path.exists('models'):
            os.makedirs('models')
            print("Utworzono katalog 'models' do przechowywania modeli.")
    
    def aktualizuj_model_docelowy(self):
        """Aktualizacja wag modelu docelowego."""
        self.model_docelowy.load_state_dict(self.model.state_dict())
    
    def zapamiętaj(self, state, action, reward, next_state, done):
        """Zapisuje doświadczenie w buforze pamięci."""
        self.pamięć.dodaj(state, action, reward, next_state, done)
        # Wstępne pobranie danych do przyszłego uczenia
        self.pamięć.pobierz_wstępnie_batch(self.rozmiar_partii)
    
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
        
        # Sprawdzamy czy mamy wstępnie pobrany partia
        partia = self.pamięć.pobierz_wstępnie_pobrany_batch()
        if partia is None:
            # Jeśli nie, pobieramy nowy
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
        # Przycinanie gradientów dla stabilności (szczególnie ważne dla GPU)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Aktualizacja modelu docelowego co określoną liczbę kroków
        if self.wykonane_kroki % self.częstotliwość_aktualizacji_docelowej == 0:
            self.aktualizuj_model_docelowy()
            
        # Zmniejszanie wartości epsilon w czasie
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.spadek_epsilon
        
        # Wstępne pobranie danych do kolejnego uczenia
        self.pamięć.pobierz_wstępnie_batch(self.rozmiar_partii)
        
        return strata.item()
    
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
            'wykonane_kroki': self.wykonane_kroki,
            'device_type': self.device.type,
            'UŻYJ_FLOAT16': UŻYJ_FLOAT16
        }
        torch.save(stan_modelu, file_name)
        print(f"Model został zapisany do {file_name}")

    def wczytaj(self, file_name='models/dqn_model.pth'):
        """
        Wczytuje model z pliku.
        
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
            
            # Wczytywanie stanu modelu
            self.model.load_state_dict(punkt_kontrolny['model_state_dict'])
            self.optimizer.load_state_dict(punkt_kontrolny['stan_optymalizatora'])
            self.epsilon = punkt_kontrolny['epsilon']
            self.gamma = punkt_kontrolny['gamma']
            self.rozmiar_stanu = punkt_kontrolny['rozmiar_stanu']
            self.rozmiar_akcji = punkt_kontrolny['rozmiar_akcji']
            
            if 'wykonane_kroki' in punkt_kontrolny:
                self.wykonane_kroki = punkt_kontrolny['wykonane_kroki']
            
            # Obsługa różnych typów urządzeń
            wczytane_urządzenie = punkt_kontrolny.get('device_type', 'cpu')
            if wczytane_urządzenie != self.device.type:
                print(f"UWAGA: Model został wytrenowany na {wczytane_urządzenie}, a obecnie używasz {self.device.type}.")
                
            # Obsługa różnych precyzji
            wczytane_float16 = punkt_kontrolny.get('UŻYJ_FLOAT16', False)
            if wczytane_float16 != UŻYJ_FLOAT16 and self.device.type == 'cuda':
                print(f"UWAGA: Model używał float16={wczytane_float16}, a obecnie float16={UŻYJ_FLOAT16}.")
                if UŻYJ_FLOAT16 and not wczytane_float16:
                    print("Konwertuję model do half precision (float16)...")
                    self.model = self.model.half()
                elif not UŻYJ_FLOAT16 and wczytane_float16:
                    print("Konwertuję model do full precision (float32)...")
                    self.model = self.model.float()
            
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
    def continue_training(self, train_function, game_params, n_episodes=100, save_interval=10, 
                      force_device=None, update_learning_params=False, **train_kwargs):
        """
        Kontynuuje trening istniejącego modelu z możliwością zmiany urządzenia obliczeniowego.
    
        Args:
        train_function: Funkcja trenująca, która będzie używana (train_hybrid lub train_cpu_only)
        game_params: Parametry dla gry
        n_episodes: Liczba epizodów do kontynuacji treningu
        save_interval: Co ile epizodów zapisywać model
        force_device: Jeśli podano, wymusza użycie określonego urządzenia ('cpu' lub 'cuda')
        update_learning_params: Czy aktualizować parametry uczenia (gamma, epsilon, itp.)
        **train_kwargs: Dodatkowe parametry dla funkcji trenującej
    
        Returns:
        tuple: Para (wyniki, historia epsilon) z kontynuowanego treningu
     """
    # Zapamiętaj oryginalne urządzenie
        original_device = self.device
    
    # Obsługa wymuszonego urządzenia
        if force_device:
            if force_device == 'cuda' and not torch.cuda.is_available():
                print("UWAGA: GPU nie jest dostępne. Używam CPU.")
                force_device = 'cpu'
        
            if force_device in ['cpu', 'cuda']:
                new_device = torch.device(force_device)
                if new_device != self.device:
                    print(f"Przenoszenie modelu z {self.device} na {new_device}...")
                
                    # Przenieś modele na nowe urządzenie
                    self.device = new_device
                    self.model = self.model.to(new_device)
                    self.model_docelowy = self.model_docelowy.to(new_device)
                
                # Obsługa precyzji dla GPU
                    if new_device.type == 'cuda' and UŻYJ_FLOAT16:
                        print("Konwertuję model do half precision (float16)...")
                        self.model = self.model.half()
                        self.model_docelowy = self.model_docelowy.half()
                        self.typ_danych = torch.float16
                    elif new_device.type == 'cpu' and self.typ_danych == torch.float16:
                        print("Konwertuję model do full precision (float32)...")
                        self.model = self.model.float()
                        self.model_docelowy = self.model_docelowy.float()
                        self.typ_danych = torch.float32
                    
                    # Zaktualizuj optymalizator
                    if new_device.type == 'cuda':
                        współczynnik_uczenia = self.optimizer.param_groups[0]['lr']
                        self.optimizer = optim.AdamW(self.model.parameters(), współczynnik_uczenia=współczynnik_uczenia, weight_decay=1e-5)
                    else:
                        współczynnik_uczenia = self.optimizer.param_groups[0]['lr']
                        self.optimizer = optim.Adam(self.model.parameters(), współczynnik_uczenia=współczynnik_uczenia)
    
        # Aktualizacja parametrów uczenia, jeśli wymagane
        if update_learning_params:
            print("Aktualizuję parametry uczenia...")
            # Tutaj można dodać dialog z użytkownikiem o parametrach lub wczytać je z konfiguracji
            self.gamma = float(input(f"Podaj współczynnik dyskontowania (obecny: {self.gamma}): ") or self.gamma)
            self.epsilon = float(input(f"Podaj początkowy współczynnik eksploracji (obecny: {self.epsilon}): ") or self.epsilon)
            self.epsilon_min = float(input(f"Podaj minimalny współczynnik eksploracji (obecny: {self.epsilon_min}): ") or self.epsilon_min)
            self.spadek_epsilon = float(input(f"Podaj współczynnik zmniejszania eksploracji (obecny: {self.spadek_epsilon}): ") or self.spadek_epsilon)
            self.rozmiar_partii = int(input(f"Podaj rozmiar batcha (obecny: {self.rozmiar_partii}): ") or self.rozmiar_partii)
        
        # Dostosuj rozmiar batcha do urządzenia
        if self.device.type == 'cuda':
            self.rozmiar_partii = max(self.rozmiar_partii, 128)  # Minimum 128 dla GPU
        else:
            self.rozmiar_partii = min(self.rozmiar_partii, 64)  # Maximum 64 dla CPU
    
        print(f"Kontynuuję trening na urządzeniu: {self.device}")
        print(f"Gamma: {self.gamma}, Epsilon: {self.epsilon}, partia size: {self.rozmiar_partii}")
        print(f"Trening przez {n_episodes} epizodów...")
    
        # Uruchom funkcję trenującą
        try:
            scores, eps_history = train_function(self, game_params, 
                                                n_episodes=n_episodes, 
                                                save_interval=save_interval, 
                                                **train_kwargs)
        
            print(f"Kontynuacja treningu zakończona. Model zapisany.")
            return scores, eps_history
        
        except Exception as e:
            print(f"Błąd podczas kontynuacji treningu: {e}")
            # W razie błędu przywróć oryginalne urządzenie
            if self.device != original_device:
                print(f"Przywracam model na oryginalne urządzenie: {original_device}")
                self.device = original_device
                self.model = self.model.to(original_device)
                self.model_docelowy = self.model_docelowy.to(original_device)
            raise e
    