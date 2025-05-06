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
from config import USE_GPU, USE_FLOAT16, EXPERIENCE_BUFFER_SIZE

#minimalne wymagania dla GPU
class ExperienceBuffer:
    """
    Bufor doświadczeń z optymalizacjami dla szybkiego przetwarzania.
    
    Atrybuty:
        capacity (int): Maksymalna liczba przechowywanych doświadczeń.
        buffer (deque): Kolejka przechowująca doświadczenia.
        device (torch.device): Urządzenie do obliczeń (CPU/GPU).
        dtype (torch.dtype): Typ danych dla tensorów.
    """
    def __init__(self, capacity=EXPERIENCE_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if USE_GPU else "cpu")
        self.dtype = torch.float16 if USE_FLOAT16 and USE_GPU else torch.float32
        self.prefetched_batch = None
        self.prefetch_lock = threading.Lock()
        self.is_prefetching = False
    
    def push(self, state, action, reward, next_state, done):
        """Dodaje nowe doświadczenie do bufora."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Pobiera losową próbkę z bufora i konwertuje na tensory."""
        if len(self.buffer) < batch_size:
            return None
            
        # Wybór losowej próbki
        batch = random.sample(self.buffer, batch_size)
        
        # Rozpakowanie danych
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Konwersja na tensory (na CPU dla uniknięcia wąskich gardeł transferu)
        states_tensor = torch.tensor(np.array(states), dtype=self.dtype).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=self.dtype).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=self.dtype).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
    
    def prefetch_batch(self, batch_size):
        """Wstępnie pobiera batch w osobnym wątku dla szybszego dostępu."""
        if not self.is_prefetching and len(self.buffer) >= batch_size:
            self.is_prefetching = True
            threading.Thread(target=self._prefetch_batch_thread, args=(batch_size,)).start()
    
    def _prefetch_batch_thread(self, batch_size):
        """Wątek pobierający batch."""
        batch = self.sample(batch_size)
        with self.prefetch_lock:
            self.prefetched_batch = batch
            self.is_prefetching = False
    
    def get_prefetched_batch(self):
        """Pobiera wcześniej pobrany batch lub tworzy nowy jeśli nie ma."""
        with self.prefetch_lock:
            batch = self.prefetched_batch
            self.prefetched_batch = None
        return batch
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agent używający algorytmu Deep Q-Network z optymalizacjami.
    
    Atrybuty:
        state_size (int): Wymiar wektora stanu.
        action_size (int): Liczba możliwych akcji.
        memory (ExperienceBuffer): Bufor przechowujący doświadczenia.
        model (QNetwork): Główna sieć neuronowa.
        target_model (QNetwork): Sieć docelowa do stabilizacji uczenia.
    """
    def __init__(self, state_size, action_size, hidden_size=256, lr=0.0003):
        self.state_size = state_size
        self.action_size = action_size
        
        # Parametry uczenia
        self.gamma = 0.99 # Współczynnik dyskontowania przyszłych nagród
        self.epsilon = 1.0  # Współczynnik eksploracji
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.batch_size = 256 if USE_GPU else 64  # Większy batch na GPU, mniejszy na CPU
        self.update_frequency = 2  # Co ile kroków aktualizować model
        self.steps_done = 0
        self.target_update_freq = 1000  # Co ile kroków aktualizować model docelowy
        
        # Urządzenie (CPU/GPU)
        self.device = torch.device("cuda" if USE_GPU else "cpu")
        self.dtype = torch.float16 if USE_FLOAT16 and USE_GPU else torch.float32
        
        # Bufor doświadczeń
        self.memory = ExperienceBuffer()
        
        # Model sieci neuronowej (policy network)
        self.model = QNetwork(state_size, hidden_size, action_size).to(self.device)
        if USE_FLOAT16 and USE_GPU:
            self.model = self.model.half()  # Konwersja do float16 dla GPU
        
        # Model sieci docelowej (target network)
        self.target_model = QNetwork(state_size, hidden_size, action_size).to(self.device)
        if USE_FLOAT16 and USE_GPU:
            self.target_model = self.target_model.half()
            
        self.update_target_model()
        
        # Optymalizator - dla CPU używamy Adam, dla GPU AdamW (lepszy dla GPU)
        if USE_GPU:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Funkcja straty - Huber jest bardziej stabilna
        self.criterion = nn.SmoothL1Loss()
        
        print(f"Używam urządzenia: {self.device}")
        if self.device.type == 'cuda':
            print(f"Model GPU: {torch.cuda.get_device_name(0)}")
            if USE_FLOAT16:
                print("Używam precyzji float16 dla szybszego treningu")
        
        # Utworzenie katalogu na modele
        if not os.path.exists('models'):
            os.makedirs('models')
            print("Utworzono katalog 'models' do przechowywania modeli.")
    
    def update_target_model(self):
        """Aktualizacja wag modelu docelowego."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Zapisuje doświadczenie w buforze pamięci."""
        self.memory.push(state, action, reward, next_state, done)
        # Wstępne pobranie danych do przyszłego uczenia
        self.memory.prefetch_batch(self.batch_size)
    
    def get_action(self, state):
        """
        Wybiera akcję zgodnie z polityką epsilon-greedy.
        
        Args:
            state (numpy.ndarray): Stan gry.
            
        Returns:
            int: Indeks wybranej akcji.
        """
        # Eksploracja - wybór losowej akcji
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Eksploatacja - wybór akcji o najwyższej wartości Q
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=self.dtype).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
    
    def learn(self):
        """
        Uczenie na podstawie próbki doświadczeń z pamięci.
        
        Returns:
            float or None: Wartość straty jeśli uczenie miało miejsce, None w przeciwnym razie.
        """
        self.steps_done += 1
        
        # Aktualizacja tylko co kilka kroków dla szybszego treningu
        if self.steps_done % self.update_frequency != 0:
            return None
        
        # Sprawdzamy czy mamy wstępnie pobrany batch
        batch = self.memory.get_prefetched_batch()
        if batch is None:
            # Jeśli nie, pobieramy nowy
            batch = self.memory.sample(self.batch_size)
            if batch is None:
                return None  # Za mało doświadczeń
        
        states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = batch
        
        # Obliczanie przewidywanych wartości Q dla obecnych stanów
        current_q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Obliczanie wartości docelowych z podwójnym DQN dla lepszej stabilności
        with torch.no_grad():
            # Podwójny DQN: wybieramy akcje za pomocą sieci głównej
            next_actions = self.model(next_states_tensor).max(1)[1].unsqueeze(1)
            # Ale wartości Q bierzemy z sieci docelowej
            next_q_values = self.target_model(next_states_tensor).gather(1, next_actions).squeeze(1)
            # Obliczamy docelowe wartości Q
            target_q_values = rewards_tensor + self.gamma * next_q_values * (~dones_tensor)
        
        # Obliczanie straty i aktualizacja modelu
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optymalizacja
        self.optimizer.zero_grad()
        loss.backward()
        # Przycinanie gradientów dla stabilności (szczególnie ważne dla GPU)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Aktualizacja modelu docelowego co określoną liczbę kroków
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_model()
            
        # Zmniejszanie wartości epsilon w czasie
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Wstępne pobranie danych do kolejnego uczenia
        self.memory.prefetch_batch(self.batch_size)
        
        return loss.item()
    
    def save(self, file_name='models/dqn_model.pth'):
        """
        Zapisuje model do pliku.
        
        Args:
            file_name (str): Ścieżka do pliku, w którym zostanie zapisany model.
        """
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'steps_done': self.steps_done,
            'device_type': self.device.type,
            'use_float16': USE_FLOAT16
        }
        torch.save(model_state, file_name)
        print(f"Model został zapisany do {file_name}")

    def load(self, file_name='models/dqn_model.pth'):
        """
        Wczytuje model z pliku.
        
        Args:
            file_name (str): Ścieżka do pliku z zapisanym modelem.
            
        Returns:
            bool: True jeśli wczytanie się powiodło, False w przeciwnym razie.
        """
        try:
            checkpoint = torch.load(file_name, map_location=self.device)
            
            # Sprawdzenie zgodności modelu
            if checkpoint['state_size'] != self.state_size or checkpoint['action_size'] != self.action_size:
                print(f"UWAGA: Niezgodność wymiarów modelu! Oczekiwano: {self.state_size}x{self.action_size}, "
                      f"Wczytano: {checkpoint['state_size']}x{checkpoint['action_size']}")
                if input("Czy chcesz kontynuować wczytywanie? (t/n): ").lower() != 't':
                    return False
            
            # Wczytywanie stanu modelu
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.gamma = checkpoint['gamma']
            self.state_size = checkpoint['state_size']
            self.action_size = checkpoint['action_size']
            
            if 'steps_done' in checkpoint:
                self.steps_done = checkpoint['steps_done']
            
            # Obsługa różnych typów urządzeń
            loaded_device = checkpoint.get('device_type', 'cpu')
            if loaded_device != self.device.type:
                print(f"UWAGA: Model został wytrenowany na {loaded_device}, a obecnie używasz {self.device.type}.")
                
            # Obsługa różnych precyzji
            loaded_float16 = checkpoint.get('use_float16', False)
            if loaded_float16 != USE_FLOAT16 and self.device.type == 'cuda':
                print(f"UWAGA: Model używał float16={loaded_float16}, a obecnie float16={USE_FLOAT16}.")
                if USE_FLOAT16 and not loaded_float16:
                    print("Konwertuję model do half precision (float16)...")
                    self.model = self.model.half()
                elif not USE_FLOAT16 and loaded_float16:
                    print("Konwertuję model do full precision (float32)...")
                    self.model = self.model.float()
            
            # Aktualizacja modelu docelowego
            self.update_target_model()
            
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
                    self.target_model = self.target_model.to(new_device)
                
                # Obsługa precyzji dla GPU
                    if new_device.type == 'cuda' and USE_FLOAT16:
                        print("Konwertuję model do half precision (float16)...")
                        self.model = self.model.half()
                        self.target_model = self.target_model.half()
                        self.dtype = torch.float16
                    elif new_device.type == 'cpu' and self.dtype == torch.float16:
                        print("Konwertuję model do full precision (float32)...")
                        self.model = self.model.float()
                        self.target_model = self.target_model.float()
                        self.dtype = torch.float32
                    
                    # Zaktualizuj optymalizator
                    if new_device.type == 'cuda':
                        lr = self.optimizer.param_groups[0]['lr']
                        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
                    else:
                        lr = self.optimizer.param_groups[0]['lr']
                        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
        # Aktualizacja parametrów uczenia, jeśli wymagane
        if update_learning_params:
            print("Aktualizuję parametry uczenia...")
            # Tutaj można dodać dialog z użytkownikiem o parametrach lub wczytać je z konfiguracji
            self.gamma = float(input(f"Podaj współczynnik dyskontowania (obecny: {self.gamma}): ") or self.gamma)
            self.epsilon = float(input(f"Podaj początkowy współczynnik eksploracji (obecny: {self.epsilon}): ") or self.epsilon)
            self.epsilon_min = float(input(f"Podaj minimalny współczynnik eksploracji (obecny: {self.epsilon_min}): ") or self.epsilon_min)
            self.epsilon_decay = float(input(f"Podaj współczynnik zmniejszania eksploracji (obecny: {self.epsilon_decay}): ") or self.epsilon_decay)
            self.batch_size = int(input(f"Podaj rozmiar batcha (obecny: {self.batch_size}): ") or self.batch_size)
        
        # Dostosuj rozmiar batcha do urządzenia
        if self.device.type == 'cuda':
            self.batch_size = max(self.batch_size, 128)  # Minimum 128 dla GPU
        else:
            self.batch_size = min(self.batch_size, 64)  # Maximum 64 dla CPU
    
        print(f"Kontynuuję trening na urządzeniu: {self.device}")
        print(f"Gamma: {self.gamma}, Epsilon: {self.epsilon}, Batch size: {self.batch_size}")
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
                self.target_model = self.target_model.to(original_device)
            raise e
    