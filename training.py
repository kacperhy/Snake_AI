"""
Moduł zawierający funkcje do trenowania agenta.
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from snake_game import SnakeGameSimple
from config import USE_GPU
import pygame
from agent import DQNAgent
from typing import List, Tuple
from typing import Dict
from typing import Any
from typing import Optional
from typing import Union
from typing import Callable
from typing import Type
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union        

def run_episode(agent, game_params, max_steps=10000):
    """
    Przeprowadza pojedynczy epizod i zwraca zebrane doświadczenia.
    
    Args:
        agent (DQNAgent): Agent podejmujący decyzje.
        game_params (dict): Parametry do inicjalizacji gry.
        max_steps (int): Maksymalna liczba kroków w epizodzie.
        
    Returns:
        tuple: Trójka (doświadczenia, całkowita nagroda, wynik).
    """
    # Inicjalizacja gry bez interfejsu graficznego
    game = SnakeGameSimple(**game_params)
    state = game.reset()
    done = False
    step_count = 0
    total_reward = 0
    experiences = []
    
    while not done and step_count < max_steps:
        # Wybór akcji przez agenta
        action = agent.get_action(state)
        
        # Wykonanie akcji w środowisku
        next_state, reward, done, score = game.step(action)
        
        # Zapisanie doświadczenia do późniejszego użycia
        experiences.append((state, action, reward, next_state, done))
        
        # Przejście do nowego stanu
        state = next_state
        total_reward += reward
        step_count += 1
    
    return experiences, total_reward, score


def train_hybrid(agent, game_params, n_episodes=1000, target_update=10, save_interval=100, n_parallel=4):
    """
    Trenuje agenta z wykorzystaniem zarówno CPU jak i GPU dla maksymalnej wydajności.
    
    Args:
        agent (DQNAgent): Agent do trenowania.
        game_params (dict): Parametry do inicjalizacji gry.
        n_episodes (int): Liczba epizodów treningu.
        target_update (int): Co ile epizodów aktualizować model docelowy.
        save_interval (int): Co ile epizodów zapisywać model.
        n_parallel (int): Liczba równoległych gier.
        
    Returns:
        tuple: Para (wyniki, historia epsilon).
    """
    scores = []
    eps_history = []
    best_score = 0
    avg_loss = 0
    
    # Określamy tryb treningu na podstawie dostępności GPU
    if USE_GPU:
        print(f"Trening hybrydowy: zbieranie doświadczeń na CPU, trening na GPU.")
    else:
        print(f"Trening na CPU z {n_parallel} równoległymi grami.")
    
    # Liczba grup epizodów do przeprowadzenia
    n_chunks = (n_episodes + n_parallel - 1) // n_parallel
    
    with tqdm(total=n_episodes, desc="Trening") as pbar:
        for chunk in range(n_chunks):
            # Rzeczywista liczba epizodów w tej grupie
            actual_n = min(n_parallel, n_episodes - chunk * n_parallel)
            
            # Uruchomienie wielu epizodów równolegle (symulacja wielowątkowości)
            all_experiences = []
            chunk_scores = []
            
            # Wykonujemy n_parallel gier "równolegle"
            for _ in range(actual_n):
                experiences, _, score = run_episode(agent, game_params)
                all_experiences.extend(experiences)
                chunk_scores.append(score)
            
            # Aktualizacja statystyk
            scores.extend(chunk_scores)
            eps_history.append(agent.epsilon)
            
            # Dodanie wszystkich doświadczeń do pamięci agenta
            for exp in all_experiences:
                agent.remember(*exp)
            
            # Uczenie agenta na zebranych doświadczeniach
            # W trybie GPU trenujemy intensywniej
            train_iterations = min(len(all_experiences), 2000 if USE_GPU else 1000)
            
            total_loss = 0
            loss_count = 0
            for _ in range(train_iterations):
                loss = agent.learn()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
            
            if loss_count > 0:
                avg_loss = total_loss / loss_count
            
            # Aktualizacja paska postępu
            pbar.update(actual_n)
            
            # Wyświetlanie postępów co save_interval chunk'ów
            current_episode = (chunk + 1) * n_parallel
            if chunk % (save_interval // max(1, n_parallel)) == 0 or chunk == n_chunks - 1:
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                recent_avg = np.mean(chunk_scores)
                pbar.set_postfix({
                    'Śr.wynik': f'{avg_score:.2f}',
                    'Ost.wynik': f'{recent_avg:.2f}',
                    'Epsilon': f'{agent.epsilon:.4f}',
                    'Loss': f'{avg_loss:.4f}'
                })
                
                # Zapisanie modelu co save_interval epizodów
                if current_episode <= n_episodes:
                    agent.save(f"models/snake_model_episode_{current_episode}.pth")
            
            # Sprawdzenie, czy mamy nowy najlepszy wynik
            max_score = max(chunk_scores) if chunk_scores else 0
            if max_score > best_score:
                best_score = max_score
                agent.save("models/snake_model_best.pth")
                pbar.write(f"Nowy najlepszy wynik: {best_score}! Model zapisany jako 'snake_model_best.pth'")
    
    # Zapisanie ostatecznego modelu
    agent.save("models/snake_model_final.pth")
    print("Trening zakończony. Ostateczny model zapisany jako 'snake_model_final.pth'")
    
    return scores, eps_history


def train_cpu_only(agent, game, n_episodes=1000, save_interval=100):
    """
    Trenuje agenta wyłącznie na CPU, bez zrównoleglenia.
    
    Args:
        agent (DQNAgent): Agent do trenowania.
        game (SnakeGame): Środowisko gry.
        n_episodes (int): Liczba epizodów treningu.
        save_interval (int): Co ile epizodów zapisywać model.
        
    Returns:
        tuple: Para (wyniki, historia epsilon).
    """
    scores = []
    eps_history = []
    best_score = 0
    avg_loss = 0
    
    print("Trening tylko na CPU z pojedynczą grą.")
    
    for e in tqdm(range(n_episodes), desc="Trening"):
        # Resetowanie gry i pobranie stanu początkowego
        state = game.reset()
        done = False
        score = 0
        total_loss = 0
        step_count = 0
        loss_count = 0
        
        while not done:
            # Wybór akcji przez agenta
            action = agent.get_action(state)
            
            # Wykonanie akcji w środowisku
            next_state, reward, done, info = game.step(action)
            
            # Zapisanie doświadczenia w pamięci agenta
            agent.remember(state, action, reward, next_state, done)
            
            # Przejście do nowego stanu
            state = next_state
            
            # Uczenie agenta
            loss = agent.learn()
            if loss is not None:
                total_loss += loss
                loss_count += 1
            
            # Aktualizacja wyniku
            score = info  # info to obecny wynik
            step_count += 1
            
        # Zapisanie wyniku i wartości epsilon dla tego epizodu
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        # Obliczenie średniej straty
        if loss_count > 0:
            avg_loss = total_loss / loss_count
        
        # Wyświetlanie postępów co save_interval epizodów
        if e % save_interval == 0 or e == n_episodes - 1:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            tqdm.write(f"Epizod {e}, Wynik: {score}, Średni wynik: {avg_score:.2f}, "
                      f"Epsilon: {agent.epsilon:.4f}, Loss: {avg_loss:.4f}")
            
            # Zapisanie modelu co save_interval epizodów
            agent.save(f"models/snake_model_episode_{e}.pth")
            
        # Zapisanie najlepszego modelu
        if score > best_score:
            best_score = score
            agent.save("models/snake_model_best.pth")
            tqdm.write(f"Nowy najlepszy wynik: {score}! Model zapisany jako 'snake_model_best.pth'")
    
    # Zapisanie ostatecznego modelu
    agent.save("models/snake_model_final.pth")
    print("Trening zakończony. Ostateczny model zapisany jako 'snake_model_final.pth'")
    
    return scores, eps_history


def plot_results(scores, eps_history):
    """
    Tworzy wykresy wyników treningu.
    
    Args:
        scores (list): Lista wyników z każdego epizodu.
        eps_history (list): Lista wartości epsilon z każdego epizodu.
    """
    plt.figure(figsize=(16, 5))
    
    # Wykres wyników
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.axhline(y=np.mean(scores), color='r', linestyle='--', label=f'Średnia: {np.mean(scores):.2f}')
    plt.xlabel('Epizod')
    plt.ylabel('Wynik')
    plt.title('Wyniki w czasie treningu')
    plt.legend()
    
    # Wykres średniej ruchomej
    plt.subplot(1, 3, 2)
    window_size = min(100, len(scores))
    moving_avg = [np.mean(scores[max(0, i-window_size):i+1]) for i in range(len(scores))]
    plt.plot(moving_avg)
    plt.xlabel('Epizod')
    plt.ylabel('Średnia z ostatnich 100 epizodów')
    plt.title('Średnia ruchoma wyników')
    
    # Wykres zmian epsilon
    plt.subplot(1, 3, 3)
    plt.plot(eps_history)
    plt.xlabel('Epizod')
    plt.ylabel('Epsilon')
    plt.title('Zmiana współczynnika eksploracji')
    
    plt.tight_layout()
    plt.savefig('models/training_results.png')
    print("Wykresy wyników treningu zapisane do 'models/training_results.png'")
    plt.show()


def test_agent(agent, game, n_games=5, delay=100):
    """
    Testuje wytrenowanego agenta.
    
    Args:
        agent (DQNAgent): Wytrenowany agent.
        game (SnakeGame): Środowisko gry.
        n_games (int): Liczba gier testowych.
        delay (int): Opóźnienie między krokami (ms).
    """
    total_score = 0
    max_score = 0
    scores = []
    
    for game_idx in range(n_games):
        state = game.reset()
        done = False
        
        while not done:
            # Agent wybiera akcję bez eksploracji
            state_tensor = torch.tensor(state, dtype=agent.dtype).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.model(state_tensor)
            action = torch.argmax(q_values).item()
            
            # Wykonanie akcji
            state, reward, done, score = game.step(action)
            
            # Opóźnienie, aby można było obserwować grę
            if delay > 0:
                pygame.time.delay(delay)
            
        total_score += score
        scores.append(score)
        max_score = max(max_score, score)
        print(f"Gra {game_idx+1}, Wynik: {score}")
    
    avg_score = total_score / n_games
    print(f"Średni wynik po {n_games} grach: {avg_score:.2f}")
    print(f"Najlepszy wynik: {max_score}")
    print(f"Wszystkie wyniki: {scores}")
    
    return avg_score, max_score, scores
            