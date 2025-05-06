"""
Główny moduł programu Snake AI z PyTorch.
"""
import torch
import os
import pygame
from snake_game import SnakeGame
from model import QNetwork
from agent import DQNAgent
from training import train_hybrid, train_cpu_only, plot_results, test_agent
from config import (
    USE_GPU, USE_FLOAT16, CPU_THREAD_COUNT, WINDOW_WIDTH, 
    WINDOW_HEIGHT, BLOCK_SIZE, GAME_SPEED, HIDDEN_SIZE, MODELS_DIR
)


def setup_directories():
    """Tworzy potrzebne katalogi, jeśli nie istnieją."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Utworzono katalog '{MODELS_DIR}' do przechowywania modeli.")
    
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Utworzono katalog 'results' do przechowywania wyników.")


def get_available_models():
    """Zwraca listę dostępnych modeli."""
    if not os.path.exists(MODELS_DIR):
        return []
    
    return [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]


def main():
    """Główna funkcja programu."""
    # Inicjalizacja Pygame
    pygame.init()
    
    # Utworzenie katalogów
    setup_directories()
    
    # Inicjalizacja gry
    game = SnakeGame()
    
    # Parametry agenta
    state_size = 11  # Liczba cech w reprezentacji stanu
    action_size = 3  # Liczba możliwych akcji (prosto, w prawo, w lewo)
    
    # Parametry dla zrównoleglonego uczenia
    game_params = {
        'width': WINDOW_WIDTH,
        'height': WINDOW_HEIGHT,
        'block_size': BLOCK_SIZE
    }
    
    # Menu wyboru
    print("\n===== Snake AI z Deep Q-Network (PyTorch) =====")
    print("Wersja zoptymalizowana dla CPU i GPU")
    print(f"Używanie GPU: {'TAK' if USE_GPU else 'NIE'}")
    print(f"Precyzja float16: {'TAK' if USE_FLOAT16 and USE_GPU else 'NIE'}")
    print(f"Liczba wątków CPU: {CPU_THREAD_COUNT}")
    print("\n1. Wczytaj istniejący model")
    print("2. Trenuj nowy model (tryb CPU)")
    print("3. Trenuj nowy model (tryb hybrydowy - zoptymalizowany)")
    print("4. Zmień ustawienia sprzętowe")
    print("5. Kontynuuj trening istniejącego modelu")
    print("6. Wyjście")
    print("===============================================")
    
    wybor = input("\nWybierz opcję (1-5): ")
    
    if wybor == '1':
        # Wczytywanie istniejącego modelu
        print("\nDostępne modele:")
        
        # Inicjalizacja agenta DQN
        agent = DQNAgent(state_size, action_size, HIDDEN_SIZE)
        
        # Wyświetlenie listy dostępnych modeli
        models = get_available_models()
        
        if models:
            for i, model in enumerate(models):
                print(f"{i+1}. {model}")
            
            model_idx = input("\nWybierz numer modelu (lub naciśnij Enter, aby wpisać własną nazwę pliku): ")
            
            if model_idx.isdigit() and 1 <= int(model_idx) <= len(models):
                file_name = os.path.join(MODELS_DIR, models[int(model_idx)-1])
            else:
                file_name = input("Podaj ścieżkę do pliku modelu: ")
            
            if agent.load(file_name):
                # Testowanie modelu
                n_games = int(input("\nPodaj liczbę gier testowych: ") or "5")
                delay = int(input("Opóźnienie między krokami (ms, 0-500): ") or "100")
                test_agent(agent, game, n_games, delay)
        else:
            print("Brak dostępnych modeli.")
            if input("Czy chcesz trenować nowy model? (t/n): ").lower() == 't':
                # Trenowanie agenta
                n_episodes = int(input("\nPodaj liczbę epizodów treningu: ") or "1000")
                save_interval = int(input("Co ile epizodów zapisywać model: ") or "100")
                
                scores, eps_history = train_hybrid(agent, game_params, n_episodes, save_interval=save_interval)
                
                # Wizualizacja wyników treningu
                plot_results(scores, eps_history)
                
                # Test wytrenowanego agenta
                n_games = int(input("\nPodaj liczbę gier testowych: ") or "5")
                delay = int(input("Opóźnienie między krokami (ms, 0-500): ") or "100")
                test_agent(agent, game, n_games, delay)
    
    elif wybor == '2':
        # Trenowanie nowego modelu (tylko CPU)
        # Wyłączamy GPU dla tego trybu, nawet jeśli jest dostępne
        import config
        old_use_gpu = config.USE_GPU
        config.USE_GPU = False
        
        n_episodes = int(input("\nPodaj liczbę epizodów treningu: ") or "1000")
        save_interval = int(input("Co ile epizodów zapisywać model: ") or "100")
        
        # Inicjalizacja agenta DQN (wymuszamy CPU)
        agent = DQNAgent(state_size, action_size, HIDDEN_SIZE)
        
        print("\nRozpoczynanie treningu na CPU...")
        print("Ten tryb używa pojedynczej gry i wizualizacji.")
        
        # Trenowanie agenta
        scores, eps_history = train_cpu_only(agent, game, n_episodes, save_interval=save_interval)
        
        # Przywracamy poprzednie ustawienie GPU
        config.USE_GPU = old_use_gpu
        
        # Wizualizacja wyników treningu
        plot_results(scores, eps_history)
        
        # Test wytrenowanego agenta
        n_games = int(input("\nPodaj liczbę gier testowych: ") or "5")
        delay = int(input("Opóźnienie między krokami (ms, 0-500): ") or "100")
        test_agent(agent, game, n_games, delay)
    
    elif wybor == '3':
        # Trenowanie nowego modelu (tryb hybrydowy - zoptymalizowany)
        n_episodes = int(input("\nPodaj liczbę epizodów treningu: ") or "1000")
        save_interval = int(input("Co ile epizodów zapisywać model: ") or "100")
        n_parallel = int(input(f"Podaj liczbę równoległych gier (zalecane: {CPU_THREAD_COUNT}-16): ") or str(CPU_THREAD_COUNT))
        
        # Inicjalizacja agenta DQN
        agent = DQNAgent(state_size, action_size, HIDDEN_SIZE)
        
        print(f"\nRozpoczynanie treningu hybrydowego z {n_parallel} równoległymi grami...")
        print("Ten tryb nie wyświetla gry, co znacznie przyspiesza uczenie.")
        if USE_GPU:
            print("GPU zostanie użyte do trenowania sieci neuronowej.")
            print("Doświadczenia będą zbierane na CPU dla maksymalnej wydajności.")
        else:
            print("Trening będzie prowadzony wyłącznie na CPU.")
        
        # Trenowanie agenta w trybie hybrydowym
        scores, eps_history = train_hybrid(agent, game_params, n_episodes, save_interval=save_interval, n_parallel=n_parallel)
        
        # Wizualizacja wyników treningu
        plot_results(scores, eps_history)
        
        # Test wytrenowanego agenta
        n_games = int(input("\nPodaj liczbę gier testowych: ") or "5")
        delay = int(input("Opóźnienie między krokami (ms, 0-500): ") or "100")
        test_agent(agent, game, n_games, delay)
    
    elif wybor == '4':
        # Zmiana ustawień sprzętowych
        import config
        
        print("\n--- Aktualne ustawienia ---")
        print(f"Używanie GPU: {'TAK' if config.USE_GPU else 'NIE'}")
        print(f"Precyzja float16: {'TAK' if config.USE_FLOAT16 and config.USE_GPU else 'NIE'}")
        print(f"Liczba wątków CPU: {config.CPU_THREAD_COUNT}")
        
        if torch.cuda.is_available():
            use_gpu = input("\nCzy używać GPU do treningu? (t/n): ").lower()
            config.USE_GPU = use_gpu == 't' or use_gpu == 'tak'
            
            if config.USE_GPU:
                use_fp16 = input("Czy używać niższej precyzji (float16) dla szybszego treningu? (t/n): ").lower()
                config.USE_FLOAT16 = use_fp16 == 't' or use_fp16 == 'tak'
        else:
            print("\nGPU nie jest dostępne na tym komputerze.")
            config.USE_GPU = False
            config.USE_FLOAT16 = False
        
        cpu_threads = input(f"Podaj liczbę wątków CPU (aktualna: {config.CPU_THREAD_COUNT}): ")
        if cpu_threads.isdigit() and int(cpu_threads) > 0:
            config.CPU_THREAD_COUNT = int(cpu_threads)
        
        speed = input(f"Podaj szybkość gry [10-60] (aktualna: {config.GAME_SPEED}): ")
        if speed.isdigit() and 10 <= int(speed) <= 60:
            config.GAME_SPEED = int(speed)
        
        print("\n--- Nowe ustawienia ---")
        print(f"Używanie GPU: {'TAK' if config.USE_GPU else 'NIE'}")
        print(f"Precyzja float16: {'TAK' if config.USE_FLOAT16 and config.USE_GPU else 'NIE'}")
        print(f"Liczba wątków CPU: {config.CPU_THREAD_COUNT}")
        print(f"Szybkość gry: {config.GAME_SPEED}")
        
        # Powrót do menu głównego
        input("\nNaciśnij Enter, aby wrócić do menu głównego...")
        main()
    
    elif wybor == '5':
        # Kontynuacja treningu istniejącego modelu
        print("\nDostępne modele:")
    
        # Inicjalizacja agenta DQN
        agent = DQNAgent(state_size, action_size, HIDDEN_SIZE)
    
        # Wyświetlenie listy dostępnych modeli
        models = get_available_models()
    
        if models:
            for i, model in enumerate(models):
                print(f"{i+1}. {model}")
        
            model_idx = input("\nWybierz numer modelu (lub naciśnij Enter, aby wpisać własną nazwę pliku): ")
        
            if model_idx.isdigit() and 1 <= int(model_idx) <= len(models):
                file_name = os.path.join(MODELS_DIR, models[int(model_idx)-1])
            else:
                file_name = input("Podaj ścieżkę do pliku modelu: ")
        
            if agent.load(file_name):
                # Wybór trybu treningu
                print("\nWybierz tryb kontynuacji treningu:")
                print("1. Tryb CPU")
                print("2. Tryb hybrydowy (GPU jeśli dostępne)")
                train_mode = input("Wybierz tryb (1-2): ")
            
                # Parametry kontynuacji treningu
                n_episodes = int(input("\nPodaj liczbę epizodów treningu: ") or "100")
                save_interval = int(input("Co ile epizodów zapisywać model: ") or "10")
                force_device = None
            
                # Wybór urządzenia, jeśli użytkownik chce zmienić
                if input("Czy chcesz wymusić konkretne urządzenie? (t/n): ").lower() == 't':
                    if torch.cuda.is_available():
                        device_wybor = input("Wybierz urządzenie (cpu/gpu): ").lower()
                        if device_wybor in ['gpu', 'cuda']:
                            force_device = 'cuda'
                        else:
                            force_device = 'cpu'
                    else:
                        print("GPU nie jest dostępne. Używam CPU.")
                        force_device = 'cpu'
            
                # Aktualizacja parametrów uczenia
                update_params = input("Czy chcesz zaktualizować parametry uczenia? (t/n): ").lower() == 't'
            
                # Kontynuacja treningu
                if train_mode == '1':
                    # Tryb CPU
                    train_function = train_cpu_only
                    scores, eps_history = agent.continue_training(
                        train_function, game, n_episodes=n_episodes, 
                        save_interval=save_interval, force_device=force_device,
                        update_learning_params=update_params
                    )
                else:
                    # Tryb hybrydowy
                    n_parallel = int(input(f"Podaj liczbę równoległych gier (zalecane: {CPU_THREAD_COUNT}-16): ") or str(CPU_THREAD_COUNT))
                    train_function = train_hybrid
                    scores, eps_history = agent.continue_training(
                        train_function, game_params, n_episodes=n_episodes, 
                        save_interval=save_interval, force_device=force_device,
                        update_learning_params=update_params, n_parallel=n_parallel
                    )
            
                # Wizualizacja wyników treningu
                plot_results(scores, eps_history)
            
                # Test wytrenowanego agenta
                n_games = int(input("\nPodaj liczbę gier testowych: ") or "5")
                delay = int(input("Opóźnienie między krokami (ms, 0-500): ") or "100")
                test_agent(agent, game, n_games, delay)
        else:
            print("Brak dostępnych modeli. Najpierw wytrenuj jakiś model.")

    elif wybor == '6':
        print("Wyjście z programu.")
    
    else:
        print("Nieprawidłowy wybór. Wyjście z programu.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram przerwany przez użytkownika.")
    except Exception as e:
        print(f"\nWystąpił błąd: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        print("Do widzenia!")