"""
Główny moduł programu Snake AI z PyTorch.
"""
import torch
import os
import pygame
from snake_game import SnakeGame
from model import QNetwork
from agent import DQNAgent
from training import trenuj_hybrydowo, trenuj_tylko_cpu, rysuj_wyniki, testuj_agenta
from config import (
    UŻYJ_GPU, UŻYJ_FLOAT16, LICZBA_WĄTKÓW_CPU, SZEROKOŚĆ_OKNA, 
    WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU, PRĘDKOŚĆ_GRY, ROZMIAR_UKRYTY, KATALOG_MODELI
)


def przygotuj_katalogi():
    """Tworzy potrzebne katalogi, jeśli nie istnieją."""
    if not os.path.exists(KATALOG_MODELI):
        os.makedirs(KATALOG_MODELI)
        print(f"Utworzono katalog '{KATALOG_MODELI}' do przechowywania modeli.")
    
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Utworzono katalog 'results' do przechowywania wyników.")


def pobierz_dostępne_modele():
    """Zwraca listę dostępnych modeli."""
    if not os.path.exists(KATALOG_MODELI):
        return []
    
    return [f for f in os.listdir(KATALOG_MODELI) if f.endswith('.pth')]


def main():
    """Główna funkcja programu."""
    # Inicjalizacja Pygame
    pygame.init()
    
    # Utworzenie katalogów
    przygotuj_katalogi()
    
    # Inicjalizacja gry
    game = SnakeGame()
    
    # Parametry agenta
    rozmiar_stanu = 11  # Liczba cech w reprezentacji stanu
    rozmiar_akcji = 3  # Liczba możliwych akcji (prosto, w prawo, w lewo)
    
    # Parametry dla zrównoleglonego uczenia
    parametry_gry = {
        'szerokość': SZEROKOŚĆ_OKNA,
        'wysokość': WYSOKOŚĆ_OKNA,
        'rozmiar_bloku': ROZMIAR_BLOKU
    }
    
    # Menu wyboru
    print("\n===== Snake AI z Deep Q-Network (PyTorch) =====")
    print("Wersja zoptymalizowana dla CPU i GPU")
    print(f"Używanie GPU: {'TAK' if UŻYJ_GPU else 'NIE'}")
    print(f"Precyzja float16: {'TAK' if UŻYJ_FLOAT16 and UŻYJ_GPU else 'NIE'}")
    print(f"Liczba wątków CPU: {LICZBA_WĄTKÓW_CPU}")
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
        agent = DQNAgent(rozmiar_stanu, rozmiar_akcji, ROZMIAR_UKRYTY)
        
        # Wyświetlenie listy dostępnych modeli
        modele = pobierz_dostępne_modele()
        
        if modele:
            for i, model in enumerate(modele):
                print(f"{i+1}. {model}")
            
            indeks_modelu = input("\nWybierz numer modelu (lub naciśnij Enter, aby wpisać własną nazwę pliku): ")
            
            if indeks_modelu.isdigit() and 1 <= int(indeks_modelu) <= len(modele):
                nazwa_pliku = os.path.join(KATALOG_MODELI, modele[int(indeks_modelu)-1])
            else:
                nazwa_pliku = input("Podaj ścieżkę do pliku modelu: ")
            
            if agent.wczytaj(nazwa_pliku):
                # Testowanie modelu
                liczba_gier = int(input("\nPodaj liczbę gier testowych: ") or "5")
                opóźnienie = int(input("Opóźnienie między krokami (ms, 0-500): ") or "100")
                testuj_agenta(agent, game, liczba_gier, opóźnienie)
        else:
            print("Brak dostępnych modeli.")
            if input("Czy chcesz trenować nowy model? (t/n): ").lower() == 't':
                # Trenowanie agenta
                liczba_epizodów = int(input("\nPodaj liczbę epizodów treningu: ") or "1000")
                interwał_zapisu = int(input("Co ile epizodów zapisywać model: ") or "100")
                
                wyniki, historia_ep = trenuj_hybrydowo(agent, parametry_gry, liczba_epizodów, interwał_zapisu=interwał_zapisu)
                
                # Wizualizacja wyników treningu
                rysuj_wyniki(wyniki, historia_ep)
                
                # Test wytrenowanego agenta
                liczba_gier = int(input("\nPodaj liczbę gier testowych: ") or "5")
                opóźnienie = int(input("Opóźnienie między krokami (ms, 0-500): ") or "100")
                testuj_agenta(agent, game, liczba_gier, opóźnienie)
    
    elif wybor == '2':
        # Trenowanie nowego modelu (tylko CPU)
        # Wyłączamy GPU dla tego trybu, nawet jeśli jest dostępne
        import config
        stare_użyj_gpu = config.UŻYJ_GPU
        config.UŻYJ_GPU = False
        
        liczba_epizodów = int(input("\nPodaj liczbę epizodów treningu: ") or "1000")
        interwał_zapisu = int(input("Co ile epizodów zapisywać model: ") or "100")
        
        # Inicjalizacja agenta DQN (wymuszamy CPU)
        agent = DQNAgent(rozmiar_stanu, rozmiar_akcji, ROZMIAR_UKRYTY)
        
        print("\nRozpoczynanie treningu na CPU...")
        print("Ten tryb używa pojedynczej gry i wizualizacji.")
        
        # Trenowanie agenta
        wyniki, historia_ep = trenuj_tylko_cpu(agent, game, liczba_epizodów, interwał_zapisu=interwał_zapisu)
        
        # Przywracamy poprzednie ustawienie GPU
        config.UŻYJ_GPU = stare_użyj_gpu
        
        # Wizualizacja wyników treningu
        rysuj_wyniki(wyniki, historia_ep)
        
        # Test wytrenowanego agenta
        liczba_gier = int(input("\nPodaj liczbę gier testowych: ") or "5")
        opóźnienie = int(input("Opóźnienie między krokami (ms, 0-500): ") or "100")
        testuj_agenta(agent, game, liczba_gier, opóźnienie)
    
    elif wybor == '3':
        # Trenowanie nowego modelu (tryb hybrydowy - zoptymalizowany)
        liczba_epizodów = int(input("\nPodaj liczbę epizodów treningu: ") or "1000")
        interwał_zapisu = int(input("Co ile epizodów zapisywać model: ") or "100")
        liczba_równoległych = int(input(f"Podaj liczbę równoległych gier (zalecane: {LICZBA_WĄTKÓW_CPU}-16): ") or str(LICZBA_WĄTKÓW_CPU))
        
        # Inicjalizacja agenta DQN
        agent = DQNAgent(rozmiar_stanu, rozmiar_akcji, ROZMIAR_UKRYTY)
        
        print(f"\nRozpoczynanie treningu hybrydowego z {liczba_równoległych} równoległymi grami...")
        print("Ten tryb nie wyświetla gry, co znacznie przyspiesza uczenie.")
        if UŻYJ_GPU:
            print("GPU zostanie użyte do trenowania sieci neuronowej.")
            print("Doświadczenia będą zbierane na CPU dla maksymalnej wydajności.")
        else:
            print("Trening będzie prowadzony wyłącznie na CPU.")
        
        # Trenowanie agenta w trybie hybrydowym
        wyniki, historia_ep = trenuj_hybrydowo(agent, parametry_gry, liczba_epizodów, interwał_zapisu=interwał_zapisu, liczba_równoległych=liczba_równoległych)
        
        # Wizualizacja wyników treningu
        rysuj_wyniki(wyniki, historia_ep)
        
        # Test wytrenowanego agenta
        liczba_gier = int(input("\nPodaj liczbę gier testowych: ") or "5")
        opóźnienie = int(input("Opóźnienie między krokami (ms, 0-500): ") or "100")
        testuj_agenta(agent, game, liczba_gier, opóźnienie)
    
    elif wybor == '4':
        # Zmiana ustawień sprzętowych
        import config
        
        print("\n--- Aktualne ustawienia ---")
        print(f"Używanie GPU: {'TAK' if config.UŻYJ_GPU else 'NIE'}")
        print(f"Precyzja float16: {'TAK' if config.UŻYJ_FLOAT16 and config.UŻYJ_GPU else 'NIE'}")
        print(f"Liczba wątków CPU: {config.LICZBA_WĄTKÓW_CPU}")
        
        if torch.cuda.is_available():
            użyj_gpu = input("\nCzy używać GPU do treningu? (t/n): ").lower()
            config.UŻYJ_GPU = użyj_gpu == 't' or użyj_gpu == 'tak'
            
            if config.UŻYJ_GPU:
                use_fp16 = input("Czy używać niższej precyzji (float16) dla szybszego treningu? (t/n): ").lower()
                config.UŻYJ_FLOAT16 = use_fp16 == 't' or use_fp16 == 'tak'
        else:
            print("\nGPU nie jest dostępne na tym komputerze.")
            config.UŻYJ_GPU = False
            config.UŻYJ_FLOAT16 = False
        
        cpu_threads = input(f"Podaj liczbę wątków CPU (aktualna: {config.LICZBA_WĄTKÓW_CPU}): ")
        if cpu_threads.isdigit() and int(cpu_threads) > 0:
            config.LICZBA_WĄTKÓW_CPU = int(cpu_threads)
        
        speed = input(f"Podaj szybkość gry [10-60] (aktualna: {config.PRĘDKOŚĆ_GRY}): ")
        if speed.isdigit() and 10 <= int(speed) <= 60:
            config.PRĘDKOŚĆ_GRY = int(speed)
        
        print("\n--- Nowe ustawienia ---")
        print(f"Używanie GPU: {'TAK' if config.UŻYJ_GPU else 'NIE'}")
        print(f"Precyzja float16: {'TAK' if config.UŻYJ_FLOAT16 and config.UŻYJ_GPU else 'NIE'}")
        print(f"Liczba wątków CPU: {config.LICZBA_WĄTKÓW_CPU}")
        print(f"Szybkość gry: {config.PRĘDKOŚĆ_GRY}")
        
        # Powrót do menu głównego
        input("\nNaciśnij Enter, aby wrócić do menu głównego...")
        main()
    
    elif wybor == '5':
        # Kontynuacja treningu istniejącego modelu
        print("\nDostępne modele:")
    
        # Inicjalizacja agenta DQN
        agent = DQNAgent(rozmiar_stanu, rozmiar_akcji, ROZMIAR_UKRYTY)
    
        # Wyświetlenie listy dostępnych modeli
        modele = pobierz_dostępne_modele()
    
        if modele:
            for i, model in enumerate(modele):
                print(f"{i+1}. {model}")
        
            indeks_modelu = input("\nWybierz numer modelu (lub naciśnij Enter, aby wpisać własną nazwę pliku): ")
        
            if indeks_modelu.isdigit() and 1 <= int(indeks_modelu) <= len(modele):
                nazwa_pliku = os.path.join(KATALOG_MODELI, modele[int(indeks_modelu)-1])
            else:
                nazwa_pliku = input("Podaj ścieżkę do pliku modelu: ")
        
            if agent.wczytaj(nazwa_pliku):
                # Wybór trybu treningu
                print("\nWybierz tryb kontynuacji treningu:")
                print("1. Tryb CPU")
                print("2. Tryb hybrydowy (GPU jeśli dostępne)")
                train_mode = input("Wybierz tryb (1-2): ")
            
                # Parametry kontynuacji treningu
                liczba_epizodów = int(input("\nPodaj liczbę epizodów treningu: ") or "100")
                interwał_zapisu = int(input("Co ile epizodów zapisywać model: ") or "10")
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
                    train_function = trenuj_tylko_cpu
                    wyniki, historia_ep = agent.continue_training(
                        train_function, game, liczba_epizodów=liczba_epizodów, 
                        interwał_zapisu=interwał_zapisu, force_device=force_device,
                        update_learning_params=update_params
                    )
                else:
                    # Tryb hybrydowy
                    liczba_równoległych = int(input(f"Podaj liczbę równoległych gier (zalecane: {LICZBA_WĄTKÓW_CPU}-16): ") or str(LICZBA_WĄTKÓW_CPU))
                    train_function = trenuj_hybrydowo
                    wyniki, historia_ep = agent.continue_training(
                        train_function, parametry_gry, liczba_epizodów=liczba_epizodów, 
                        interwał_zapisu=interwał_zapisu, force_device=force_device,
                        update_learning_params=update_params, liczba_równoległych=liczba_równoległych
                    )
            
                # Wizualizacja wyników treningu
                rysuj_wyniki(wyniki, historia_ep)
            
                # Test wytrenowanego agenta
                liczba_gier = int(input("\nPodaj liczbę gier testowych: ") or "5")
                opóźnienie = int(input("Opóźnienie między krokami (ms, 0-500): ") or "100")
                testuj_agenta(agent, game, liczba_gier, opóźnienie)
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