"""
Moduł zawierający funkcje do trenowania agenta.
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from snake_game import UproszczonySnake
from config import UŻYJ_GPU
import pygame
from agent import DQNAgent
      

def uruchom_epizod(agent, parametry_gry, maks_kroków=10000):
    """
    Przeprowadza pojedynczy epizod i zwraca zebrane doświadczenia.
    
    Args:
        agent (DQNAgent): Agent podejmujący decyzje.
        parametry_gry (dict): Parametry do inicjalizacji gry.
        maks_kroków (int): Maksymalna liczba kroków w epizodzie.
        
    Returns:
        tuple: Trójka (doświadczenia, całkowita nagroda, wynik).
    """
    # Inicjalizacja gry bez interfejsu graficznego
    game = UproszczonySnake(**parametry_gry)
    stan = game.reset()
    zakończone = False
    liczba_kroków = 0
    łączna_nagroda = 0
    doświadczenia = []
    
    while not zakończone and liczba_kroków < maks_kroków:
        # Wybór akcji przez agenta
        akcja = agent.pobierz_akcję(stan)
        
        # Wykonanie akcji w środowisku
        następny_stan, nagroda, zakończone, wynik = game.krok(akcja)
        
        # Zapisanie doświadczenia do późniejszego użycia
        doświadczenia.append((stan, akcja, nagroda, następny_stan, zakończone))
        
        # Przejście do nowego stanu
        stan = następny_stan
        łączna_nagroda += nagroda
        liczba_kroków += 1
    
    return doświadczenia, łączna_nagroda, wynik


def trenuj_hybrydowo(agent, parametry_gry, liczba_epizodów=1000, aktualizacja_docelowa=10, interwał_zapisu=100, liczba_równoległych=4):
    """
    Trenuje agenta z wykorzystaniem zarówno CPU jak i GPU dla maksymalnej wydajności.
    
    Args:
        agent (DQNAgent): Agent do trenowania.
        parametry_gry (dict): Parametry do inicjalizacji gry.
        liczba_epizodów (int): Liczba epizodów treningu.
        aktualizacja_docelowa (int): Co ile epizodów aktualizować model docelowy.
        interwał_zapisu (int): Co ile epizodów zapisywać model.
        liczba_równoległych (int): Liczba równoległych gier.
        
    Returns:
        tuple: Para (wyniki, historia epsilon).
    """
    wyniki = []
    historia_ep = []
    najlepszy_wynik = 0
    średnia_strata = 0
    
    # Określamy tryb treningu na podstawie dostępności GPU
    if UŻYJ_GPU:
        print(f"Trening hybrydowy: zbieranie doświadczeń na CPU, trening na GPU.")
    else:
        print(f"Trening na CPU z {liczba_równoległych} równoległymi grami.")
    
    # Liczba grup epizodów do przeprowadzenia
    liczba_fragmentów = (liczba_epizodów + liczba_równoległych - 1) // liczba_równoległych
    
    with tqdm(total=liczba_epizodów, desc="Trening") as pbar:
        for fragment in range(liczba_fragmentów):
            # Rzeczywista liczba epizodów w tej grupie
            rzeczywista_n = min(liczba_równoległych, liczba_epizodów - fragment * liczba_równoległych)
            
            # Uruchomienie wielu epizodów równolegle (symulacja wielowątkowości)
            wszystkie_doświadczenia = []
            wyniki_fragmentu = []
            
            # Wykonujemy liczba_równoległych gier "równolegle"
            for _ in range(rzeczywista_n):
                doświadczenia, _, wynik = uruchom_epizod(agent, parametry_gry)
                wszystkie_doświadczenia.extend(doświadczenia)
                wyniki_fragmentu.append(wynik)
            
            # Aktualizacja statystyk
            wyniki.extend(wyniki_fragmentu)
            historia_ep.append(agent.epsilon)
            
            # Dodanie wszystkich doświadczeń do pamięci agenta
            for exp in wszystkie_doświadczenia:
                agent.zapamiętaj(*exp)
            
            # Uczenie agenta na zebranych doświadczeniach
            # W trybie GPU trenujemy intensywniej
            iteracje_treningu = min(len(wszystkie_doświadczenia), 2000 if UŻYJ_GPU else 1000)
            
            łączna_strata = 0
            liczba_strat = 0
            for _ in range(iteracje_treningu):
                przegrana = agent.ucz_się()
                if przegrana is not None:
                    łączna_strata += przegrana
                    liczba_strat += 1
            
            if liczba_strat > 0:
                średnia_strata = łączna_strata / liczba_strat
            
            # Aktualizacja paska postępu
            pbar.update(rzeczywista_n)
            
            # Wyświetlanie postępów co interwał_zapisu fragment'ów
            obecny_epizod = (fragment + 1) * liczba_równoległych
            if fragment % (interwał_zapisu // max(1, liczba_równoległych)) == 0 or fragment == liczba_fragmentów - 1:
                średni_wynik = np.mean(wyniki[-100:]) if len(wyniki) >= 100 else np.mean(wyniki)
                ostatnio_średnia = np.mean(wyniki_fragmentu)
    
                print(f"\nEpizod {obecny_epizod}/{liczba_epizodów}:")
                print(f"  Średni wynik: {średni_wynik:.4f}")
                print(f"  Ostatni średni wynik: {ostatnio_średnia:.4f}")
                print(f"  Epsilon: {agent.epsilon:.6f}")
                print(f"  Strata: {średnia_strata:.6f}")
                print(f"  Najlepszy wynik do tej pory: {najlepszy_wynik}")
    
                pbar.set_postfix({
                        'Śr.wynik': f'{średni_wynik:.2f}',
                        'Ost.wynik': f'{ostatnio_średnia:.2f}',
                        'Epsilon': f'{agent.epsilon:.4f}',
                        'Loss': f'{średnia_strata:.4f}'
                })
                
                pbar.update(0)
                # Zapisanie modelu co interwał_zapisu epizodów
                if obecny_epizod <= liczba_epizodów:
                    agent.save(f"models/snake_model_episode_{obecny_epizod}.pth")
            
            # Sprawdzenie, czy mamy nowy najlepszy wynik
            maks_wynik = max(wyniki_fragmentu) if wyniki_fragmentu else 0
            if maks_wynik > najlepszy_wynik:
                najlepszy_wynik = maks_wynik
                agent.save("models/snake_model_best.pth")
                pbar.write(f"Nowy najlepszy wynik: {najlepszy_wynik}! Model zapisany jako 'snake_model_best.pth'")
    
    # Zapisanie ostatecznego modelu
    agent.save("models/snake_model_final.pth")
    print("Trening zakończony. Ostateczny model zapisany jako 'snake_model_final.pth'")
    
    return wyniki, historia_ep


def trenuj_tylko_cpu(agent, game, liczba_epizodów=1000, interwał_zapisu=100):
    """
    Trenuje agenta wyłącznie na CPU, bez zrównoleglenia.
    
    Args:
        agent (DQNAgent): Agent do trenowania.
        game (SnakeGame): Środowisko gry.
        liczba_epizodów (int): Liczba epizodów treningu.
        interwał_zapisu (int): Co ile epizodów zapisywać model.
        
    Returns:
        tuple: Para (wyniki, historia epsilon).
    """
    wyniki = []
    historia_ep = []
    najlepszy_wynik = 0
    średnia_strata = 0
    
    print("Trening tylko na CPU z pojedynczą grą.")
    
    for e in tqdm(range(liczba_epizodów), desc="Trening"):
        # Resetowanie gry i pobranie stanu początkowego
        stan = game.reset()
        zakończone = False
        wynik = 0
        łączna_strata = 0
        liczba_kroków = 0
        liczba_strat = 0
        
        while not zakończone:
            # Wybór akcji przez agenta
            akcja = agent.pobierz_akcję(stan)
            
            # Wykonanie akcji w środowisku
            następny_stan, nagroda, zakończone, info = game.krok(akcja)
            
            # Zapisanie doświadczenia w pamięci agenta
            agent.zapamiętaj(stan, akcja, nagroda, następny_stan, zakończone)
            
            # Przejście do nowego stanu
            stan = następny_stan
            
            # Uczenie agenta
            przegrana = agent.ucz_się()
            if przegrana is not None:
                łączna_strata += przegrana
                liczba_strat += 1
            
            # Aktualizacja wyniku
            wynik = info  # info to obecny wynik
            liczba_kroków += 1
            
        # Zapisanie wyniku i wartości epsilon dla tego epizodu
        wyniki.append(wynik)
        historia_ep.append(agent.epsilon)
        
        # Obliczenie średniej straty
        if liczba_strat > 0:
            średnia_strata = łączna_strata / liczba_strat
        
        # Wyświetlanie postępów co interwał_zapisu epizodów
        if e % interwał_zapisu == 0 or e == liczba_epizodów - 1:
            średni_wynik = np.mean(wyniki[-100:]) if len(wyniki) >= 100 else np.mean(wyniki)
            tqdm.write(f"Epizod {e}, Wynik: {wynik}, Średni wynik: {średni_wynik:.2f}, "
                      f"Epsilon: {agent.epsilon:.4f}, Loss: {średnia_strata:.4f}")
            
            # Zapisanie modelu co interwał_zapisu epizodów
            agent.save(f"models/snake_model_episode_{e}.pth")
            
        # Zapisanie najlepszego modelu
        if wynik > najlepszy_wynik:
            najlepszy_wynik = wynik
            agent.save("models/snake_model_best.pth")
            tqdm.write(f"Nowy najlepszy wynik: {wynik}! Model zapisany jako 'snake_model_best.pth'")
    
    # Zapisanie ostatecznego modelu
    agent.save("models/snake_model_final.pth")
    print("Trening zakończony. Ostateczny model zapisany jako 'snake_model_final.pth'")
    
    return wyniki, historia_ep


def rysuj_wyniki(wyniki, historia_ep):
    """
    Tworzy wykresy wyników treningu.
    
    Args:
        wyniki (list): Lista wyników z każdego epizodu.
        historia_ep (list): Lista wartości epsilon z każdego epizodu.
    """
    plt.figure(figsize=(16, 5))
    
    # Wykres wyników
    plt.subplot(1, 3, 1)
    plt.plot(wyniki)
    plt.axhline(y=np.mean(wyniki), color='r', linestyle='--', label=f'Średnia: {np.mean(wyniki):.2f}')
    plt.xlabel('Epizod')
    plt.ylabel('Wynik')
    plt.title('Wyniki w czasie treningu')
    plt.legend()
    
    # Wykres średniej ruchomej
    plt.subplot(1, 3, 2)
    rozmiar_okna = min(100, len(wyniki))
    ruchoma_średnia = [np.mean(wyniki[max(0, i-rozmiar_okna):i+1]) for i in range(len(wyniki))]
    plt.plot(ruchoma_średnia)
    plt.xlabel('Epizod')
    plt.ylabel('Średnia z ostatnich 100 epizodów')
    plt.title('Średnia ruchoma wyników')
    
    # Wykres zmian epsilon
    plt.subplot(1, 3, 3)
    plt.plot(historia_ep)
    plt.xlabel('Epizod')
    plt.ylabel('Epsilon')
    plt.title('Zmiana współczynnika eksploracji')
    
    plt.tight_layout()
    plt.savefig('models/training_results.png')
    print("Wykresy wyników treningu zapisane do 'models/training_results.png'")
    plt.show()


def testuj_agenta(agent, game, liczba_gier=5, opóźnienie=100):
    """
    Testuje wytrenowanego agenta.
    
    Args:
        agent (DQNAgent): Wytrenowany agent.
        game (SnakeGame): Środowisko gry.
        liczba_gier (int): Liczba gier testowych.
        opóźnienie (int): Opóźnienie między krokami (ms).
    """
    łączny_wynik = 0
    maks_wynik = 0
    wyniki = []
    
    for indeks_gry in range(liczba_gier):
        stan = game.reset()
        zakończone = False
        
        while not zakończone:
            # Agent wybiera akcję bez eksploracji
            tensor_stanu = torch.tensor(stan, dtype=agent.typ_danych).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                wartości_q = agent.model(tensor_stanu)
            akcja = torch.argmax(wartości_q).item()
            
            # Wykonanie akcji
            stan, nagroda, zakończone, wynik = game.krok(akcja)
            
            # Opóźnienie, aby można było obserwować grę
            if opóźnienie > 0:
                pygame.time.delay(opóźnienie)
            
        łączny_wynik += wynik
        wyniki.append(wynik)
        maks_wynik = max(maks_wynik, wynik)
        print(f"Gra {indeks_gry+1}, Wynik: {wynik}")
    
    średni_wynik = łączny_wynik / liczba_gier
    print(f"Średni wynik po {liczba_gier} grach: {średni_wynik:.2f}")
    print(f"Najlepszy wynik: {maks_wynik}")
    print(f"Wszystkie wyniki: {wyniki}")
    
    return średni_wynik, maks_wynik, wyniki
            