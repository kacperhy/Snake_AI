"""
Benchmark: jaki rozmiar sieci i partii najlepiej wykorzysta Twój sprzęt.

Uruchom z katalogu projektu:
    python Testy/benchmark_gpu.py            (pełny, ok. 3–6 min)
    python Testy/benchmark_gpu.py --szybko   (mniej kombinacji, ok. 1 min)

Mierzy:
 1. UCZENIE – ile kroków ucz_się() na sekundę dla różnych rozmiarów warstwy
    ukrytej (ROZMIAR_UKRYTY) i partii (ROZMIAR_PARTII), na CPU i na GPU.
 2. ZBIERANIE – ile ruchów na sekundę przy różnej liczbie gier równoległych
    (opcja „Gier równolegle” w GUI).
Na końcu wypisuje rekomendację do config.py.
"""

import os
import sys
import io
import time
import argparse
import contextlib

KATALOG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KATALOG)
os.chdir(KATALOG)

import numpy as np
import torch

import config
from agent import DQNAgent
from training import uruchom_epizody


def liczba_parametrów(h, wejście=11, wyjście=3):
    """Wagi + biasy sieci QNetwork: wejście→h→h→h/2→wyjście."""
    return (wejście * h + h) + (h * h + h) + (h * (h // 2) + h // 2) + ((h // 2) * wyjście + wyjście)


def nowy_agent(urządzenie, h):
    with contextlib.redirect_stdout(io.StringIO()):     # bez komunikatów agenta
        return DQNAgent(11, 3, h, urządzenie=urządzenie)


def synchronizuj(urządzenie):
    if urządzenie == 'cuda':
        torch.cuda.synchronize()


def zmierz_uczenie(urządzenie, h, partia, czas=1.0):
    """Zwraca liczbę kroków uczenia na sekundę."""
    agent = nowy_agent(urządzenie, h)
    agent.rozmiar_partii = partia
    agent.częstotliwość_aktualizacji = 1          # każde wywołanie = krok uczenia

    n = max(20000, partia * 2)                     # losowe, ale realistyczne dane
    rng = np.random.default_rng(0)
    stany = rng.integers(0, 2, (n, 11)).astype(np.float32)
    następne = rng.integers(0, 2, (n, 11)).astype(np.float32)
    akcje = rng.integers(0, 3, n)
    nagrody = rng.choice([-10.0, -0.1, 0.1, 10.0], n)
    zakończone = rng.random(n) < 0.02
    for i in range(n):
        agent.zapamiętaj(stany[i], int(akcje[i]), float(nagrody[i]), następne[i], bool(zakończone[i]))

    for _ in range(5):                             # rozgrzewka (pierwsze kroki są wolniejsze)
        agent.ucz_się()
    synchronizuj(urządzenie)

    kroki, t0 = 0, time.perf_counter()
    while True:
        agent.ucz_się()
        kroki += 1
        if kroki % 10 == 0 or kroki < 10:
            synchronizuj(urządzenie)
            upłynęło = time.perf_counter() - t0
            if upłynęło >= czas and kroki >= 3:
                return kroki / upłynęło


def zmierz_zbieranie(urządzenie, h, liczba_gier, czas=1.5):
    """Zwraca liczbę ruchów (decyzji sieci) na sekundę."""
    agent = nowy_agent(urządzenie, h)
    agent.epsilon = 0.0                            # każda decyzja idzie przez sieć
    parametry = dict(szerokość=config.SZEROKOŚĆ_OKNA, wysokość=config.WYSOKOŚĆ_OKNA,
                     rozmiar_bloku=config.ROZMIAR_BLOKU)
    uruchom_epizody(agent, parametry, liczba_gier, maks_kroków=20)   # rozgrzewka

    ruchy, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < czas:
        doświadczenia, _ = uruchom_epizody(agent, parametry, liczba_gier, maks_kroków=200)
        ruchy += len(doświadczenia)
    return ruchy / (time.perf_counter() - t0)


def tabela(tytuł, wiersze, kolumny, wartości, format_wartości):
    print(f"\n{tytuł}")
    nagłówek = f"{'':>18}" + "".join(f"{k:>12}" for k in kolumny)
    print(nagłówek)
    print("-" * len(nagłówek))
    for w in wiersze:
        komórki = "".join(
            f"{format_wartości(wartości[(w, k)]):>12}" if (w, k) in wartości else f"{'—':>12}"
            for k in kolumny)
        print(f"{w:>18}{komórki}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--szybko', action='store_true', help='mniej kombinacji (~1 min)')
    args = parser.parse_args()

    if args.szybko:
        ukryte, partie, gry = [256, 1024], [256, 1024], [1, 32]
    else:
        ukryte, partie, gry = [128, 256, 512, 1024, 2048], [64, 256, 1024, 4096], [1, 8, 32, 128]

    urządzenia = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    print("=" * 66)
    print(" Benchmark Snake DQN")
    print("=" * 66)
    print(f" PyTorch {torch.__version__}")
    if 'cuda' in urządzenia:
        p = torch.cuda.get_device_properties(0)
        print(f" Karta:  {p.name} ({p.total_memory / 1024**3:.0f} GB)")
    else:
        print(" Karta:  brak CUDA – mierzę tylko procesor")
    print(f" Obecnie w config.py: ROZMIAR_UKRYTY={config.ROZMIAR_UKRYTY}, "
          f"ROZMIAR_PARTII={config.ROZMIAR_PARTII}")

    print("\n Liczba parametrów sieci dla rozmiaru warstwy ukrytej:")
    for h in ukryte:
        print(f"   {h:>5} → {liczba_parametrów(h):>12,} wag".replace(',', ' '))

    # Uczenie
    uczenie = {}
    for urz in urządzenia:
        for h in ukryte:
            for b in partie:
                print(f"\r Mierzę uczenie: {urz.upper()} warstwa {h}, partia {b}…      ",
                      end='', flush=True)
                try:
                    uczenie[(urz, h, b)] = zmierz_uczenie(urz, h, b)
                except torch.cuda.OutOfMemoryError:
                    print(f"\n   brak pamięci karty dla {h}/{b} – pomijam")
                    torch.cuda.empty_cache()
    print("\r" + " " * 70)

    kol = [f"partia {b}" for b in partie]
    for urz in urządzenia:
        wart = {(f"warstwa {h}", f"partia {b}"): uczenie[(urz, h, b)]
                for h in ukryte for b in partie if (urz, h, b) in uczenie}
        tabela(f" UCZENIE na {urz.upper()} – kroków uczenia na sekundę (więcej = lepiej)",
               [f"warstwa {h}" for h in ukryte], kol, wart, lambda v: f"{v:,.0f}".replace(',', ' '))

    if 'cuda' in urządzenia:
        wart = {(f"warstwa {h}", f"partia {b}"): uczenie[('cuda', h, b)] / uczenie[('cpu', h, b)]
                for h in ukryte for b in partie
                if ('cuda', h, b) in uczenie and ('cpu', h, b) in uczenie}
        tabela(" PRZYSPIESZENIE GPU względem CPU (×)",
               [f"warstwa {h}" for h in ukryte], kol, wart, lambda v: f"{v:.1f}×")

    # Zbieranie
    h0 = config.ROZMIAR_UKRYTY
    zbieranie = {}
    for urz in urządzenia:
        for n in gry:
            print(f"\r Mierzę zbieranie: {urz.upper()}, {n} gier równolegle…      ",
                  end='', flush=True)
            zbieranie[(urz, n)] = zmierz_zbieranie(urz, h0, n)
    print("\r" + " " * 70)
    wart = {(urz.upper(), f"{n} gier"): zbieranie[(urz, n)] for urz in urządzenia for n in gry}
    tabela(f" ZBIERANIE – ruchów na sekundę (sieć {h0}, więcej = lepiej)",
           [u.upper() for u in urządzenia], [f"{n} gier" for n in gry], wart,
           lambda v: f"{v:,.0f}".replace(',', ' '))

    # Rekomendacja
    print("\n" + "=" * 66)
    print(" REKOMENDACJA")
    print("=" * 66)
    urz = 'cuda' if 'cuda' in urządzenia else 'cpu'
    bazowe = uczenie.get((urz, 256, 256))
    if bazowe:
        # Największa sieć x partia, która jest najwyżej o 25% wolniejsza od 256×256
        tanie = [(h * b, h, b) for (u, h, b), v in uczenie.items()
                 if u == urz and v >= 0.75 * bazowe]
        _, h_rek, b_rek = max(tanie)
        print(f" Na {'karcie' if urz == 'cuda' else 'procesorze'} sieć {h_rek} z partią {b_rek} "
              f"kosztuje prawie tyle co 256×256")
        print(f" ({uczenie[(urz, h_rek, b_rek)]:.0f} vs {bazowe:.0f} kroków/s), "
              f"a każdy krok przetwarza {h_rek * b_rek / (256 * 256):.0f}× więcej obliczeń.")
        print(f"\n   W config.py możesz ustawić:")
        print(f"     ROZMIAR_UKRYTY = {h_rek}")
        print(f"     ROZMIAR_PARTII = {b_rek}")
    najlepsze_n = max(gry, key=lambda n: zbieranie[(urz, n)])
    print(f"\n   W GUI „Gier równolegle”: {najlepsze_n} "
          f"({zbieranie[(urz, najlepsze_n)]:.0f} ruchów/s vs "
          f"{zbieranie[(urz, gry[0])]:.0f} przy {gry[0]})")


if __name__ == '__main__':
    main()
