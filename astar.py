"""
Algorytm A* do sterowania wężem.

Strategia:
1. A* do jedzenia + sprawdzenie bezpieczeństwa (czy po zjedzeniu ogon osiągalny)
2. Fallback: A* do ogona (przetrwanie gdy droga do jedzenia niebezpieczna)
3. Ostatnia szansa: pierwszy dowolny wolny ruch
"""

import heapq
from snake_game import Kierunek

_RUCHY = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _szukaj(start, cel, zablokowane, gw, gh):
    """A* – zwraca listę komórek (col, row) od start do cel lub None."""
    stos = [(0, start)]
    skąd = {start: None}
    koszt = {start: 0}

    while stos:
        _, bież = heapq.heappop(stos)
        if bież == cel:
            ścieżka = []
            while bież is not None:
                ścieżka.append(bież)
                bież = skąd[bież]
            ścieżka.reverse()
            return ścieżka

        for dx, dy in _RUCHY:
            nb = (bież[0] + dx, bież[1] + dy)
            if (0 <= nb[0] < gw and
                    0 <= nb[1] < gh and
                    nb not in zablokowane):
                nk = koszt[bież] + 1
                if nb not in koszt or nk < koszt[nb]:
                    koszt[nb] = nk
                    h = abs(nb[0] - cel[0]) + abs(nb[1] - cel[1])
                    heapq.heappush(stos, (nk + h, nb))
                    skąd[nb] = bież
    return None


def wybierz_akcję(game):
    """
    Wybiera akcję dla węża (0=prosto, 1=prawo, 2=lewo).

    Returns:
        tuple: (akcja: int, ścieżka: list[(col,row)] lub None)
               ścieżka jest zwracana tylko gdy idziemy do jedzenia (do wizualizacji)
    """
    b  = game.rozmiar_bloku
    gw = game.szerokość // b
    gh = game.wysokość // b

    def k(seg): return (seg[0] // b, seg[1] // b)

    głowa    = k(game.snake[0])
    jedzenie = k(game.jedzenie)
    ogon     = k(game.snake[-1])

    ciało_bez_ogona = {k(s) for s in game.snake[1:-1]}
    ciało           = {k(s) for s in game.snake[1:]}

    # ── 1. Droga do jedzenia ───────────────────────────────────────────────
    ścieżka = _szukaj(głowa, jedzenie, ciało_bez_ogona, gw, gh)

    if ścieżka and len(ścieżka) >= 2:
        następna = ścieżka[1]
        # Bezpieczeństwo: po zjedzeniu wąż rośnie – ogon zostaje.
        po_zjedzeniu = {k(s) for s in game.snake[:-1]}
        if _szukaj(jedzenie, ogon, po_zjedzeniu, gw, gh) is not None:
            return _na_akcję(game, następna, b), ścieżka

    # ── 2. Fallback: goń ogon żeby przeżyć ────────────────────────────────────
    ścieżka_ogon = _szukaj(głowa, ogon, ciało_bez_ogona, gw, gh)
    if ścieżka_ogon and len(ścieżka_ogon) >= 2:
        return _na_akcję(game, ścieżka_ogon[1], b), None

    # ── 3. Ostatnia szansa: dowolny wolny ruch ───────────────────────────
    for dx, dy in _RUCHY:
        nb = (głowa[0] + dx, głowa[1] + dy)
        if 0 <= nb[0] < gw and 0 <= nb[1] < gh and nb not in ciało:
            return _na_akcję(game, nb, b), None

    return 0, None  # całkowicie zablokowany


def _na_akcję(game, następna, b):
    """Tłumaczy docelową komórkę siatki na względną akcję (0/1/2)."""
    głowa = game.snake[0]
    dx = następna[0] * b - głowa[0]
    dy = następna[1] * b - głowa[1]

    if   dx > 0: cel = Kierunek.PRAWO
    elif dx < 0: cel = Kierunek.LEWO
    elif dy > 0: cel = Kierunek.DÓŁ
    else:         cel = Kierunek.GÓRA

    zegar = [Kierunek.PRAWO, Kierunek.DÓŁ, Kierunek.LEWO, Kierunek.GÓRA]
    diff  = (zegar.index(cel) - zegar.index(game.Kierunek)) % 4
    return 0 if diff == 0 else (1 if diff == 1 else (2 if diff == 3 else 0))
