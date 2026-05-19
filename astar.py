"""
Snake AI z cyklem Hamiltonowskim — gwarantuje zbieranie każdego jedzenia.

Strategia:
1. Skrót A*: jeśli droga do jedzenia przechodzi tylko przez komórki
   'przed' jedzeniem w cyklu → idź skrótem (szybciej)
2. Fallback: podążaj cyklem Hamiltonowskim krok po kroku
3. Zapas: flood fill dla siatek bez cyklu (oba wymiary nieparzyste)
"""

import heapq
from snake_game import Kierunek

_RUCHY = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ham_cache: dict = {}


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
            if (0 <= nb[0] < gw and 0 <= nb[1] < gh and nb not in zablokowane):
                nk = koszt[bież] + 1
                if nb not in koszt or nk < koszt[nb]:
                    koszt[nb] = nk
                    h = abs(nb[0] - cel[0]) + abs(nb[1] - cel[1])
                    heapq.heappush(stos, (nk + h, nb))
                    skąd[nb] = bież
    return None


def _flood_fill(start, zablokowane, gw, gh):
    # Liczy komórki osiągalne z start (BFS).
    odwiedzone = {start}
    kolejka = [start]
    while kolejka:
        nowe = []
        for bież in kolejka:
            for dx, dy in _RUCHY:
                nb = (bież[0] + dx, bież[1] + dy)
                if (0 <= nb[0] < gw and 0 <= nb[1] < gh and
                        nb not in zablokowane and nb not in odwiedzone):
                    odwiedzone.add(nb)
                    nowe.append(nb)
        kolejka = nowe
    return len(odwiedzone)


def _hamiltonian_order(gw, gh):
    """
    Zwraca {(col,row): pozycja_w_cyklu} dla siatki gw×gh lub None jeśli
    oba wymiary nieparzyste (cykl Hamiltonowski nie istnieje).
   """
    key = (gw, gh)
    if key in _ham_cache:
        return _ham_cache[key]

    if gw % 2 != 0 and gh % 2 != 0:
        _ham_cache[key] = None
        return None

    if gw % 2 != 0:
        # Transpozycja: oblicz dla gh×gw, zamień współrzędne
        tmp = _hamiltonian_order(gh, gw)
        result = {(r, c): p for (c, r), p in tmp.items()}
        _ham_cache[key] = result
        return result

    # gw parzyste — główna konstrukcja
    order = {}
    pos = 0

    order[(0, 0)] = pos; pos += 1

    for row in range(1, gh):               # kolumna 0: w dół
        order[(0, row)] = pos; pos += 1

    for c in range(1, gw - 1):            # środkowe kolumny: cik-cak
        if c % 2 == 1:                     #   nieparzyste: w górę
            for row in range(gh - 1, 0, -1):
                order[(c, row)] = pos; pos += 1
        else:                              #   parzyste: w dół
            for row in range(1, gh):
                order[(c, row)] = pos; pos += 1

    for row in range(gh - 1, -1, -1):     # ostatnia kolumna: wszystkie wiersze w górę
        order[(gw - 1, row)] = pos; pos += 1

    for c in range(gw - 2, 0, -1):        # wiersz 0: w lewo
        order[(c, 0)] = pos; pos += 1

    _ham_cache[key] = order
    return order


def wybierz_akcję(game):
    # Wybiera akcję dla węża (0=prosto, 1=prawo, 2=lewo).
    
    b  = game.rozmiar_bloku
    gw = game.szerokość // b
    gh = game.wysokość // b
    N  = gw * gh

    def k(seg): return (seg[0] // b, seg[1] // b)

    głowa           = k(game.snake[0])
    jedzenie        = k(game.jedzenie)
    ciało_bez_ogona = {k(s) for s in game.snake[1:-1]}

    order = _hamiltonian_order(gw, gh)

    if order is not None:
        # ── Ścieżka Hamiltonowska dostępna ───────────────────────────────
        cycle = [None] * N
        for cell, p in order.items():
            cycle[p] = cell

        h_pos = order[głowa]
        f_pos = order[jedzenie]

        def przed_jedzeniem(p):
            #Czy pozycja p leży między h_pos (wyłącznie) a f_pos (włącznie) w cyklu.
            if h_pos < f_pos:
                return h_pos < p <= f_pos
            else:               # zawinięcie przez 0
                return p > h_pos or p <= f_pos

        # Skrót A*: jeśli cała ścieżka mieści się w bezpiecznym przedziale cyklu
        ścieżka = _szukaj(głowa, jedzenie, ciało_bez_ogona, gw, gh)
        if ścieżka and len(ścieżka) >= 2:
            if all(przed_jedzeniem(order[c]) for c in ścieżka[1:]):
                return _na_akcję(game, ścieżka[1], b), ścieżka

        # Fallback: następny krok w cyklu Hamiltonowskim
        next_cell = cycle[(h_pos + 1) % N]
        return _na_akcję(game, next_cell, b), None

    # ── Zapas: flood fill dla siatek bez cyklu (oba wymiary nieparzyste) ─
    ogon   = k(game.snake[-1])
    po_ruchu = {k(s) for s in game.snake[:-1]}

    sąsiedzi = [
        (głowa[0] + dx, głowa[1] + dy)
        for dx, dy in _RUCHY
        if (0 <= głowa[0] + dx < gw and 0 <= głowa[1] + dy < gh
            and (głowa[0] + dx, głowa[1] + dy) not in ciało_bez_ogona)
    ]

    bezpieczne = _szukaj(jedzenie, ogon, po_ruchu, gw, gh) is not None
    if bezpieczne:
        best_ff, best_nb, best_ścieżka = -1, None, None
        for nb in sąsiedzi:
            p = _szukaj(nb, jedzenie, ciało_bez_ogona, gw, gh)
            if p is None:
                continue
            ff = _flood_fill(nb, po_ruchu, gw, gh)
            if ff > best_ff:
                best_ff, best_nb, best_ścieżka = ff, nb, [głowa] + p
        if best_nb is not None:
            return _na_akcję(game, best_nb, b), best_ścieżka

    best_ff, best_nb = -1, None
    for nb in sąsiedzi:
        if _szukaj(nb, ogon, ciało_bez_ogona, gw, gh) is not None:
            ff = _flood_fill(nb, po_ruchu, gw, gh)
            if ff > best_ff:
                best_ff, best_nb = ff, nb
    if best_nb is not None:
        return _na_akcję(game, best_nb, b), None

    ciało_pełne = {k(s) for s in game.snake[1:]}
    best_ff, best_nb = -1, None
    for dx, dy in _RUCHY:
        nb = (głowa[0] + dx, głowa[1] + dy)
        if 0 <= nb[0] < gw and 0 <= nb[1] < gh and nb not in ciało_pełne:
            ff = _flood_fill(nb, po_ruchu, gw, gh)
            if ff > best_ff:
                best_ff, best_nb = ff, nb
    if best_nb is not None:
        return _na_akcję(game, best_nb, b), None

    return 0, None


def _na_akcję(game, następna, b):
    # Tłumaczy docelową komórkę siatki na względną akcję (0/1/2).
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
