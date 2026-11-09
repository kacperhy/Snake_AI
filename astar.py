"""
Algorytm A* + cykl Hamiltona do sterowania wężem.

1. Budujemy cykl Hamiltona – trasę przechodzącą przez KAŻDĄ komórkę planszy
   dokładnie raz i wracającą do startu. Wąż jadący tylko po cyklu nigdy się
   nie zderzy i zawsze wypełni całą planszę – ale jest bardzo wolny.
2. Dlatego A* szuka skrótów do jedzenia.
3. Skrót jest dozwolony tylko wtedy, gdy NIE wyprzedza ogona w kolejności
   cyklu. Wtedy ciało dalej leży "w kolejce" za głową, a przed głową (aż do
   ogona) są same wolne pola – powrót na cykl jest zawsze bezpieczny.

Wymaganie: przynajmniej jeden wymiar planszy (w kratkach) musi być parzysty,
inaczej cykl Hamiltona na siatce nie istnieje.
"""

import heapq
from snake_game import Kierunek

_RUCHY = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ZEGAR = [Kierunek.PRAWO, Kierunek.DÓŁ, Kierunek.LEWO, Kierunek.GÓRA]


# Cykl Hamiltona

def _cykl_parzysta_wysokość(gw, gh):
    """
    Cykl dla gh parzystego. Przykład 4x4 (numery = kolejność odwiedzin):

         0 15 14 13
         1 10 11 12
         2  9  8  7
         3  4  5  6

    Lewa kolumna w dół, potem "wężykiem" w górę po pozostałych kolumnach.
    """
    cykl = [(0, r) for r in range(gh)]
    for i, r in enumerate(range(gh - 1, -1, -1)):
        kolumny = range(1, gw) if i % 2 == 0 else range(gw - 1, 0, -1)
        cykl.extend((c, r) for c in kolumny)
    return cykl


def zbuduj_cykl(gw, gh):
    #Zwraca listę komórek (col, row) tworzących cykl Hamiltona
    if gw < 2 or gh < 2:
        raise ValueError("Zbyt mała plansza.")
    if gh % 2 == 0:
        return _cykl_parzysta_wysokość(gw, gh)
    if gw % 2 == 0:
        # Transpozycja: budujemy dla zamienionych wymiarów i zamieniamy współrzędne
        return [(r, c) for c, r in _cykl_parzysta_wysokość(gh, gw)]
    raise ValueError(f"Plansza {gw}x{gh}: oba wymiary nieparzyste, cykl Hamiltona potrzebuje parz.")


class _Planer:
    #Trzyma cykl i numer pozycji każdej komórki w cyklu

    def __init__(self, gw, gh, ciało):
        self.gw, self.gh = gw, gh
        self.cykl = zbuduj_cykl(gw, gh)
        self.N = len(self.cykl)
        self.poz = {c: i for i, c in enumerate(self.cykl)}

        # Kierunek cyklu dopasowujemy do początkowego ułożenia węża
        # żeby głowa była przed ogonem w kolejności cyklu
        if len(ciało) >= 2 and self.odl(ciało[1], ciało[0]) != 1:
            self.cykl.reverse()
            self.poz = {c: i for i, c in enumerate(self.cykl)}

    def odl(self, a, b):
        #Ile kroków po cyklu z a do b (zawsze do przodu)
        return (self.poz[b] - self.poz[a]) % self.N

    def następnik(self, a):
        return self.cykl[(self.poz[a] + 1) % self.N]

# A*

def _szukaj(start, cel, zablokowane, gw, gh):
    # algorytm zwraca listę komórek od start do cel lub None
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

# Wybór ruchu
def _planer_dla(game, gw, gh, ciało):
    #Planer jest liczony raz i trzymany na obiekcie gry 
    planer = getattr(game, "_planer_astar", None)
    nowa_gra = getattr(game, "iteracja_klatki", 0) == 0
    if planer is None or (planer.gw, planer.gh) != (gw, gh) or nowa_gra:
        planer = _Planer(gw, gh, ciało)
        game._planer_astar = planer
    return planer


def wybierz_akcję(game):
    """
    Wybiera akcję dla węża (0=prosto, 1=prawo, 2=lewo).

    Zwraca krotkę: (akcja: int, ścieżka: list[(col,row)] lub None)
    ścieżka jest zwracana, gdy wąż idzie skrótem 
    """
    b = game.rozmiar_bloku
    gw = game.szerokość // b
    gh = game.wysokość // b

    def k(seg):
        return (seg[0] // b, seg[1] // b)

    segmenty = [k(s) for s in game.snake]
    planer = _planer_dla(game, gw, gh, segmenty)

    głowa, ogon = segmenty[0], segmenty[-1]
    jedzenie = k(game.jedzenie)
    ciało_bez_ogona = set(segmenty[1:-1])

    d_ogon = planer.odl(głowa, ogon)
    d_jedz = planer.odl(głowa, jedzenie)

    def bezpieczny_skrót(n):
        """
        Skrót na pole n jest OK, jeśli:
          - n jest przed ogonem w kolejności cyklu
          - n nie jest za jedzeniem
        """
        d = planer.odl(głowa, n)
        return n not in ciało_bez_ogona and 0 < d < d_ogon and d <= d_jedz

    # A* do jedzenia jeśli pierwszy krok jest bezpiecznym skrótem
    ścieżka = _szukaj(głowa, jedzenie, ciało_bez_ogona, gw, gh)
    if ścieżka and len(ścieżka) >= 2 and bezpieczny_skrót(ścieżka[1]):
        return _na_akcję(game, ścieżka[1], b), ścieżka

    # Najdalszy bezpieczny skok po cyklu w stronę jedzenia
    najlepszy, najlepsza_odl = None, 0
    for dx, dy in _RUCHY:
        n = (głowa[0] + dx, głowa[1] + dy)
        if 0 <= n[0] < gw and 0 <= n[1] < gh and bezpieczny_skrót(n):
            d = planer.odl(głowa, n)
            if d > najlepsza_odl:
                najlepszy, najlepsza_odl = n, d
    if najlepszy is not None:
        return _na_akcję(game, najlepszy, b), None

    # Zwykły krok po cyklu który jest zawsze bezpieczny
    return _na_akcję(game, planer.następnik(głowa), b), None


def _na_akcję(game, następna, b):
    # Tłumaczy docelową komórkę siatki na akcję 
    głowa = game.snake[0]
    dx = następna[0] * b - głowa[0]
    dy = następna[1] * b - głowa[1]

    if dx > 0:
        cel = Kierunek.PRAWO
    elif dx < 0:
        cel = Kierunek.LEWO
    elif dy > 0:
        cel = Kierunek.DÓŁ
    else:
        cel = Kierunek.GÓRA

    diff = (_ZEGAR.index(cel) - _ZEGAR.index(game.Kierunek)) % 4
    return {0: 0, 1: 1, 3: 2}.get(diff, 0)