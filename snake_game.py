"""
Moduł zawierający implementację gry Snake.
"""

import pygame
import random
import numpy as np
from enum import Enum
from config import (
    UŻYJ_GPU, UŻYJ_FLOAT16, LICZBA_WĄTKÓW_CPU, SZEROKOŚĆ_OKNA, 
    WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU, PRĘDKOŚĆ_GRY, ROZMIAR_UKRYTY, KATALOG_MODELI
)

# Definicja kolorów
BIAŁY = (255, 255, 255)
CZARNY = (0, 0, 0)
CZERWONY = (255, 0, 0)
ZIELONY = (0, 255, 0)
NIEBIESKI = (0, 0, 255)
CIEMNY_ZIELONY = (0, 200, 0)

# Definicja kierunków
class Kierunek(Enum):
    PRAWO = 0
    DÓŁ = 1
    LEWO = 2
    GÓRA = 3

class SnakeGame:
    """Pełna implementacja gry Snake z interfejsem graficznym do testowania."""
    def __init__(self):
        # Inicjalizacja parametrów gry
        self.szerokość = SZEROKOŚĆ_OKNA
        self.wysokość = WYSOKOŚĆ_OKNA
        self.rozmiar_bloku = ROZMIAR_BLOKU
        self.ekran = pygame.ekran.set_mode((SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA))
        pygame.ekran.set_caption('Snake AI - PyTorch')
        self.zegar = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        # Resetowanie gry do stanu początkowego
        self.Kierunek = Kierunek.PRAWO
        
        # Wąż zaczyna na środku planszy
        self.głowa = [self.szerokość // (2 * self.rozmiar_bloku) * self.rozmiar_bloku, 
                     self.wysokość // (2 * self.rozmiar_bloku) * self.rozmiar_bloku]
        
        # Początkowe segmenty węża
        self.snake = [
            self.głowa,
            [self.głowa[0] - self.rozmiar_bloku, self.głowa[1]],
            [self.głowa[0] - 2 * self.rozmiar_bloku, self.głowa[1]]
        ]
        
        self.wynik = 0
        self.jedzenie = None
        self._umieść_jedzenie()
        self.iteracja_klatki = 0
        self.kroki_bez_jedzenia = 0
        return self._pobierz_stan()
    
    def _umieść_jedzenie(self):
        # Umieszczenie jedzenia w losowym miejscu na planszy, ale nie na wężu
        max_x = (self.szerokość // self.rozmiar_bloku) - 1
        max_y = (self.wysokość // self.rozmiar_bloku) - 1
        
        while True:
            x = random.randint(0, max_x) * self.rozmiar_bloku
            y = random.randint(0, max_y) * self.rozmiar_bloku
            self.jedzenie = [x, y]
            if self.jedzenie not in self.snake:
                break
    
    def _pobierz_stan(self):
        # Zwraca obecny stan gry jako tablicę cech
        głowa = self.snake[0]
        
        # Punkty wokół głowy
        punkt_l = [głowa[0] - self.rozmiar_bloku, głowa[1]]
        punkt_pr = [głowa[0] + self.rozmiar_bloku, głowa[1]]
        punkt_g = [głowa[0], głowa[1] - self.rozmiar_bloku]
        punkt_d = [głowa[0], głowa[1] + self.rozmiar_bloku]
        
        # Aktualne kierunki
        kier_l = self.Kierunek == Kierunek.LEWO
        kier_pr = self.Kierunek == Kierunek.PRAWO
        kier_g = self.Kierunek == Kierunek.GÓRA
        kier_d = self.Kierunek == Kierunek.DÓŁ
        
        # Stan jako lista cech
        stan = [
            # Niebezpieczeństwo przed sobą
            (kier_pr and self._czy_kolizja(punkt_pr)) or
            (kier_l and self._czy_kolizja(punkt_l)) or
            (kier_g and self._czy_kolizja(punkt_g)) or
            (kier_d and self._czy_kolizja(punkt_d)),
            
            # Niebezpieczeństwo po prawej
            (kier_g and self._czy_kolizja(punkt_pr)) or
            (kier_d and self._czy_kolizja(punkt_l)) or
            (kier_l and self._czy_kolizja(punkt_g)) or
            (kier_pr and self._czy_kolizja(punkt_d)),
            
            # Niebezpieczeństwo po lewej
            (kier_d and self._czy_kolizja(punkt_pr)) or
            (kier_g and self._czy_kolizja(punkt_l)) or
            (kier_pr and self._czy_kolizja(punkt_g)) or
            (kier_l and self._czy_kolizja(punkt_d)),
            
            # Kierunek ruchu
            kier_l,
            kier_pr,
            kier_g,
            kier_d,
            
            # Lokalizacja jedzenia względem głowy
            self.jedzenie[0] < głowa[0],  # jedzenie po lewej
            self.jedzenie[0] > głowa[0],  # jedzenie po prawej
            self.jedzenie[1] < głowa[1],  # jedzenie powyżej
            self.jedzenie[1] > głowa[1]   # jedzenie poniżej
        ]
        
        return np.array(stan, dtype=np.float32)
    
    def _czy_kolizja(self, point=None):
        # Sprawdza, czy nastąpiła kolizja
        if point is None:
            point = self.snake[0]
            
        # Uderzenie w ścianę
        if (point[0] < 0 or point[0] >= self.szerokość or 
            point[1] < 0 or point[1] >= self.wysokość):
            return True
        
        # Uderzenie w siebie
        if point in self.snake[1:]:
            return True
            
        return False
    
    def krok(self, akcja):
        # Wykonanie akcji i przejście do następnego stanu
        self.iteracja_klatki += 1
        self.kroki_bez_jedzenia += 1
        
        # Obsługa zdarzeń pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Aktualizacja kierunku na podstawie akcji
        # [prosto, w prawo, w lewo]
        clock_wise = [Kierunek.PRAWO, Kierunek.DÓŁ, Kierunek.LEWO, Kierunek.GÓRA]
        indeks = clock_wise.index(self.Kierunek)
        
        if akcja == 0:  # Prosto
            nowy_kier = clock_wise[indeks]
        elif akcja == 1:  # W prawo
            next_idx = (indeks + 1) % 4
            nowy_kier = clock_wise[next_idx]
        else:  # W lewo
            next_idx = (indeks - 1) % 4
            nowy_kier = clock_wise[next_idx]
            
        self.Kierunek = nowy_kier
        
        # Aktualizacja pozycji głowy
        x = self.snake[0][0]
        y = self.snake[0][1]
        if self.Kierunek == Kierunek.PRAWO:
            x += self.rozmiar_bloku
        elif self.Kierunek == Kierunek.LEWO:
            x -= self.rozmiar_bloku
        elif self.Kierunek == Kierunek.DÓŁ:
            y += self.rozmiar_bloku
        elif self.Kierunek == Kierunek.GÓRA:
            y -= self.rozmiar_bloku
            
        self.głowa = [x, y]
        self.snake.insert(0, self.głowa)
        
        # Sprawdzenie, czy gra się zakończyła
        nagroda = 0
        koniec_gry = False
        
        # Kolizja lub przekroczenie limitu ruchów bez jedzenia
        # Bardziej agresywny limit dla długich węży (zmniejsza trenowanie na "chodzeniu w kółko")
        maks_kroków_bez_jedzenia = 100 * len(self.snake)
        if len(self.snake) > 10:
            maks_kroków_bez_jedzenia = 50 * len(self.snake)
            
        if self._czy_kolizja() or self.kroki_bez_jedzenia > maks_kroków_bez_jedzenia:
            koniec_gry = True
            nagroda = -10
            return self._pobierz_stan(), nagroda, koniec_gry, self.wynik
            
        # Zjedzenie jedzenia
        if self.głowa == self.jedzenie:
            self.wynik += 1
            nagroda = 10
            self.kroki_bez_jedzenia = 0
            self._umieść_jedzenie()
        else:
            self.snake.pop()  # usunięcie ostatniego segmentu węża, jeśli nie zjadł jedzenia
            
            # Dodatkowe nagrody za zbliżanie się do jedzenia (kształtowanie nagrody)
            poprz_odl_do_jedzenia = abs(self.snake[1][0] - self.jedzenie[0]) + abs(self.snake[1][1] - self.jedzenie[1])
            obecna_odl_do_jedzenia = abs(self.głowa[0] - self.jedzenie[0]) + abs(self.głowa[1] - self.jedzenie[1])
            
            if obecna_odl_do_jedzenia < poprz_odl_do_jedzenia:
                nagroda = 0.1  # Mała nagroda za zbliżanie się do jedzenia
            elif obecna_odl_do_jedzenia > poprz_odl_do_jedzenia:
                nagroda = -0.1  # Mała kara za oddalanie się od jedzenia
        
        # Aktualizacja wyświetlania
        self._aktualizuj_ui()
        self.zegar.tick(20)  # Kontrola szybkości gry
        
        # Zwrócenie nowego stanu, nagrody i informacji, czy gra się zakończyła
        return self._pobierz_stan(), nagroda, koniec_gry, self.wynik
    
    def _aktualizuj_ui(self):
        # Aktualizacja interfejsu graficznego
        self.ekran.fill(CZARNY)
        
        # Rysowanie węża
        for pt in self.snake:
            pygame.draw.rect(self.ekran, ZIELONY, pygame.Rect(pt[0], pt[1], self.rozmiar_bloku, self.rozmiar_bloku))
            pygame.draw.rect(self.ekran, CIEMNY_ZIELONY, pygame.Rect(pt[0] + 4, pt[1] + 4, self.rozmiar_bloku - 8, self.rozmiar_bloku - 8))
            
        # Rysowanie jedzenia
        pygame.draw.rect(self.ekran, CZERWONY, pygame.Rect(self.jedzenie[0], self.jedzenie[1], self.rozmiar_bloku, self.rozmiar_bloku))
        
        # Wyświetlanie wyniku
        czcionka = pygame.czcionka.SysFont('arial', 25)
        tekst = czcionka.render(f"Wynik: {self.wynik}", True, BIAŁY)
        self.ekran.blit(tekst, [0, 0])
        pygame.ekran.flip()


class UproszczonySnake:
    """Uproszczona wersja gry Snake bez interfejsu graficznego, zoptymalizowana pod kątem szybkości."""
    def __init__(self):
        # Inicjalizacja parametrów gry bez interfejsu graficznego
        self.szerokość = SZEROKOŚĆ_OKNA
        self.wysokość = WYSOKOŚĆ_OKNA
        self.rozmiar_bloku = ROZMIAR_BLOKU
        # Prekompilacja stałych
        self.max_x = (SZEROKOŚĆ_OKNA // ROZMIAR_BLOKU) - 1
        self.max_y = (WYSOKOŚĆ_OKNA // ROZMIAR_BLOKU) - 1
        self.reset()
    
    def reset(self):
        # Resetowanie gry do stanu początkowego
        self.Kierunek = Kierunek.PRAWO
        
        # Wąż zaczyna na środku planszy
        self.głowa = [self.szerokość // (2 * self.rozmiar_bloku) * self.rozmiar_bloku, 
                     self.wysokość // (2 * self.rozmiar_bloku) * self.rozmiar_bloku]
        
        # Początkowe segmenty węża
        self.snake = [
            self.głowa.copy(),  # Używamy kopii, aby uniknąć referencji
            [self.głowa[0] - self.rozmiar_bloku, self.głowa[1]],
            [self.głowa[0] - 2 * self.rozmiar_bloku, self.głowa[1]]
        ]
        
        self.wynik = 0
        self.jedzenie = None
        self._umieść_jedzenie()
        self.iteracja_klatki = 0
        self.kroki_bez_jedzenia = 0
        return self._pobierz_stan()
    
    def _umieść_jedzenie(self):
        # Umieszczenie jedzenia w losowym miejscu na planszy, ale nie na wężu
        while True:
            x = random.randint(0, self.max_x) * self.rozmiar_bloku
            y = random.randint(0, self.max_y) * self.rozmiar_bloku
            self.jedzenie = [x, y]
            if self.jedzenie not in self.snake:
                break
    
    def _pobierz_stan(self):
        # Zwraca obecny stan gry jako tablicę cech (identyczna funkcja jak w SnakeGame)
        głowa = self.snake[0]
        
        # Punkty wokół głowy
        punkt_l = [głowa[0] - self.rozmiar_bloku, głowa[1]]
        punkt_pr = [głowa[0] + self.rozmiar_bloku, głowa[1]]
        punkt_g = [głowa[0], głowa[1] - self.rozmiar_bloku]
        punkt_d = [głowa[0], głowa[1] + self.rozmiar_bloku]
        
        # Aktualne kierunki
        kier_l = self.Kierunek == Kierunek.LEWO
        kier_pr = self.Kierunek == Kierunek.PRAWO
        kier_g = self.Kierunek == Kierunek.GÓRA
        kier_d = self.Kierunek == Kierunek.DÓŁ
        
        # Stan jako lista cech
        stan = [
            # Niebezpieczeństwo przed sobą
            (kier_pr and self._czy_kolizja(punkt_pr)) or
            (kier_l and self._czy_kolizja(punkt_l)) or
            (kier_g and self._czy_kolizja(punkt_g)) or
            (kier_d and self._czy_kolizja(punkt_d)),
            
            # Niebezpieczeństwo po prawej
            (kier_g and self._czy_kolizja(punkt_pr)) or
            (kier_d and self._czy_kolizja(punkt_l)) or
            (kier_l and self._czy_kolizja(punkt_g)) or
            (kier_pr and self._czy_kolizja(punkt_d)),
            
            # Niebezpieczeństwo po lewej
            (kier_d and self._czy_kolizja(punkt_pr)) or
            (kier_g and self._czy_kolizja(punkt_l)) or
            (kier_pr and self._czy_kolizja(punkt_g)) or
            (kier_l and self._czy_kolizja(punkt_d)),
            
            # Kierunek ruchu
            kier_l,
            kier_pr,
            kier_g,
            kier_d,
            
            # Lokalizacja jedzenia względem głowy
            self.jedzenie[0] < głowa[0],  # jedzenie po lewej
            self.jedzenie[0] > głowa[0],  # jedzenie po prawej
            self.jedzenie[1] < głowa[1],  # jedzenie powyżej
            self.jedzenie[1] > głowa[1]   # jedzenie poniżej
        ]
        
        return np.array(stan, dtype=np.float32)
    
    def _czy_kolizja(self, point=None):
        # Sprawdza, czy nastąpiła kolizja (optymalizacja pod kątem szybkości)
        if point is None:
            point = self.snake[0]
            
        # Uderzenie w ścianę
        if (point[0] < 0 or point[0] >= self.szerokość or 
            point[1] < 0 or point[1] >= self.wysokość):
            return True
        
        # Uderzenie w siebie
        if point in self.snake[1:]:
            return True
            
        return False
    
    def krok(self, akcja):
        # Wykonanie akcji i przejście do następnego stanu (bez renderowania)
        self.iteracja_klatki += 1
        self.kroki_bez_jedzenia += 1
        
        # Aktualizacja kierunku na podstawie akcji - używamy prekompilowanych stałych
        # [prosto, w prawo, w lewo]
        clock_wise = [Kierunek.PRAWO, Kierunek.DÓŁ, Kierunek.LEWO, Kierunek.GÓRA]
        indeks = clock_wise.index(self.Kierunek)
        
        if akcja == 0:  # Prosto
            nowy_kier = clock_wise[indeks]
        elif akcja == 1:  # W prawo
            next_idx = (indeks + 1) % 4
            nowy_kier = clock_wise[next_idx]
        else:  # W lewo
            next_idx = (indeks - 1) % 4
            nowy_kier = clock_wise[next_idx]
            
        self.Kierunek = nowy_kier
        
        # Aktualizacja pozycji głowy
        x = self.snake[0][0]
        y = self.snake[0][1]
        if self.Kierunek == Kierunek.PRAWO:
            x += self.rozmiar_bloku
        elif self.Kierunek == Kierunek.LEWO:
            x -= self.rozmiar_bloku
        elif self.Kierunek == Kierunek.DÓŁ:
            y += self.rozmiar_bloku
        elif self.Kierunek == Kierunek.GÓRA:
            y -= self.rozmiar_bloku
            
        self.głowa = [x, y]
        self.snake.insert(0, self.głowa.copy())  # Używamy kopii, aby uniknąć referencji
        
        # Sprawdzenie, czy gra się zakończyła
        nagroda = 0
        koniec_gry = False
        
        # Kolizja lub przekroczenie limitu ruchów bez jedzenia
        maks_kroków_bez_jedzenia = 100 * len(self.snake)
        if len(self.snake) > 10:
            maks_kroków_bez_jedzenia = 50 * len(self.snake)
            
        if self._czy_kolizja() or self.kroki_bez_jedzenia > maks_kroków_bez_jedzenia:
            koniec_gry = True
            nagroda = -10
            return self._pobierz_stan(), nagroda, koniec_gry, self.wynik
            
        # Zjedzenie jedzenia
        if self.głowa == self.jedzenie:
            self.wynik += 1
            nagroda = 10
            self.kroki_bez_jedzenia = 0
            self._umieść_jedzenie()
        else:
            self.snake.pop()  # usunięcie ostatniego segmentu węża, jeśli nie zjadł jedzenia
            
            # Dodatkowe nagrody za zbliżanie się do jedzenia
            poprz_odl_do_jedzenia = abs(self.snake[1][0] - self.jedzenie[0]) + abs(self.snake[1][1] - self.jedzenie[1])
            obecna_odl_do_jedzenia = abs(self.głowa[0] - self.jedzenie[0]) + abs(self.głowa[1] - self.jedzenie[1])
            
            if obecna_odl_do_jedzenia < poprz_odl_do_jedzenia:
                nagroda = 0.1  # Mała nagroda za zbliżanie się do jedzenia
            elif obecna_odl_do_jedzenia > poprz_odl_do_jedzenia:
                nagroda = -0.1  # Mała kara za oddalanie się od jedzenia
        
        # Zwrócenie nowego stanu, nagrody i informacji, czy gra się zakończyła
        return self._pobierz_stan(), nagroda, koniec_gry, self.wynik
