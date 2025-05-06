"""
Moduł zawierający implementację gry Snake.
"""

import pygame
import random
import numpy as np
from enum import Enum
from config import (
    USE_GPU, USE_FLOAT16, CPU_THREAD_COUNT, WINDOW_WIDTH, 
    WINDOW_HEIGHT, BLOCK_SIZE, GAME_SPEED, HIDDEN_SIZE, MODELS_DIR
)

# Definicja kolorów
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_GREEN = (0, 200, 0)

# Definicja kierunków
class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class SnakeGame:
    """Pełna implementacja gry Snake z interfejsem graficznym do testowania."""
    def __init__(self):
        # Inicjalizacja parametrów gry
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.block_size = BLOCK_SIZE
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Snake AI - PyTorch')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        # Resetowanie gry do stanu początkowego
        self.direction = Direction.RIGHT
        
        # Wąż zaczyna na środku planszy
        self.head = [self.width // (2 * self.block_size) * self.block_size, 
                     self.height // (2 * self.block_size) * self.block_size]
        
        # Początkowe segmenty węża
        self.snake = [
            self.head,
            [self.head[0] - self.block_size, self.head[1]],
            [self.head[0] - 2 * self.block_size, self.head[1]]
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.steps_without_food = 0
        return self._get_state()
    
    def _place_food(self):
        # Umieszczenie jedzenia w losowym miejscu na planszy, ale nie na wężu
        max_x = (self.width // self.block_size) - 1
        max_y = (self.height // self.block_size) - 1
        
        while True:
            x = random.randint(0, max_x) * self.block_size
            y = random.randint(0, max_y) * self.block_size
            self.food = [x, y]
            if self.food not in self.snake:
                break
    
    def _get_state(self):
        # Zwraca obecny stan gry jako tablicę cech
        head = self.snake[0]
        
        # Punkty wokół głowy
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]
        
        # Aktualne kierunki
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        # Stan jako lista cech
        state = [
            # Niebezpieczeństwo przed sobą
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),
            
            # Niebezpieczeństwo po prawej
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),
            
            # Niebezpieczeństwo po lewej
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),
            
            # Kierunek ruchu
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Lokalizacja jedzenia względem głowy
            self.food[0] < head[0],  # jedzenie po lewej
            self.food[0] > head[0],  # jedzenie po prawej
            self.food[1] < head[1],  # jedzenie powyżej
            self.food[1] > head[1]   # jedzenie poniżej
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _is_collision(self, point=None):
        # Sprawdza, czy nastąpiła kolizja
        if point is None:
            point = self.snake[0]
            
        # Uderzenie w ścianę
        if (point[0] < 0 or point[0] >= self.width or 
            point[1] < 0 or point[1] >= self.height):
            return True
        
        # Uderzenie w siebie
        if point in self.snake[1:]:
            return True
            
        return False
    
    def step(self, action):
        # Wykonanie akcji i przejście do następnego stanu
        self.frame_iteration += 1
        self.steps_without_food += 1
        
        # Obsługa zdarzeń pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Aktualizacja kierunku na podstawie akcji
        # [prosto, w prawo, w lewo]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if action == 0:  # Prosto
            new_dir = clock_wise[idx]
        elif action == 1:  # W prawo
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # W lewo
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        # Aktualizacja pozycji głowy
        x = self.snake[0][0]
        y = self.snake[0][1]
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
            
        self.head = [x, y]
        self.snake.insert(0, self.head)
        
        # Sprawdzenie, czy gra się zakończyła
        reward = 0
        game_over = False
        
        # Kolizja lub przekroczenie limitu ruchów bez jedzenia
        # Bardziej agresywny limit dla długich węży (zmniejsza trenowanie na "chodzeniu w kółko")
        max_steps_without_food = 100 * len(self.snake)
        if len(self.snake) > 10:
            max_steps_without_food = 50 * len(self.snake)
            
        if self._is_collision() or self.steps_without_food > max_steps_without_food:
            game_over = True
            reward = -10
            return self._get_state(), reward, game_over, self.score
            
        # Zjedzenie jedzenia
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.steps_without_food = 0
            self._place_food()
        else:
            self.snake.pop()  # usunięcie ostatniego segmentu węża, jeśli nie zjadł jedzenia
            
            # Dodatkowe nagrody za zbliżanie się do jedzenia (kształtowanie nagrody)
            prev_dist_to_food = abs(self.snake[1][0] - self.food[0]) + abs(self.snake[1][1] - self.food[1])
            cur_dist_to_food = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
            
            if cur_dist_to_food < prev_dist_to_food:
                reward = 0.1  # Mała nagroda za zbliżanie się do jedzenia
            elif cur_dist_to_food > prev_dist_to_food:
                reward = -0.1  # Mała kara za oddalanie się od jedzenia
        
        # Aktualizacja wyświetlania
        self._update_ui()
        self.clock.tick(20)  # Kontrola szybkości gry
        
        # Zwrócenie nowego stanu, nagrody i informacji, czy gra się zakończyła
        return self._get_state(), reward, game_over, self.score
    
    def _update_ui(self):
        # Aktualizacja interfejsu graficznego
        self.display.fill(BLACK)
        
        # Rysowanie węża
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt[0], pt[1], self.block_size, self.block_size))
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(pt[0] + 4, pt[1] + 4, self.block_size - 8, self.block_size - 8))
            
        # Rysowanie jedzenia
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        
        # Wyświetlanie wyniku
        font = pygame.font.SysFont('arial', 25)
        text = font.render(f"Wynik: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


class SnakeGameSimple:
    """Uproszczona wersja gry Snake bez interfejsu graficznego, zoptymalizowana pod kątem szybkości."""
    def __init__(self, width=640, height=480, block_size=20):
        # Inicjalizacja parametrów gry bez interfejsu graficznego
        self.width = width
        self.height = height
        self.block_size = block_size
        # Prekompilacja stałych
        self.max_x = (width // block_size) - 1
        self.max_y = (height // block_size) - 1
        self.reset()
    
    def reset(self):
        # Resetowanie gry do stanu początkowego
        self.direction = Direction.RIGHT
        
        # Wąż zaczyna na środku planszy
        self.head = [self.width // (2 * self.block_size) * self.block_size, 
                     self.height // (2 * self.block_size) * self.block_size]
        
        # Początkowe segmenty węża
        self.snake = [
            self.head.copy(),  # Używamy kopii, aby uniknąć referencji
            [self.head[0] - self.block_size, self.head[1]],
            [self.head[0] - 2 * self.block_size, self.head[1]]
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.steps_without_food = 0
        return self._get_state()
    
    def _place_food(self):
        # Umieszczenie jedzenia w losowym miejscu na planszy, ale nie na wężu
        while True:
            x = random.randint(0, self.max_x) * self.block_size
            y = random.randint(0, self.max_y) * self.block_size
            self.food = [x, y]
            if self.food not in self.snake:
                break
    
    def _get_state(self):
        # Zwraca obecny stan gry jako tablicę cech (identyczna funkcja jak w SnakeGame)
        head = self.snake[0]
        
        # Punkty wokół głowy
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]
        
        # Aktualne kierunki
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        # Stan jako lista cech
        state = [
            # Niebezpieczeństwo przed sobą
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),
            
            # Niebezpieczeństwo po prawej
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),
            
            # Niebezpieczeństwo po lewej
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),
            
            # Kierunek ruchu
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Lokalizacja jedzenia względem głowy
            self.food[0] < head[0],  # jedzenie po lewej
            self.food[0] > head[0],  # jedzenie po prawej
            self.food[1] < head[1],  # jedzenie powyżej
            self.food[1] > head[1]   # jedzenie poniżej
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _is_collision(self, point=None):
        # Sprawdza, czy nastąpiła kolizja (optymalizacja pod kątem szybkości)
        if point is None:
            point = self.snake[0]
            
        # Uderzenie w ścianę
        if (point[0] < 0 or point[0] >= self.width or 
            point[1] < 0 or point[1] >= self.height):
            return True
        
        # Uderzenie w siebie
        if point in self.snake[1:]:
            return True
            
        return False
    
    def step(self, action):
        # Wykonanie akcji i przejście do następnego stanu (bez renderowania)
        self.frame_iteration += 1
        self.steps_without_food += 1
        
        # Aktualizacja kierunku na podstawie akcji - używamy prekompilowanych stałych
        # [prosto, w prawo, w lewo]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if action == 0:  # Prosto
            new_dir = clock_wise[idx]
        elif action == 1:  # W prawo
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # W lewo
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        # Aktualizacja pozycji głowy
        x = self.snake[0][0]
        y = self.snake[0][1]
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
            
        self.head = [x, y]
        self.snake.insert(0, self.head.copy())  # Używamy kopii, aby uniknąć referencji
        
        # Sprawdzenie, czy gra się zakończyła
        reward = 0
        game_over = False
        
        # Kolizja lub przekroczenie limitu ruchów bez jedzenia
        max_steps_without_food = 100 * len(self.snake)
        if len(self.snake) > 10:
            max_steps_without_food = 50 * len(self.snake)
            
        if self._is_collision() or self.steps_without_food > max_steps_without_food:
            game_over = True
            reward = -10
            return self._get_state(), reward, game_over, self.score
            
        # Zjedzenie jedzenia
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.steps_without_food = 0
            self._place_food()
        else:
            self.snake.pop()  # usunięcie ostatniego segmentu węża, jeśli nie zjadł jedzenia
            
            # Dodatkowe nagrody za zbliżanie się do jedzenia
            prev_dist_to_food = abs(self.snake[1][0] - self.food[0]) + abs(self.snake[1][1] - self.food[1])
            cur_dist_to_food = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
            
            if cur_dist_to_food < prev_dist_to_food:
                reward = 0.1  # Mała nagroda za zbliżanie się do jedzenia
            elif cur_dist_to_food > prev_dist_to_food:
                reward = -0.1  # Mała kara za oddalanie się od jedzenia
        
        # Zwrócenie nowego stanu, nagrody i informacji, czy gra się zakończyła
        return self._get_state(), reward, game_over, self.score
