"""
GUI dla Snake AI.
Każdy algorytm to osobna zakładka (DQN, A*, może coś w przyszłości).
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import os
import time
import math
import platform
import subprocess
import numpy as np
import torch

from snake_game import UproszczonySnake, Kierunek
from agent import DQNAgent
from training import uruchom_epizody
import config
from config import (
    SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU,
    ROZMIAR_UKRYTY, KATALOG_MODELI, ITERACJE_UCZENIA,
)

# ── Stałe wyświetlania ────────────────────────────────────────────────
SCALE = 2                          # mnożnik rozdzielczości podglądu
BLOCK = ROZMIAR_BLOKU * SCALE      # 80 px na blok
CW    = SZEROKOŚĆ_OKNA  * SCALE    # 640 px – szerokość canvasu
CH    = WYSOKOŚĆ_OKNA   * SCALE    # 480 px – wysokość canvasu

# Paleta kolorów (hex – dla tkinter)
C = {
    'bg':      '#0b0b14',
    'panel':   '#131320',
    'panel2':  '#1a1a2a',
    'border':  '#2c2c42',
    'accent':  '#00d264',
    'accent2': '#3c91ff',
    'accent3': '#ffa528',
    'danger':  '#c33232',
    'text':    '#dadae4',
    'text2':   '#87879e',
    'text3':   '#4c4c66',
    'btn':     '#202034',
    'grn':     '#009444',
    'grn_h':   '#00b054',
    'blue':    '#2662c3',
    'blue_h':  '#3878d9',
    'stop':    '#962626',
    'stop_h':  '#be3232',
    'snake_h': '#00e66c',
    'snake_b': '#00b654',
    'snake_i': '#009644',
    'food':    '#e13737',
    'food_i':  '#ff6e48',
    'grid':    '#101018',
    'gridl':   '#151523',
    'gline':   '#00d264',
    'gavg':    '#3c91ff',
    'gbg':     '#0d0d17',
}


# ══════════════════════════════════════════════════════════════════════
# Wykrywanie sprzętu
# ══════════════════════════════════════════════════════════════════════

def _nazwa_procesora() -> str:
    """Pełna nazwa procesora, np. 'AMD Ryzen 7 5800H with Radeon Graphics'."""
    system = platform.system()
    try:
        if system == 'Windows':
            import winreg
            klucz = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(klucz, 'ProcessorNameString')[0].strip()
        if system == 'Linux':
            with open('/proc/cpuinfo', encoding='utf-8') as f:
                for linia in f:
                    if linia.startswith('model name'):
                        return linia.split(':', 1)[1].strip()
        if system == 'Darwin':
            return subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    except Exception:
        pass
    return platform.processor() or 'nieznany procesor'


def wykryj_sprzęt() -> dict:
    """Zwraca {'cpu': opis procesora, 'gpu': opis karty lub None}."""
    n = os.cpu_count() or 0
    # polska odmiana: 2–4, 22–24… → „wątki”, reszta (5–21, 25…) → „wątków”
    słowo = 'wątki' if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else 'wątków'
    cpu = f'{_nazwa_procesora()} ({n} {słowo})'
    gpu = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = f'{p.name} ({p.total_memory / 1024**3:.0f} GB VRAM)'
    return {'cpu': cpu, 'gpu': gpu}


class Podpowiedź:
    """Dymek z opisem, który pojawia się po najechaniu myszką na widżet."""

    def __init__(self, widget, tekst: str):
        self.w, self.tekst, self.okno = widget, tekst, None
        widget.bind('<Enter>', self._pokaż, add='+')
        widget.bind('<Leave>', self._ukryj, add='+')
        widget.bind('<ButtonPress>', self._ukryj, add='+')

    def _pokaż(self, _e=None):
        if self.okno:
            return
        x = self.w.winfo_rootx() + 10
        y = self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.okno = tk.Toplevel(self.w)
        self.okno.wm_overrideredirect(True)        # bez ramki okna
        self.okno.wm_geometry(f'+{x}+{y}')
        tk.Label(self.okno, text=self.tekst, justify='left',
                 bg='#26263a', fg=C['text'], font=('Segoe UI', 9),
                 relief='solid', borderwidth=1, wraplength=320,
                 padx=8, pady=5).pack()

    def _ukryj(self, _e=None):
        if self.okno:
            self.okno.destroy()
            self.okno = None


# ══════════════════════════════════════════════════════════════════════
# GameCanvas – podgląd gry
# ══════════════════════════════════════════════════════════════════════

class GameCanvas(tk.Frame):
    """Rysuje stan gry Snake na tk.Canvas przy użyciu prostokątów.
    Rozmiar siatki jest dynamiczny.
    """

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C['panel'], **kw)
        self.canvas = tk.Canvas(
            self, width=CW, height=CH,
            bg=C['bg'],
            highlightthickness=1,
            highlightbackground=C['border'],
        )
        self.canvas.pack(padx=2, pady=2, fill='both', expand=True)
        self._grid_n     = 8           # komórek na bok
        self._game_block = ROZMIAR_BLOKU   # px/komórkę w logice gry (zawsze 40)
        self._cw, self._ch = CW, CH    # aktualny rozmiar canvasu (zmienia się z oknem)
        self._last_gs, self._last_tick = None, 0
        self._recalc()
        self._draw_grid()
        self.canvas.bind('<Configure>', self._on_resize)

    def _on_resize(self, e):
        """Okno zmieniło rozmiar – przeskaluj planszę i narysuj ją od nowa."""
        if (e.width, e.height) == (self._cw, self._ch):
            return
        self._cw, self._ch = e.width, e.height
        self._recalc()
        self.canvas.delete('all')
        self._draw_grid()
        if self._last_gs is not None:
            self.update_state(self._last_gs, self._last_tick)

    def _recalc(self):
        """Przelicza rozmiar bloku wyświetlania i offset centrowania."""
        self._disp_block = max(4, min(self._cw, self._ch) // self._grid_n)
        area = self._disp_block * self._grid_n
        self._ox = (self._cw - area) // 2
        self._oy = (self._ch - area) // 2

    def set_grid(self, grid_n: int):
        """Zmienia rozmiar siatki i przerysowuje siatkę tła."""
        self._grid_n = grid_n
        self._recalc()
        self.canvas.delete('all')
        self._draw_grid()

    def _draw_grid(self):
        db, n    = self._disp_block, self._grid_n
        ox, oy   = self._ox, self._oy
        area     = db * n
        self.canvas.create_rectangle(ox, oy, ox + area, oy + area,
                                     fill=C['grid'], outline='', tags='grid')
        for i in range(n + 1):
            self.canvas.create_line(ox + i*db, oy, ox + i*db, oy + area,
                                    fill=C['gridl'], tags='grid')
            self.canvas.create_line(ox, oy + i*db, ox + area, oy + i*db,
                                    fill=C['gridl'], tags='grid')

    def update_state(self, gs, tick: int = 0):
        """
        gs = {'snake': [[px,py],...], 'food': [px,py], 'score': int,
              'path': [(col,row),...] lub None}
        px/py to współrzędne pikselowe z UproszczonySnake (wielokrotności _game_block).
        """
        self.canvas.delete('dyn')
        self._last_gs, self._last_tick = gs, tick
        if gs is None:
            return

        db = self._disp_block
        gb = self._game_block
        ox, oy = self._ox, self._oy

        # Ścieżka A* (komórki pośrednie – bez głowy i jedzenia)
        path = gs.get('path')
        if path and len(path) > 2:
            r = max(2, db * 3 // 10)
            for col, row in path[1:-1]:
                cx, cy = ox + col * db + db // 2, oy + row * db + db // 2
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill='#1a4a70', outline='', tags='dyn')

        # Wąż
        for i, (px, py) in enumerate(gs['snake']):
            cx, cy = px // gb, py // gb        # piksel → indeks komórki
            x1, y1 = ox + cx * db, oy + cy * db
            x2, y2 = x1 + db, y1 + db
            fill = C['snake_h'] if i == 0 else C['snake_b']
            inn  = C['snake_i'] if i > 0  else '#00af54'
            self.canvas.create_rectangle(x1, y1, x2, y2,
                                         fill=fill, outline='', tags='dyn')
            pad = max(2, db // 10)
            self.canvas.create_rectangle(x1 + pad, y1 + pad,
                                         x2 - pad, y2 - pad,
                                         fill=inn, outline='', tags='dyn')

        self._draw_face(gs['snake'], ox, oy, db, gb, tick)

        # Jedzenie z pulsowaniem (po wygranej plansza jest pełna – brak jedzenia)
        if not gs.get('won'):
            fx, fy = gs['food'][0] // gb, gs['food'][1] // gb
            x1, y1 = ox + fx * db, oy + fy * db
            pulse  = int(abs(math.sin(tick * 0.1)) * max(1, db // 8))
            self.canvas.create_rectangle(x1, y1, x1 + db, y1 + db,
                                         fill=C['food'], outline='', tags='dyn')
            p2 = max(2, db // 5) + pulse
            self.canvas.create_rectangle(x1 + p2, y1 + p2,
                                         x1 + db - p2, y1 + db - p2,
                                         fill=C['food_i'], outline='', tags='dyn')

        # Komunikat na planszy (pauza / koniec gry / wygrana)
        overlay = gs.get('overlay') or (['WYGRANA!'] if gs.get('won') else None)
        if overlay:
            self._draw_overlay(overlay)

    def _draw_face(self, snake, ox, oy, db, gb, tick):
        """Oczy i język na głowie, obrócone w stronę ruchu."""
        hx, hy = snake[0][0] // gb, snake[0][1] // gb
        if len(snake) > 1:
            nx, ny = snake[1][0] // gb, snake[1][1] // gb
            dx, dy = hx - nx, hy - ny        # (1,0)=prawo, (0,-1)=góra itd.
        else:
            dx, dy = 1, 0
        cx = ox + hx * db + db / 2           # środek głowy w pikselach
        cy = oy + hy * db + db / 2
        px, py = -dy, dx                     # kierunek "na boki"

        # Oczy: białko + źrenica patrząca do przodu
        for strona in (-1, 1):
            ex = cx + dx * db * 0.15 + px * strona * db * 0.22
            ey = cy + dy * db * 0.15 + py * strona * db * 0.22
            r = db * 0.13
            self.canvas.create_oval(ex - r, ey - r, ex + r, ey + r,
                                    fill='white', outline='', tags='dyn')
            r2 = r * 0.55
            zx, zy = ex + dx * r * 0.35, ey + dy * r * 0.35
            self.canvas.create_oval(zx - r2, zy - r2, zx + r2, zy + r2,
                                    fill='black', outline='', tags='dyn')

        # Język – miga (widoczny przez pół sekundy co sekundę)
        if (tick // 15) % 2 == 0:
            tx, ty = cx + dx * db * 0.5, cy + dy * db * 0.5
            self.canvas.create_line(tx, ty,
                                    tx + dx * db * 0.25, ty + dy * db * 0.25,
                                    fill='#ff3355',
                                    width=max(2, int(db * 0.06)), tags='dyn')

    def _draw_overlay(self, lines: list):
        """Półprzezroczysty pas z tekstem; pierwsza linia duża."""
        h = 40 + 28 * len(lines)
        y0 = self._ch // 2 - h // 2
        self.canvas.create_rectangle(0, y0, self._cw, y0 + h, fill='#000000',
                                     stipple='gray50', outline='', tags='dyn')
        cy = y0 + 34
        for i, line in enumerate(lines):
            font = ('Segoe UI', 28, 'bold') if i == 0 else ('Segoe UI', 12)
            self.canvas.create_text(self._cw // 2, cy, text=line, fill='#ffffff',
                                    font=font, tags='dyn')
            cy += 40 if i == 0 else 26

    def show_message(self, lines: list, color: str = C['text3']):
        self.canvas.delete('dyn')
        cy = self._ch // 2 - len(lines) * 15
        for line in lines:
            col  = color if line else C['text3']
            size = 13 if line else 8
            self.canvas.create_text(
                self._cw // 2, cy, text=line,
                fill=col, font=('Segoe UI', size), tags='dyn',
            )
            cy += 28


# ══════════════════════════════════════════════════════════════════════
# StatsGraph – wykres live na tk.Canvas
# ══════════════════════════════════════════════════════════════════════

class StatsGraph(tk.Canvas):
    """Wykres wyników: linia surowa (zielona) + średnia ruchoma 20 ep (niebieska)."""

    def __init__(self, parent, **kw):
        kw.setdefault('bg', C['gbg'])
        kw.setdefault('highlightthickness', 1)
        kw.setdefault('highlightbackground', C['border'])
        super().__init__(parent, **kw)
        self.data: list = []
        self.avg:  list = []
        self._placeholder()

    def _placeholder(self):
        self.delete('all')
        self.create_text(150, 25, text='Oczekiwanie na dane…',
                         fill=C['text3'], font=('Segoe UI', 9))

    def add(self, v: float):
        self.data.append(float(v))
        w = min(20, len(self.data))
        self.avg.append(float(np.mean(self.data[-w:])))
        self._redraw()

    def clear(self):
        self.data.clear()
        self.avg.clear()
        self._placeholder()

    def _redraw(self):
        self.delete('all')
        w   = self.winfo_width()  or 300
        h   = self.winfo_height() or 100
        pad = 8
        pts = self.data[-200:]
        avs = self.avg[-200:]
        n   = len(pts)
        if n < 2:
            return
        iw  = w - 2 * pad
        ih  = h - 2 * pad - 14
        lo  = min(min(pts), 0)
        hi  = max(max(pts), 1)
        rng = hi - lo or 1

        def coord(i, v):
            return (pad + i / (n - 1) * iw,
                    pad + (1 - (v - lo) / rng) * ih)

        raw = [coord(i, v) for i, v in enumerate(pts)]
        avg = [coord(i, v) for i, v in enumerate(avs)]

        if len(raw) >= 2:
            flat = [c for p in raw for c in p]
            self.create_line(*flat, fill=C['gline'], width=1)
        if len(avg) >= 2:
            flat = [c for p in avg for c in p]
            self.create_line(*flat, fill=C['gavg'], width=1)

        self.create_text(pad, h - 4, text=f'Max {int(hi)}',
                         fill=C['text3'], font=('Segoe UI', 7), anchor='w')
        self.create_text(w - pad, h - 4, text=f'n={len(self.data)}',
                         fill=C['text3'], font=('Segoe UI', 7), anchor='e')


# ══════════════════════════════════════════════════════════════════════
# ModelList – lista plików .pth
# ══════════════════════════════════════════════════════════════════════

class ModelList(tk.Frame):
    """Przeglądarka zapisanych modeli z katalogu models/."""

    def __init__(self, parent, on_select=None, **kw):
        super().__init__(parent, bg=C['panel'], **kw)
        self.on_select = on_select

        hdr = tk.Frame(self, bg=C['panel'])
        hdr.pack(fill='x', padx=8, pady=(6, 2))
        tk.Label(hdr, text='Dostępne modele',
                 bg=C['panel'], fg=C['text'],
                 font=('Segoe UI', 10, 'bold')).pack(side='left')
        tk.Button(hdr, text='↺', command=self.reload,
                  bg=C['btn'], fg=C['text2'],
                  font=('Segoe UI', 9), relief='flat',
                  padx=4, cursor='hand2').pack(side='right')

        frame = tk.Frame(self, bg=C['panel'])
        frame.pack(fill='both', expand=True, padx=6, pady=(0, 6))

        sb = tk.Scrollbar(frame)
        sb.pack(side='right', fill='y')

        self.lb = tk.Listbox(
            frame,
            yscrollcommand=sb.set,
            bg=C['panel2'], fg=C['text'],
            selectbackground='#004b2a',
            selectforeground=C['accent'],
            font=('Segoe UI', 9),
            borderwidth=0, highlightthickness=0,
            activestyle='none',
        )
        self.lb.pack(fill='both', expand=True)
        sb.config(command=self.lb.yview)
        self.lb.bind('<<ListboxSelect>>', self._sel)

        self.reload()

    def reload(self):
        self.lb.delete(0, 'end')
        if os.path.exists(KATALOG_MODELI):
            models = sorted(
                [f for f in os.listdir(KATALOG_MODELI) if f.endswith('.pth')],
                key=lambda f: os.path.getmtime(os.path.join(KATALOG_MODELI, f)),
                reverse=True,
            )
            for m in models:
                self.lb.insert('end', m)
        if self.lb.size() > 0:
            self.lb.selection_set(0)
            if self.on_select:
                self.on_select(self.lb.get(0))

    def _sel(self, _):
        sel = self.lb.curselection()
        if sel and self.on_select:
            self.on_select(self.lb.get(sel[0]))

    def selected(self):
        sel = self.lb.curselection()
        return self.lb.get(sel[0]) if sel else None


# ══════════════════════════════════════════════════════════════════════
# DQNPanel – konfiguracja i sterowanie DQN
# ══════════════════════════════════════════════════════════════════════

OPISY_STATYSTYK = {
    'best':    'Najwyższy wynik w tym treningu – przy każdym nowym rekordzie '
               'model zapisuje się jako <nazwa>_best_<siatka>.pth.',
    'epsilon': 'Szansa na losowy ruch (eksploracja). Startuje od 1.0 i maleje – '
               'agent coraz bardziej ufa temu, czego się nauczył.',
    'loss':    'Błąd sieci: jak bardzo przewidywane wartości Q różnią się od '
               'docelowych. Powinien maleć i się stabilizować.',
}


class DQNPanel(tk.Frame):
    """Panel zakładki DQN: konfiguracja, statystyki, wykres, przyciski."""

    EP_OPTS    = [50, 100, 200, 500, 1000, 5000, 10000, 25000, 50000, 100000]
    PAR_OPTS   = [1, 2, 4, 8, 16, 32, 64, 128]
    DELAY_OPTS = [1, 10, 25, 50, 100, 200]   # ms opóźnienia między krokami CPU
    GRID_OPTS  = [6, 8, 10, 12, 16, 20]       # komórek na bok (kwadratowa siatka)

    def __init__(self, parent, app, **kw):
        super().__init__(parent, bg=C['panel'], **kw)
        self.app     = app
        self.ep_i    = 2   # domyślnie 200 epizodów
        self.par_i   = 0
        # Urządzenie dla sieci: GPU tylko jeśli PyTorch widzi kartę z CUDA
        self.DEV_OPTS = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
        self.dev_i   = 0 if config.UŻYJ_GPU else len(self.DEV_OPTS) - 1
        self.delay_i = 1   # domyślnie 10 ms
        self.grid_i  = 1   # domyślnie 8×8
        self.sv: dict = {}
        self._build()

    # ── Budowa UI ────────────────────────────────────────────────────

    def _build(self):
        pad = {'padx': 10, 'pady': (6, 2)}

        # Konfiguracja
        cfg = tk.LabelFrame(self, text=' Konfiguracja ',
                            bg=C['panel'], fg=C['text2'],
                            font=('Segoe UI', 9), relief='groove')
        cfg.pack(fill='x', **pad)

        self._ep_var    = tk.StringVar(value=str(self.EP_OPTS[self.ep_i]))
        self._par_var   = tk.StringVar(value=str(self.PAR_OPTS[self.par_i]))
        self._delay_var = tk.StringVar(value=f'{self.DELAY_OPTS[self.delay_i]} ms')
        self._grid_var  = tk.StringVar(value=f'{self.GRID_OPTS[self.grid_i]}×{self.GRID_OPTS[self.grid_i]}')

        self._stepper(cfg, 'Epizody:', self._ep_var,
                      self._ep_minus, self._ep_plus, row=0,
                      opis='Ile gier łącznie rozegra agent podczas treningu.')
        self._stepper(cfg, 'Gier równolegle:', self._par_var,
                      self._par_minus, self._par_plus, row=2,
                      opis='Tylko trening w tle: tyle gier toczy się naraz, a decyzje '
                           'dla wszystkich liczy jedno zapytanie do sieci. Potem agent '
                           'uczy się na zebranych ruchach. Więcej gier = lepsze '
                           'wykorzystanie karty graficznej.')
        self._stepper(cfg, 'Opóźnienie podglądu:', self._delay_var,
                      self._delay_minus, self._delay_plus, row=4,
                      opis='Tylko trening z podglądem: pauza między ruchami, żeby '
                           'dało się oglądać grę. 1 ms = najszybciej.')
        self._stepper(cfg, 'Siatka:', self._grid_var,
                      self._grid_minus, self._grid_plus, row=6,
                      opis='Rozmiar planszy w kratkach. Większa plansza = '
                           'trudniejsza gra i dłuższy trening.')
        self._dev_var = tk.StringVar(value=self._nazwa_urządzenia())
        self._stepper(cfg, 'Sieć liczona na:', self._dev_var,
                      self._dev_zmień, self._dev_zmień, row=8,
                      opis=('GPU = karta graficzna, CPU = procesor. Dotyczy uczenia '
                            'i decyzji sieci; sama gra zawsze działa na procesorze.')
                           if len(self.DEV_OPTS) > 1 else
                           'Brak karty z obsługą CUDA – dostępny tylko procesor.')

        # Nazwa zapisywanych modeli (puste pole = nazwa domyślna)
        tk.Label(cfg, text='Nazwa modelu:', bg=C['panel'], fg=C['text2'],
                 font=('Segoe UI', 9)).grid(row=10, column=0, sticky='w',
                                            padx=(4, 8), pady=(3, 0))
        self._nazwa_var = tk.StringVar()
        tk.Entry(cfg, textvariable=self._nazwa_var, width=16,
                 bg=C['panel2'], fg=C['accent'], insertbackground=C['text'],
                 relief='flat', font=('Segoe UI', 10, 'bold')).grid(
            row=10, column=1, columnspan=3, sticky='we', pady=(3, 0))
        self._pliki_var = tk.StringVar()
        tk.Label(cfg, textvariable=self._pliki_var, bg=C['panel'], fg=C['text3'],
                 font=('Segoe UI', 8), justify='left', wraplength=380).grid(
            row=11, column=0, columnspan=4, sticky='w', padx=4, pady=(0, 4))
        self._nazwa_var.trace_add('write', lambda *_: self._odśwież_pliki())
        self._odśwież_pliki()

        # Sprzęt
        hw = tk.LabelFrame(self, text=' Sprzęt ',
                           bg=C['panel'], fg=C['text2'],
                           font=('Segoe UI', 9), relief='groove')
        hw.pack(fill='x', **pad)
        sprzęt = wykryj_sprzęt()

        wiersze = [
            ('Procesor:', sprzęt['cpu'], C['text']),
            ('Karta graficzna:', sprzęt['gpu'] or 'brak obsługi CUDA',
             C['text'] if sprzęt['gpu'] else C['text3']),
        ]
        for r, (etykieta, wartość, kolor) in enumerate(wiersze):
            tk.Label(hw, text=etykieta, bg=C['panel'], fg=C['text2'],
                     font=('Segoe UI', 9)).grid(row=r, column=0, sticky='nw',
                                                 padx=(4, 8), pady=2)
            tk.Label(hw, text=wartość, bg=C['panel'], fg=kolor,
                     font=('Segoe UI', 9, 'bold'), justify='left',
                     wraplength=260).grid(row=r, column=1, sticky='w', pady=2)
        tk.Label(hw, text='Logika gry zawsze działa na procesorze; karta '
                          'graficzna przyspiesza tylko obliczenia sieci.',
                 bg=C['panel'], fg=C['text3'], font=('Segoe UI', 8),
                 justify='left', wraplength=380).grid(
            row=len(wiersze), column=0, columnspan=2, sticky='w',
            padx=4, pady=(0, 4))

        # Statystyki
        sts = tk.LabelFrame(self, text=' Statystyki ',
                            bg=C['panel'], fg=C['text2'],
                            font=('Segoe UI', 9), relief='groove')
        sts.pack(fill='x', **pad)

        for k in ('score', 'best', 'episode', 'epsilon', 'loss'):
            self.sv[k] = tk.StringVar(value='—')

        colors = {
            'score':   C['accent'],
            'best':    C['accent3'],
            'episode': C['text'],
            'epsilon': C['accent2'],
            'loss':    C['text2'],
        }
        pairs = [
            ('Wynik:',    'score',   0),
            ('Najlepszy:', 'best',   1),
            ('Epizod:',   'episode', 0),
            ('Epsilon:',  'epsilon', 1),
        ]
        for idx, (lbl, key, col_idx) in enumerate(pairs):
            row = idx // 2
            c   = col_idx * 2
            etykieta = tk.Label(sts, text=lbl, bg=C['panel'], fg=C['text2'],
                                font=('Segoe UI', 9))
            etykieta.grid(row=row, column=c, sticky='w', padx=(4, 2), pady=2)
            if key in OPISY_STATYSTYK:
                Podpowiedź(etykieta, OPISY_STATYSTYK[key])
            tk.Label(sts, textvariable=self.sv[key],
                     bg=C['panel'], fg=colors[key],
                     font=('Segoe UI', 11, 'bold')).grid(
                row=row, column=c + 1, sticky='w', padx=(0, 14))

        etykieta = tk.Label(sts, text='Loss:', bg=C['panel'], fg=C['text2'],
                            font=('Segoe UI', 9))
        etykieta.grid(row=2, column=0, sticky='w', padx=(4, 2))
        Podpowiedź(etykieta, OPISY_STATYSTYK['loss'])
        tk.Label(sts, textvariable=self.sv['loss'],
                 bg=C['panel'], fg=C['text2'],
                 font=('Segoe UI', 9)).grid(
            row=2, column=1, columnspan=3, sticky='w')

        # Wykres
        self.graph = StatsGraph(self, height=100)
        self.graph.pack(fill='x', padx=10, pady=(2, 4))

        # Przyciski
        bf = tk.Frame(self, bg=C['panel'])
        bf.pack(fill='x', padx=10, pady=(2, 6))

        def btn(parent, text, cmd, bg, bg_h=None):
            return tk.Button(
                parent, text=text, command=cmd,
                bg=bg, fg=C['text'], relief='flat',
                font=('Segoe UI', 10), pady=5,
                activebackground=bg_h or bg,
                activeforeground='#ffffff',
                cursor='hand2',
                disabledforeground=C['text3'],
            )

        self.btn_hybrid = btn(bf, '▶  Trening w tle',
                              lambda: self.app.go_train('hybrid'),
                              C['grn'], C['grn_h'])
        self.btn_hybrid.pack(fill='x', pady=2)

        self.btn_cpu = btn(bf, '▶  Trening z podglądem',
                           lambda: self.app.go_train('cpu'),
                           C['btn'], '#32324c')
        self.btn_cpu.pack(fill='x', pady=2)

        self.btn_test = btn(bf, '▶  Testuj',
                            self.app.go_test, C['blue'], C['blue_h'])
        self.btn_test.pack(fill='x', pady=2)

        row2 = tk.Frame(bf, bg=C['panel'])
        row2.pack(fill='x', pady=2)
        self.btn_cont_h = btn(row2, '↺  Kontynuuj w tle',
                              lambda: self.app.go_continue('hybrid'),
                              C['btn'], '#32324c')
        self.btn_cont_h.pack(side='left', fill='x', expand=True, padx=(0, 2))
        self.btn_cont_c = btn(row2, '↺  Kontynuuj z podgl.',
                              lambda: self.app.go_continue('cpu'),
                              C['btn'], '#32324c')
        self.btn_cont_c.pack(side='left', fill='x', expand=True, padx=(2, 0))

        self.btn_stop = btn(bf, '■  ZATRZYMAJ',
                            self.app.stop, C['stop'], C['stop_h'])
        self.btn_stop.pack(fill='x', pady=(6, 2))
        self.btn_stop.config(state='disabled')

        self.lbl_status = tk.Label(bf, text='Gotowy',
                                   bg=C['panel'], fg=C['text2'],
                                   font=('Segoe UI', 9))
        self.lbl_status.pack(anchor='w')

        # Opisy przycisków – pokazują się po najechaniu myszką
        for przycisk, tekst in (
            (self.btn_hybrid, 'Agent rozgrywa naraz tyle gier, ile ustawiono w '
                              '„Gier równolegle” (bez podglądu), potem uczy się na '
                              'zebranych ruchach. Najszybszy tryb.'),
            (self.btn_cpu,    'Agent uczy się po każdym ruchu, a gra jest '
                              'widoczna na planszy. Wolniej, ale widać postępy. '
                              'Urządzenie wybierasz w „Sieć liczona na”.'),
            (self.btn_test,   'Gra modelem zaznaczonym na liście – bez uczenia '
                              'i bez losowych ruchów.'),
            (self.btn_cont_h, 'Wczytuje zaznaczony model i trenuje go dalej '
                              'w tle.'),
            (self.btn_cont_c, 'Wczytuje zaznaczony model i trenuje go dalej '
                              'z podglądem.'),
            (self.btn_stop,   'Przerywa trening lub test. Przerwany trening '
                              'też zapisuje model finalny (<nazwa>_final_<siatka>.pth).'),
        ):
            Podpowiedź(przycisk, tekst)

    def _stepper(self, parent, label, var, minus_cmd, plus_cmd, row: int,
                 opis: str = ''):
        """Wiersz '−  wartość  +'; pod nim (row+1) szary opis opcji."""
        tk.Label(parent, text=label, bg=C['panel'], fg=C['text2'],
                 font=('Segoe UI', 9)).grid(
            row=row, column=0, sticky='w', padx=(4, 8), pady=(3, 0))
        if opis:
            tk.Label(parent, text=opis, bg=C['panel'], fg=C['text3'],
                     font=('Segoe UI', 8), justify='left',
                     wraplength=380).grid(row=row + 1, column=0, columnspan=4,
                                          sticky='w', padx=4, pady=(0, 4))
        tk.Button(parent, text='−', command=minus_cmd,
                  bg=C['btn'], fg=C['text'], relief='flat',
                  font=('Segoe UI', 10, 'bold'), width=2,
                  cursor='hand2').grid(row=row, column=1, padx=(0, 2))
        tk.Label(parent, textvariable=var,
                 bg=C['panel2'], fg=C['accent'],
                 font=('Segoe UI', 10, 'bold'), width=5,
                 anchor='center').grid(row=row, column=2, padx=2)
        tk.Button(parent, text='+', command=plus_cmd,
                  bg=C['btn'], fg=C['text'], relief='flat',
                  font=('Segoe UI', 10, 'bold'), width=2,
                  cursor='hand2').grid(row=row, column=3, padx=(2, 0))

    # ── Steppers ─────────────────────────────────────────────────────

    def _ep_minus(self):
        self.ep_i = max(0, self.ep_i - 1)
        self._ep_var.set(str(self.EP_OPTS[self.ep_i]))

    def _ep_plus(self):
        self.ep_i = min(len(self.EP_OPTS) - 1, self.ep_i + 1)
        self._ep_var.set(str(self.EP_OPTS[self.ep_i]))

    def _par_minus(self):
        self.par_i = max(0, self.par_i - 1)
        self._par_var.set(str(self.PAR_OPTS[self.par_i]))

    def _par_plus(self):
        self.par_i = min(len(self.PAR_OPTS) - 1, self.par_i + 1)
        self._par_var.set(str(self.PAR_OPTS[self.par_i]))

    def _delay_minus(self):
        self.delay_i = max(0, self.delay_i - 1)
        self._delay_var.set(f'{self.DELAY_OPTS[self.delay_i]} ms')

    def _delay_plus(self):
        self.delay_i = min(len(self.DELAY_OPTS) - 1, self.delay_i + 1)
        self._delay_var.set(f'{self.DELAY_OPTS[self.delay_i]} ms')

    def _grid_minus(self):
        self.grid_i = max(0, self.grid_i - 1)
        n = self.GRID_OPTS[self.grid_i]
        self._grid_var.set(f'{n}×{n}')
        self._odśwież_pliki()
        self.app.podgląd_siatki(n)

    def _grid_plus(self):
        self.grid_i = min(len(self.GRID_OPTS) - 1, self.grid_i + 1)
        n = self.GRID_OPTS[self.grid_i]
        self._grid_var.set(f'{n}×{n}')
        self._odśwież_pliki()
        self.app.podgląd_siatki(n)

    # ── Nazwy zapisywanych modeli ────────────────────────────────────

    DOMYŚLNA_NAZWA = 'snake_model'

    def _nazwa_bazowa(self) -> str:
        """Nazwa z pola bez znaków zakazanych w nazwach plików Windows."""
        nazwa = self._nazwa_var.get().strip()
        if nazwa.lower().endswith('.pth'):
            nazwa = nazwa[:-4]
        nazwa = ''.join('_' if z in '<>:"/\\|?*' else z for z in nazwa)
        nazwa = nazwa.strip(' .')
        return nazwa or self.DOMYŚLNA_NAZWA

    def nazwy_plików(self) -> tuple:
        """(najlepszy, finalny), np. ('snake_model_best_8x8.pth', ...)."""
        n = self.grid_size
        baza = self._nazwa_bazowa()
        return f'{baza}_best_{n}x{n}.pth', f'{baza}_final_{n}x{n}.pth'

    def _odśwież_pliki(self):
        best, final = self.nazwy_plików()
        puste = '' if self._nazwa_var.get().strip() else 'Puste pole = nazwa domyślna. '
        self._pliki_var.set(f'{puste}Zapisze: {best} i {final}')

    # ── API dla App ──────────────────────────────────────────────────

    def update_stats(self, d: dict):
        """Wywoływana wyłącznie z głównego wątku (via after)."""
        for k, v in d.items():
            if k not in self.sv:
                continue
            if k == 'epsilon':
                self.sv[k].set(f'{float(v):.4f}')
            elif k == 'loss':
                self.sv[k].set(f'{float(v):.5f}')
            else:
                self.sv[k].set(str(v))
        if 'score' in d:
            self.graph.add(float(d['score']))

    def set_active(self, active: bool):
        off = 'disabled' if active else 'normal'
        on  = 'normal'   if active else 'disabled'
        for b in (self.btn_hybrid, self.btn_cpu, self.btn_test,
                  self.btn_cont_h, self.btn_cont_c):
            b.config(state=off)
        self.btn_stop.config(state=on)

    def set_status(self, text: str, color: str = C['text2']):
        self.lbl_status.config(text=text, fg=color)

    @property
    def episodes(self) -> int:
        return self.EP_OPTS[self.ep_i]

    def _nazwa_urządzenia(self) -> str:
        return 'GPU' if self.DEV_OPTS[self.dev_i] == 'cuda' else 'CPU'

    def _dev_zmień(self):
        self.dev_i = (self.dev_i + 1) % len(self.DEV_OPTS)
        self._dev_var.set(self._nazwa_urządzenia())

    @property
    def urządzenie(self) -> str:
        return self.DEV_OPTS[self.dev_i]

    @property
    def parallel(self) -> int:
        return self.PAR_OPTS[self.par_i]

    @property
    def step_delay(self) -> float:
        return self.DELAY_OPTS[self.delay_i] / 1000.0

    @property
    def grid_size(self) -> int:
        return self.GRID_OPTS[self.grid_i]


# ══════════════════════════════════════════════════════════════════════
# AStarPanel – sterowanie i statystyki algorytmu A*
# ══════════════════════════════════════════════════════════════════════

class SiatkaMixin:
    """Wybór rozmiaru siatki dla paneli A* i Gracz (każdy ma własny).

    Wszystkie opcje są parzyste – A* potrzebuje cyklu Hamiltona,
    a ten na kwadratowej siatce n×n istnieje tylko dla parzystego n.
    """
    GRID_OPTS = [6, 8, 10, 12, 16, 20]

    def _zbuduj_siatkę(self, cfg, row: int):
        self.grid_i = 1   # domyślnie 8×8
        n = self.GRID_OPTS[self.grid_i]
        self._grid_var = tk.StringVar(value=f'{n}×{n}')
        tk.Label(cfg, text='Siatka:', bg=C['panel'], fg=C['text2'],
                 font=('Segoe UI', 9)).grid(row=row, column=0, sticky='w',
                                             padx=(4, 8), pady=3)
        tk.Button(cfg, text='−', command=lambda: self._zmień_siatkę(-1),
                  bg=C['btn'], fg=C['text'], relief='flat',
                  font=('Segoe UI', 10, 'bold'), width=2, takefocus=0,
                  cursor='hand2').grid(row=row, column=1, padx=(0, 2))
        tk.Label(cfg, textvariable=self._grid_var,
                 bg=C['panel2'], fg=C['accent'],
                 font=('Segoe UI', 10, 'bold'), width=6,
                 anchor='center').grid(row=row, column=2, padx=2)
        tk.Button(cfg, text='+', command=lambda: self._zmień_siatkę(+1),
                  bg=C['btn'], fg=C['text'], relief='flat',
                  font=('Segoe UI', 10, 'bold'), width=2, takefocus=0,
                  cursor='hand2').grid(row=row, column=3, padx=(2, 0))

    def _zmień_siatkę(self, krok: int):
        self.grid_i = min(max(0, self.grid_i + krok), len(self.GRID_OPTS) - 1)
        n = self.GRID_OPTS[self.grid_i]
        self._grid_var.set(f'{n}×{n}')
        self.app.podgląd_siatki(n)

    @property
    def grid_size(self) -> int:
        return self.GRID_OPTS[self.grid_i]


class AStarPanel(SiatkaMixin, tk.Frame):
    DELAY_OPTS = [0, 10, 25, 50, 100, 200, 500]  # ms między krokami

    def __init__(self, parent, app, **kw):
        super().__init__(parent, bg=C['panel'], **kw)
        self.app     = app
        self.delay_i = 3   # domyślnie 50 ms
        self.sv: dict = {}
        self._build()

    def _build(self):
        pad = {'padx': 10, 'pady': (6, 2)}

        # Konfiguracja
        cfg = tk.LabelFrame(self, text=' Konfiguracja ',
                            bg=C['panel'], fg=C['text2'],
                            font=('Segoe UI', 9), relief='groove')
        cfg.pack(fill='x', **pad)

        self._delay_var = tk.StringVar(value=f'{self.DELAY_OPTS[self.delay_i]} ms')

        tk.Label(cfg, text='Szybkość:', bg=C['panel'], fg=C['text2'],
                 font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w',
                                             padx=(4, 8), pady=3)
        tk.Button(cfg, text='−', command=self._delay_minus,
                  bg=C['btn'], fg=C['text'], relief='flat',
                  font=('Segoe UI', 10, 'bold'), width=2,
                  cursor='hand2').grid(row=0, column=1, padx=(0, 2))
        tk.Label(cfg, textvariable=self._delay_var,
                 bg=C['panel2'], fg=C['accent'],
                 font=('Segoe UI', 10, 'bold'), width=5,
                 anchor='center').grid(row=0, column=2, padx=2)
        tk.Button(cfg, text='+', command=self._delay_plus,
                  bg=C['btn'], fg=C['text'], relief='flat',
                  font=('Segoe UI', 10, 'bold'), width=2,
                  cursor='hand2').grid(row=0, column=3, padx=(2, 0))
        self._zbuduj_siatkę(cfg, row=1)

        # Statystyki
        sts = tk.LabelFrame(self, text=' Statystyki ',
                            bg=C['panel'], fg=C['text2'],
                            font=('Segoe UI', 9), relief='groove')
        sts.pack(fill='x', **pad)

        for key in ('score', 'best', 'game', 'wins'):
            self.sv[key] = tk.StringVar(value='—')

        info = [('Wynik:', 'score', C['accent'],   0, 0),
                ('Najlepszy:', 'best', C['accent3'], 0, 2),
                ('Gra:',  'game',  C['text'],    1, 0),
                ('Wygrane:', 'wins', C['accent2'], 1, 2)]
        for lbl, key, col, row, c in info:
            tk.Label(sts, text=lbl, bg=C['panel'], fg=C['text2'],
                     font=('Segoe UI', 9)).grid(
                row=row, column=c, sticky='w', padx=(4, 2), pady=2)
            tk.Label(sts, textvariable=self.sv[key],
                     bg=C['panel'], fg=col,
                     font=('Segoe UI', 11, 'bold')).grid(
                row=row, column=c + 1, sticky='w', padx=(0, 14))

        # Wykres
        self.graph = StatsGraph(self, height=100)
        self.graph.pack(fill='x', padx=10, pady=(2, 4))

        # Przyciski
        bf = tk.Frame(self, bg=C['panel'])
        bf.pack(fill='x', padx=10, pady=(2, 6))

        def btn(text, cmd, bg, bg_h=None):
            return tk.Button(bf, text=text, command=cmd,
                             bg=bg, fg=C['text'], relief='flat',
                             font=('Segoe UI', 10), pady=5,
                             activebackground=bg_h or bg,
                             activeforeground='#ffffff',
                             cursor='hand2',
                             disabledforeground=C['text3'])

        self.btn_start = btn('▶  Uruchom A*', self.app.go_astar,
                             C['blue'], C['blue_h'])
        self.btn_start.pack(fill='x', pady=2)

        self.btn_stop = btn('■  ZATRZYMAJ', self.app.stop,
                            C['stop'], C['stop_h'])
        self.btn_stop.pack(fill='x', pady=(4, 2))
        self.btn_stop.config(state='disabled')

        self.lbl_status = tk.Label(bf, text='Gotowy',
                                   bg=C['panel'], fg=C['text2'],
                                   font=('Segoe UI', 9))
        self.lbl_status.pack(anchor='w')

    def _delay_minus(self):
        self.delay_i = max(0, self.delay_i - 1)
        self._delay_var.set(f'{self.DELAY_OPTS[self.delay_i]} ms')

    def _delay_plus(self):
        self.delay_i = min(len(self.DELAY_OPTS) - 1, self.delay_i + 1)
        self._delay_var.set(f'{self.DELAY_OPTS[self.delay_i]} ms')

    def update_stats(self, score: int, best: int, game_n: int, wins: int = 0):
        self.sv['score'].set(str(score))
        self.sv['best'].set(str(best))
        self.sv['game'].set(str(game_n))
        self.sv['wins'].set(f'{wins}/{game_n}')
        self.graph.add(float(score))

    def set_active(self, active: bool):
        self.btn_start.config(state='disabled' if active else 'normal')
        self.btn_stop.config(state='normal' if active else 'disabled')

    def set_status(self, text: str, color: str = C['text2']):
        self.lbl_status.config(text=text, fg=color)

    @property
    def step_delay(self) -> float:
        return self.DELAY_OPTS[self.delay_i] / 1000.0


# ══════════════════════════════════════════════════════════════════════
# PlayerPanel – samodzielna gra z klawiatury
# ══════════════════════════════════════════════════════════════════════

class PlayerPanel(SiatkaMixin, tk.Frame):
    """Panel zakładki Gracz: szybkość, sterowanie, statystyki, przyciski."""

    SPEED_OPTS = [200, 150, 120, 100, 80, 60]   # ms na jeden krok węża

    def __init__(self, parent, app, **kw):
        super().__init__(parent, bg=C['panel'], **kw)
        self.app     = app
        self.speed_i = 2   # domyślnie 120 ms
        self.sv: dict = {}
        self._build()

    def _build(self):
        pad = {'padx': 10, 'pady': (6, 2)}

        # Konfiguracja
        cfg = tk.LabelFrame(self, text=' Konfiguracja ',
                            bg=C['panel'], fg=C['text2'],
                            font=('Segoe UI', 9), relief='groove')
        cfg.pack(fill='x', **pad)

        self._speed_var = tk.StringVar(value=f'{self.SPEED_OPTS[self.speed_i]} ms')
        tk.Label(cfg, text='Krok co:', bg=C['panel'], fg=C['text2'],
                 font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w',
                                             padx=(4, 8), pady=3)
        tk.Button(cfg, text='−', command=self._speed_minus,
                  bg=C['btn'], fg=C['text'], relief='flat',
                  font=('Segoe UI', 10, 'bold'), width=2,
                  cursor='hand2').grid(row=0, column=1, padx=(0, 2))
        tk.Label(cfg, textvariable=self._speed_var,
                 bg=C['panel2'], fg=C['accent'],
                 font=('Segoe UI', 10, 'bold'), width=6,
                 anchor='center').grid(row=0, column=2, padx=2)
        tk.Button(cfg, text='+', command=self._speed_plus,
                  bg=C['btn'], fg=C['text'], relief='flat',
                  font=('Segoe UI', 10, 'bold'), width=2,
                  cursor='hand2').grid(row=0, column=3, padx=(2, 0))
        self._zbuduj_siatkę(cfg, row=1)

        # Sterowanie
        ctl = tk.LabelFrame(self, text=' Sterowanie ',
                            bg=C['panel'], fg=C['text2'],
                            font=('Segoe UI', 9), relief='groove')
        ctl.pack(fill='x', **pad)
        for txt in ('← ↑ ↓ →  lub  W A S D  –  kierunek',
                    'Spacja  –  start / pauza / nowa gra'):
            tk.Label(ctl, text=txt, bg=C['panel'], fg=C['text'],
                     font=('Segoe UI', 9)).pack(anchor='w', padx=6, pady=1)

        # Statystyki
        sts = tk.LabelFrame(self, text=' Statystyki ',
                            bg=C['panel'], fg=C['text2'],
                            font=('Segoe UI', 9), relief='groove')
        sts.pack(fill='x', **pad)

        for key in ('score', 'best', 'game'):
            self.sv[key] = tk.StringVar(value='—')

        info = [('Wynik:', 'score', C['accent'],   0, 0),
                ('Najlepszy:', 'best', C['accent3'], 0, 2),
                ('Gra:',  'game',  C['text'],    1, 0)]
        for lbl, key, col, row, c in info:
            tk.Label(sts, text=lbl, bg=C['panel'], fg=C['text2'],
                     font=('Segoe UI', 9)).grid(
                row=row, column=c, sticky='w', padx=(4, 2), pady=2)
            tk.Label(sts, textvariable=self.sv[key],
                     bg=C['panel'], fg=col,
                     font=('Segoe UI', 11, 'bold')).grid(
                row=row, column=c + 1, sticky='w', padx=(0, 14))

        # Wykres
        self.graph = StatsGraph(self, height=100)
        self.graph.pack(fill='x', padx=10, pady=(2, 4))

        # Przyciski
        bf = tk.Frame(self, bg=C['panel'])
        bf.pack(fill='x', padx=10, pady=(2, 6))

        def btn(text, cmd, bg, bg_h=None):
            return tk.Button(bf, text=text, command=cmd,
                             bg=bg, fg=C['text'], relief='flat',
                             font=('Segoe UI', 10), pady=5,
                             activebackground=bg_h or bg,
                             activeforeground='#ffffff',
                             cursor='hand2',
                             disabledforeground=C['text3'],
                             takefocus=0)   # Spacja nie "klika" przycisków

        self.btn_start = btn('▶  Zagraj', self.app.go_player,
                             C['grn'], C['grn_h'])
        self.btn_start.pack(fill='x', pady=2)

        self.btn_stop = btn('■  ZAKOŃCZ', self.app.stop,
                            C['stop'], C['stop_h'])
        self.btn_stop.pack(fill='x', pady=(4, 2))
        self.btn_stop.config(state='disabled')

        self.lbl_status = tk.Label(bf, text='Gotowy',
                                   bg=C['panel'], fg=C['text2'],
                                   font=('Segoe UI', 9))
        self.lbl_status.pack(anchor='w')

    def _speed_minus(self):
        self.speed_i = max(0, self.speed_i - 1)
        self._speed_var.set(f'{self.SPEED_OPTS[self.speed_i]} ms')

    def _speed_plus(self):
        self.speed_i = min(len(self.SPEED_OPTS) - 1, self.speed_i + 1)
        self._speed_var.set(f'{self.SPEED_OPTS[self.speed_i]} ms')

    def update_stats(self, score: int, best: int, game_n: int):
        self.sv['score'].set(str(score))
        self.sv['best'].set(str(best))
        self.sv['game'].set(str(game_n))
        self.graph.add(float(score))

    def set_active(self, active: bool):
        self.btn_start.config(state='disabled' if active else 'normal')
        self.btn_stop.config(state='normal' if active else 'disabled')

    def set_status(self, text: str, color: str = C['text2']):
        self.lbl_status.config(text=text, fg=color)

    @property
    def step_ms(self) -> int:
        return self.SPEED_OPTS[self.speed_i]


# ══════════════════════════════════════════════════════════════════════
# App – główne okno
# ══════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    """Główna aplikacja – zarządza wątkami i spina komponenty."""

    def __init__(self):
        super().__init__()
        self.title('Snake AI  ·  Modular GUI')
        self.configure(bg=C['bg'])
        self.resizable(True, True)
        self.minsize(900, 620)

        # Stan współdzielony z wątkami
        self.agent:      object           = None
        self.worker:     threading.Thread = None
        self.stop_ev:    threading.Event  = threading.Event()
        self.sq:         queue.Queue      = queue.Queue()
        self.game_state: dict             = None
        self.glock:      threading.Lock   = threading.Lock()
        self._tick:        int              = 0
        self._active:      bool             = False
        self._sel_model:   str              = None
        self._last_mode:   str              = 'hybrid'
        self._active_mode: str              = 'dqn'
        self._pl_sesja:    int              = 0   # chroni przed podwójną pętlą gracza

        self._build_ui()
        self._dopasuj_do_ekranu()
        self._poll()

    # ── Layout ───────────────────────────────────────────────────────

    def _build_ui(self):
        # nagłówek
        hdr = tk.Frame(self, bg='#0e0e1a', height=44)
        hdr.pack(fill='x', side='top')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='Snake AI  ·  Deep Q-Network',
                 bg='#0e0e1a', fg=C['accent'],
                 font=('Segoe UI', 13, 'bold')).pack(side='left', padx=14)
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', side='top')

        # stopka
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', side='bottom')
        ftr = tk.Frame(self, bg='#0c0c16', height=26)
        ftr.pack(fill='x', side='bottom')
        ftr.pack_propagate(False)
        self.lbl_log = tk.Label(ftr, text='',
                                bg='#0c0c16', fg=C['text3'],
                                font=('Segoe UI', 8))
        self.lbl_log.pack(side='left', padx=10, pady=4)

        # zawartość
        main = tk.Frame(self, bg=C['bg'])
        main.pack(fill='both', expand=True, side='top')

        # prawy panel: notebook algorytmów (stała szerokość)
        right = tk.Frame(main, bg=C['panel'], width=420)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)

        # separator pionowy
        tk.Frame(main, bg=C['border'], width=1).pack(side='right', fill='y')

        # lewy panel: canvas gry (rośnie z oknem) + lista modeli na dole
        left = tk.Frame(main, bg=C['panel'])
        left.pack(side='left', fill='both', expand=True)

        self.model_list = ModelList(left, on_select=self._on_model_sel)
        self.model_list.lb.config(height=6)
        self.model_list.pack(side='bottom', fill='x')

        self._model_sep = tk.Frame(left, bg=C['border'], height=1)
        self._model_sep.pack(side='bottom', fill='x')
        self._lista_widoczna = True

        self.game_canvas = GameCanvas(left)
        self.game_canvas.pack(fill='both', expand=True)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook',
                        background=C['panel'], borderwidth=0)
        style.configure('TNotebook.Tab',
                        background=C['panel2'], foreground=C['text2'],
                        font=('Segoe UI', 10), padding=[14, 6])
        style.map('TNotebook.Tab',
                  background=[('selected', C['panel'])],
                  foreground=[('selected', C['text'])])
        style.configure('TFrame', background=C['panel'])

        nb = ttk.Notebook(right)
        nb.pack(fill='both', expand=True)

        # zakładka DQN
        dqn_f = tk.Frame(nb, bg=C['panel'])
        nb.add(dqn_f, text='  DQN  ')
        self.dqn = DQNPanel(dqn_f, app=self)
        self.dqn.pack(fill='both', expand=True)

        # zakładka A*
        astar_f = tk.Frame(nb, bg=C['panel'])
        nb.add(astar_f, text='  A*  ')
        self.astar_panel = AStarPanel(astar_f, app=self)
        self.astar_panel.pack(fill='both', expand=True)

        # zakładka Gracz
        player_f = tk.Frame(nb, bg=C['panel'])
        nb.add(player_f, text='  Gracz  ')
        self.player_panel = PlayerPanel(player_f, app=self)
        self.player_panel.pack(fill='both', expand=True)

        # Wyjście z zakładki Gracz kończy grę – inaczej przyciski DQN/A*
        # zostają wyszarzone, bo tryb gracza wciąż jest aktywny.
        self.nb = nb
        self._zakładki = {str(dqn_f): 'dqn', str(astar_f): 'astar',
                          str(player_f): 'player'}
        nb.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    # ── Pętla pollingu ─────────────────────────────────────

    def _poll(self):
        self._tick += 1

        # Drenaż kolejki statystyk z wątków
        while not self.sq.empty():
            d = self.sq.get_nowait()
            if d.get('done'):
                self._set_active(False)
                self.dqn.set_status('Gotowy')
                self.astar_panel.set_status('Gotowy')
                self.player_panel.set_status('Gotowy')
                if self._active_mode == 'dqn':
                    self.model_list.reload()
                    if d.get('zapisane'):
                        self._log(f"Zakończono. Zapisano: {d['zapisane']}")
                    else:
                        self._log('Zakończono.')
                elif self._active_mode == 'player':
                    self._log('Gra zakończona.')
                else:
                    self._log('A* zatrzymany.')
                with self.glock:
                    self.game_state = None
            elif d.get('benchmark'):
                self._log(d['benchmark'])
            elif d.get('astar'):
                self.astar_panel.update_stats(d['score'], d['best'], d['game'],
                                              d.get('wins', 0))
            elif d.get('player'):
                self.player_panel.update_stats(d['score'], d['best'], d['game'])
            else:
                ep  = d.get('episode', 0)
                tot = d.get('total', 0)
                d2  = dict(d)
                d2['episode'] = str(ep) + (f' / {tot}' if tot else '')
                self.dqn.update_stats(d2)

        # Odświeżenie gry
        if self._active:
            with self.glock:
                gs = self.game_state
            if gs:
                self.game_canvas.update_state(gs, self._tick)
                self.dqn.sv['score'].set(str(gs['score']))
            else:
                ep_s = self.dqn.sv['episode'].get()
                self.game_canvas.show_message(
                    ['Trening hybrydowy', 'działa w tle…', '', f'Epizod: {ep_s}'],
                    color=C['accent'],
                )

        # Wykryj zakończenie wątku bez sygnału done
        if self._active and self.worker and not self.worker.is_alive():
            self._set_active(False)
            self.dqn.set_status('Gotowy')
            self.model_list.reload()
            with self.glock:
                self.game_state = None

        self.after(33, self._poll)

    # ── Pomoce ──────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.lbl_log.config(text=msg)

    def _on_model_sel(self, name: str):
        self._sel_model = name

    def _set_active(self, v: bool):
        self._active = v
        self.dqn.set_active(v)
        self.astar_panel.set_active(v)
        self.player_panel.set_active(v)
        if not v:
            self.game_canvas.update_state(None)

    def _dopasuj_do_ekranu(self):
        """Otwiera okno zmaksymalizowane (z paskiem zadań, nie pełny ekran)."""
        try:
            self.state('zoomed')                     # Windows / macOS
        except tk.TclError:
            try:
                self.attributes('-zoomed', True)     # Linux
            except tk.TclError:
                self.geometry(f'{self.winfo_screenwidth()}x'
                              f'{self.winfo_screenheight()}+0+0')

    def _on_tab_changed(self, _event=None):
        if not hasattr(self, 'game_canvas'):
            return                          # zdarzenie przy budowie okna
        zakładka = self._zakładki.get(self.nb.select(), 'dqn')

        # Wyjście z zakładki Gracz kończy grę – inaczej przyciski DQN/A*
        # zostają wyszarzone, bo tryb gracza wciąż jest aktywny.
        if self._active and self._active_mode == 'player' and zakładka != 'player':
            self.stop()

        # Lista modeli ma sens tylko w DQN – gdzie indziej plansza dostaje miejsce
        self._pokaż_listę_modeli(zakładka == 'dqn')

        # Podgląd siatki wybranej w tej zakładce
        panel = {'dqn': self.dqn, 'astar': self.astar_panel,
                 'player': self.player_panel}[zakładka]
        self.podgląd_siatki(panel.grid_size)

    def _pokaż_listę_modeli(self, widoczna: bool):
        if widoczna == self._lista_widoczna:
            return
        if widoczna:
            # 'before' przywraca pierwotną kolejność: lista na dole, nad nią kreska
            self.model_list.pack(side='bottom', fill='x', before=self.game_canvas)
            self._model_sep.pack(side='bottom', fill='x', before=self.game_canvas)
        else:
            self.model_list.pack_forget()
            self._model_sep.pack_forget()
        self._lista_widoczna = widoczna

    def podgląd_siatki(self, n: int):
        """Pokazuje pustą siatkę n×n – tylko gdy nic nie działa."""
        if not self._active:
            self.game_canvas.set_grid(n)

    def _selected(self):
        return self._sel_model or self.model_list.selected()

    def _clear_queue(self):
        while not self.sq.empty():
            self.sq.get_nowait()

    # ── Akcje wywoływane z UI ────────────────────────────────────────

    def go_train(self, mode: str, agent=None):
        if self._active:
            return
        self._last_mode   = mode
        self._active_mode = 'dqn'
        self.stop_ev.clear()
        self._clear_queue()
        self.agent = agent or DQNAgent(11, 3, ROZMIAR_UKRYTY, urządzenie=self.dqn.urządzenie)
        self.dqn.graph.clear()
        with self.glock:
            self.game_state = None
        self._set_active(True)
        self.dqn.set_status('Trening…', C['accent3'])

        grid_n = self.dqn.grid_size
        best_f, final_f = self.dqn.nazwy_plików()
        self._plik_best  = os.path.join(KATALOG_MODELI, best_f)
        self._plik_final = os.path.join(KATALOG_MODELI, final_f)
        os.makedirs(KATALOG_MODELI, exist_ok=True)
        tryb = 'w tle' if mode == 'hybrid' else 'z podglądem'
        urz  = 'GPU' if self.agent.device.type == 'cuda' else 'CPU'
        self._log(f'Trening {tryb} | sieć na {urz} | {self.dqn.episodes} ep | '
                  f'siatka {grid_n}×{grid_n} | zapis: {best_f}, {final_f}')
        self.game_canvas.set_grid(grid_n)

        params = dict(szerokość=grid_n * ROZMIAR_BLOKU,
                      wysokość=grid_n  * ROZMIAR_BLOKU,
                      rozmiar_bloku=ROZMIAR_BLOKU)
        self._game_params = params
        ep, par = self.dqn.episodes, self.dqn.parallel

        if mode == 'hybrid':
            t = threading.Thread(target=self._th_hybrid,
                                 args=(params, ep, par), daemon=True)
        else:
            t = threading.Thread(target=self._th_cpu,
                                 args=(ep,), daemon=True)
        self.worker = t
        t.start()

    def go_test(self):
        sel = self._selected()
        if not sel or self._active:
            return
        path  = os.path.join(KATALOG_MODELI, sel)
        agent = DQNAgent(11, 3, ROZMIAR_UKRYTY, urządzenie=self.dqn.urządzenie)
        if not agent.wczytaj(path):
            self._log(f'Błąd wczytywania: {sel}')
            return
        self.agent = agent
        self.stop_ev.clear()
        self._clear_queue()
        self.dqn.graph.clear()
        self._set_active(True)
        self.dqn.set_status('Testowanie', C['accent'])

        grid_n = self.dqn.grid_size
        self._log(f'Testowanie: {sel} | siatka {grid_n}×{grid_n}')
        self.game_canvas.set_grid(grid_n)
        self._game_params = dict(szerokość=grid_n * ROZMIAR_BLOKU,
                                 wysokość=grid_n  * ROZMIAR_BLOKU,
                                 rozmiar_bloku=ROZMIAR_BLOKU)

        self.worker = threading.Thread(target=self._th_test, daemon=True)
        self.worker.start()

    def go_continue(self, mode: str = None):
        sel = self._selected()
        if not sel or self._active:
            return
        path  = os.path.join(KATALOG_MODELI, sel)
        agent = DQNAgent(11, 3, ROZMIAR_UKRYTY, urządzenie=self.dqn.urządzenie)
        if not agent.wczytaj(path):
            self._log(f'Błąd wczytywania: {sel}')
            return
        chosen = mode or self._last_mode
        self._log(f'Kontynuuję: {sel} ({chosen})')
        self.go_train(chosen, agent)

    def stop(self):
        self.stop_ev.set()
        self.dqn.set_status('Zatrzymywanie…', C['danger'])
        self.astar_panel.set_status('Zatrzymywanie…', C['danger'])
        self.player_panel.set_status('Zatrzymywanie…', C['danger'])

    # ── Wątki robocze ────────────────────────────────────────────────

    def _push(self, score, best, ep, total, loss):
        """Wątek → kolejka → główny wątek."""
        self.sq.put(dict(
            score=score, best=best, episode=ep, total=total,
            epsilon=self.agent.epsilon, loss=loss,
        ))

    def _th_hybrid(self, params, total, parallel):
        best, ep = 0, 0
        chunks = (total + parallel - 1) // parallel
        t_collect_total = 0.0
        t_train_total   = 0.0

        for ch in range(chunks):
            if self.stop_ev.is_set():
                break
            real_n = min(parallel, total - ch * parallel)
            exps, scores = [], []

            t0 = time.perf_counter()
            exps, scores = uruchom_epizody(self.agent, params, real_n,
                                           stop_ev=self.stop_ev)
            t_collect_total += time.perf_counter() - t0

            ep += len(scores)
            for e in exps:
                self.agent.zapamiętaj(*e)

            iters = min(len(exps), ITERACJE_UCZENIA)
            tl, nl = 0.0, 0
            t1 = time.perf_counter()
            for i in range(iters):
                if self.stop_ev.is_set():
                    break
                l = self.agent.ucz_się()
                if l is not None:
                    tl += l
                    nl += 1
                if nl > 0 and i % 200 == 199:
                    self._push(scores[-1] if scores else 0,
                               best, ep, total, tl / nl)

            t_train_total += time.perf_counter() - t1
            avg_l = tl / nl if nl else 0.0
            bh    = max(scores) if scores else 0
            if bh > best:
                best = bh
                self.agent.save(self._plik_best)
            self._push(scores[-1] if scores else 0, best, ep, total, avg_l)

            if (ch + 1) % 10 == 0:
                total_t = t_collect_total + t_train_total
                if total_t > 0:
                    pct_c = t_collect_total / total_t * 100
                    pct_t = t_train_total   / total_t * 100
                    self.sq.put({'benchmark':
                        f'Ep {ep}: zbieranie {pct_c:.0f}%  trening {pct_t:.0f}%  '
                        f'({t_collect_total:.1f}s / {t_train_total:.1f}s)'})
                t_collect_total = 0.0
                t_train_total   = 0.0

        self.agent.save(self._plik_final)
        self.sq.put({'done': True, 'zapisane': self._zapisane_pliki()})

    def _zapisane_pliki(self) -> str:
        pliki = [self._plik_best, self._plik_final]
        return ', '.join(os.path.basename(f) for f in pliki if os.path.exists(f))

    def _th_cpu(self, total):
        p    = self._game_params
        game = UproszczonySnake(p['szerokość'], p['wysokość'], p['rozmiar_bloku'])
        best = 0

        for ep_i in range(total):
            if self.stop_ev.is_set():
                break
            stan = game.reset()
            done = False
            score = 0
            tl, nl = 0.0, 0

            while not done and not self.stop_ev.is_set():
                akcja = self.agent.pobierz_akcję(stan)
                nst, nagroda, done, score = game.krok(akcja)
                self.agent.zapamiętaj(stan, akcja, nagroda, nst, done)
                stan = nst
                l = self.agent.ucz_się()
                if l is not None:
                    tl += l
                    nl += 1
                with self.glock:
                    self.game_state = dict(
                        snake=[list(p) for p in game.snake],
                        food=list(game.jedzenie),
                        score=game.wynik,
                    )
                d = self.dqn.step_delay
                if d > 0:
                    time.sleep(d)

            avg_l = tl / nl if nl else 0.0
            if score > best:
                best = score
                self.agent.save(self._plik_best)
            self._push(score, best, ep_i + 1, total, avg_l)

        self.agent.save(self._plik_final)
        self.sq.put({'done': True, 'zapisane': self._zapisane_pliki()})

    def _th_test(self):
        p     = self._game_params
        game  = UproszczonySnake(p['szerokość'], p['wysokość'], p['rozmiar_bloku'])
        agent = self.agent
        best, ep = 0, 0
        agent.model.eval()   # wyłącza dropout – deterministyczne decyzje

        while not self.stop_ev.is_set():
            stan = game.reset()
            done = False
            score = 0

            while not done and not self.stop_ev.is_set():
                t = torch.tensor(
                    stan, dtype=agent.typ_danych,
                ).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    akcja = torch.argmax(agent.model(t)).item()
                stan, _, done, score = game.krok(akcja)

                with self.glock:
                    self.game_state = dict(
                        snake=[list(p) for p in game.snake],
                        food=list(game.jedzenie),
                        score=game.wynik,
                    )
                time.sleep(0.05)   # ~20 fps w trybie testowania

            if score > best:
                best = score
            ep += 1
            self.sq.put(dict(score=score, best=best, episode=ep,
                             total=0, epsilon=0.0, loss=0.0))

    # ── A* ───────────────────────────────────────────────────────────

    def go_astar(self):
        if self._active:
            return
        self._active_mode = 'astar'
        self.stop_ev.clear()
        self._clear_queue()
        self.astar_panel.graph.clear()
        with self.glock:
            self.game_state = None
        self._set_active(True)
        self.astar_panel.set_status('A* działa…', C['accent'])

        grid_n = self.astar_panel.grid_size
        self._log(f'A* | siatka {grid_n}×{grid_n}')
        self.game_canvas.set_grid(grid_n)
        self._game_params = dict(szerokość=grid_n * ROZMIAR_BLOKU,
                                 wysokość=grid_n  * ROZMIAR_BLOKU,
                                 rozmiar_bloku=ROZMIAR_BLOKU)

        self.worker = threading.Thread(target=self._th_astar, daemon=True)
        self.worker.start()

    def _th_astar(self):
        from astar import wybierz_akcję as astar_akcja
        p    = self._game_params
        game = UproszczonySnake(p['szerokość'], p['wysokość'], p['rozmiar_bloku'])
        best, game_n, wins = 0, 0, 0

        while not self.stop_ev.is_set():
            game.reset()
            done  = False
            score = 0

            while not done and not self.stop_ev.is_set():
                akcja, ścieżka = astar_akcja(game)
                _, _, done, score = game.krok(akcja)

                with self.glock:
                    self.game_state = dict(
                        snake=[list(s) for s in game.snake],
                        food=list(game.jedzenie),
                        score=game.wynik,
                        path=ścieżka,
                        won=game.wygrana,
                    )

                d = self.astar_panel.step_delay
                if d > 0:
                    time.sleep(d)

            if score > best:
                best = score
            game_n += 1
            if game.wygrana:
                wins += 1
                # chwila na obejrzenie pełnej planszy (przerywalna STOP-em)
                self.stop_ev.wait(2.0)
            self.sq.put({'astar': True, 'score': score,
                         'best': best, 'game': game_n, 'wins': wins})

        self.sq.put({'done': True})

    # ── Gracz (sterowanie z klawiatury) ──────────────────────────────

    _ZEGAR = [Kierunek.PRAWO, Kierunek.DÓŁ, Kierunek.LEWO, Kierunek.GÓRA]
    _KLAWISZE = {
        'Up': Kierunek.GÓRA,  'w': Kierunek.GÓRA,  'W': Kierunek.GÓRA,
        'Down': Kierunek.DÓŁ, 's': Kierunek.DÓŁ,   'S': Kierunek.DÓŁ,
        'Left': Kierunek.LEWO, 'a': Kierunek.LEWO, 'A': Kierunek.LEWO,
        'Right': Kierunek.PRAWO, 'd': Kierunek.PRAWO, 'D': Kierunek.PRAWO,
    }

    def go_player(self):
        if self._active:
            return
        self._active_mode = 'player'
        self.stop_ev.clear()
        self._clear_queue()
        self.player_panel.graph.clear()
        self.worker = None                # gra działa w głównym wątku (after)
        self._set_active(True)
        self.player_panel.set_status('Gra trwa', C['accent'])

        grid_n = self.player_panel.grid_size
        self._log(f'Gracz | siatka {grid_n}×{grid_n}')
        self.game_canvas.set_grid(grid_n)
        self._pl_game = UproszczonySnake(grid_n * ROZMIAR_BLOKU,
                                         grid_n * ROZMIAR_BLOKU,
                                         ROZMIAR_BLOKU)
        self._pl_best, self._pl_n = 0, 0
        self._pl_nowa_gra()

        # Klawisze łapiemy globalnie; fokus na canvas, żeby strzałki nie
        # przełączały zakładek notebooka.
        self.bind_all('<KeyPress>', self._pl_klawisz)
        self.game_canvas.canvas.focus_set()

        self._pl_sesja += 1
        self._pl_tick(self._pl_sesja)

    def _pl_nowa_gra(self):
        self._pl_game.reset()
        self._pl_skręty = []              # bufor wciśniętych kierunków
        self._pl_stan   = 'gotowy'        # gotowy → gra → pauza / koniec
        self._pl_publikuj(['GOTOWY?', 'Strzałka lub Spacja – start'])

    def _pl_publikuj(self, overlay=None):
        g = self._pl_game
        with self.glock:
            self.game_state = dict(
                snake=[list(s) for s in g.snake],
                food=list(g.jedzenie),
                score=g.wynik,
                won=g.wygrana,
                overlay=overlay,
            )

    def _pl_klawisz(self, e):
        if not self._active or self._active_mode != 'player':
            return

        if e.keysym == 'space':
            if self._pl_stan == 'gra':
                self._pl_stan = 'pauza'
                self._pl_publikuj(['PAUZA', 'Spacja – wznów'])
            elif self._pl_stan in ('gotowy', 'pauza'):
                self._pl_stan = 'gra'
            elif self._pl_stan == 'koniec':
                self._pl_nowa_gra()
            return 'break'

        kier = self._KLAWISZE.get(e.keysym)
        if kier is None or self._pl_stan == 'koniec':
            return

        # Porównujemy z ostatnim zakolejkowanym kierunkiem, nie z obecnym –
        # dzięki temu szybkie "góra, lewo" w jednej klatce działa poprawnie,
        # a zawrócenie o 180° jest ignorowane.
        ostatni = self._pl_skręty[-1] if self._pl_skręty else self._pl_game.Kierunek
        z = self._ZEGAR
        zawrót = (z.index(kier) - z.index(ostatni)) % 4 == 2
        if kier != ostatni and not zawrót and len(self._pl_skręty) < 3:
            self._pl_skręty.append(kier)

        if self._pl_stan in ('gotowy', 'pauza'):
            self._pl_stan = 'gra'
        return 'break'

    def _pl_tick(self, sesja):
        if sesja != self._pl_sesja:
            return                         # stara pętla po restarcie – wygaś
        if self.stop_ev.is_set():
            self.unbind_all('<KeyPress>')
            self.sq.put({'done': True})
            return

        if self._pl_stan == 'gra':
            g = self._pl_game
            akcja = 0                      # domyślnie prosto
            if self._pl_skręty:
                kier = self._pl_skręty.pop(0)
                diff = (self._ZEGAR.index(kier) - self._ZEGAR.index(g.Kierunek)) % 4
                akcja = {1: 1, 3: 2}.get(diff, 0)

            _, _, done, score = g.krok(akcja)

            if done:
                self._pl_n += 1
                self._pl_best = max(self._pl_best, score)
                self._pl_stan = 'koniec'
                self.sq.put({'player': True, 'score': score,
                             'best': self._pl_best, 'game': self._pl_n})
                tytuł = 'WYGRANA!' if g.wygrana else 'KONIEC GRY'
                self._pl_publikuj([tytuł, f'Wynik: {score}', 'Spacja – nowa gra'])
            else:
                self._pl_publikuj()

        self.after(self.player_panel.step_ms, lambda: self._pl_tick(sesja))


# ── Uruchomienie ──────────────────────────────────────────────────────

if __name__ == '__main__':
    app = App()
    app.mainloop()