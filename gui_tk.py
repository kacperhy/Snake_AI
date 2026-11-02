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
import numpy as np
import torch

from snake_game import UproszczonySnake
from agent import DQNAgent
from training import uruchom_epizod
import config
from config import (
    SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU,
    ROZMIAR_UKRYTY, KATALOG_MODELI,
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
        self.canvas.pack(padx=2, pady=2)
        self._grid_n     = 8           # komórek na bok
        self._game_block = ROZMIAR_BLOKU   # px/komórkę w logice gry (zawsze 40)
        self._recalc()
        self._draw_grid()

    def _recalc(self):
        """Przelicza rozmiar bloku wyświetlania i offset centrowania."""
        self._disp_block = CH // self._grid_n
        area = self._disp_block * self._grid_n
        self._ox = (CW - area) // 2
        self._oy = (CH - area) // 2

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

        # Jedzenie z pulsowaniem
        fx, fy = gs['food'][0] // gb, gs['food'][1] // gb
        x1, y1 = ox + fx * db, oy + fy * db
        pulse  = int(abs(math.sin(tick * 0.1)) * max(1, db // 8))
        self.canvas.create_rectangle(x1, y1, x1 + db, y1 + db,
                                     fill=C['food'], outline='', tags='dyn')
        p2 = max(2, db // 5) + pulse
        self.canvas.create_rectangle(x1 + p2, y1 + p2,
                                     x1 + db - p2, y1 + db - p2,
                                     fill=C['food_i'], outline='', tags='dyn')

    def show_message(self, lines: list, color: str = C['text3']):
        self.canvas.delete('dyn')
        cy = CH // 2 - len(lines) * 15
        for line in lines:
            col  = color if line else C['text3']
            size = 13 if line else 8
            self.canvas.create_text(
                CW // 2, cy, text=line,
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

class DQNPanel(tk.Frame):
    """Panel zakładki DQN: konfiguracja, statystyki, wykres, przyciski."""

    EP_OPTS    = [50, 100, 200, 500, 1000, 5000, 10000, 25000, 50000, 100000]
    PAR_OPTS   = [1, 2, 4, 8, 16, 32]
    DELAY_OPTS = [1, 10, 25, 50, 100, 200]   # ms opóźnienia między krokami CPU
    GRID_OPTS  = [6, 8, 10, 12, 16, 20]       # komórek na bok (kwadratowa siatka)

    def __init__(self, parent, app, **kw):
        super().__init__(parent, bg=C['panel'], **kw)
        self.app     = app
        self.ep_i    = 2   # domyślnie 200 epizodów
        self.par_i   = 0
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

        self._stepper(cfg, 'Epizody:',          self._ep_var,
                      self._ep_minus,    self._ep_plus,    row=0)
        self._stepper(cfg, 'ilość gier przed uczeniem:',   self._par_var,
                      self._par_minus,   self._par_plus,   row=1)
        self._stepper(cfg, 'Opóźnienie (CPU):', self._delay_var,
                      self._delay_minus, self._delay_plus, row=2)
        self._stepper(cfg, 'Siatka:',           self._grid_var,
                      self._grid_minus,  self._grid_plus,  row=3)

        dev_text = 'GPU (CUDA)' if config.UŻYJ_GPU else 'CPU'
        dev_col  = C['accent'] if config.UŻYJ_GPU else C['text3']
        tk.Label(cfg, text=f'Urządzenie: {dev_text}',
                 bg=C['panel'], fg=dev_col,
                 font=('Segoe UI', 9)).grid(
            row=4, column=0, columnspan=5, sticky='w',
            pady=(4, 2), padx=4)

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
            tk.Label(sts, text=lbl, bg=C['panel'], fg=C['text2'],
                     font=('Segoe UI', 9)).grid(
                row=row, column=c, sticky='w', padx=(4, 2), pady=2)
            tk.Label(sts, textvariable=self.sv[key],
                     bg=C['panel'], fg=colors[key],
                     font=('Segoe UI', 11, 'bold')).grid(
                row=row, column=c + 1, sticky='w', padx=(0, 14))

        tk.Label(sts, text='Loss:', bg=C['panel'], fg=C['text2'],
                 font=('Segoe UI', 9)).grid(
            row=2, column=0, sticky='w', padx=(4, 2))
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

        self.btn_hybrid = btn(bf, '▶  Trening Hybrydowy',
                              lambda: self.app.go_train('hybrid'),
                              C['grn'], C['grn_h'])
        self.btn_hybrid.pack(fill='x', pady=2)

        self.btn_cpu = btn(bf, '▶  Trening CPU',
                           lambda: self.app.go_train('cpu'),
                           C['btn'], '#32324c')
        self.btn_cpu.pack(fill='x', pady=2)

        self.btn_test = btn(bf, '▶  Testuj',
                            self.app.go_test, C['blue'], C['blue_h'])
        self.btn_test.pack(fill='x', pady=2)

        row2 = tk.Frame(bf, bg=C['panel'])
        row2.pack(fill='x', pady=2)
        self.btn_cont_h = btn(row2, '↺  Kontynuuj Hybr.',
                              lambda: self.app.go_continue('hybrid'),
                              C['btn'], '#32324c')
        self.btn_cont_h.pack(side='left', fill='x', expand=True, padx=(0, 2))
        self.btn_cont_c = btn(row2, '↺  Kontynuuj CPU',
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

    def _stepper(self, parent, label, var, minus_cmd, plus_cmd, row: int):
        tk.Label(parent, text=label, bg=C['panel'], fg=C['text2'],
                 font=('Segoe UI', 9)).grid(
            row=row, column=0, sticky='w', padx=(4, 8), pady=3)
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

    def _grid_plus(self):
        self.grid_i = min(len(self.GRID_OPTS) - 1, self.grid_i + 1)
        n = self.GRID_OPTS[self.grid_i]
        self._grid_var.set(f'{n}×{n}')

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

class AStarPanel(tk.Frame):
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
        tk.Label(cfg, text='Siatka – ustawiana w zakładce DQN',
                 bg=C['panel'], fg=C['text3'],
                 font=('Segoe UI', 8)).grid(row=1, column=0, columnspan=5,
                                             sticky='w', padx=4, pady=(0, 4))

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
    def step_delay(self) -> float:
        return self.DELAY_OPTS[self.delay_i] / 1000.0


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

        self._build_ui()
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
        tk.Label(hdr,
                 text='GPU ON' if config.UŻYJ_GPU else 'CPU ONLY',
                 bg='#0e0e1a',
                 fg=C['accent'] if config.UŻYJ_GPU else C['text3'],
                 font=('Segoe UI', 9)).pack(side='right', padx=14)
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

        # lewy panel: canvas gry + lista modeli
        left = tk.Frame(main, bg=C['panel'], width=CW + 8)
        left.pack(side='left', fill='y')
        left.pack_propagate(False)

        self.game_canvas = GameCanvas(left)
        self.game_canvas.pack(fill='x')

        tk.Frame(left, bg=C['border'], height=1).pack(fill='x')

        self.model_list = ModelList(left, on_select=self._on_model_sel)
        self.model_list.pack(fill='both', expand=True)

        # separator pionowy
        tk.Frame(main, bg=C['border'], width=1).pack(side='left', fill='y')

        # prawy panel: notebook algorytmów
        right = tk.Frame(main, bg=C['panel'])
        right.pack(side='left', fill='both', expand=True)

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
                if self._active_mode == 'dqn':
                    self.model_list.reload()
                    self._log('Zakończono. Modele zapisane.')
                else:
                    self._log('A* zatrzymany.')
                with self.glock:
                    self.game_state = None
            elif d.get('benchmark'):
                self._log(d['benchmark'])
            elif d.get('astar'):
                self.astar_panel.update_stats(d['score'], d['best'], d['game'])
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
        if not v:
            self.game_canvas.update_state(None)

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
        self.agent = agent or DQNAgent(11, 3, ROZMIAR_UKRYTY)
        self.dqn.graph.clear()
        with self.glock:
            self.game_state = None
        self._set_active(True)
        self.dqn.set_status('Trening…', C['accent3'])

        grid_n = self.dqn.grid_size
        self._log(f'Start {mode.upper()} | {self.dqn.episodes} ep | siatka {grid_n}×{grid_n}')
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
        agent = DQNAgent(11, 3, ROZMIAR_UKRYTY)
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
        agent = DQNAgent(11, 3, ROZMIAR_UKRYTY)
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
            for _ in range(real_n):
                if self.stop_ev.is_set():
                    break
                d, _, sc = uruchom_epizod(self.agent, params)
                exps.extend(d)
                scores.append(sc)
            t_collect_total += time.perf_counter() - t0

            ep += len(scores)
            for e in exps:
                self.agent.zapamiętaj(*e)

            iters = min(len(exps), 2000 if config.UŻYJ_GPU else 1000)
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
                self.agent.save(os.path.join(KATALOG_MODELI, 'snake_model_best.pth'))
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

        self.agent.save(os.path.join(KATALOG_MODELI, 'snake_model_final.pth'))
        self.sq.put({'done': True})

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
                self.agent.save(os.path.join(KATALOG_MODELI, 'snake_model_best.pth'))
            self._push(score, best, ep_i + 1, total, avg_l)

        self.agent.save(os.path.join(KATALOG_MODELI, 'snake_model_final.pth'))
        self.sq.put({'done': True})

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

        grid_n = self.dqn.grid_size
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
        best, game_n = 0, 0

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
                    )

                d = self.astar_panel.step_delay
                if d > 0:
                    time.sleep(d)

            if score > best:
                best = score
            game_n += 1
            self.sq.put({'astar': True, 'score': score,
                         'best': best, 'game': game_n})

        self.sq.put({'done': True})


# ── Uruchomienie ──────────────────────────────────────────────────────

if __name__ == '__main__':
    app = App()
    app.mainloop()
