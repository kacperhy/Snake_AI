"""
GUI dla Snake AI - Deep Q-Network.
Uruchom: python gui.py
"""

import pygame
import threading
import queue
import os
import math
import torch
import numpy as np

from snake_game import UproszczonySnake
from agent import DQNAgent
from training import uruchom_epizod
import config
from config import (
    SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU,
    ROZMIAR_UKRYTY, KATALOG_MODELI, LICZBA_WĄTKÓW_CPU
)

# ── Wymiary ──────────────────────────────────────────────────────────
WIN_W, WIN_H  = 720, 568
LPANEL_W      = 342
RPANEL_W      = WIN_W - LPANEL_W
HEADER_H      = 48
FOOTER_H      = 34
CONTENT_H     = WIN_H - HEADER_H - FOOTER_H

# ── Kolory ───────────────────────────────────────────────────────────
C = {
    'bg':        (11, 11, 20),
    'panel':     (19, 19, 32),
    'panel2':    (26, 26, 42),
    'border':    (44, 44, 66),
    'accent':    (0,  210, 100),
    'accent2':   (60, 145, 255),
    'accent3':   (255, 165, 40),
    'danger':    (195, 50, 50),
    'text':      (218, 218, 228),
    'text2':     (135, 135, 158),
    'text3':     (76, 76, 102),
    'btn':       (32, 32, 52),
    'btn_h':     (50, 50, 76),
    'grn':       (0, 148, 68),
    'grn_h':     (0, 182, 85),
    'blue':      (38, 98, 195),
    'blue_h':    (52, 118, 225),
    'stop':      (150, 38, 38),
    'stop_h':    (190, 52, 52),
    'sel':       (0, 75, 42),
    'sel_b':     (0, 155, 75),
    'snake_h':   (0, 230, 108),
    'snake_b':   (0, 182, 84),
    'snake_i':   (0, 142, 64),
    'food':      (225, 55, 55),
    'food_i':    (255, 110, 72),
    'grid':      (16, 16, 28),
    'gridl':     (21, 21, 35),
    'gbg':       (13, 13, 23),
    'gline':     (0, 210, 100),
    'gavg':      (60, 145, 255),
}

STATE_MENU     = 'menu'
STATE_TESTING  = 'testing'
STATE_TRAINING = 'training'


# ── Widgety ──────────────────────────────────────────────────────────

class Btn:
    """Prosty przycisk pygame."""
    def __init__(self, rect, label, bg, bg_h, font, radius=6):
        self.rect    = pygame.Rect(rect)
        self.label   = label
        self.bg      = bg
        self.bg_h    = bg_h
        self.font    = font
        self.radius  = radius
        self.hover   = False
        self.enabled = True

    def update(self, pos):
        self.hover = self.rect.collidepoint(pos) and self.enabled

    def draw(self, surf):
        bg = (self.bg_h if self.hover else self.bg)
        if not self.enabled:
            bg = tuple(max(0, v - 18) for v in self.bg)
        pygame.draw.rect(surf, bg, self.rect, border_radius=self.radius)
        pygame.draw.rect(surf, C['border'], self.rect, 1, border_radius=self.radius)
        tc = C['text'] if self.enabled else C['text3']
        s  = self.font.render(self.label, True, tc)
        surf.blit(s, s.get_rect(center=self.rect.center))

    def clicked(self, ev):
        return (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and self.rect.collidepoint(ev.pos) and self.enabled)


class Graph:
    """Wykres wyników — linia + średnia ruchoma."""
    def __init__(self, max_pts=200):
        self.rect    = pygame.Rect(0, 0, 100, 60)
        self.data    = []
        self.avg     = []
        self.max_pts = max_pts

    def add(self, v):
        self.data.append(float(v))
        w = min(20, len(self.data))
        self.avg.append(float(np.mean(self.data[-w:])))

    def clear(self):
        self.data.clear()
        self.avg.clear()

    def draw(self, surf, f_sm):
        r = self.rect
        pygame.draw.rect(surf, C['gbg'],   r, border_radius=4)
        pygame.draw.rect(surf, C['border'], r, 1, border_radius=4)

        if len(self.data) < 2:
            s = f_sm.render("Oczekiwanie na dane…", True, C['text3'])
            surf.blit(s, s.get_rect(center=r.center))
            return

        pts  = self.data[-self.max_pts:]
        avgs = self.avg[-self.max_pts:]
        n    = len(pts)
        pad  = 6
        iw   = r.w - 2 * pad
        ih   = r.h - 2 * pad - 12
        lo   = min(pts + [0])
        hi   = max(pts + [1])
        rng  = hi - lo or 1

        def px(i, v):
            return (r.x + pad + int(i / max(n - 1, 1) * iw),
                    r.y + pad + int((1 - (v - lo) / rng) * ih))

        raw = [px(i, v) for i, v in enumerate(pts)]
        avg = [px(i, v) for i, v in enumerate(avgs)]
        if len(raw) >= 2:
            pygame.draw.lines(surf, C['gline'], False, raw, 1)
        if len(avg) >= 2:
            pygame.draw.lines(surf, C['gavg'],  False, avg, 1)

        s1 = f_sm.render(f"Max {int(hi)}", True, C['text3'])
        s2 = f_sm.render(f"n={len(self.data)}", True, C['text3'])
        surf.blit(s1, (r.x + pad,                    r.y + r.h - 12))
        surf.blit(s2, (r.right - s2.get_width() - pad, r.y + r.h - 12))


# ── Główna klasa ─────────────────────────────────────────────────────

class GUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Snake AI  ·  Deep Q-Network")
        self.clock  = pygame.time.Clock()

        self.f_title = pygame.font.SysFont('segoeui', 21, bold=True)
        self.f_lg    = pygame.font.SysFont('segoeui', 17, bold=True)
        self.f_md    = pygame.font.SysFont('segoeui', 14)
        self.f_sm    = pygame.font.SysFont('segoeui', 12)

        self.running    = True
        self.state      = STATE_MENU
        self.mode       = 'hybrid'
        self.tick       = 0

        self.agent      = None
        self.game_state = None
        self.glock      = threading.Lock()

        self.worker     = None
        self.stop_ev    = threading.Event()
        self.sq         = queue.Queue()

        self.stats = dict(score=0, best=0, episode=0, total=0,
                          epsilon=1.0, loss=0.0, status='Gotowy')
        self.graph = Graph()
        self.log   = []

        # Parametry treningu
        self.ep_opts  = [50, 100, 200, 500, 1000, 2000]
        self.ep_i     = 2        # 200
        self.par_opts = [2, 4, 8, 16]
        self.par_i    = 1        # 4
        self.save_ev  = 50

        # Modele
        self.models    = []
        self.sel       = 0
        self.scroll    = 0
        self._reload()

        # Rects +/− (ustawiane przy rysowaniu)
        self._rm_ep  = pygame.Rect(0, 0, 0, 0)
        self._rp_ep  = pygame.Rect(0, 0, 0, 0)
        self._rm_par = pygame.Rect(0, 0, 0, 0)
        self._rp_par = pygame.Rect(0, 0, 0, 0)

        self._build()

    def _build(self):
        f = self.f_md
        fs = self.f_sm
        self.btn_hybrid  = Btn((0,0,1,1), "▶  Trening Hybrydowy", C['grn'],  C['grn_h'],  f)
        self.btn_cpu     = Btn((0,0,1,1), "▶  Trening CPU",        C['btn'],  C['btn_h'],  f)
        self.btn_test    = Btn((0,0,1,1), "▶  Testuj Model",        C['blue'], C['blue_h'], f)
        self.btn_cont    = Btn((0,0,1,1), "↺  Kontynuuj Trening",  C['btn'],  C['btn_h'],  f)
        self.btn_reload  = Btn((0,0,1,1), "↺  Odśwież",            C['btn'],  C['btn_h'],  fs)
        self.btn_stop    = Btn((0,0,1,1), "■  ZATRZYMAJ",           C['stop'], C['stop_h'], f)
        self._all_btns = [self.btn_hybrid, self.btn_cpu, self.btn_test,
                          self.btn_cont, self.btn_reload, self.btn_stop]

    # ── Modele ───────────────────────────────────────────────────────

    def _reload(self):
        if os.path.exists(KATALOG_MODELI):
            self.models = sorted(
                [f for f in os.listdir(KATALOG_MODELI) if f.endswith('.pth')],
                key=lambda f: os.path.getmtime(os.path.join(KATALOG_MODELI, f)),
                reverse=True)
        else:
            self.models = []
        self.sel = min(self.sel, max(0, len(self.models) - 1))

    def _log(self, msg):
        self.log.append(msg)
        if len(self.log) > 30:
            self.log.pop(0)

    # ── Wątki treningowe ─────────────────────────────────────────────

    def _go_train(self, mode, agent=None):
        if self.worker and self.worker.is_alive():
            return
        self.stop_ev.clear()
        while not self.sq.empty():
            self.sq.get_nowait()

        self.agent = agent or DQNAgent(11, 3, ROZMIAR_UKRYTY)
        ep = self.ep_opts[self.ep_i]
        self.stats.update(episode=0, total=ep, best=0,
                          epsilon=self.agent.epsilon, status='Trening…')
        self.graph.clear()
        self.mode  = mode
        self.state = STATE_TRAINING
        dev = 'GPU' if config.UŻYJ_GPU else 'CPU'
        self._log(f"Start {mode.upper()} | {ep} ep | {dev}")

        params = dict(szerokość=SZEROKOŚĆ_OKNA, wysokość=WYSOKOŚĆ_OKNA,
                      rozmiar_bloku=ROZMIAR_BLOKU)

        def run():
            if mode == 'hybrid':
                self._th_hybrid(params, ep, self.par_opts[self.par_i])
            else:
                self._th_cpu(ep)

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _push(self, score, best, ep, loss):
        """Bezpieczne wysłanie statystyk do kolejki."""
        self.sq.put(dict(score=score, best=best, episode=ep,
                         epsilon=self.agent.epsilon, loss=loss))

    def _th_hybrid(self, params, total, parallel):
        best, ep = 0, 0
        chunks = (total + parallel - 1) // parallel

        for ch in range(chunks):
            if self.stop_ev.is_set():
                break
            real_n = min(parallel, total - ch * parallel)
            exps, scores = [], []

            for _ in range(real_n):
                if self.stop_ev.is_set():
                    break
                d, _, sc = uruchom_epizod(self.agent, params)
                exps.extend(d)
                scores.append(sc)

            ep += len(scores)
            for e in exps:
                self.agent.zapamiętaj(*e)

            iters = min(len(exps), 2000 if config.UŻYJ_GPU else 1000)
            tl, nl = 0.0, 0

            for i in range(iters):
                if self.stop_ev.is_set():
                    break
                l = self.agent.ucz_się()
                if l is not None:
                    tl += l
                    nl += 1
                # Wysyłaj aktualizację epsilon co 200 kroków treningu
                if nl > 0 and i % 200 == 199:
                    self._push(scores[-1] if scores else 0, best, ep, tl / nl)

            avg_l = tl / nl if nl else 0.0
            best_h = max(scores) if scores else 0
            if best_h > best:
                best = best_h
                self.agent.save(os.path.join(KATALOG_MODELI, "snake_model_best.pth"))

            self._push(scores[-1] if scores else 0, best, ep, avg_l)

            if ep % max(1, self.save_ev) < parallel:
                self.agent.save(os.path.join(KATALOG_MODELI,
                                             f"snake_model_episode_{ep}.pth"))

        self.agent.save(os.path.join(KATALOG_MODELI, "snake_model_final.pth"))
        self.sq.put(dict(done=True))

    def _th_cpu(self, total):
        game = UproszczonySnake(SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU)
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
                        score=game.wynik)

            avg_l = tl / nl if nl else 0.0
            if score > best:
                best = score
                self.agent.save(os.path.join(KATALOG_MODELI, "snake_model_best.pth"))

            self._push(score, best, ep_i + 1, avg_l)

            if (ep_i + 1) % max(1, self.save_ev) == 0:
                self.agent.save(os.path.join(KATALOG_MODELI,
                                             f"snake_model_episode_{ep_i+1}.pth"))

        self.agent.save(os.path.join(KATALOG_MODELI, "snake_model_final.pth"))
        self.sq.put(dict(done=True))

    def _go_test(self):
        if not self.models or (self.worker and self.worker.is_alive()):
            return
        path  = os.path.join(KATALOG_MODELI, self.models[self.sel])
        agent = DQNAgent(11, 3, ROZMIAR_UKRYTY)
        if not agent.wczytaj(path):
            self._log(f"Błąd wczytywania: {self.models[self.sel]}")
            return
        agent._loaded_from = path
        self.agent = agent
        self.stop_ev.clear()
        while not self.sq.empty():
            self.sq.get_nowait()
        self.stats.update(episode=0, total=0, best=0,
                          epsilon=0.0, status='Testowanie')
        self.graph.clear()
        self.mode  = 'cpu'
        self.state = STATE_TESTING
        self._log(f"Testowanie: {self.models[self.sel]}")

        def run():
            game = UproszczonySnake(SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA, ROZMIAR_BLOKU)
            best, ep = 0, 0
            while not self.stop_ev.is_set():
                stan = game.reset()
                done = False
                score = 0
                while not done and not self.stop_ev.is_set():
                    t = torch.tensor(stan, dtype=agent.typ_danych).unsqueeze(0).to(agent.device)
                    with torch.no_grad():
                        akcja = torch.argmax(agent.model(t)).item()
                    stan, _, done, score = game.krok(akcja)
                    with self.glock:
                        self.game_state = dict(
                            snake=[list(p) for p in game.snake],
                            food=list(game.jedzenie),
                            score=game.wynik)
                    pygame.time.delay(55)
                if score > best:
                    best = score
                ep += 1
                self.sq.put(dict(score=score, best=best, episode=ep,
                                 epsilon=0.0, loss=0.0))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _go_continue(self):
        if not self.models:
            return
        path  = os.path.join(KATALOG_MODELI, self.models[self.sel])
        agent = DQNAgent(11, 3, ROZMIAR_UKRYTY)
        if not agent.wczytaj(path):
            self._log(f"Błąd wczytywania: {self.models[self.sel]}")
            return
        self._log(f"Kontynuuję: {self.models[self.sel]}")
        self._go_train('hybrid', agent)

    def _stop(self):
        self.stop_ev.set()
        self.stats['status'] = 'Zatrzymywanie…'

    # ── Kolejka statystyk ────────────────────────────────────────────

    def _flush(self):
        while not self.sq.empty():
            d = self.sq.get_nowait()
            if d.get('done'):
                self.state = STATE_MENU
                self.stats['status'] = 'Gotowy'
                self._reload()
                self._log("Zakończono. Modele zapisane.")
                with self.glock:
                    self.game_state = None
            else:
                for k in ('score', 'best', 'episode', 'epsilon', 'loss'):
                    if k in d:
                        self.stats[k] = d[k]
                if 'score' in d:
                    self.graph.add(d['score'])
                ep = self.stats['episode']
                if ep % 25 == 0 and ep > 0:
                    self._log(f"Ep {ep}  wynik={d['score']}  "
                              f"ε={self.stats['epsilon']:.3f}  "
                              f"loss={self.stats['loss']:.4f}")

        if (self.state in (STATE_TRAINING, STATE_TESTING)
                and self.worker and not self.worker.is_alive()):
            self.state = STATE_MENU
            self.stats['status'] = 'Gotowy'
            self._reload()
            with self.glock:
                self.game_state = None

    # ── Zdarzenia ────────────────────────────────────────────────────

    def _events(self):
        pos = pygame.mouse.get_pos()
        for b in self._all_btns:
            b.update(pos)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
                self.stop_ev.set()
                return

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE and self.state != STATE_MENU:
                    self._stop()

            if ev.type == pygame.MOUSEWHEEL and self.state == STATE_MENU:
                self.scroll = max(0, min(max(0, len(self.models) - 1),
                                         self.scroll - ev.y))

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos

                # Kliknięcie na listę modeli
                if self.state == STATE_MENU and mx < LPANEL_W and my > HEADER_H + 38:
                    ih  = 34
                    rel = my - (HEADER_H + 38)
                    idx = rel // ih + self.scroll
                    if 0 <= idx < len(self.models):
                        self.sel = idx

                # +/− epizody
                if self._rp_ep.collidepoint(mx, my):
                    self.ep_i = min(self.ep_i + 1, len(self.ep_opts) - 1)
                elif self._rm_ep.collidepoint(mx, my):
                    self.ep_i = max(self.ep_i - 1, 0)

                # +/− równoległe
                if self._rp_par.collidepoint(mx, my):
                    self.par_i = min(self.par_i + 1, len(self.par_opts) - 1)
                elif self._rm_par.collidepoint(mx, my):
                    self.par_i = max(self.par_i - 1, 0)

            # Przyciski menu
            if self.state == STATE_MENU:
                if self.btn_hybrid.clicked(ev):
                    self._go_train('hybrid')
                elif self.btn_cpu.clicked(ev):
                    self._go_train('cpu')
                elif self.btn_test.clicked(ev):
                    self._go_test()
                elif self.btn_cont.clicked(ev):
                    self._go_continue()
                elif self.btn_reload.clicked(ev):
                    self._reload()
                    self._log("Lista modeli odświeżona.")

            # Stop
            if self.state in (STATE_TRAINING, STATE_TESTING):
                if self.btn_stop.clicked(ev):
                    self._stop()

    # ── Rysowanie ────────────────────────────────────────────────────

    def _hdr(self):
        pygame.draw.rect(self.screen, (14, 14, 26),
                         pygame.Rect(0, 0, WIN_W, HEADER_H))
        pygame.draw.line(self.screen, C['border'],
                         (0, HEADER_H - 1), (WIN_W, HEADER_H - 1))

        t = self.f_title.render("Snake AI  ·  Deep Q-Network", True, C['accent'])
        self.screen.blit(t, (14, HEADER_H // 2 - t.get_height() // 2))

        col = {'Gotowy': C['text2'], 'Trening…': C['accent3'],
               'Testowanie': C['accent'],
               'Zatrzymywanie…': C['danger']}.get(self.stats['status'], C['text2'])
        st = self.f_md.render(self.stats['status'], True, col)
        self.screen.blit(st, (WIN_W - st.get_width() - 14,
                              HEADER_H // 2 - st.get_height() // 2))

    def _ftr(self):
        fy = WIN_H - FOOTER_H
        pygame.draw.rect(self.screen, (12, 12, 22),
                         pygame.Rect(0, fy, WIN_W, FOOTER_H))
        pygame.draw.line(self.screen, C['border'], (0, fy), (WIN_W, fy))
        if self.log:
            s = self.f_sm.render(self.log[-1], True, C['text3'])
            self.screen.blit(s, (10, fy + FOOTER_H // 2 - s.get_height() // 2))
        inf = f"GPU: {'ON' if config.UŻYJ_GPU else 'OFF'}  |  FPS: {int(self.clock.get_fps())}"
        g = self.f_sm.render(inf, True, C['text3'])
        self.screen.blit(g, (WIN_W - g.get_width() - 10,
                             fy + FOOTER_H // 2 - g.get_height() // 2))

    # Lewy panel ─────────────────────────────────────────────────────

    def _left(self):
        pygame.draw.rect(self.screen, C['panel'],
                         pygame.Rect(0, HEADER_H, LPANEL_W, CONTENT_H))
        pygame.draw.line(self.screen, C['border'],
                         (LPANEL_W, HEADER_H), (LPANEL_W, WIN_H - FOOTER_H))

        if self.state in (STATE_TESTING, STATE_TRAINING):
            self._canvas()
            self._canvas_info()
        else:
            self._model_list()

    def _canvas(self):
        cx = (LPANEL_W - SZEROKOŚĆ_OKNA) // 2
        cy = HEADER_H + 14
        cr = pygame.Rect(cx, cy, SZEROKOŚĆ_OKNA, WYSOKOŚĆ_OKNA)

        pygame.draw.rect(self.screen, C['grid'], cr)
        for gx in range(0, SZEROKOŚĆ_OKNA + 1, ROZMIAR_BLOKU):
            pygame.draw.line(self.screen, C['gridl'],
                             (cx + gx, cy), (cx + gx, cy + WYSOKOŚĆ_OKNA))
        for gy in range(0, WYSOKOŚĆ_OKNA + 1, ROZMIAR_BLOKU):
            pygame.draw.line(self.screen, C['gridl'],
                             (cx, cy + gy), (cx + SZEROKOŚĆ_OKNA, cy + gy))

        with self.glock:
            gs = self.game_state

        if gs:
            for i, pt in enumerate(gs['snake']):
                rx_, ry_ = cx + pt[0], cy + pt[1]
                c1 = C['snake_h'] if i == 0 else C['snake_b']
                c2 = C['snake_i'] if i > 0  else (0, 175, 84)
                pygame.draw.rect(self.screen, c1,
                                 (rx_, ry_, ROZMIAR_BLOKU, ROZMIAR_BLOKU))
                pygame.draw.rect(self.screen, c2,
                                 (rx_ + 3, ry_ + 3,
                                  ROZMIAR_BLOKU - 6, ROZMIAR_BLOKU - 6))

            pulse = abs(math.sin(self.tick * 0.07)) * 2
            fx_, fy_ = cx + gs['food'][0], cy + gs['food'][1]
            pygame.draw.rect(self.screen, C['food'],
                             (fx_, fy_, ROZMIAR_BLOKU, ROZMIAR_BLOKU))
            s  = ROZMIAR_BLOKU - 8 - int(pulse)
            o  = 4 + int(pulse / 2)
            pygame.draw.rect(self.screen, C['food_i'],
                             (fx_ + o, fy_ + o, s, s))

            sc = self.f_lg.render(str(gs['score']), True, C['text'])
            self.screen.blit(sc, (cx + 4, cy + 4))

        elif self.state == STATE_TRAINING and self.mode == 'hybrid':
            pulse = abs(math.sin(self.tick * 0.05))
            col   = tuple(int(30 + pulse * 50) for _ in range(3))
            col   = (int(0 + pulse * 60), int(120 + pulse * 80), int(50 + pulse * 50))
            lines = [
                "Trening hybrydowy",
                "działa w tle",
                "",
                f"Ep: {self.stats['episode']} / {self.stats['total']}",
                f"ε = {self.stats['epsilon']:.4f}",
            ]
            for i, line in enumerate(lines):
                c = col if i < 2 else C['text3']
                s = self.f_md.render(line, True, c)
                self.screen.blit(s, s.get_rect(
                    center=(cx + SZEROKOŚĆ_OKNA // 2, cy + 70 + i * 26)))

        pygame.draw.rect(self.screen, C['border'], cr, 1)

    def _canvas_info(self):
        y = HEADER_H + WYSOKOŚĆ_OKNA + 22
        x = 10
        mname = '—'
        if self.agent and hasattr(self.agent, '_loaded_from'):
            mname = os.path.basename(self.agent._loaded_from)
        for txt in [f"Model: {mname}",
                    "Tryb: " + ("Test" if self.state == STATE_TESTING else "Trening CPU")]:
            s = self.f_sm.render(txt, True, C['text2'])
            self.screen.blit(s, (x, y))
            y += 16
        y += 4
        for msg in self.log[-5:]:
            s = self.f_sm.render(msg, True, C['text3'])
            self.screen.blit(s, (x, y))
            y += 14

    def _model_list(self):
        x, y = 12, HEADER_H + 12
        h = self.f_lg.render("Dostępne modele", True, C['text'])
        self.screen.blit(h, (x, y))
        y += 28

        if not self.models:
            s1 = self.f_md.render("Brak modeli w katalogu models/", True, C['text3'])
            s2 = self.f_sm.render("Wytrenuj nowy model →", True, C['text3'])
            self.screen.blit(s1, (x, y))
            self.screen.blit(s2, (x, y + 22))
            return

        ih = 34
        vis = (CONTENT_H - 45) // ih
        for i in range(vis):
            idx = i + self.scroll
            if idx >= len(self.models):
                break
            m   = self.models[idx]
            r   = pygame.Rect(x - 4, y, LPANEL_W - 16, ih - 2)
            sel = idx == self.sel
            pygame.draw.rect(self.screen,
                             C['sel'] if sel else C['panel2'], r, border_radius=4)
            if sel:
                pygame.draw.rect(self.screen, C['sel_b'], r, 1, border_radius=4)
            ns = self.f_md.render(m, True, C['text'] if sel else C['text2'])
            self.screen.blit(ns, (x + 4, y + 7))
            fp = os.path.join(KATALOG_MODELI, m)
            if os.path.exists(fp):
                sz = self.f_sm.render(f"{os.path.getsize(fp)//1024} KB",
                                      True, C['text3'])
                self.screen.blit(sz, (LPANEL_W - sz.get_width() - 12, y + 9))
            y += ih
        if len(self.models) > vis:
            s = self.f_sm.render(f"↑↓ przewiń  ({len(self.models)} plików)",
                                 True, C['text3'])
            self.screen.blit(s, (x, WIN_H - FOOTER_H - 18))

    # Prawy panel ────────────────────────────────────────────────────

    def _right(self):
        pygame.draw.rect(self.screen, C['panel'],
                         pygame.Rect(LPANEL_W, HEADER_H, RPANEL_W, CONTENT_H))
        if self.state == STATE_MENU:
            self._right_menu()
        else:
            self._right_active()

    def _right_menu(self):
        rx  = LPANEL_W + 14
        y   = HEADER_H + 16
        bw  = RPANEL_W - 28       # szerokość pełna dla dużych przycisków
        bwh = (bw - 6) // 2      # połówki
        bh  = 34
        gap = 7

        # ── Konfiguracja treningu ──
        lbl = self.f_lg.render("Konfiguracja treningu", True, C['text'])
        self.screen.blit(lbl, (rx, y))
        y += 26

        def stepper(label, val_str, rm_rect_attr, rp_rect_attr, row_y):
            """Rysuje wiersz z etykietą, [−] wartość [+]."""
            ls = self.f_md.render(label, True, C['text2'])
            self.screen.blit(ls, (rx, row_y + 4))

            pw = 26  # szerokość przycisku +/−
            val_x = rx + 148
            minus_r = pygame.Rect(val_x,       row_y, pw, bh - 4)
            val_r   = pygame.Rect(val_x + pw + 2, row_y, 54, bh - 4)
            plus_r  = pygame.Rect(val_x + pw + 58, row_y, pw, bh - 4)

            # Tło pola wartości
            pygame.draw.rect(self.screen, C['panel2'], val_r, border_radius=4)
            pygame.draw.rect(self.screen, C['border'], val_r, 1, border_radius=4)

            # Przyciski − +
            for r_, sym in [(minus_r, '−'), (plus_r, '+')]:
                pos = pygame.mouse.get_pos()
                hov = r_.collidepoint(pos)
                bg  = C['btn_h'] if hov else C['btn']
                pygame.draw.rect(self.screen, bg, r_, border_radius=4)
                pygame.draw.rect(self.screen, C['border'], r_, 1, border_radius=4)
                s = self.f_lg.render(sym, True, C['text'])
                self.screen.blit(s, s.get_rect(center=r_.center))

            vs = self.f_md.render(val_str, True, C['accent'])
            self.screen.blit(vs, vs.get_rect(center=val_r.center))

            # Zapisz rects do obsługi kliknięć
            setattr(self, rm_rect_attr, minus_r)
            setattr(self, rp_rect_attr, plus_r)

        stepper("Epizody",       str(self.ep_opts[self.ep_i]),
                '_rm_ep',  '_rp_ep',  y)
        y += bh + 2

        stepper("Równoległe gry", str(self.par_opts[self.par_i]),
                '_rm_par', '_rp_par', y)
        y += bh + 2

        # GPU status
        dev_col = C['accent'] if config.UŻYJ_GPU else C['text3']
        dev_s   = self.f_sm.render(
            "GPU: " + ("TAK  (CUDA)" if config.UŻYJ_GPU else "NIE  (tylko CPU)"),
            True, dev_col)
        self.screen.blit(dev_s, (rx, y + 4))
        y += 22

        # Separator
        pygame.draw.line(self.screen, C['border'], (rx - 4, y), (WIN_W - 8, y))
        y += 10

        # ── Duże przyciski treningu ──
        self.btn_hybrid.rect = pygame.Rect(rx, y, bw, bh + 2)
        self.btn_hybrid.draw(self.screen)
        y += bh + 2 + gap

        self.btn_cpu.rect = pygame.Rect(rx, y, bw, bh)
        self.btn_cpu.draw(self.screen)
        y += bh + gap + 6

        # Separator
        pygame.draw.line(self.screen, C['border'], (rx - 4, y), (WIN_W - 8, y))
        y += 10

        # ── Przyciski pomocnicze ──
        self.btn_test.enabled = bool(self.models)
        self.btn_cont.enabled = bool(self.models)

        self.btn_test.rect   = pygame.Rect(rx,          y, bwh, bh)
        self.btn_cont.rect   = pygame.Rect(rx + bwh + 6, y, bwh, bh)
        self.btn_test.draw(self.screen)
        self.btn_cont.draw(self.screen)
        y += bh + gap

        self.btn_reload.rect = pygame.Rect(rx, y, 100, bh - 4)
        self.btn_reload.draw(self.screen)

        # Wykres (jeśli są dane)
        avail = WIN_H - FOOTER_H - y - bh - 18
        if len(self.graph.data) > 1 and avail > 50:
            y2 = WIN_H - FOOTER_H - 8 - min(avail, 110)
            self.graph.rect = pygame.Rect(rx - 4, y2, RPANEL_W - 20, min(avail, 110))
            self.graph.draw(self.screen, self.f_sm)

    def _right_active(self):
        rx  = LPANEL_W + 14
        y   = HEADER_H + 16
        bw  = RPANEL_W - 28
        bh  = 34

        # ── Nagłówek ──
        lbl = self.f_lg.render("Statystyki", True, C['text'])
        self.screen.blit(lbl, (rx, y))
        y += 28

        # ── Dwa bloki 2×2 ──
        def stat_box(lbl, val, col, bx, by, bw_, bh_):
            r = pygame.Rect(bx, by, bw_, bh_)
            pygame.draw.rect(self.screen, C['panel2'], r, border_radius=5)
            pygame.draw.rect(self.screen, C['border'], r, 1, border_radius=5)
            ls = self.f_sm.render(lbl, True, C['text3'])
            vs = self.f_lg.render(str(val), True, col)
            self.screen.blit(ls, (bx + 6, by + 4))
            self.screen.blit(vs, (bx + 6, by + 16))

        hw = (bw - 6) // 2
        hh = 44
        stat_box("WYNIK",     self.stats['score'], C['accent'],  rx,         y, hw, hh)
        stat_box("NAJLEPSZY", self.stats['best'],  C['accent3'], rx + hw + 6, y, hw, hh)
        y += hh + 6

        ep_str = (f"{self.stats['episode']}"
                  + (f" / {self.stats['total']}" if self.stats['total'] else ""))
        stat_box("EPIZOD",   ep_str,                       C['text'],  rx,         y, hw, hh)
        stat_box("EPSILON",  f"{self.stats['epsilon']:.4f}", C['accent2'], rx + hw + 6, y, hw, hh)
        y += hh + 6

        # Loss jako pasek
        loss_r = pygame.Rect(rx, y, bw, 30)
        pygame.draw.rect(self.screen, C['panel2'], loss_r, border_radius=5)
        pygame.draw.rect(self.screen, C['border'], loss_r, 1, border_radius=5)
        ll = self.f_sm.render("LOSS", True, C['text3'])
        lv = self.f_md.render(f"{self.stats['loss']:.5f}", True, C['text2'])
        self.screen.blit(ll, (rx + 6, y + 4))
        self.screen.blit(lv, (rx + 60, y + 6))
        y += 36

        # Separator
        pygame.draw.line(self.screen, C['border'], (rx - 4, y), (WIN_W - 8, y))
        y += 8

        # Wykres
        avail = WIN_H - FOOTER_H - y - bh - 24
        gh    = max(50, min(120, avail))
        self.graph.rect = pygame.Rect(rx - 4, y, RPANEL_W - 20, gh)
        self.graph.draw(self.screen, self.f_sm)
        y += gh + 10

        # Separator
        pygame.draw.line(self.screen, C['border'], (rx - 4, y), (WIN_W - 8, y))
        y += 8

        # Stop
        self.btn_stop.rect = pygame.Rect(rx, y, bw, bh)
        self.btn_stop.draw(self.screen)
        y += bh + 4

        hint = self.f_sm.render("lub naciśnij ESC", True, C['text3'])
        self.screen.blit(hint, (rx, y))

    # ── Pętla główna ─────────────────────────────────────────────────

    def run(self):
        while self.running:
            self.clock.tick(30)
            self.tick += 1
            self._events()
            self._flush()

            self.screen.fill(C['bg'])
            self._hdr()
            self._left()
            self._right()
            self._ftr()
            pygame.display.flip()

        pygame.quit()


if __name__ == '__main__':
    gui = GUI()
    gui.run()
