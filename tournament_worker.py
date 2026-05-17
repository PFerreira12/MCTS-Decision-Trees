"""
tournament_worker.py — função worker para o torneio MCTS paralelizado.

PORQUÊ ficheiro separado e não inline no notebook?
  No Windows, o multiprocessing usa 'spawn': cada worker inicia um
  interpretador Python limpo e importa o módulo onde a função está
  definida. Se a função estiver em __main__ (o kernel Jupyter), o
  worker tenta importar o kernel → crash imediato.
  Ao colocar a função neste ficheiro regular (.py), os workers importam
  tournament_worker sem qualquer dependência do notebook.

Não referencia globals do notebook. Tudo chega via o dict 'task'.
"""

from __future__ import annotations

import os
import sys

# Garante que o diretório do projeto está no path dos workers
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import random
import time

import numpy as np

from logic import PopOutGame
from mcts import MCTS, MCTS_RAVE, MCTSTopK, MCTSWithHeuristics

_STRATEGY_CLS = {
    'Standard' : MCTS,
    'RAVE'     : MCTS_RAVE,
    'Top-K'    : MCTSTopK,
    'Heuristic': MCTSWithHeuristics,
}


def _make_agent(cfg: dict):
    """Cria um agente MCTS a partir de um dict de configuração."""
    cls = _STRATEGY_CLS[cfg['strategy']]
    kwargs: dict = {
        'iterations': cfg['iterations'],
        'time_limit': cfg['time_limit'],
        'c':          cfg['c'],
    }
    if cfg['strategy'] == 'Top-K':
        kwargs['k'] = cfg['k']
    return cls(**kwargs)


def _diag(agent) -> dict:
    if not hasattr(agent, 'root') or agent.root is None:
        return {}
    ch = agent.root.children
    if not ch:
        return {}
    vis = np.array([c.visits for c in ch], dtype=float)
    tot = vis.sum()
    if tot == 0:
        return {}
    pr  = vis / tot
    ent = float(-np.sum(pr * np.log2(pr + 1e-12)))
    mv  = max(ch, key=lambda c: c.visits)
    bc  = agent.root.most_visited_child()
    return {
        'root_sims':     int(tot),
        'n_children':    len(ch),
        'best_visit_wr': round(mv.wins / mv.visits if mv.visits else 0.0, 4),
        'chosen_wr':     round(bc.wins / bc.visits if bc.visits else 0.0, 4),
        'entropy':       round(ent, 4),
    }


def _phase(mn: int) -> str:
    f = mn / 40
    return 'opening' if f < 0.33 else ('midgame' if f < 0.67 else 'endgame')


def _play_game(a1, a2) -> dict:
    game = PopOutGame(rows=6, cols=7)
    ags  = {1: a1, 2: a2}
    mts  = {1: [], 2: []}
    log  = []
    p1p = p2p = 0
    mn = 0
    while not game.game_over:
        p  = game.current_player
        t0 = time.time()
        mv = ags[p].choose_move(game)
        ms = (time.time() - t0) * 1000
        mts[p].append(ms)
        if mv is None:
            break
        d = _diag(ags[p])
        log.append({
            'move_number':   mn,
            'player':        p,
            'move_type':     mv[0],
            'col':           mv[1],
            'move_time_ms':  round(ms, 2),
            'phase':         _phase(mn),
            'root_sims':     d.get('root_sims',     None),
            'n_children':    d.get('n_children',    None),
            'best_visit_wr': d.get('best_visit_wr', None),
            'chosen_wr':     d.get('chosen_wr',     None),
            'entropy':       d.get('entropy',       None),
        })
        if mv[0] == 'pop':
            if p == 1: p1p += 1
            else:      p2p += 1
        game.make_move(mv[0], mv[1])
        mn += 1
    w = game.winner
    return {
        'winner':      w,
        'n_moves':     mn,
        'move_log':    log,
        'p1_pops':     p1p,
        'p2_pops':     p2p,
        'p1_pops_win': p1p if w == 1 else 0,
        'p2_pops_win': p2p if w == 2 else 0,
        'p1_times_ms': mts[1],
        'p2_times_ms': mts[2],
    }


def run_one_game(task: dict) -> dict:
    """
    Ponto de entrada do worker. Recebe um dict 'task' com:
      n1, n2     : nomes das estratégias (str)
      cfg1, cfg2 : dicts com keys: strategy, c, k (só Top-K), iterations, time_limit
      gi         : índice do jogo (int)
      half       : n_games // 2  — determina quem joga como P1
      seed       : semente aleatória (int)
    Devolve dict com resultado agregado do jogo.
    """
    random.seed(task['seed'])
    np.random.seed(task['seed'] % (2 ** 31))

    a1 = _make_agent(task['cfg1'])
    a2 = _make_agent(task['cfg2'])

    p1_is_a1 = task['gi'] < task['half']
    res = _play_game(a1, a2) if p1_is_a1 else _play_game(a2, a1)

    if p1_is_a1:
        a1t  = res['p1_times_ms'];  a2t  = res['p2_times_ms']
        a1p  = res['p1_pops'];      a2p  = res['p2_pops']
        a1pw = res['p1_pops_win'];  a2pw = res['p2_pops_win']
        a1m  = sum(1 for e in res['move_log'] if e['player'] == 1)
        a2m  = sum(1 for e in res['move_log'] if e['player'] == 2)
        a1e  = [e['entropy'] for e in res['move_log']
                if e['player'] == 1 and e['entropy'] is not None]
        a2e  = [e['entropy'] for e in res['move_log']
                if e['player'] == 2 and e['entropy'] is not None]
        w    = res['winner']
        out  = 'draw' if w is None else ('a1_win' if w == 1 else 'a2_win')
    else:
        a1t  = res['p2_times_ms'];  a2t  = res['p1_times_ms']
        a1p  = res['p2_pops'];      a2p  = res['p1_pops']
        a1pw = res['p2_pops_win'];  a2pw = res['p1_pops_win']
        a1m  = sum(1 for e in res['move_log'] if e['player'] == 2)
        a2m  = sum(1 for e in res['move_log'] if e['player'] == 1)
        a1e  = [e['entropy'] for e in res['move_log']
                if e['player'] == 2 and e['entropy'] is not None]
        a2e  = [e['entropy'] for e in res['move_log']
                if e['player'] == 1 and e['entropy'] is not None]
        w    = res['winner']
        out  = 'draw' if w is None else ('a1_win' if w == 2 else 'a2_win')

    return {
        'pair':    (task['n1'], task['n2']),
        'gi':      task['gi'],
        'result':  out,
        'n_moves': res['n_moves'],
        'a1t':     a1t[1:],  # trim warm-up (1º movimento)
        'a2t':     a2t[1:],
        'a1p':     a1p,   'a2p':  a2p,
        'a1pw':    a1pw,  'a2pw': a2pw,
        'a1m':     a1m,   'a2m':  a2m,
        'a1e':     a1e,   'a2e':  a2e,
    }
