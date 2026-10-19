#!/usr/bin/env python3
"""
skill_econ.py — modèle économique du PROGRAMME II.

Question : à budget mémoire fixé, vaut-il mieux dépenser un gigaoctet en
POIDS (un plus gros modèle) ou en COMPÉTENCES (une bibliothèque de procédures
exécutables et vérifiées) ?

Ce que fait ce script
---------------------
Il combine trois modèles explicites :

  1. DEMANDE — la fréquence d'usage des compétences suit une loi de Zipf.
     La couverture d'une bibliothèque de N compétences est donc
     H(N)/H(M), où H est le nombre harmonique généralisé et M le nombre
     total de procédures possibles. C'est le modèle le plus défendable
     dont on dispose pour une longue traîne, et il prédit une saturation
     logarithmique — donc des rendements décroissants.

  2. CAPACITÉ — le taux de réussite d'un cortex suit une loi de puissance
     en N (comme Chinchilla), et la couverture apporte un gain fixe par
     requête résolue (la vérification garantit la justesse).

  3. COÛT — une requête résolue par compétence coûte quelques tokens de
     planification + une exécution ; une requête non couverte coûte une
     génération complète. L'énergie est dominée par la plateforme, donc
     proportionnelle au temps, donc au nombre de tokens.

CE QUE CE SCRIPT N'EST PAS
--------------------------
Ce n'est pas une mesure de modèle réel. Les paramètres (exposant de Zipf,
taux de réussite, joules par token) sont des HYPOTHÈSES. Le script sert à
deux choses :
  * trouver les SEUILS à partir desquels la bibliothèque gagne ;
  * montrer quels paramètres il faut mesurer en priorité (E-Ω1, E-Ω5).

Toute sortie doit être citée avec ses hypothèses.

Auteur : KAIROS · 2026-10-19 · stdlib pure
"""

import argparse
import math
import os

# ---------------------------------------------------------------------------
# 1. Modèles
# ---------------------------------------------------------------------------

def harmonic(n: float, gamma: float) -> float:
    """Nombre harmonique généralisé H_n^(gamma).

    Pour gamma == 1 on utilise l'approximation ln(n) + 0.5772 (Euler-Mascheroni),
    exacte à 1 % près dès n = 10. Sinon, intégrale : n^(1-gamma)/(1-gamma).
    """
    if n <= 1:
        return 1.0
    if abs(gamma - 1.0) < 1e-9:
        return math.log(n) + 0.5772156649
    return (n ** (1.0 - gamma)) / (1.0 - gamma)


def coverage(n_skills: float, n_total: float = 1e9, gamma: float = 1.0,
             decompositions: int = 1) -> float:
    """Fraction des requêtes qu'une bibliothèque de n_skills peut servir.

    * Zipf : la probabilité qu'une requête tirée au hasard porte sur l'une des
      n_skills procédures les plus demandées est H(n)/H(M).
    * `decompositions` : une requête peut souvent être résolue par plusieurs
      décompositions indépendantes. S'il en existe k, la probabilité d'échec
      est (1-p)^k, donc la couverture effective monte.
    """
    p = harmonic(n_skills, gamma) / harmonic(n_total, gamma)
    p = min(1.0, max(0.0, p))
    if decompositions > 1:
        p = 1.0 - (1.0 - p) ** decompositions
    return p


def cortex_success(n_params: float, s_max: float = 0.95, alpha: float = 0.217,
                   a: float = 56.4) -> float:
    """Taux de réussite d'un modèle de n_params, calibré sur des ordres de
    grandeur publics (8 B ≈ 0,55 ; 70 B ≈ 0,70 ; 400 B ≈ 0,78 sur un mélange
    de benchmarks). Ce sont des ordres de grandeur, pas une mesure."""
    s = s_max - a * (n_params ** (-alpha))
    return max(0.0, min(s_max, s))


def params_per_gb(bits: float) -> float:
    """Paramètres stockables dans 1 Go à `bits` bits effectifs."""
    return 8.0e9 / bits


# ---------------------------------------------------------------------------
# 2. Coût et capacité
# ---------------------------------------------------------------------------

class Assumptions:
    def __init__(self, e_token=0.5, plan_tokens=10, gen_tokens=300,
                 exec_j=0.2, verify_j=0.5, retrieval_j=0.3,
                 s_skill=0.98, bytes_per_skill=5000, bits=1.7):
        self.e_token = e_token          # J/token du cortex
        self.plan_tokens = plan_tokens  # tokens de planification
        self.gen_tokens = gen_tokens    # tokens d'une génération complète
        self.exec_j = exec_j            # énergie d'exécution d'une skill
        self.verify_j = verify_j        # énergie de vérification
        self.retrieval_j = retrieval_j  # énergie d'un retrieval
        self.s_skill = s_skill          # réussite d'une skill vérifiée
        self.bytes_per_skill = bytes_per_skill
        self.bits = bits


def query_cost(p: float, cortex: float, a: Assumptions) -> float:
    """Joules par requête pour une couverture p et une réussite cortex."""
    plan = a.plan_tokens * a.e_token
    fixed = plan + a.retrieval_j
    covered = a.exec_j + a.verify_j
    uncovered = a.gen_tokens * a.e_token
    return fixed + p * covered + (1.0 - p) * uncovered


def system_success(p: float, cortex: float, a: Assumptions) -> float:
    """Taux de réussite du système complet."""
    return p * a.s_skill + (1.0 - p) * cortex


# ---------------------------------------------------------------------------
# 3. Rendement marginal
# ---------------------------------------------------------------------------

def marginal_weights(success_fn, n_params: float, d_params: float) -> float:
    """Points de réussite gagnés par paramètre ajouté (dérivée numérique)."""
    h = n_params * 1e-3
    return (success_fn(n_params + h) - success_fn(n_params - h)) / (2 * h)


def report(args) -> None:
    a = Assumptions(e_token=args.e_token, bits=args.bits,
                    bytes_per_skill=args.bytes_per_skill)
    print("=" * 78)
    print("  PROGRAMME II — rendement d'un gigaoctet : poids ou compétences ?")
    print("=" * 78)
    print(f"  Hypothèses : {a.e_token} J/token · plan {a.plan_tokens} tokens · "
          f"génération {a.gen_tokens} tokens")
    print(f"               exécution {a.exec_j} J · vérification {a.verify_j} J · "
          f"retrieval {a.retrieval_j} J")
    print(f"               Zipf γ={args.gamma} · {args.decompositions} "
          f"décomposition(s) · {a.bytes_per_skill/1000:.0f} Ko/skill")
    print()

    # --- table : taille de bibliothèque -> couverture -> coût -> réussite ---
    print("BIBLIOTHÈQUE : couverture et coût par requête")
    head = (f"{'skills':>10} {'Go':>7} {'couverture':>11} {'J/requête':>10} "
            f"{'réussite':>9} {'réussite/J':>11}")
    print(head)
    print("-" * len(head))
    cortex = cortex_success(args.cortex_params)
    for n in (1e4, 1e5, 1e6, 1e7, 3e7, 1e8):
        p = coverage(n, args.n_total, args.gamma, args.decompositions)
        gb = n * a.bytes_per_skill / 1e9
        j = query_cost(p, cortex, a)
        s = system_success(p, cortex, a)
        print(f"{n:>10.0f} {gb:>7.2f} {p:>11.1%} {j:>10.1f} "
              f"{s:>9.1%} {s/j:>11.5f}")
    print()

    # --- comparaison des systèmes ---
    print("SYSTÈMES COMPARÉS (mêmes joules, même matériel)")
    print(f"{'système':<44} {'réussite':>9} {'J/requête':>10} {'réussite/J':>11}")

    small = cortex_success(8e9)
    small_j = a.gen_tokens * a.e_token
    print(f"{'cortex 8 B seul (baseline)':<44} {small:>9.1%} {small_j:>10.1f} "
          f"{small/small_j:>11.5f}")

    big = cortex_success(400e9)
    big_j = 400 * 2.0
    print(f"{'gros modèle 400 B (HECATON-I, 2 J/tok)':<44} {big:>9.1%} "
          f"{big_j:>10.1f} {big/big_j:>11.5f}")

    for n in (1e6, 1e7):
        p = coverage(n, args.n_total, args.gamma, args.decompositions)
        j = query_cost(p, cortex, a)
        s = system_success(p, cortex, a)
        label = f"tri-système (cortex 8 B + {n:.0e} skills)"
        print(f"{label:<44} {s:>9.1%} {j:>10.1f} {s/j:>11.5f}")
    print()

    # --- seuil de couverture ---
    base = small / small_j
    print("SEUIL : à partir de quelle couverture le tri-système bat-il le cortex seul ?")
    for p in [i / 100 for i in range(1, 101)]:
        j = query_cost(p, cortex, a)
        s = system_success(p, cortex, a)
        if s / j > base:
            print(f"  couverture minimale = {p:.0%}  "
                  f"(réussite {s:.1%}, {j:.1f} J, {s/j:.5f} réussite/J)")
            break
    else:
        print("  aucune : même à 100 % de couverture, le système ne gagne pas.")
    print()

    # --- rendement marginal : poids vs bibliothèque ---
    print("RENDEMENT MARGINAL D'UN GIGAOCTET (points de réussite par Go)")
    print(f"{'taille du cortex':>18} {'Go':>7} {'poids':>12} {'bibliothèque':>14}")
    print("-" * 55)
    for n_params in (8e9, 70e9, 200e9, 400e9):
        gb_cortex = n_params * a.bits / 8e9
        d_per_param = marginal_weights(cortex_success, n_params, 1.0)
        d_per_gb_weights = d_per_param * params_per_gb(a.bits)

        # rendement de la bibliothèque au point où elle fait ~50 Go
        n_skills = 50e9 / a.bytes_per_skill
        lo = coverage(n_skills * 0.99, args.n_total, args.gamma, args.decompositions)
        hi = coverage(n_skills * 1.01, args.n_total, args.gamma, args.decompositions)
        d_p_per_skill = (hi - lo) / (n_skills * 0.02)
        d_p_per_gb = d_p_per_skill * (1e9 / a.bytes_per_skill)
        d_succ_per_gb = d_p_per_gb * (a.s_skill - cortex)
        print(f"{n_params/1e9:>13.0f} B {gb_cortex:>7.0f} "
              f"{d_per_gb_weights:>12.4f} {d_succ_per_gb:>14.4f}")
    print()
    print("Lecture : 'poids' = points gagnés par Go de poids supplémentaires ;")
    print("'bibliothèque' = points gagnés par Go de compétences, au point 50 Go.")
    print("Si les deux colonnes se croisent, la répartition optimale change.")
    print()
    print("⚠  Toutes ces valeurs sont des HYPOTHÈSES paramétrées, pas des mesures.")
    print("   L'expérience E-Ω5 est précisément celle qui les mesure.")


# ---------------------------------------------------------------------------
# 4. SVG
# ---------------------------------------------------------------------------

def svg(args, path: str) -> None:
    a = Assumptions(e_token=args.e_token, bits=args.bits,
                    bytes_per_skill=args.bytes_per_skill)
    cortex = cortex_success(args.cortex_params)
    W, H = 880, 460
    ml, mr, mt, mb = 80, 190, 60, 64
    pw, ph = W - ml - mr, H - mt - mb

    pts = []
    for i in range(0, 121):
        n = 10 ** (3 + 5 * i / 120.0)     # 1e3 -> 1e8
        p = coverage(n, args.n_total, args.gamma, args.decompositions)
        j = query_cost(p, cortex, a)
        s = system_success(p, cortex, a)
        pts.append((n, p, s / j))

    ymax = max(v for _, _, v in pts) * 1.15
    xmin, xmax = math.log10(pts[0][0]), math.log10(pts[-1][0])

    def sx(n):
        return ml + pw * (math.log10(n) - xmin) / (xmax - xmin)

    def sy(v):
        return mt + ph * (1 - v / ymax)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>',
           f'<text x="24" y="30" font-size="17" font-weight="700">'
           f'Rendement capacité/joule d\'une bibliothèque de compétences</text>',
           f'<text x="24" y="48" font-size="12" fill="#555">'
           f'cortex 8 B · Zipf γ={args.gamma} · {args.decompositions} décomposition(s) '
           f'· hypothèses, pas mesures</text>']

    for i in range(6):
        v = ymax * i / 5
        y = sy(v)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#eee"/>')
        out.append(f'<text x="{ml-8}" y="{y+4:.1f}" font-size="10.5" fill="#666" '
                   f'text-anchor="end">{v:.3f}</text>')
    for e in range(3, 9):
        x = sx(10 ** e)
        out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="#f4f4f4"/>')
        out.append(f'<text x="{x:.1f}" y="{mt+ph+20}" font-size="10.5" fill="#666" '
                   f'text-anchor="middle">1e{e}</text>')

    d = " ".join(("M" if i == 0 else "L") + f"{sx(n):.1f},{sy(v):.1f}"
                 for i, (n, _, v) in enumerate(pts))
    out.append(f'<path d="{d}" fill="none" stroke="#1f4e79" stroke-width="2.4"/>')

    # point particulier : 10 M de skills
    n0 = 1e7
    p0 = coverage(n0, args.n_total, args.gamma, args.decompositions)
    j0 = query_cost(p0, cortex, a)
    s0 = system_success(p0, cortex, a)
    out.append(f'<circle cx="{sx(n0):.1f}" cy="{sy(s0/j0):.1f}" r="5" fill="#c55a11"/>')
    out.append(f'<text x="{sx(n0)+8:.1f}" y="{sy(s0/j0)-8:.1f}" font-size="11" fill="#c55a11">'
               f'10 M skills · couverture {p0:.0%} · {s0/j0:.4f}</text>')

    # référence : cortex seul
    base = cortex_success(args.cortex_params) / (a.gen_tokens * a.e_token)
    yb = sy(base)
    out.append(f'<line x1="{ml}" y1="{yb:.1f}" x2="{ml+pw}" y2="{yb:.1f}" '
               f'stroke="#999" stroke-dasharray="6 4"/>')
    out.append(f'<text x="{ml+6}" y="{yb-6:.1f}" font-size="10.5" fill="#777">'
               f'cortex seul (sans bibliothèque)</text>')

    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>')
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>')
    out.append(f'<text x="{ml+pw/2}" y="{H-18}" font-size="12" fill="#333" '
               f'text-anchor="middle">taille de la bibliothèque (nombre de compétences)</text>')
    out.append(f'<text x="20" y="{mt+ph/2}" font-size="12" fill="#333" '
               f'transform="rotate(-90 20 {mt+ph/2})" text-anchor="middle">'
               f'réussite par joule</text>')
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma", type=float, default=1.0, help="exposant de Zipf")
    ap.add_argument("--n-total", type=float, default=1e9,
                    help="nombre total de procédures possibles")
    ap.add_argument("--decompositions", type=int, default=1,
                    help="nombre de décompositions alternatives par requête")
    ap.add_argument("--cortex-params", type=float, default=8e9)
    ap.add_argument("--e-token", type=float, default=0.5, help="J/token du cortex")
    ap.add_argument("--bits", type=float, default=1.7)
    ap.add_argument("--bytes-per-skill", type=float, default=5000)
    ap.add_argument("--svg", default=None)
    args = ap.parse_args()
    if args.svg:
        os.makedirs(os.path.dirname(args.svg) or ".", exist_ok=True)
        svg(args, args.svg)
        print(f"écrit : {args.svg}")
    else:
        report(args)


if __name__ == "__main__":
    main()
