#!/usr/bin/env python3
"""toy_shared_basis.py — capacité et composition d'une base partagée.

Question : si une compétence est un delta de rang faible dans une base
partagée ``W_k = U diag(c_k) V``, combien de rangs faut-il pour représenter
des compétences arbitraires, et que vaut la composition ``c_A + c_B`` ?

Ce n'est PAS un LLM. C'est un jouet synthétique, volontairement petit
(d=32, r jusqu'à 32), qui mesure trois choses :

1. l'erreur d'approximation d'une matrice dense aléatoire par la famille
   ``U diag(c) V`` en fonction du rang r ;
2. l'erreur quand la compétence est réellement de rang faible dans la base
   (le cas favorable de LoRA), avec du bruit ;
3. l'erreur de composition : ``c_A + c_B`` contre le meilleur ajustement
   direct de ``W_A + W_B``.

Sortie : un tableau texte + un JSON. Aucune dépendance (stdlib).

Usage :
    python3 experiments/mnemosyne/toy_shared_basis.py
    python3 experiments/mnemosyne/toy_shared_basis.py --d 32 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

Matrix = List[List[float]]


def zeros(n: int, m: int) -> Matrix:
    return [[0.0] * m for _ in range(n)]


def transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    n, k, m = len(a), len(b), len(b[0])
    out = zeros(n, m)
    for i in range(n):
        ai = a[i]
        for t in range(k):
            v = ai[t]
            if v == 0.0:
                continue
            bt = b[t]
            for j in range(m):
                out[i][j] += v * bt[j]
    return out


def frob(a: Matrix) -> float:
    return math.sqrt(sum(x * x for row in a for x in row))


def vec(a: Matrix) -> List[float]:
    return [x for row in a for x in row]


def unvec(v: Sequence[float], n: int, m: int) -> Matrix:
    return [list(v[i * m:(i + 1) * m]) for i in range(n)]


def gauss_jordan_solve(a: Matrix, b: List[float]) -> List[float]:
    """Résout a x = b par élimination de Gauss avec pivot partiel."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            raise ValueError("matrice singulière")
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def orthonormal_columns(d: int, r: int, rng: random.Random) -> Matrix:
    """d x r, colonnes orthonormées (Gram-Schmidt)."""
    cols: List[List[float]] = []
    for _ in range(r):
        v = [rng.gauss(0.0, 1.0) for _ in range(d)]
        for u in cols:
            dot = sum(v[i] * u[i] for i in range(d))
            v = [v[i] - dot * u[i] for i in range(d)]
        norm = math.sqrt(sum(x * x for x in v))
        cols.append([x / norm for x in v])
    return [[cols[j][i] for j in range(r)] for i in range(d)]


def orthonormal_rows(r: int, d: int, rng: random.Random) -> Matrix:
    """r x d, lignes orthonormées (Gram-Schmidt)."""
    rows: List[List[float]] = []
    for _ in range(r):
        v = [rng.gauss(0.0, 1.0) for _ in range(d)]
        for u in rows:
            dot = sum(v[i] * u[i] for i in range(d))
            v = [v[i] - dot * u[i] for i in range(d)]
        norm = math.sqrt(sum(x * x for x in v))
        rows.append([x / norm for x in v])
    return rows


def design_matrix(u: Matrix, v: Matrix) -> Matrix:
    """A (d^2 x r) telle que vec(U diag(c) V) = A c."""
    d = len(u)
    r = len(v)
    a = zeros(d * d, r)
    for i in range(d):
        for j in range(d):
            row = i * d + j
            for l in range(r):
                a[row][l] = u[i][l] * v[l][j]
    return a


def fit_coeffs(a: Matrix, target: Matrix, ridge: float = 1e-10) -> List[float]:
    """Moindres carrés ridge : c = (A^T A + λI)^-1 A^T vec(W)."""
    at = transpose(a)
    ata = matmul(at, a)
    atb = matmul(at, [[x] for x in vec(target)])
    r = len(ata)
    for i in range(r):
        ata[i][i] += ridge
    return gauss_jordan_solve(ata, [row[0] for row in atb])


def relative_error(pred: Matrix, target: Matrix) -> float:
    denom = frob(target)
    if denom == 0:
        return 0.0
    return frob([[pred[i][j] - target[i][j] for j in range(len(pred[0]))]
                 for i in range(len(pred))]) / denom


def random_dense(d: int, rng: random.Random) -> Matrix:
    scale = 1.0 / math.sqrt(d)
    return [[rng.gauss(0.0, scale) for _ in range(d)] for _ in range(d)]


def make_in_basis(u: Matrix, v: Matrix, c: Sequence[float]) -> Matrix:
    return matmul(matmul(u, [[c[l] if l == j else 0.0 for j in range(len(c))]
                             for l in range(len(c))]), v)


def experiment(d: int, r_max: int, seeds: int) -> dict:
    ranks = [r for r in (1, 2, 4, 8, 16, 32) if r <= r_max]
    results = {
        "d": d, "seeds": seeds, "ranks": ranks,
        "random_dense": [],
        "in_basis_sigma0": [],
        "in_basis_sigma01": [],
        "own_basis": [],
        "composition": [],
        "storage": [],
    }

    for r in ranks:
        errs_dense, errs_in0, errs_in01, errs_own, errs_comp, errs_comp_direct = [], [], [], [], [], []
        for seed in range(seeds):
            rng = random.Random(1000 * seed + r)
            u = orthonormal_columns(d, r, rng)
            v = orthonormal_rows(r, d, rng)
            a = design_matrix(u, v)

            # 1. matrice dense aléatoire : quelle part est dans la famille ?
            w_dense = random_dense(d, rng)
            c = fit_coeffs(a, w_dense)
            errs_dense.append(relative_error(make_in_basis(u, v, c), w_dense))

            # 2. compétence réellement de rang r dans la base, sans bruit
            c_true = [rng.gauss(0.0, 1.0) for _ in range(r)]
            w_in = make_in_basis(u, v, c_true)
            c_hat = fit_coeffs(a, w_in)
            errs_in0.append(relative_error(make_in_basis(u, v, c_hat), w_in))

            # 3. même compétence + bruit dense 10 %
            noise = random_dense(d, rng)
            w_noisy = [[w_in[i][j] + 0.1 * noise[i][j] for j in range(d)] for i in range(d)]
            c_noisy = fit_coeffs(a, w_noisy)
            errs_in01.append(relative_error(make_in_basis(u, v, c_noisy), w_noisy))

            # 4. compétence de rang r dans une AUTRE base (mésappariement)
            u2 = orthonormal_columns(d, r, rng)
            v2 = orthonormal_rows(r, d, rng)
            c2 = [rng.gauss(0.0, 1.0) for _ in range(r)]
            w_own = make_in_basis(u2, v2, c2)
            c_own = fit_coeffs(a, w_own)
            errs_own.append(relative_error(make_in_basis(u, v, c_own), w_own))

            # 5. composition : deux compétences dans la base + bruit
            ca = [rng.gauss(0.0, 1.0) for _ in range(r)]
            cb = [rng.gauss(0.0, 1.0) for _ in range(r)]
            wa = make_in_basis(u, v, ca)
            wb = make_in_basis(u, v, cb)
            noise2 = random_dense(d, rng)
            w_sum = [[wa[i][j] + wb[i][j] + 0.1 * noise2[i][j] for j in range(d)] for i in range(d)]
            c_sum_direct = fit_coeffs(a, w_sum)
            c_sum_add = [ca[l] + cb[l] for l in range(r)]
            errs_comp.append(relative_error(make_in_basis(u, v, c_sum_add), w_sum))
            errs_comp_direct.append(relative_error(make_in_basis(u, v, c_sum_direct), w_sum))

        def mean(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        def std(xs: List[float]) -> float:
            if len(xs) < 2:
                return 0.0
            m = mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

        results["random_dense"].append({
            "r": r, "mean": mean(errs_dense), "std": std(errs_dense)})
        results["in_basis_sigma0"].append({
            "r": r, "mean": mean(errs_in0), "std": std(errs_in0)})
        results["in_basis_sigma01"].append({
            "r": r, "mean": mean(errs_in01), "std": std(errs_in01)})
        results["own_basis"].append({
            "r": r, "mean": mean(errs_own), "std": std(errs_own)})
        results["composition"].append({
            "r": r, "mean_add": mean(errs_comp), "std_add": std(errs_comp),
            "mean_direct": mean(errs_comp_direct), "std_direct": std(errs_comp_direct)})
        # stockage : U,V partagés + K cartouches de r coefficients,
        # comparé à K matrices denses d x d. K = 20.
        k = 20
        shared = 2 * d * r + k * r
        dense = k * d * d
        results["storage"].append({
            "r": r, "shared_params": shared, "dense_params": dense,
            "ratio": shared / dense})

    return results


def render(res: dict) -> str:
    lines = []
    lines.append("BASE PARTAGÉE — capacité et composition (jouet synthétique)")
    lines.append("=" * 78)
    lines.append(f"d={res['d']} · {res['seeds']} graines · rangs {res['ranks']}")
    lines.append("Erreur relative de Frobenius (moyenne ± écart-type)")
    lines.append("")
    lines.append(f"{'r':>3} {'dense aléatoire':>18} {'dans la base':>16} "
                 f"{'+ bruit 10%':>14} {'autre base':>14}")
    lines.append("-" * 78)
    for i, r in enumerate(res["ranks"]):
        a = res["random_dense"][i]
        b = res["in_basis_sigma0"][i]
        c = res["in_basis_sigma01"][i]
        e = res["own_basis"][i]
        lines.append(f"{r:>3} {a['mean']:>10.3f}±{a['std']:<6.3f} "
                     f"{b['mean']:>10.3f}±{b['std']:<5.3f} "
                     f"{c['mean']:>8.3f}±{c['std']:<5.3f} "
                     f"{e['mean']:>8.3f}±{e['std']:<5.3f}")
    lines.append("")
    lines.append("COMPOSITION : c_A + c_B contre le meilleur ajustement direct")
    lines.append(f"{'r':>3} {'addition':>18} {'ajustement direct':>20} {'écart':>8}")
    lines.append("-" * 78)
    for i, r in enumerate(res["ranks"]):
        c = res["composition"][i]
        gap = c["mean_add"] - c["mean_direct"]
        lines.append(f"{r:>3} {c['mean_add']:>10.3f}±{c['std_add']:<6.3f} "
                     f"{c['mean_direct']:>12.3f}±{c['std_direct']:<6.3f} "
                     f"{gap:>8.4f}")
    lines.append("")
    lines.append("STOCKAGE : 20 compétences, U,V partagés + 20 cartouches vs 20 denses")
    lines.append(f"{'r':>3} {'partagé':>10} {'dense':>10} {'ratio':>8}")
    lines.append("-" * 78)
    for s in res["storage"]:
        lines.append(f"{s['r']:>3} {s['shared_params']:>10} {s['dense_params']:>10} "
                     f"{s['ratio']:>8.4f}")
    lines.append("")
    lines.append("Lecture :")
    lines.append("- 'dense aléatoire' mesure la perte inévitable de la famille : même")
    lines.append("  à r = d, U diag(c) V ne peut pas représenter toutes les matrices")
    lines.append("  (seulement celles de rang ≤ r dans une base fixée).")
    lines.append("- 'dans la base' est le cas favorable de LoRA : la compétence est")
    lines.append("  réellement un delta de rang r dans la base ; l'erreur doit être ~0.")
    lines.append("- 'autre base' mesure le mésappariement : une compétence de rang r")
    lines.append("  dans une autre base n'est pas représentable exactement.")
    lines.append("- 'addition' vs 'direct' mesure l'interférence de composition :")
    lines.append("  si l'écart est petit, c_A + c_B est un bon composeur.")
    lines.append("")
    lines.append("Ce jouet ne dit RIEN de la qualité d'un LLM. Il dit ce que la")
    lines.append("paramétrisation peut représenter, et où sont les limites.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--r-max", type=int, default=32)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--json", type=str, default="results/toy-shared-basis.json")
    p.add_argument("--txt", type=str, default="results/toy-shared-basis.txt")
    args = p.parse_args()

    res = experiment(args.d, args.r_max, args.seeds)
    txt = render(res)
    print(txt)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    Path(args.txt).write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
