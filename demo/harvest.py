#!/usr/bin/env python3
"""
harvest.py — et si on n'écrivait pas les compétences à la main ?

La critique est juste : 10 M de compétences écrites à 5 minutes chacune,
c'est 40 ans-homme. Le programme meurt là.

Ce script teste l'autre voie : **on ne rédige pas les compétences, on les
moissonne**. Le monde contient déjà des millions de fonctions testées par
leurs auteurs (bibliothèque standard, paquets, outils en ligne de commande).
Le travail n'est plus d'écrire la procédure, mais de :

  1. découvrir la fonction et sa signature ;
  2. générer une fiche (nom, signature, docstring, types) ;
  3. générer un test d'appel automatiquement et vérifier qu'elle s'exécute ;
  4. indexer la fiche.

On mesure le débit réel de cette chaîne, puis on projette.

Usage
-----
  python3 demo/harvest.py index
  python3 demo/harvest.py ask "PGCD de 84 et 126"
  python3 demo/harvest.py ask "logarithme base 2 de 1024"
  python3 demo/harvest.py --json

Auteur : KAIROS · 2026-10-19 · stdlib pure.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import binascii
import calendar
import hashlib
import inspect
import json
import math
import statistics
import sys
import textwrap
import time
import unicodedata
import urllib.parse
import zlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

MODULES = [
    math, statistics, base64, binascii, hashlib, json, calendar,
    textwrap, unicodedata, urllib.parse, zlib,
]

# Modules dont l'API est faite de classes : on moissonne leurs méthodes
# publiques plutôt que des fonctions de module.
CLASSES = [calendar.TextCalendar, calendar.HTMLCalendar]


# ---------------------------------------------------------------------------
# 1. Découverte et fiche
# ---------------------------------------------------------------------------

@dataclass
class Card:
    module: str
    name: str
    qualified: str
    signature: str
    doc: str
    arity: int
    tested: bool
    ok: bool
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def public_callables(module) -> List[Tuple[str, Any]]:
    out = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            out.append((name, obj))
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            # on ne garde que les méthodes statiques/instanciables triviales
            for mname, meth in vars(obj).items():
                if mname.startswith("_") or not callable(meth):
                    continue
                if isinstance(meth, (staticmethod, classmethod)):
                    continue
                if inspect.isfunction(meth):
                    out.append((f"{name}.{mname}", meth))
    return out


def _positional(p: inspect.Parameter) -> bool:
    return p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)


def _sample_for(param: inspect.Parameter, module_name: str) -> Any:
    """Valeur d'exemple dérivée du nom et de l'annotation. Jamais inventée
    au hasard : on reste dans les types que la fonction annonce."""
    ann = param.annotation
    if ann is int:
        return 7
    if ann is float:
        return 2.5
    if ann is str:
        return "bonjour"
    if ann is bytes:
        return b"abc"
    if ann is bool:
        return True
    if ann in (list, List):
        return [1, 2, 3]
    if ann in (dict, Dict):
        return {"a": 1, "b": 2}

    n = param.name.lower()
    if n in ("n", "k", "x", "a", "b", "i", "j", "num", "value", "val"):
        return 7
    if n in ("y", "z", "f", "p"):
        return 3
    if n in ("s", "text", "string", "sep", "prefix", "suffix", "word"):
        return "bonjour"
    if n in ("data", "b", "bytes", "payload"):
        return b"abc"
    if n in ("seq", "iterable", "values", "items", "population", "numbers"):
        return [1, 2, 3, 4]
    if n in ("year",):
        return 2024
    if n in ("month",):
        return 2
    if n in ("day",):
        return 15
    if n in ("base",):
        return 2
    if n in ("width",):
        return 40
    return None


def make_card(module, qualified: str, fn) -> Card:
    mod_name = module.__name__
    doc = (inspect.getdoc(fn) or "").strip().split("\n")[0][:160]
    try:
        sig = str(inspect.signature(fn))
        params = list(inspect.signature(fn).parameters.values())
    except (ValueError, TypeError):
        # fonction C sans signature exposée : on essaie 0, 1, 2 puis 3 arguments
        for args in ((), (7,), (7, 2), (7, 2, 3)):
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    fn(*args)
                return Card(mod_name, qualified, qualified, "(?)", doc,
                            len(args), True, True, "")
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"[:120]
        return Card(mod_name, qualified, qualified, "(?)", doc, -1, True, False, last)

    arity = len([p for p in params
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                 and p.default is p.empty])

    # Jeu d'arguments candidats : les noms d'abord, puis des profils simples.
    profiles = [
        [_sample_for(p, mod_name) for p in params if _positional(p)],
        [7 if p.default is p.empty else p.default for p in params if _positional(p)],
        ["bonjour" for p in params if _positional(p)],
        [b"abc" for p in params if _positional(p)],
        [[1, 2, 3] for p in params if _positional(p)],
        [2.5 for p in params if _positional(p)],
    ]

    tested, ok, err = False, False, ""
    for args in profiles:
        if args is None or any(a is None for a in args):
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                fn(*args)
            tested, ok, err = True, True, ""
            break
        except Exception as e:  # noqa: BLE001
            tested, ok, err = True, False, f"{type(e).__name__}: {e}"[:120]
    if not tested:
        err = "aucun jeu d'arguments admissible"
    return Card(mod_name, qualified, qualified, sig, doc, arity, tested, ok, err)


def harvest() -> Tuple[List[Card], Dict[str, Any]]:
    t0 = time.perf_counter()
    cards: List[Card] = []
    seen = set()
    n_candidates = 0
    for module in MODULES:
        try:
            items = public_callables(module)
        except Exception:  # noqa: BLE001
            continue
        for qualified, fn in items:
            n_candidates += 1
            if qualified in seen:
                continue
            seen.add(qualified)
            cards.append(make_card(module, qualified, fn))
    dt = time.perf_counter() - t0

    ok = [c for c in cards if c.ok]
    size = len(json.dumps([c.as_dict() for c in cards], ensure_ascii=False).encode("utf-8"))
    stats = {
        "modules": len(MODULES),
        "candidats": n_candidates,
        "fiches": len(cards),
        "testees_ok": len(ok),
        "echecs": len(cards) - len(ok),
        "secondes": dt,
        "fiches_par_seconde": len(cards) / dt if dt else 0.0,
        "ms_par_fiche": 1000 * dt / len(cards) if cards else 0.0,
        "octets_par_fiche": size / len(cards) if cards else 0.0,
        "projection_10M_heures": (10_000_000 / (len(cards) / dt)) / 3600 if dt and cards else 0.0,
        "index_go_10M": 10_000_000 * (size / len(cards)) / 1e9 if cards else 0.0,
    }
    return cards, stats


# ---------------------------------------------------------------------------
# 2. Routage : le vrai travail qui reste
# ---------------------------------------------------------------------------

# Un routeur lexical suffit pour la démonstration, mais c'est le point clé :
# le travail restant n'est pas d'ÉCRIRE la compétence, c'est de la TROUVER
# et de remplir ses arguments. C'est exactement le rôle du cortex.
SYNONYMES = {
    "pgcd": ["gcd"],
    "ppcm": ["lcm"],
    "logarithme": ["log"],
    "log": ["log"],
    "racine": ["sqrt", "isqrt"],
    "puissance": ["pow"],
    "arrondi": ["round", "ceil", "floor"],
    "moyenne": ["mean"],
    "mediane": ["median"],
    "ecart-type": ["pstdev", "stdev"],
    "base64": ["b64encode", "encodebytes"],
    "sha256": ["sha256"],
    "md5": ["md5"],
    "compression": ["compress"],
    "jours": ["monthrange", "weekday", "day_name"],
    "mois": ["monthrange", "month_name"],
}


def _numbers(q: str) -> List[int]:
    import re
    return [int(x) for x in re.findall(r"-?\d+", q)]


def _texte(q: str) -> Optional[str]:
    import re
    m = re.search(r'["«“](.+?)["»”]', q)
    if m:
        return m.group(1)
    m = re.search(r"\bde\s+([a-zA-ZÀ-ÿ]{2,})\b", q)
    return m.group(1) if m else None


def route_and_call(cards: List[Card], q: str) -> Tuple[Optional[str], str, str]:
    """Retourne (résultat, fonction utilisée, preuve) ou (None, '', raison)."""
    import re
    n = q.lower()
    nums = _numbers(q)
    target = None
    by_name = {}
    for c in cards:
        if c.ok:
            by_name.setdefault(c.name.split(".")[-1], c)
    for word, fns in SYNONYMES.items():
        if word in n:
            # priorité à l'ordre de la liste de synonymes, pas à l'ordre du module
            for fname in fns:
                if fname in by_name:
                    target = by_name[fname]
                    break
            if target:
                break

    if target is None:
        return None, "", "aucune fonction moissonnée ne correspond"

    name = target.name.split(".")[-1]
    qualified = target.qualified
    ns = {}
    for module in MODULES:
        if module.__name__ == target.module:
            ns = vars(module)
            break
    fn = ns.get(name)
    if fn is None:
        return None, "", f"fonction {qualified} introuvable"

    try:
        if name == "gcd" and len(nums) >= 2:
            r = fn(nums[0], nums[1])
            return str(r), qualified, f"{qualified}({nums[0]}, {nums[1]}) = {r}"
        if name == "lcm" and len(nums) >= 2:
            r = fn(nums[0], nums[1])
            return str(r), qualified, f"{qualified}({nums[0]}, {nums[1]}) = {r}"
        if name == "log" and len(nums) >= 2:
            # « logarithme base B de X » -> log(X, B)
            m = re.search(r"base\s*(\d+)\s*(?:de|d'|:)?\s*(\d+)", n)
            if m:
                b, x = int(m.group(1)), int(m.group(2))
            else:
                x, b = nums[0], nums[1]
            r = fn(x, b)
            return (("%.6f" % r).rstrip("0").rstrip("."), qualified,
                    f"{qualified}({x}, {b}) = {r}")
        if name in ("sqrt", "isqrt") and nums:
            r = fn(nums[0])
            return str(r), qualified, f"{qualified}({nums[0]}) = {r}"
        if name == "pow" and len(nums) >= 2:
            r = fn(nums[0], nums[1])
            return str(r), qualified, f"{qualified}({nums[0]}, {nums[1]}) = {r}"
        if name == "monthrange":
            MOIS = {"janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
                    "juin": 6, "juillet": 7, "aout": 8, "septembre": 9,
                    "octobre": 10, "novembre": 11, "decembre": 12}
            nq = unicodedata.normalize("NFD", n)
            nq = "".join(c for c in nq if unicodedata.category(c) != "Mn")
            month = next((v for k, v in MOIS.items() if k in nq), None)
            year = next((x for x in nums if 1000 <= x <= 3000), None)
            if month and year:
                _, ndays = fn(year, month)
                return str(ndays), qualified, f"{qualified}({year}, {month}) → {ndays} jours"
        if name in ("sha256", "md5"):
            txt = _texte(q) or "abc"
            h = fn(txt.encode("utf-8")).hexdigest()
            return h, qualified, f"{qualified}({txt!r}.encode()).hexdigest()"
        if name in ("b64encode", "encodebytes"):
            txt = _texte(q) or "bonjour"
            r = fn(txt.encode("utf-8")).decode("ascii")
            return r, qualified, f"{qualified}({txt!r}.encode())"
        if name == "compress" and nums:
            r = len(fn(b"x" * nums[0]))
            return f"{r} octets", qualified, f"len({qualified}(b'x'*{nums[0]}))"
        if name == "mean":
            vals = nums or [1, 2, 3]
            r = fn(vals)
            return str(r), qualified, f"{qualified}({vals}) = {r}"
        if name == "median":
            vals = nums or [1, 2, 3]
            r = fn(vals)
            return str(r), qualified, f"{qualified}({vals}) = {r}"
    except Exception as e:  # noqa: BLE001
        return None, "", f"{qualified} a échoué : {e}"
    return None, "", "arguments non déductibles de la question"


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------

def fmt_stats(st: Dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        "  MOISSONNAGE DE COMPÉTENCES DANS LA BIBLIOTHÈQUE STANDARD",
        "=" * 72,
        f"  modules explorés ............ {st['modules']}",
        f"  fonctions publiques vues .... {st['candidats']}",
        f"  fiches produites ............ {st['fiches']}",
        f"  fiches testées avec succès .. {st['testees_ok']}",
        f"  échecs (signature/args) ..... {st['echecs']}",
        "",
        f"  temps total ................. {st['secondes'] * 1000:.0f} ms",
        f"  débit ....................... {st['fiches_par_seconde']:.0f} fiches/s "
        f"({st['ms_par_fiche']:.2f} ms par fiche)",
        f"  taille d'une fiche .......... {st['octets_par_fiche']:.0f} octets",
        "",
        f"  PROJECTION à 10 M de fiches : {st['projection_10M_heures']:.2f} heures "
        f"de moissonnage, {st['index_go_10M']:.1f} Go d'index.",
        "",
        "  À comparer : 10 M de compétences écrites à la main à 5 min chacune",
        "  = 833 000 heures = 40 années-personnes.",
        "=" * 72,
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="moissonnage de compétences")
    ap.add_argument("cmd", choices=["index", "ask"])
    ap.add_argument("question", nargs="*")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="afficher N fiches")
    args = ap.parse_args()

    cards, st = harvest()
    if args.cmd == "index":
        print(fmt_stats(st))
        if args.show:
            print("\n  Exemples de fiches :")
            for c in [c for c in cards if c.ok][:args.show]:
                print(f"    {c.qualified:38} {c.signature:40} {c.doc[:60]}")
        if args.json:
            print(json.dumps(st, indent=2, ensure_ascii=False))
        return 0

    q = " ".join(args.question)
    if not q:
        ap.error("poser une question")
    t0 = time.perf_counter()
    res, fn_name, proof = route_and_call(cards, q)
    dt = (time.perf_counter() - t0) * 1000
    if res is None:
        print(f"Je ne sais pas : {proof}")
        return 1
    print(res)
    print(f"  [moissonnée · {fn_name} · {dt:.2f} ms]")
    print(f"  preuve : {proof}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
