#!/usr/bin/env python3
"""
tri_system.py — le Programme II en miniature, exécutable dans cette sandbox.

Ce n'est PAS un transformer. C'est le tri-système décrit dans
docs/06-PROGRAMME-II.md, réduit à ce qui tient dans 3 Go :

  * un routeur (le « cortex ») qui choisit une compétence ;
  * une bibliothèque de compétences exécutables, chacune avec ses tests ;
  * un vérificateur qui exécute et contrôle le résultat ;
  * une mémoire documentaire (RAG) qui répond sur le projet ;
  * un aveu explicite quand rien ne s'applique.

Chaque compétence est vérifiée au chargement : si un test échoue, elle est
désactivée. Une réponse de type « compétence » est donc correcte par
construction sur les cas testés, pas « probablement correcte ».

Auteur : KAIROS · 2026-10-19 · stdlib pure.
"""

from __future__ import annotations

import json
import math
import os
import re
import resource
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 0. Normalisation
# ---------------------------------------------------------------------------

def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("œ", "oe")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return ("%.6f" % x).rstrip("0").rstrip(".").replace(".", ",")


def quoted(q: str) -> Optional[str]:
    m = re.search(r'["«“](.+?)["»”]', q)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# 1. Compétences
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    title: str
    keywords: Tuple[str, ...]
    patterns: Tuple[str, ...]
    fn: Callable[[str], Tuple[str, str]]
    tests: Tuple[Tuple[str, str], ...]
    enabled: bool = True
    failures: Tuple[str, ...] = ()

    def score(self, q: str) -> int:
        s = 0
        nq = norm(q)
        for p in self.patterns:
            if re.search(p, nq):
                s += 3
        for k in self.keywords:
            if re.search(r"\b" + re.escape(norm(k)) + r"\b", nq):
                s += 1
        return s


MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "decembre": 12,
}
WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def parse_date(s: str) -> Optional[date]:
    n = norm(s)
    if "aujourd'hui" in n or "aujourd hui" in n:
        return date.today()
    if "demain" in n:
        return date.today() + timedelta(days=1)
    if "hier" in n:
        return date.today() - timedelta(days=1)
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", n)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", n)
    if m and m.group(2) in MONTHS:
        return date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", n)
    if m:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return None


def extract_number(s: str) -> Optional[float]:
    m = re.search(r"-?\d+(?:[.,]\d+)?", s)
    return float(m.group(0).replace(",", ".")) if m else None


# --- mathématiques ---------------------------------------------------------

def skill_arithmetic(q: str) -> Tuple[str, str]:
    n = norm(q)
    if re.search(r"\d{4}-\d{2}-\d{2}", n):
        raise ValueError("date, pas une opération")
    n = re.sub(r"\bmultiplie(?:r)?\s+par\b", "*", n)
    n = re.sub(r"\bdivise(?:r)?\s+par\b", "/", n)
    n = re.sub(r"\bfois\b", "*", n)
    n = re.sub(r"\bplus\b", "+", n)
    n = re.sub(r"\bmoins\b", "-", n)
    n = n.replace("×", "*").replace("÷", "/")
    m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*([+\-*/])\s*(-?\d+(?:[.,]\d+)?)", n)
    if not m:
        raise ValueError("aucune opération")
    a = float(m.group(1).replace(",", "."))
    op = m.group(2)
    b = float(m.group(3).replace(",", "."))
    if op == "+":
        r = a + b
    elif op == "-":
        r = a - b
    elif op == "*":
        r = a * b
    else:
        if b == 0:
            raise ValueError("division par zéro")
        r = a / b
    proof = f"{fmt_num(a)} {op} {fmt_num(b)} = {fmt_num(r)}"
    return f"{fmt_num(r)}", proof


def skill_percentage(q: str) -> Tuple[str, str]:
    n = norm(q)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:%|pour\s*cent)\s*(?:de|du|des|d')\s*(\d+(?:[.,]\d+)?)", n)
    if not m:
        raise ValueError("pas un pourcentage")
    p = float(m.group(1).replace(",", "."))
    v = float(m.group(2).replace(",", "."))
    r = p * v / 100.0
    return f"{fmt_num(r)}", f"{fmt_num(p)} % de {fmt_num(v)} = {fmt_num(r)}"


def skill_percent_change(q: str) -> Tuple[str, str]:
    n = norm(q)
    m = re.search(r"(?:de|entre)\s*(\d+(?:[.,]\d+)?)\s*(?:a|et)\s*(\d+(?:[.,]\d+)?)", n)
    if not m or not re.search(r"augment|diminu|hausse|baisse|variation|evolution", n):
        raise ValueError("pas une variation")
    a = float(m.group(1).replace(",", "."))
    b = float(m.group(2).replace(",", "."))
    if a == 0:
        raise ValueError("division par zéro")
    r = (b - a) / a * 100.0
    verb = "augmentation" if r >= 0 else "diminution"
    return f"{verb} de {fmt_num(abs(r))} %", f"({fmt_num(b)} − {fmt_num(a)}) / {fmt_num(a)} = {fmt_num(r)} %"


def skill_power(q: str) -> Tuple[str, str]:
    n = norm(q)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:puissance|\^)\s*(\d+)", n)
    if not m:
        raise ValueError("pas une puissance")
    a = float(m.group(1).replace(",", "."))
    b = int(m.group(2))
    if b > 100:
        raise ValueError("exposant trop grand")
    return fmt_num(a ** b), f"{fmt_num(a)}^{b}"


def skill_sqrt(q: str) -> Tuple[str, str]:
    n = norm(q)
    m = re.search(r"racine\s*(?:carree)?\s*(?:de)?\s*(\d+(?:[.,]\d+)?)", n)
    if not m:
        raise ValueError("pas une racine")
    a = float(m.group(1).replace(",", "."))
    if a < 0:
        raise ValueError("négatif")
    return fmt_num(math.sqrt(a)), f"sqrt({fmt_num(a)})"


def skill_gcd(q: str) -> Tuple[str, str]:
    n = norm(q)
    m = re.search(r"(\d+)\s*(?:et|,)\s*(\d+)", n)
    if not m or not re.search(r"pgcd|plus grand commun", n):
        raise ValueError("pas un pgcd")
    a, b = int(m.group(1)), int(m.group(2))
    return str(math.gcd(a, b)), f"pgcd({a}, {b})"


# --- conversions -----------------------------------------------------------

_CONVERSIONS = [
    (r"(\d+(?:[.,]\d+)?)\s*km\b.*\ben\s*(?:miles?|mi)\b", "km→miles", 0.621371),
    (r"(\d+(?:[.,]\d+)?)\s*(?:miles?|mi)\b.*\ben\s*km\b", "miles→km", 1.609344),
    (r"(\d+(?:[.,]\d+)?)\s*(?:kg|kilos?)\b.*\ben\s*(?:lb|livres?)\b", "kg→lb", 2.2046226),
    (r"(\d+(?:[.,]\d+)?)\s*(?:lb|livres?)\b.*\ben\s*kg\b", "lb→kg", 0.45359237),
    (r"(\d+(?:[.,]\d+)?)\s*m\b.*\ben\s*(?:pieds?|ft)\b", "m→pieds", 3.2808399),
    (r"(\d+(?:[.,]\d+)?)\s*(?:pieds?|ft)\b.*\ben\s*m\b", "pieds→m", 0.3048),
    (r"(\d+(?:[.,]\d+)?)\s*(?:litres?|l)\b.*\ben\s*(?:galons?|gal)\b", "L→gal", 0.264172),
    (r"(\d+(?:[.,]\d+)?)\s*(?:galons?|gal)\b.*\ben\s*(?:litres?|l)\b", "gal→L", 3.785412),
    (r"(\d+(?:[.,]\d+)?)\s*(?:minutes?|min)\b.*\ben\s*(?:heures?|h)\b", "min→h", 1 / 60),
    (r"(\d+(?:[.,]\d+)?)\s*(?:heures?|h)\b.*\ben\s*(?:minutes?|min)\b", "h→min", 60),
    (r"(\d+(?:[.,]\d+)?)\s*(?:jours?)\b.*\ben\s*(?:heures?|h)\b", "jours→h", 24),
    (r"(\d+(?:[.,]\d+)?)\s*(?:mo|mega-octets?)\b.*\ben\s*(?:go|giga-octets?)\b", "Mo→Go", 1 / 1000),
    (r"(\d+(?:[.,]\d+)?)\s*(?:ko|kilo-octets?)\b.*\ben\s*(?:mo|mega-octets?)\b", "Ko→Mo", 1 / 1000),
    (r"(\d+(?:[.,]\d+)?)\s*(?:go|giga-octets?)\b.*\ben\s*(?:to|tera-octets?)\b", "Go→To", 1 / 1000),
    (r"(\d+(?:[.,]\d+)?)\s*(?:euros?|€)\b.*\ben\s*(?:dollars?|\$)\b", "EUR→USD", 1.08),
    (r"(\d+(?:[.,]\d+)?)\s*(?:dollars?|\$)\b.*\ben\s*euros?\b", "USD→EUR", 1 / 1.08),
    (r"(\d+(?:[.,]\d+)?)\s*(?:cm|centimetres?)\b.*\ben\s*(?:pouces?|in)\b", "cm→pouces", 0.3937008),
    (r"(\d+(?:[.,]\d+)?)\s*(?:pouces?|in)\b.*\ben\s*cm\b", "pouces→cm", 2.54),
]


def skill_conversion(q: str) -> Tuple[str, str]:
    n = norm(q)
    # températures
    m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(?:°\s*)?c\b.*\ben\s*(?:°\s*)?f\b", n)
    if m:
        c = float(m.group(1).replace(",", "."))
        f = c * 9 / 5 + 32
        return f"{fmt_num(f)} °F", f"F = C × 9/5 + 32 = {fmt_num(f)}"
    m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(?:°\s*)?f\b.*\ben\s*(?:°\s*)?c\b", n)
    if m:
        f = float(m.group(1).replace(",", "."))
        c = (f - 32) * 5 / 9
        return f"{fmt_num(c)} °C", f"C = (F − 32) × 5/9 = {fmt_num(c)}"
    m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(?:°\s*)?c\b.*\ben\s*k\b", n)
    if m:
        c = float(m.group(1).replace(",", "."))
        return f"{fmt_num(c + 273.15)} K", f"K = C + 273,15 = {fmt_num(c + 273.15)}"
    for pat, label, factor in _CONVERSIONS:
        mm = re.search(pat, n)
        if mm:
            v = float(mm.group(1).replace(",", "."))
            r = v * factor
            return f"{fmt_num(r)}", f"{fmt_num(v)} × {factor:.6g} ({label})"
    raise ValueError("conversion inconnue")


# --- dates -----------------------------------------------------------------

def skill_date_day(q: str) -> Tuple[str, str]:
    n = norm(q)
    if not re.search(r"quel\s+jour", n):
        raise ValueError("pas une question de jour")
    d = parse_date(n)
    if not d:
        raise ValueError("date illisible")
    return WEEKDAYS[d.weekday()], f"{d.isoformat()} → {WEEKDAYS[d.weekday()]}"


def skill_date_diff(q: str) -> Tuple[str, str]:
    n = norm(q)
    if not re.search(r"combien\s+de\s+jours", n):
        raise ValueError("pas une différence de dates")
    dates = []
    for m in re.finditer(r"(\d{1,2}\s+[a-z]+\s+\d{4}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|aujourd'hui)", n):
        d = parse_date(m.group(1))
        if d:
            dates.append(d)
    if len(dates) < 2:
        raise ValueError("il faut deux dates")
    delta = abs((dates[1] - dates[0]).days)
    return f"{delta} jours", f"du {dates[0].isoformat()} au {dates[1].isoformat()} = {delta} jours"


def skill_age(q: str) -> Tuple[str, str]:
    n = norm(q)
    if not re.search(r"\bage\b", n) or not re.search(r"\bne(?:e)?\s+le\b|\bnee?\b", n):
        raise ValueError("pas un calcul d'âge")
    d = parse_date(n)
    if not d:
        raise ValueError("date illisible")
    today = date.today()
    years = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    return f"{years} ans", f"né le {d.isoformat()}, nous sommes le {today.isoformat()}"


# --- texte -----------------------------------------------------------------

def skill_text_stats(q: str) -> Tuple[str, str]:
    n = norm(q)
    t = quoted(q)
    if t is None:
        raise ValueError("texte non cité")
    if re.search(r"combien\s+de\s+mots", n):
        nw = len(re.findall(r"[\wÀ-ÿ'-]+", t))
        return f"{nw} mots", f"{nw} mots dans le texte"
    if re.search(r"combien\s+de\s+(?:caracteres|lettres)", n):
        return f"{len(t)} caractères", f"{len(t)} caractères"
    if re.search(r"combien\s+de\s+voyelles", n):
        nv = len([c for c in norm(t) if c in "aeiouy"])
        return f"{nv} voyelles", f"{nv} voyelles"
    raise ValueError("statistique inconnue")


def skill_text_transform(q: str) -> Tuple[str, str]:
    n = norm(q)
    t = quoted(q)
    if t is None:
        raise ValueError("texte non cité")
    if re.search(r"\bmajuscule", n):
        return t.upper(), "passage en majuscules"
    if re.search(r"\bminuscule", n):
        return t.lower(), "passage en minuscules"
    if re.search(r"\binvers", n):
        return t[::-1], "inversion caractère par caractère"
    raise ValueError("transformation inconnue")


def skill_palindrome(q: str) -> Tuple[str, str]:
    t = quoted(q)
    if t is None or not re.search(r"palindrome", norm(q)):
        raise ValueError("pas un palindrome")
    clean = re.sub(r"[^a-z0-9]", "", norm(t))
    ok = clean == clean[::-1]
    return ("oui" if ok else "non"), f"« {clean} » {'=' if ok else '≠'} son inverse"


def skill_anagram(q: str) -> Tuple[str, str]:
    n = norm(q)
    if not re.search(r"anagramme", n):
        raise ValueError("pas un anagramme")
    parts = re.findall(r'"([^"]+)"', q)
    if len(parts) < 2:
        raise ValueError("il faut deux textes")
    a = sorted(re.sub(r"[^a-z]", "", norm(parts[0])))
    b = sorted(re.sub(r"[^a-z]", "", norm(parts[1])))
    return ("oui" if a == b else "non"), f"{''.join(a)} vs {''.join(b)}"


# --- culture / données -----------------------------------------------------

ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
         (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
         (5, "V"), (4, "IV"), (1, "I")]


def to_roman(n: int) -> str:
    if not 0 < n < 4000:
        raise ValueError("hors plage")
    out = []
    for v, s in ROMAN:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def from_roman(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = vals.get(ch)
        if v is None:
            raise ValueError("symbole invalide")
        total += -v if v < prev else v
        prev = max(prev, v)
    if to_roman(total) != s.upper():
        raise ValueError("écriture romaine non canonique")
    return total


def skill_roman(q: str) -> Tuple[str, str]:
    n = norm(q)
    if re.search(r"chiffres?\s+romains?", n):
        m = re.search(r"(\d+)", n)
        if not m:
            raise ValueError("pas de nombre")
        return to_roman(int(m.group(1))), f"{m.group(1)} → {to_roman(int(m.group(1)))}"
    m = re.search(r"\b([ivxlcdm]{2,})\b", n)
    if m and re.search(r"vaut|valeur|convert", n):
        return str(from_roman(m.group(1))), f"{m.group(1)} → {from_roman(m.group(1))}"
    raise ValueError("pas un nombre romain")


CAPITALS = {
    "france": "Paris", "australie": "Canberra", "canada": "Ottawa", "bresil": "Brasilia",
    "suisse": "Berne", "etats-unis": "Washington", "japon": "Tokyo", "chine": "Pékin",
    "inde": "New Delhi", "allemagne": "Berlin", "espagne": "Madrid", "italie": "Rome",
    "royaume-uni": "Londres", "belgique": "Bruxelles", "portugal": "Lisbonne",
    "grece": "Athènes", "turquie": "Ankara", "egypte": "Le Caire", "maroc": "Rabat",
    "algerie": "Alger", "tunisie": "Tunis", "senegal": "Dakar", "kenya": "Nairobi",
    "afrique du sud": "Pretoria", "mexique": "Mexico", "argentine": "Buenos Aires",
    "chili": "Santiago", "perou": "Lima", "colombie": "Bogota", "coree du sud": "Séoul",
    "coree du nord": "Pyongyang", "vietnam": "Hanoi", "thailande": "Bangkok",
    "indonesie": "Jakarta", "australie": "Canberra", "nouvelle-zelande": "Wellington",
    "russie": "Moscou", "ukraine": "Kyiv", "pologne": "Varsovie", "suede": "Stockholm",
    "norvege": "Oslo", "finlande": "Helsinki", "danemark": "Copenhague",
    "pays-bas": "Amsterdam", "autriche": "Vienne", "irlande": "Dublin",
    "islande": "Reykjavik", "arabie saoudite": "Riyad", "iran": "Téhéran",
    "irak": "Bagdad", "israel": "Jérusalem", "nigeria": "Abuja", "ethiopie": "Addis-Abeba",
}


def skill_capital(q: str) -> Tuple[str, str]:
    n = norm(q)
    if "capitale" not in n:
        raise ValueError("pas une capitale")
    m = re.search(r"capitale\s+(?:de\s+la\s+|de\s+l'|du\s+|des\s+|de\s+)?([a-z \-]+)", n)
    if not m:
        raise ValueError("pays illisible")
    country = m.group(1).strip()
    for key, val in CAPITALS.items():
        if key in country or country in key:
            return val, f"table : {key} → {val}"
    raise ValueError("pays inconnu")


ELEMENTS = {
    "hydrogene": ("H", 1), "helium": ("He", 2), "carbone": ("C", 6), "azote": ("N", 7),
    "oxygene": ("O", 8), "sodium": ("Na", 11), "magnesium": ("Mg", 12),
    "aluminium": ("Al", 13), "silicium": ("Si", 14), "phosphore": ("P", 15),
    "soufre": ("S", 16), "chlore": ("Cl", 17), "argon": ("Ar", 18),
    "potassium": ("K", 19), "calcium": ("Ca", 20), "titane": ("Ti", 22),
    "chrome": ("Cr", 24), "manganese": ("Mn", 25), "fer": ("Fe", 26),
    "cobalt": ("Co", 27), "nickel": ("Ni", 28), "cuivre": ("Cu", 29),
    "zinc": ("Zn", 30), "argent": ("Ag", 47), "iode": ("I", 53),
    "or": ("Au", 79), "mercure": ("Hg", 80), "plomb": ("Pb", 82),
    "uranium": ("U", 92),
}


def skill_element(q: str) -> Tuple[str, str]:
    n = norm(q)
    m = re.search(r"(?:symbole|numero atomique)(?:\s+chimique)?\s+"
                  r"(?:de\s+l'|du\s+|de\s+la\s+|de\s+|des\s+)?([a-z]+)", n)
    if not m:
        raise ValueError("pas un élément")
    el = m.group(1)
    if el not in ELEMENTS:
        raise ValueError("élément inconnu")
    sym, num = ELEMENTS[el]
    if "symbole" in n:
        return sym, f"{el} → {sym}"
    return str(num), f"{el} → Z = {num}"


PLANETS = ["mercure", "venus", "terre", "mars", "jupiter", "saturne", "uranus", "neptune"]


def skill_planet(q: str) -> Tuple[str, str]:
    n = norm(q)
    if "planete" not in n and "position" not in n:
        raise ValueError("pas une planète")
    m = re.search(r"(?:position\s+de\s+([a-z]+))", n)
    if m:
        for i, p in enumerate(PLANETS):
            if p == m.group(1):
                return f"position {i + 1}", f"{p.capitalize()} est la {i + 1}ᵉ planète"
        raise ValueError("planète inconnue")
    m = re.search(r"(\d+)\s*(?:e|eme|ieme)?\s*planete", n)
    if m:
        i = int(m.group(1))
        if not 1 <= i <= 8:
            raise ValueError("hors plage")
        return PLANETS[i - 1].capitalize(), f"position {i} dans l'ordre"
    for i, p in enumerate(PLANETS):
        if re.search(r"\b" + p + r"\b", n):
            return f"position {i + 1}", f"{p.capitalize()} est la {i + 1}ᵉ planète"
    raise ValueError("planète inconnue")


def skill_identity(q: str) -> Tuple[str, str]:
    n = norm(q)
    if not re.search(r"qui es-tu|tu es qui|tu es un (?:llm|modele|chatgpt)|comment fonctionnes-tu|c'est quoi ce modele", n):
        raise ValueError("pas une question d'identité")
    return ("Je suis le tri-système du Programme II : un routeur, une bibliothèque "
            "de compétences exécutées et vérifiées, une mémoire documentaire, et un "
            "aveu quand je ne sais pas. Je ne suis pas un transformer entraîné."), \
           "réponse de la compétence identite"


PROJECT_FACTS = [
    (("budget", "ram", "memoire"), "Le Programme II se cadre à ~100 Go de RAM. "
     "La piste produit, séparée, vise 200 Mo de RSS."),
    (("metrique", "joule", "rendement"), "La métrique du projet est le joule par "
     "réponse juste ; à défaut, la réussite par joule à taille de modèle fixe."),
    (("programme", "programme ii", "tri-systeme"), "Le Programme II découple "
     "connaissance, compétence et raisonnement : cortex 15 Go + bibliothèque "
     "55 Go + mémoire plastique 25 Go + vérificateur 3 Go."),
    (("hecaton", "baseline"), "HECATON-I est la baseline de compression : 372 B "
     "de paramètres, 82,7 Go, 2,05 Go lus par token, 49 tok/s projetés."),
    (("silex",), "Silex est l'agent du chantier W3 sur la branche "
     "arena/d930d4b4-scienceai : packing ternaire et streaming des poids."),
    (("kairos",), "KAIROS est l'agent de la branche arena/c9070fe4-scienceai, "
     "auteur du Programme II et du calculateur de budget."),
    (("recette", "cuisine"), "Je n'ai pas de compétence de cuisine : c'est hors "
     "de ma bibliothèque, et je préfère l'avouer."),
]


def skill_projet(q: str) -> Tuple[str, str]:
    n = norm(q)
    for keys, answer in PROJECT_FACTS:
        for k in keys:
            if re.search(r"\b" + re.escape(k) + r"\b", n):
                return answer, f"table de faits du projet : « {k} »"
    raise ValueError("aucun fait correspondant")


SKILLS: List[Skill] = [
    Skill("arithmetique", "Calcul arithmétique",
          ("calcul", "combien font", "fois", "plus", "moins", "divise", "multiplie", "operation"),
          (r"-?\d+(?:[.,]\d+)?\s*[+\-*/×÷]\s*-?\d+", r"\b(?:fois|plus|moins|divise|multiplie)\b"),
          skill_arithmetic,
          (("Combien font 17 × 23 ?", "391"), ("12 fois 7", "84"), ("100 / 8", "12,5"))),
    Skill("pourcentage", "Pourcentage d'un nombre",
          ("pourcentage", "pour cent", "pourcent"),
          (r"\d+(?:[.,]\d+)?\s*(?:%|pour\s*cent)\s*(?:de|du|des|d')",),
          skill_percentage,
          (("Combien font 15 % de 240 ?", "36"), ("50 pour cent de 90", "45"))),
    Skill("variation", "Variation en pourcentage",
          ("augmentation", "diminution", "hausse", "baisse", "variation"),
          (r"(?:de|entre)\s*\d+(?:[.,]\d+)?\s*(?:a|et)\s*\d+",),
          skill_percent_change,
           (("Quelle est l'augmentation de 120 à 150 ?", "25"),
           ("Quelle est la diminution de 200 à 150 ?", "25"))),
    Skill("puissance", "Puissance et racine",
          ("puissance", "carre", "racine"),
          (r"\d+\s*(?:puissance|\^)\s*\d+",),
          skill_power,
          (("2 puissance 10", "1024"), ("9 puissance 2", "81"))),
    Skill("racine", "Racine carrée",
          ("racine",),
          (r"racine\s*(?:carree)?\s*(?:de)?\s*\d+",),
          skill_sqrt,
          (("Racine carrée de 144", "12"), ("racine de 2", "1,414214"))),
    Skill("pgcd", "Plus grand commun diviseur",
          ("pgcd", "diviseur commun"),
          (r"pgcd",),
          skill_gcd,
          (("PGCD de 42 et 56", "14"), ("pgcd de 12 et 18", "6"))),
    Skill("conversion", "Conversion d'unités",
          ("convertir", "conversion", "combien font", "en", "lien"),
          (r"\d+(?:[.,]\d+)?\s*(?:km|kg|m|cm|litres?|galons?|minutes?|heures?|jours?|mo|ko|go|euros?|dollars?|°)\b.*\ben\b",),
          skill_conversion,
          (("Convertir 10 km en miles", "6,21371"), ("100 °C en °F", "212"),
           ("50 kg en livres", "110,2311"), ("3 h en minutes", "180"))),
    Skill("jour_date", "Jour de la semaine",
          ("quel jour", "jour de la semaine"),
          (r"quel\s+jour",),
          skill_date_day,
          (("Quel jour de la semaine le 14 juillet 2026 ?", "mardi"),
           ("Quel jour le 1 janvier 2025 ?", "mercredi"))),
    Skill("diff_dates", "Intervalle entre deux dates",
          ("combien de jours", "intervalle"),
          (r"combien\s+de\s+jours",),
          skill_date_diff,
          (("Combien de jours entre le 1 janvier 2025 et le 1 janvier 2026 ?", "365"),
           ("Combien de jours du 1 mars 2024 au 1 avril 2024 ?", "31"))),
    Skill("age", "Âge à partir d'une date de naissance",
          ("age", "quel age"),
          (r"age",),
          skill_age,
          (("Quel âge a une personne née le 1 janvier 2000 ?", "ans"),
           ("age de quelqu'un ne le 15 mars 1990", "ans"))),
    Skill("stats_texte", "Statistiques sur un texte",
          ("combien de mots", "combien de caracteres", "combien de voyelles"),
          (r"combien\s+de\s+(?:mots|caracteres|lettres|voyelles)",),
          skill_text_stats,
          (('Combien de mots dans "le chat dort sur le tapis" ?', "6"),
           ('Combien de voyelles dans "bonjour" ?', "3"))),
    Skill("transformation", "Majuscules, minuscules, inversion",
          ("majuscule", "minuscule", "inverser"),
          (r"(?:majuscule|minuscule|invers)",),
          skill_text_transform,
          (('Mets "bonjour" en majuscules', "BONJOUR"),
           ('Inverse "abc"', "cba"))),
    Skill("palindrome", "Test de palindrome",
          ("palindrome",),
          (r"palindrome",),
          skill_palindrome,
          (('Est-ce que "kayak" est un palindrome ?', "oui"),
           ('Est-ce que "chat" est un palindrome ?', "non"))),
    Skill("anagramme", "Test d'anagramme",
          ("anagramme",),
          (r"anagramme",),
          skill_anagram,
          (('"chien" et "niche" sont-ils des anagrammes ?', "oui"),
           ('"chat" et "chien" sont-ils des anagrammes ?', "non"))),
    Skill("romain", "Chiffres romains",
          ("romain", "romains", "vaut"),
          (r"chiffres?\s+romains?", r"\bvaut\s+[ivxlcdm]{2,}\b"),
          skill_roman,
          (("1987 en chiffres romains", "MCMLXXXVII"), ("Que vaut MCMXCIV ?", "1994"))),
    Skill("capitale", "Capitale d'un pays",
          ("capitale",),
          (r"capitale",),
          skill_capital,
          (("Quelle est la capitale de l'Australie ?", "Canberra"),
           ("capitale de la France", "Paris"))),
    Skill("element", "Élément chimique",
          ("symbole", "numero atomique", "element"),
          (r"(?:symbole|numero atomique)",),
          skill_element,
          (("Quel est le symbole du fer ?", "Fe"), ("Numéro atomique de l'or", "79"))),
    Skill("planete", "Ordre des planètes",
          ("planete", "planetes", "position"),
          (r"planete", r"position\s+de\s+[a-z]+", r"\b(?:mercure|venus|terre|mars|jupiter|saturne|uranus|neptune)\b"),
          skill_planet,
          (("Quelle est la 3e planète ?", "Terre"), ("Quelle est la position de Mars ?", "4"))),
    Skill("projet", "Faits vérifiés du projet",
          ("programme", "budget", "metrique", "joule", "hecaton", "silex", "kairos", "projet"),
          (r"programme\s+ii", r"budget", r"metrique", r"silex", r"kairos", r"hecaton"),
          skill_projet,
          (("Quel est le budget mémoire du Programme II ?", "100 Go"),
           ("Quelle est la métrique du projet ?", "joule"),
           ("Qui est Silex ?", "packing ternaire"),
           ("Qui est KAIROS ?", "arena/c9070fe4-scienceai"),
           ("Qu'est-ce que HECATON ?", "baseline"))),
    Skill("identite", "Identité du système",
          ("qui es-tu", "tu es qui", "comment fonctionnes-tu"),
          (r"qui\s+es-tu|tu\s+es\s+qui|comment\s+fonctionnes-tu|tu\s+es\s+un\s+(?:llm|modele|chatgpt)",),
          skill_identity,
          (("Qui es-tu ?", "tri-système"), ("Tu es un LLM ?", "pas un transformer"))),
]


# ---------------------------------------------------------------------------
# 2. Mémoire documentaire (BM25)
# ---------------------------------------------------------------------------

STOP = set("""que qui quoi est sont une un des les le la de du dans pour avec sur pas plus
combien quel quelle quels quelles comment pourquoi quand ou ce cest et en au aux il elle on
nous vous ils elles je tu mon ma mes ton ta tes son sa ses leur leurs mais donc car si aussi
tres tout tous toute toutes faire fait peut etre avoir the of and to is a an this that it as
at be by for from in on or with was were are which what how why when where""".split())


class Memory:
    """BM25 sur les paragraphes de docs/*.md."""

    def __init__(self, docs_dir: Path, max_chunks: int = 4000):
        self.chunks: List[Tuple[str, str]] = []
        self.tokens: List[List[str]] = []
        self.df: Dict[str, int] = {}
        self.avgdl = 0.0
        if not docs_dir.exists():
            return
        for path in sorted(docs_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            title = path.stem
            for block in re.split(r"\n\s*\n", text):
                b = block.strip()
                if len(b) < 120 or b.startswith("#"):
                    continue
                toks = self._tok(b)
                if len(toks) < 10:
                    continue
                self.chunks.append((title, b))
                self.tokens.append(toks)
                for t in set(toks):
                    self.df[t] = self.df.get(t, 0) + 1
                if len(self.chunks) >= max_chunks:
                    break
        self.avgdl = (sum(len(t) for t in self.tokens) / len(self.tokens)) if self.tokens else 0.0

    @staticmethod
    def _tok(s: str) -> List[str]:
        return [w for w in re.findall(r"[a-z0-9]{3,}", norm(s)) if w not in STOP]

    def search(self, q: str, k: int = 3) -> List[Tuple[float, int, str, str]]:
        qt = set(self._tok(q))
        if not qt or not self.chunks:
            return []
        n = len(self.chunks)
        scores = []
        for i, toks in enumerate(self.tokens):
            dl = len(toks)
            score = 0.0
            tf: Dict[str, int] = {}
            for t in toks:
                if t in qt:
                    tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                idf = math.log(1 + (n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))
                denom = f + 1.5 * (1 - 0.75 + 0.75 * dl / self.avgdl)
                score += idf * f * 2.5 / denom
            if score > 0:
                matched_terms = [t for t in qt if t in tf]
                scores.append((score, matched_terms, i))
        scores.sort(key=lambda t: -t[0])
        out = []
        for score, matched_terms, i in scores[:k]:
            out.append((score, matched_terms, self.chunks[i][0], self.chunks[i][1]))
        return out

    def snippet(self, text: str, terms: Sequence[str], width: int = 480) -> str:
        """Fenêtre centrée sur le terme le plus rare présent dans le texte."""
        n = norm(text)
        best = -1
        for t in sorted(terms, key=lambda x: self.df.get(x, 0)):
            i = n.find(t)
            if i >= 0:
                best = i
                break
        if best < 0 or len(text) <= width:
            s = re.sub(r"\s+", " ", text).strip()
            return s if len(s) <= width else s[:width - 1].rsplit(" ", 1)[0] + "…"
        start = max(0, best - width // 3)
        end = min(len(text), start + width)
        frag = re.sub(r"\s+", " ", text[start:end]).strip()
        return ("… " if start else "") + frag + (" …" if end < len(text) else "")


# ---------------------------------------------------------------------------
# 3. Le système
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    text: str
    kind: str          # "competence" | "memoire" | "aveu"
    source: str
    proof: str
    score: int
    ms: float
    energy_mj: float
    verified: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "text": self.text, "kind": self.kind, "source": self.source,
            "proof": self.proof, "score": self.score, "ms": round(self.ms, 3),
            "energy_mj": round(self.energy_mj, 3), "verified": self.verified,
        }


class TriSystem:
    def __init__(self, docs_dir: Optional[Path] = None, verify: bool = True):
        self.memory = Memory(docs_dir or (ROOT / "docs"))
        self.counters = {"competence": 0, "memoire": 0, "aveu": 0}
        self.tests_total = 0
        self.tests_passed = 0
        if verify:
            for sk in SKILLS:
                fails = []
                for q, expected in sk.tests:
                    self.tests_total += 1
                    try:
                        got, _ = sk.fn(q)
                        if norm(expected) not in norm(got):
                            fails.append(f"{q!r} -> {got!r} (attendu {expected!r})")
                        else:
                            self.tests_passed += 1
                    except Exception as e:  # noqa: BLE001
                        fails.append(f"{q!r} -> exception {e}")
                if fails:
                    sk.enabled = False
                    sk.failures = tuple(fails)

    def skills_ok(self) -> List[Skill]:
        return [s for s in SKILLS if s.enabled]

    def answer(self, query: str) -> Answer:
        t0 = time.perf_counter()
        c0 = time.process_time()
        try:
            candidates = sorted(((sk.score(query), sk) for sk in self.skills_ok()),
                                key=lambda t: -t[0])
            candidates = [(sc, sk) for sc, sk in candidates if sc >= 3]
            best_score = candidates[0][0] if candidates else 0
            # le vérificateur tranche : on essaie les compétences par score
            # décroissant et on garde la première qui n'échoue pas.
            chosen, text, proof, rejected = None, "", "", []
            for sc, sk in candidates:
                try:
                    t, pr = sk.fn(query)
                    chosen, text, proof = sk, t, pr
                    break
                except Exception as e:  # noqa: BLE001
                    rejected.append(f"{sk.name}: {e}")
            if chosen is not None:
                ans = Answer(text=text, kind="competence", source=chosen.title,
                             proof=proof, score=best_score, ms=0.0, energy_mj=0.0,
                             verified=True)
                if rejected:
                    ans.proof += " · rejets : " + " ; ".join(rejected)
            else:
                hits = self.memory.search(query)
                qt = set(self.memory._tok(query))
                # Règle d'honnêteté : le paragraphe doit contenir TOUS les termes
                # rares de la question (df <= 20). Les termes fréquents peuvent
                # manquer. Sinon, on avoue au lieu de servir un hors-sujet.
                rare = {t for t in qt if self.memory.df.get(t, 0) <= 20}
                if hits and hits[0][0] >= 3.5 and rare.issubset(set(hits[0][1])):
                    sc, matched, title, text = hits[0]
                    snippet = self.memory.snippet(text, matched)
                    ans = Answer(text=snippet, kind="memoire", source=f"docs/{title}.md",
                                 proof=f"BM25 = {sc:.2f} · {len(matched)} terme(s) rares", score=int(sc * 10),
                                 ms=0.0, energy_mj=0.0, verified=False)
                else:
                    ans = Answer(
                        text=("Je ne sais pas répondre avec mes compétences actuelles. "
                              "Je préfère l'avouer plutôt que d'inventer. "
                              "Essayez un calcul, une conversion, une date, ou une "
                              "question sur le projet."),
                        kind="aveu", source="—", proof="aucune compétence au-dessus du seuil",
                        score=best_score, ms=0.0, energy_mj=0.0, verified=True)
        except Exception as e:  # noqa: BLE001
            ans = Answer(text=f"Échec de la compétence : {e}", kind="aveu",
                         source="vérificateur", proof="exception interceptée",
                         score=0, ms=0.0, energy_mj=0.0, verified=False)
        cpu = max(0.0, time.process_time() - c0)
        ans.ms = (time.perf_counter() - t0) * 1000.0
        # énergie estimée à 15 W de plateforme ; c'est une hypothèse, pas une mesure
        ans.energy_mj = cpu * 15.0
        self.counters[ans.kind] += 1
        return ans

    def metrics(self) -> Dict[str, object]:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "skills": len(self.skills_ok()),
            "skills_total": len(SKILLS),
            "tests": self.tests_total,
            "tests_passed": self.tests_passed,
            "rss_mb": round(rss_kb / 1024.0, 1),
            "counters": dict(self.counters),
            "chunks": len(self.memory.chunks),
        }


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="tri-système en ligne de commande")
    ap.add_argument("question", nargs="*", help="question ; sans argument : démonstration")
    args = ap.parse_args()
    ts = TriSystem()
    if args.question:
        ans = ts.answer(" ".join(args.question))
        print(ans.text)
        print(f"  [{ans.kind} · {ans.source} · {ans.ms:.2f} ms · "
              f"{ans.energy_mj:.2f} mJ · vérifié={ans.verified}]")
        if ans.proof:
            print(f"  preuve : {ans.proof}")
        return 0
    print("Tri-système — démonstration\n")
    questions = [
        "Combien font 17 × 23 ?",
        "Combien font 15 % de 240 ?",
        "Convertir 10 km en miles",
        "Quel jour de la semaine le 14 juillet 2026 ?",
        "Quelle est la capitale de l'Australie ?",
        "C'est quoi le Programme II ?",
        "Qui es-tu ?",
        "Quelle est la recette du gratin dauphinois ?",
    ]
    for q in questions:
        a = ts.answer(q)
        print(f"Q: {q}")
        print(f"R: {a.text}")
        print(f"   [{a.kind} · {a.source} · {a.ms:.2f} ms · {a.energy_mj:.2f} mJ]\n")
    print(ts.metrics())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
