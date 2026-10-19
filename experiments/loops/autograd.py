"""autograd.py — micro-moteur de dérivation automatique en NumPy.

Écrit pour comparer des architectures sur ce CPU, sans PyTorch.
~200 lignes, float32, suffisant pour un transformer de 200 k paramètres.
"""
from __future__ import annotations

import numpy as np


class T:
    __slots__ = ("d", "g", "back", "req", "kids")

    def __init__(self, d, req=False):
        self.d = np.asarray(d, dtype=np.float32)
        self.g = None
        self.back = None
        self.req = req
        self.kids = ()

    # --- helpers -----------------------------------------------------------
    @property
    def shape(self):
        return self.d.shape

    def _acc(self, g):
        if g is None:
            return
        g = np.asarray(g, dtype=np.float32)
        self.g = g if self.g is None else self.g + g

    # --- opérations --------------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.d + o.d, self.req or o.req)
        out.kids = tuple(x for x in (self, o) if x.req)

        def back():
            if self.req:
                self._acc(_unbroadcast(out.g, self.d.shape))
            if o.req:
                o._acc(_unbroadcast(out.g, o.d.shape))
        out.back = back
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (o * -1.0 if isinstance(o, T) else T(-np.asarray(o, np.float32)))

    def __rsub__(self, o):
        return T(o) + (-self)

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.d * o.d, self.req or o.req)
        out.kids = tuple(x for x in (self, o) if x.req)

        def back():
            if self.req:
                self._acc(_unbroadcast(out.g * o.d, self.d.shape))
            if o.req:
                o._acc(_unbroadcast(out.g * self.d, o.d.shape))
        out.back = back
        return out

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self * (o ** -1.0 if isinstance(o, T) else T(np.float32(1.0 / np.float32(o))))

    def matmul(self, o):
        out = T(self.d @ o.d, self.req or o.req)
        out.kids = tuple(x for x in (self, o) if x.req)

        def back():
            if self.req:
                self._acc(_unbroadcast(out.g @ np.swapaxes(o.d, -1, -2), self.d.shape))
            if o.req:
                o._acc(_unbroadcast(np.swapaxes(self.d, -1, -2) @ out.g, o.d.shape))
        out.back = back
        return out

    def transpose(self):
        out = T(np.swapaxes(self.d, -1, -2), self.req)
        out.kids = (self,) if self.req else ()

        def back():
            self._acc(np.swapaxes(out.g, -1, -2))
        out.back = back
        return out

    def permute(self, axes):
        inv = np.argsort(axes)
        out = T(np.transpose(self.d, axes), self.req)
        out.kids = (self,) if self.req else ()

        def back():
            self._acc(np.transpose(out.g, inv))
        out.back = back
        return out

    def masked(self, mask):
        """mask : array ou T avec -1e9 hors du domaine autorisé."""
        return self + (mask if isinstance(mask, T) else T(mask))

    def reshape(self, *shape):
        out = T(self.d.reshape(shape), self.req)
        out.kids = (self,) if self.req else ()

        def back():
            self._acc(out.g.reshape(self.d.shape))
        out.back = back
        return out

    def sum(self, axis=None, keepdims=False):
        out = T(self.d.sum(axis=axis, keepdims=keepdims), self.req)
        out.kids = (self,) if self.req else ()

        def back():
            g = out.g
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.d.shape))
        out.back = back
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.d.size if axis is None else self.d.shape[axis]
        return self.sum(axis, keepdims) * np.float32(1.0 / n)

    def pow(self, p):
        out = T(self.d ** p, self.req)
        out.kids = (self,) if self.req else ()

        def back():
            self._acc(out.g * p * (self.d ** (p - 1)))
        out.back = back
        return out

    def exp(self):
        out = T(np.exp(self.d), self.req)
        out.kids = (self,) if self.req else ()

        def back():
            self._acc(out.g * out.d)
        out.back = back
        return out

    def log(self):
        out = T(np.log(self.d), self.req)
        out.kids = (self,) if self.req else ()

        def back():
            self._acc(out.g / self.d)
        out.back = back
        return out

    def relu(self):
        out = T(np.maximum(self.d, 0), self.req)
        out.kids = (self,) if self.req else ()

        def back():
            self._acc(out.g * (self.d > 0))
        out.back = back
        return out

    def gelu(self):
        c = np.float32(np.sqrt(2.0 / np.pi))
        x = self.d
        t = np.tanh(c * (x + 0.044715 * x ** 3)).astype(np.float32)
        out = T(0.5 * x * (1 + t), self.req)
        out.kids = (self,) if self.req else ()

        def back():
            dt = c * (1 + 3 * 0.044715 * x ** 2) * (1 - t ** 2)
            self._acc(out.g * (0.5 * (1 + t) + 0.5 * x * dt))
        out.back = back
        return out

    def embedding(self, idx):
        out = T(self.d[idx], self.req)
        out.kids = (self,) if self.req else ()

        def back():
            g = np.zeros_like(self.d)
            np.add.at(g, idx, out.g)
            self._acc(g)
        out.back = back
        return out

    def softmax(self, axis=-1):
        x = self.d - self.d.max(axis=axis, keepdims=True)
        e = np.exp(x)
        out = T(e / e.sum(axis=axis, keepdims=True), self.req)
        out.kids = (self,) if self.req else ()

        def back():
            s = (out.g * out.d).sum(axis=axis, keepdims=True)
            self._acc(out.d * (out.g - s))
        out.back = back
        return out

    def layernorm(self, eps=1e-5):
        mu = self.d.mean(-1, keepdims=True)
        var = ((self.d - mu) ** 2).mean(-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        out = T((self.d - mu) * inv, self.req)
        out.kids = (self,) if self.req else ()

        def back():
            g = out.g
            gx = g * inv
            gmu = (-gx).sum(-1, keepdims=True)
            gvar = (g * (self.d - mu) * (-0.5 * inv ** 3)).sum(-1, keepdims=True)
            n = self.d.shape[-1]
            self._acc(gx + gmu / n + gvar * 2 * (self.d - mu) / n)
        out.back = back
        return out

    def backward(self):
        self.g = np.ones_like(self.d)
        order, seen = [], set()

        def walk(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for c in t.kids:
                walk(c)
            order.append(t)

        walk(self)
        for node in reversed(order):
            if node.back is not None and node.g is not None:
                node.back()
        # Le graphe est un cycle : chaque enfant garde son parent vivant via la
        # fermeture `back`, et le parent garde l'enfant via `kids`.  Sans ce
        # nettoyage, seuls les passages du ramasse-miettes libèrent les
        # tenseurs, et le RSS monte à plusieurs Go pour un modèle de 58 k
        # paramètres (constaté : 2,24 Go au pas 2 100).  On coupe les arêtes
        # après la passe backward : la libération redevient immédiate.
        for node in order:
            node.kids = ()
            node.back = None


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g
