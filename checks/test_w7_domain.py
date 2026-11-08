"""Hypothesis-тесты для check_coverage: границы и разрывы по label.

Запуск: python -m pytest test_w7_domain.py -q
"""

import sys
import os

# Allow overriding the src module path via environment variable
MOD_PATH = os.environ.get("M67_TEST_MOD_PATH")
if MOD_PATH:
    # Remove current directory from path to avoid loading original src/
    sys.path = [p for p in sys.path if p != '']
    sys.path.insert(0, MOD_PATH)

from fractions import Fraction as F
from unittest import TestCase

from hypothesis import assume, example, given, settings, strategies as st

from src.m67_residuals import check_coverage


# ──────────────────────────────────────────────────────────────────────
# Тест 1: Границы
# ──────────────────────────────────────────────────────────────────────

class TestCheckCoverageBoundaries(TestCase):
    """Одна ветвь label=1 шире экспериментального диапазона [15, 18].

    Узлы: (14, 0.50) — (19, 1.50). Линия проходит (15, 0.70) и (18, 1.30).
    Любые запросы вне [15, 18] должны вызывать ValueError.
    """

    @settings(max_examples=50, derandomize=True, database=None, deadline=None,
              report_multiple_bugs=False)
    @given(delta_milli=st.integers(1, 999))
    @example(delta_milli=1)
    @example(delta_milli=500)
    @example(delta_milli=999)
    def test_outside_range_raises(self, delta_milli):
        """G = 15 - δ и G = 18 + δ должны вызывать ValueError."""
        delta = F(delta_milli, 1000)           # 0.001 .. 0.999
        colour_at = check_coverage((
            (F(14), F("0.50"), 1),
            (F(19), F("1.50"), 1),
        ))
        # внутри — нормально
        self.assertEqual(colour_at(F(15)), F("0.70"))
        self.assertEqual(colour_at(F(18)), F("1.30"))
        # снаружи — ошибка
        for g in (F(15) - delta, F(18) + delta):
            with self.assertRaises(ValueError, msg=f"G={float(g):.3f} должно быть вне диапазона"):
                colour_at(g)


# ──────────────────────────────────────────────────────────────────────
# Тест 2: Разрыв
# ──────────────────────────────────────────────────────────────────────

class TestCheckCoverageGap(TestCase):
    """Разрыв по label внутри [15, 18]: check_coverage должен поднимать ValueError при построении."""

    @settings(max_examples=50, derandomize=True, database=None, deadline=None,
              report_multiple_bugs=False)
    @given(
        a_milli=st.integers(15001, 17997),
        b_milli=st.integers(15002, 17999),
    )
    @example(a_milli=16000, b_milli=17000)
    def test_gap_rejected(self, a_milli, b_milli):
        """15 < a < b < 18. Сначала все label=1 — ок. Затем label=2 в середине — ошибка."""
        a = F(a_milli, 1000)
        b = F(b_milli, 1000)
        assume(15 < a < b < 18)

        mid = (a + b) / 2
        # Цвета на одной прямой, проходящей (15, 0.70) и (18, 1.30)
        slope = F("0.20") / 3
        c15 = F("0.70")
        c18 = F("1.30")
        ca = c15 + slope * (a - 15)
        cb = c15 + slope * (b - 15)
        cmid = c15 + slope * (mid - 15)

        # 1) Все label=1 — ветвь строится
        nodes_ok = (
            (F(15), c15, 1),
            (a, ca, 1),
            (mid, cmid, 1),
            (b, cb, 1),
            (F(18), c18, 1),
        )
        colour_at_ok = check_coverage(nodes_ok)
        # Проверяем, что работает — границы должны давать правильные цвета
        self.assertEqual(colour_at_ok(F(15)), c15)
        self.assertEqual(colour_at_ok(F(18)), c18)

        # 2) Меняем ТОЛЬКО среднюю метку на 2 — должна быть ошибка при построении
        nodes_bad = (
            (F(15), c15, 1),
            (a, ca, 1),
            (mid, cmid, 2),   # <-- разрыв
            (b, cb, 1),
            (F(18), c18, 1),
        )
        with self.assertRaises(ValueError):
            check_coverage(nodes_bad)