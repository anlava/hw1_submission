"""Независимые примеры для M67. Из корня: python -m pytest checks/test_m67_residuals.py -q"""

from fractions import Fraction as F
from unittest import TestCase

from hypothesis import example, given, settings, strategies as st

from src.m67_residuals import check_coverage, colour_residual


# Из METHOD.md. Не импортируем из программы ни контрольную кривую, ни эталон, ни допуск.
STRAIGHT = ((15, "0.70", 1), (16, "0.90", 1), (17, "1.10", 1), (18, "1.30", 1))
TOL_MAG = F(1, 10**8)


def branch(nodes):
    return check_coverage((F(g), F(colour), label) for g, colour, label in nodes)


class TestM67Residuals(TestCase):
    def assert_mag(self, actual, expected, context):
        self.assertLessEqual(
            abs(actual - expected), TOL_MAG,
            f"{context}: получено {float(actual):+.9f}, ожидалось {float(expected):+.9f} mag",
        )

    def test_one_star_against_two_curves(self):
        """Та же звезда между двумя кривыми: знак должен поменяться."""
        g, observed = F("16.50"), F("1.05")
        model_a = branch(STRAIGHT)
        # Вторую кривую задаём в обратном порядке G и с изломом у G=17.
        model_b = branch(((18, "1.40", 1), (17, "1.30", 1),
                          (16, "0.90", 1), (15, "0.70", 1)))
        # По узлам при 16 и 17: цвет A посередине = 1.00, цвет B = 1.10.
        self.assert_mag(colour_residual(g, observed, model_a), F("+0.05"), "кривая A")
        self.assert_mag(colour_residual(g, observed, model_b), F("-0.05"), "кривая B")

    @settings(max_examples=64, derandomize=True, database=None, deadline=None,
              report_multiple_bugs=False)
    @given(g_milli=st.integers(15000, 18000), shift_centi=st.sampled_from((-5, 0, 5)))
    @example(g_milli=15000, shift_centi=0)
    @example(g_milli=18000, shift_centi=0)
    @example(g_milli=16500, shift_centi=5)
    def test_known_shift_is_recovered(self, g_milli, shift_centi):
        """Если цвет звезды задан как C_ref(G) + s, программа должна вернуть s."""
        g, shift = F(g_milli, 1000), F(shift_centi, 100)
        observed = F("0.70") + F("0.20") * (g - 15) + shift
        actual = colour_residual(g, observed, branch(STRAIGHT))
        self.assert_mag(actual, shift, f"G={float(g):.3f}")

    def test_shared_vertex_gives_one_colour(self):
        # Два соседних отрезка, включая повтор той же самой точки.
        colour_at = branch((STRAIGHT[0], STRAIGHT[1], STRAIGHT[1], *STRAIGHT[2:]))
        self.assert_mag(colour_at(F(16)), F("0.90"), "общая вершина")

    def test_phase_gap_is_rejected(self):
        # Общий диапазон G достаточен, но соединять два блока label=1 через label=2 нельзя.
        with self.assertRaises(ValueError):
            branch(((15, "0.70", 1), (16, "0.90", 1), ("16.25", "1.55", 2),
                    ("16.50", "1.00", 1), (18, "1.30", 1)))

    def test_fold_with_two_colours_is_rejected(self):
        with self.assertRaises(ValueError):
            branch(((15, "0.70", 1), (18, "1.30", 1), (16, "1.20", 1)))

    def test_vertical_segment_is_rejected(self):
        with self.assertRaises(ValueError):
            branch(((15, "0.70", 1), (16, "0.90", 1),
                    (16, "1.00", 1), (18, "1.30", 1)))

    def test_extrapolation_is_rejected(self):
        colour_at = branch(STRAIGHT)
        for g in (F("14.99"), F("18.01")):
            with self.subTest(g=g), self.assertRaises(ValueError):
                colour_at(g)
