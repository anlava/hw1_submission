"""G3: масштаб цветовых остатков — mag vs миллимагнитуды.

Если выразить все цвета в миллимагнитудах (умножить на 1000),
оставив G в mag, то остаток, делённый на 1000, совпадёт с расчётом в mag.

Из корня: $M67_PY -m pytest -q test_w7_colour_scale.py
"""

from fractions import Fraction as F

from hypothesis import example, given, settings, strategies as st

from src.m67_residuals import check_coverage, colour_residual


TOL = F(1, 10**8)


def _build_branch(colours):
    """colours: [(G1, C1), (G2, C2), ...] — узлы label=1.
    Возвращает colour_at из check_coverage."""
    return check_coverage((F(g), F(c), 1) for g, c in colours)


# ---------- Hypothesis: случайные узлы, случайная звезда ----------

@given(
    n_nodes=st.integers(2, 5),
    colours=st.lists(st.integers(3000, 15000), min_size=5, max_size=5),
    g_milli=st.integers(15000, 18000),
    star_milli=st.integers(3000, 15000),
)
@settings(max_examples=64, derandomize=True, database=None, deadline=None,
          report_multiple_bugs=False)
def test_colour_scale_invariant(n_nodes, colours, g_milli, star_milli):
    """Генерируем ветвь из n_nodes узлов в [15, 18], проверяем масштаб."""
    g_min = F(15)
    g_max = F(18)
    g_range = g_max - g_min
    step = g_range / (n_nodes - 1)

    node_colours = [F(c, 10000) for c in colours[:n_nodes]]
    nodes = [(g_min + step * i, node_colours[i]) for i in range(n_nodes)]

    g = F(g_milli, 1000)
    observed = F(star_milli, 1000)

    # --- расчёт в mag ---
    colour_at = _build_branch(nodes)
    r_mag = colour_residual(g, observed, colour_at)

    # --- расчёт в mmag: все цвета умножены на 1000 ---
    colour_at_mmag = _build_branch((g, c * 1000) for g, c in nodes)
    r_mmag = colour_residual(g, observed * 1000, colour_at_mmag)

    # Делим остаток из mmag на 1000 и сравниваем с mag-остатком
    diff = abs(r_mmag / 1000 - r_mag)
    if diff > TOL:
        raise AssertionError(
            f"масштаб нарушен: n={n_nodes}, G={float(g):.3f}, "
            f"набл={float(observed):.4f}, r_mag={float(r_mag):+.9f}, "
            f"r_mmag/1000={float(r_mmag / 1000):+.9f}, "
            f"разница {float(diff):.2e} mag"
        )


# ---------- явный пример: узлы (15, 0.70) и (18, 1.30), G=16.5, цвет=1.0534 ----------

@given(
    g=st.just(F("16.5")),
    observed=st.just(F("1.0534")),
)
@settings(max_examples=1, derandomize=True, database=None, deadline=None,
          report_multiple_bugs=False)
def test_explicit_colour_scale_example(g, observed):
    """Фиксированный пример: 2 узла, G=16.5, цвет звезды 1.0534 mag.
    Ожидаемый остаток: 0.0534 mag = 53.4 mmag."""
    nodes = [(F(15), F("0.70")), (F(18), F("1.30"))]

    colour_at = _build_branch(nodes)
    r_mag = colour_residual(g, observed, colour_at)

    # Ожидаемый остаток: 1.0534 - 1.00 = 0.0534 mag
    expected = F("0.0534")
    diff = abs(r_mag - expected)
    if diff > TOL:
        raise AssertionError(
            f"явный пример: G={float(g):.1f}, "
            f"набл={float(observed):.4f}, r_mag={float(r_mag):+.9f}, "
            f"ожидалось {float(expected):+.9f}, "
            f"разница {float(diff):.2e} mag"
        )

    # --- проверка масштаба ---
    colour_at_mmag = _build_branch((g_node, c * 1000) for g_node, c in nodes)
    r_mmag = colour_residual(g, observed * 1000, colour_at_mmag)

    diff_scale = abs(r_mmag / 1000 - r_mag)
    if diff_scale > TOL:
        raise AssertionError(
            f"явный пример масштаб: r_mag={float(r_mag):+.9f}, "
            f"r_mmag/1000={float(r_mmag / 1000):+.9f}, "
            f"разница {float(diff_scale):.2e} mag"
        )