"""G2: линейная интерполяция между соседними узлами изохроны.

Проверяет, что на каждом отрезке label=1 цвет меняется как
(1-t)*C1 + t*C2 при 0 <= t <= 1. Hypothesis генерирует 4-7 узлов
с случайными цветами и доли t для проверки внутренних точек.
Допуск — 1e-8 mag.

Из корня: $M67_PY -m pytest -q test_w7_segment.py
"""

from fractions import Fraction as F

from hypothesis import given, settings, strategies as st

from src.m67_residuals import check_coverage


TOL = F(1, 10**8)


def _assert_colour(actual, expected, where, g1, c1, g2, c2, t):
    diff = abs(actual - expected)
    if diff > TOL:
        raise AssertionError(
            f"{where}: узлы ({float(g1)}, {float(c1)}) — ({float(g2)}, {float(c2)}), "
            f"t={float(t):.4f}, G={float(g1 + (g2 - g1) * t):.4f}, "
            f"ожидалось {float(expected):+.9f}, "
            f"получено {float(actual):+.9f}, "
            f"разница {float(diff):.2e} mag"
        )


def _build_branch(segments):
    """segments: [(G1, C1), (G2, C2), ...] — узлы label=1.
    Возвращает colour_at из check_coverage."""
    knots = ((g, c, 1) for g, c in segments)
    return check_coverage(knots)


# ---------- стратегия: 4-7 узлов, случайные цвета, Hypothesis ----------

@given(
    n_segments=st.integers(3, 6),
    colours=st.lists(
        st.integers(3000, 15000), min_size=7, max_size=7,
    ),
    t_extra=st.integers(min_value=1, max_value=999),
)
@settings(max_examples=64, derandomize=True, database=None, deadline=None,
          report_multiple_bugs=False)
def test_kinked_segment_linear_interpolation(n_segments, colours, t_extra):
    """Генерируем изломанную линию из n_segments отрезков в [15, 18].
    Проверяются концы, три фиксированные внутренние точки и одна
    сгенерированная (t_extra) для каждого отрезка."""
    g_min = F(15)
    g_max = F(18)
    g_range = g_max - g_min
    step = g_range / n_segments

    # Берём первые n_segments+1 цветов из сгенерированного списка.
    node_colours = [F(c, 10000) for c in colours[:n_segments + 1]]

    nodes = []
    for i in range(n_segments + 1):
        g = g_min + step * i
        nodes.append((g, node_colours[i]))

    colour_at = _build_branch(nodes)

    # Проверяем каждый отрезок.
    for seg_idx in range(n_segments):
        g1, c1 = nodes[seg_idx]
        g2, c2 = nodes[seg_idx + 1]

        # --- концы отрезка ---
        _assert_colour(colour_at(g1), c1, f"seg {seg_idx} t=0", g1, c1, g2, c2, F(0))
        _assert_colour(colour_at(g2), c2, f"seg {seg_idx} t=1", g1, c1, g2, c2, F(1))

        # --- внутренние точки: 3 фиксированных + 1 сгенерированная ---
        for t_num in (125, 500, 875, t_extra):
            t = F(t_num, 1000)
            g_test = g1 + (g2 - g1) * t
            expected = (F(1) - t) * c1 + t * c2
            _assert_colour(colour_at(g_test), expected,
                           f"seg {seg_idx} t={float(t):.3f}", g1, c1, g2, c2, t)


# ---------- явный пример с изломом: [0.645, 0.830, 0.920, 0.625], t=0.084 ----------

@given(t=st.just(F(84, 1000)))
@settings(max_examples=1, derandomize=True, database=None, deadline=None,
          report_multiple_bugs=False)
def test_explicit_kink_example(t):
    """Фиксированный излом: 4 узла, цвета [0.645, 0.830, 0.920, 0.625].
    Проверяем t=0.084 на первом отрезке."""
    nodes = [
        (F(15), F("0.645")),
        (F(16), F("0.830")),
        (F(17), F("0.920")),
        (F(18), F("0.625")),
    ]

    colour_at = _build_branch(nodes)

    g1, c1 = nodes[0]
    g2, c2 = nodes[1]

    # --- концы первого отрезка ---
    _assert_colour(colour_at(g1), c1, "explicit t=0", g1, c1, g2, c2, F(0))
    _assert_colour(colour_at(g2), c2, "explicit t=1", g1, c1, g2, c2, F(1))

    # --- внутренняя точка t=0.084 ---
    g_test = g1 + (g2 - g1) * t
    expected = (F(1) - t) * c1 + t * c2
    _assert_colour(colour_at(g_test), expected,
                   f"explicit t={float(t):.3f}", g1, c1, g2, c2, t)