"""G4: повтор подряд одного и того же узла (G, C) не меняет модельный цвет.
Если при той же G записаны два разных цвета на ветви label=1,
программа отклоняет ветвь как неоднозначную.

Допуск — 1e-8 mag.

Из корня: $M67_PY -m pytest -q test_w7_vertex.py
"""

from fractions import Fraction as F

from hypothesis import given, settings, strategies as st

from src.m67_residuals import check_coverage, colour_residual


TOL = F(1, 10**8)


def _assert_colour(actual, expected, where):
    diff = abs(actual - expected)
    if diff > TOL:
        raise AssertionError(
            f"{where}: ожидалось {float(expected):+.9f}, "
            f"получено {float(actual):+.9f}, "
            f"разница {float(diff):.2e} mag"
        )


def _build_branch(segments):
    """segments: [(G1, C1), (G2, C2), ...] — узлы label=1.
    Возвращает colour_at из check_coverage."""
    return check_coverage((F(g), F(c), 1) for g, c in segments)


# ---------- Hypothesis: повтор одной и той же вершины не меняет цвет ----------

@given(
    n_nodes=st.integers(2, 5),
    colours=st.lists(st.integers(3000, 15000), min_size=5, max_size=5),
    repeat_idx=st.integers(0, 4),
)
@settings(max_examples=64, derandomize=True, database=None, deadline=None,
          report_multiple_bugs=False)
def test_identical_vertex_repeat_does_not_change_colour(n_nodes, colours, repeat_idx):
    """Повтор подряд одной и той же вершины (G, C) не меняет модельный цвет."""
    g_min = F(15)
    g_max = F(18)
    g_range = g_max - g_min
    step = g_range / (n_nodes - 1)

    node_colours = [F(c, 10000) for c in colours[:n_nodes]]
    nodes = [(g_min + step * i, node_colours[i]) for i in range(n_nodes)]

    # Вставляем повтор выбранной вершины (только если индекс валиден)
    if repeat_idx < n_nodes:
        dup = nodes[repeat_idx]
        nodes = nodes[:repeat_idx + 1] + [dup] + nodes[repeat_idx + 1:]

        colour_at = _build_branch(nodes)

        # Цвет в узле повтора должен совпадать с оригиналом
        g_dup, c_orig = dup
        _assert_colour(colour_at(g_dup), c_orig, f"повтор вершины idx={repeat_idx}")


# ---------- Hypothesis: два разных цвета при одной G отклоняет ветвь ----------

@given(
    n_nodes=st.integers(2, 5),
    colours=st.lists(st.integers(3000, 15000), min_size=5, max_size=5),
    conflict_idx=st.integers(0, 4),
    shift_mmag=st.integers(1, 100),
)
@settings(max_examples=64, derandomize=True, database=None, deadline=None,
          report_multiple_bugs=False)
def test_two_colours_at_same_g_rejected(n_nodes, colours, conflict_idx, shift_mmag):
    """Если при той же G записаны два разных цвета на label=1 — ошибка."""
    g_min = F(15)
    g_max = F(18)
    g_range = g_max - g_min
    step = g_range / (n_nodes - 1)

    node_colours = [F(c, 10000) for c in colours[:n_nodes]]
    nodes = [(g_min + step * i, node_colours[i]) for i in range(n_nodes)]

    # Сначала строим корректную ветвь
    _build_branch(nodes)

    # Вставляем конфликт: та же G, цвет сдвинут на 1–100 mmag
    if conflict_idx < n_nodes:
        g_conflict = nodes[conflict_idx][0]
        c_conflict = nodes[conflict_idx][1] + F(shift_mmag, 10000)
        nodes = nodes[:conflict_idx + 1] + [(g_conflict, c_conflict)] + nodes[conflict_idx + 1:]
        try:
            _build_branch(nodes)
        except ValueError:
            return  # ожидаемая ошибка
        raise AssertionError(f"конфликт idx={conflict_idx}: ветвь не отклонена")


# ---------- явный пример: повтор вершины (16, 0.90) ----------

def test_explicit_identical_vertex_repeat():
    """Фиксированный пример: узлы STRAIGHT с повтором (16, 0.90)."""
    nodes = [
        (15, "0.70"),
        (16, "0.90"),
        (16, "0.90"),  # повтор той же вершины
        (17, "1.10"),
        (18, "1.30"),
    ]
    colour_at = _build_branch(nodes)
    _assert_colour(colour_at(F(16)), F("0.90"), "явный повтор вершины")


# ---------- явный пример: два цвета при G=16 ----------

def test_explicit_two_colours_at_same_g_rejected():
    """Фиксированный пример: (16, 0.90) и (16, 0.95) — должна быть ошибка."""
    nodes = [
        (15, "0.70"),
        (16, "0.90"),
        (16, "0.95"),  # другой цвет при той же G
        (17, "1.10"),
        (18, "1.30"),
    ]
    try:
        _build_branch(nodes)
    except ValueError:
        return  # ожидаемая ошибка
    raise AssertionError("явный конфликт: ветвь не отклонена")
