"""M67: этапы dry, smoke и sweep из METHOD.md."""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from importlib import metadata
from itertools import pairwise
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Cavallo": "m67_isochrone_apparent.csv",
    "Bossini": "m67_isochrone_bossini2019_apparent.csv",
}
BINS = ("[15,16)", "[16,17)", "[17,18]")
MIN_N = 20
TOL = Fraction(1, 10**8)  # Абсолютный допуск для контрольного сдвига, mag.
LABEL_SOURCE = "https://stev.oapd.inaf.it/cmd_3.9/help.html"  # 1=MS (main sequence)
LOG_FIELDS = ("timestamp", "stage", "shift", "seed", "n", "metric", "elapsed_s",
              "verdict", "detail", "python", "numpy", "pytest", "hypothesis")


def records(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or not set(fields) <= set(header):
            raise ValueError(f"{path.name}: проверьте заголовок CSV; нужны {', '.join(fields)}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}:{reader.line_num}: неверное число полей")
            yield f"{path.name}:{reader.line_num}", row


def number(value, where):
    """Читаем десятичные числа CSV точно: общая вершина двух отрезков — один цвет."""
    try:
        value = Decimal(value)
        if not value.is_finite():
            raise ValueError
        return Fraction(value)
    except (ValueError, InvalidOperation) as exc:
        raise ValueError(f"{where}: ожидалось конечное число") from exc


def bin_index(g):
    if 15 <= g < 16:
        return 0
    if 16 <= g < 17:
        return 1
    if 17 <= g <= 18:
        return 2
    return None


def member_counts(path):
    counts = [0, 0, 0]
    seen = set()
    for where, row in records(path, ("GaiaDR3", "Gmag", "BP_RP")):
        key = row["GaiaDR3"]
        if not (key.isascii() and key.isdigit()) or key in seen:
            raise ValueError(f"{where}: GaiaDR3 пуст, некорректен или повторён: {key!r}")
        seen.add(key)
        i = bin_index(number(row["Gmag"], f"{where} Gmag"))
        number(row["BP_RP"], f"{where} BP_RP")
        if i is not None:
            counts[i] += 1
    return len(seen), counts


def model_points(path):
    for where, row in records(path, ("G_app", "BP_RP", "label")):
        stage = number(row["label"], f"{where} label")
        if stage.denominator != 1:
            raise ValueError(f"{where}: label не целое")
        yield (number(row["G_app"], f"{where} G_app"),
               number(row["BP_RP"], f"{where} BP_RP"), int(stage))


def check_coverage(points, lo=Fraction(15), hi=Fraction(18)):
    """Проверяет всю ветвь и возвращает функцию модельного цвета при заданной G."""
    segments = []
    knots = {lo, hi}
    # Соседство определяем до отбора по фазе. Между разными блоками линии нет.
    for (x, y, label), (xx, yy, next_label) in pairwise(points):
        if label != 1 or next_label != 1:
            continue
        left, right = min(x, xx), max(x, xx)
        if right < lo or hi < left:
            continue
        segments.append((x, y, xx, yy, left, right))
        knots.update((max(lo, left), min(hi, right)))

    def colour_at(g):
        if not lo <= g <= hi:
            raise ValueError(f"G={float(g):.8g} вне проверенного диапазона")
        colours = set()
        for x, y, xx, yy, left, right in segments:
            if left <= g <= right:
                if x == xx:
                    colours.update((y, yy))  # Вертикальный отрезок с разными цветами.
                else:
                    colours.add(y + (yy - y) * (g - x) / (xx - x))
        if not colours:
            raise ValueError(f"нет покрытия при G={float(g):.8g}")
        if len(colours) > 1:
            raise ValueError(f"несколько цветов при G={float(g):.8g}")
        return colours.pop()

    knots = sorted(knots)
    for g in knots:
        colour_at(g)

    # Внутри интервала между узлами состав отрезков постоянен. Покрывающие
    # его прямые совпадают в обоих концах (проверено выше), значит, и внутри.
    for a, b in pairwise(knots):
        if not any(left <= a and b <= right for *_, left, right in segments):
            raise ValueError(f"разрыв покрытия между G={float(a):.8g} и G={float(b):.8g}")
    return colour_at


def colour_residual(g, observed_colour, colour_at):
    """r = цвет звезды минус цвет модели при той же видимой G, в mag."""
    return observed_colour - colour_at(g)


def environment():
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "pytest", "hypothesis"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "не установлен"
    return versions


def append_log(path, row):
    has_rows = path.exists() and path.stat().st_size > 0
    if has_rows:
        with path.open(encoding="utf-8", newline="") as stream:
            if next(csv.reader(stream), []) != list(LOG_FIELDS):
                raise ValueError(f"{path.name}: чужой заголовок; журнал оставлен без изменений")
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, LOG_FIELDS)
        if not has_rows:
            writer.writeheader()
        writer.writerow(row)


def dry(root):
    results = root / "results"
    versions = environment()
    row = dict(timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
               stage="dry", verdict="FAIL", **versions)
    start = perf_counter()
    print("Окружение:", ", ".join(f"{k}={v}" for k, v in versions.items()), flush=True)
    try:
        if sys.version_info < (3, 11) or "не установлен" in versions.values():
            raise ValueError("по METHOD.md нужен Python 3.11+ с NumPy, pytest и Hypothesis")
        cmd = [sys.executable, "-B", str(root / "checks" / "check_comparison.py")]
        status = subprocess.run(cmd, cwd=root, check=False).returncode
        if status:
            raise ValueError(f"checks/check_comparison.py: код {status}")
        total, counts = member_counts(results / "m67_members_selected.csv")
        row["n"] = sum(counts)
        print(f"Звёзд в исходной CSV: {total}. В пределах 15 ≤ G ≤ 18: {row['n']}.")
        for name, n in zip(BINS, counts):
            print(f"G {name}: {n}" + (" — недостаточно данных" if n < MIN_N else ""))
        errors = []
        for name, filename in MODELS.items():
            try:
                check_coverage(model_points(results / filename))
                print(f"{name}: при каждом G от 15 до 18 есть ровно один цвет (label=1).")
            except (OSError, csv.Error, ValueError) as exc:
                errors.append(f"{name}: {exc}")
                print(errors[-1])
        bin_detail = ", ".join(f"{name}: {n}" for name, n in zip(BINS, counts))
        if errors:
            raise ValueError(f"{bin_detail}; " + "; ".join(errors))
        row.update(verdict="PASS", detail=f"{bin_detail}; обе кривые однозначны; {LABEL_SOURCE}")
    except (OSError, csv.Error, ValueError) as exc:
        row["detail"] = str(exc)
    row["elapsed_s"] = f"{perf_counter() - start:.6f}"
    try:
        append_log(results / "week6_runs.csv", row)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"dry: НЕ ПРИНЯТО; {row['detail']}; не удалось записать журнал: {exc}")
        return 1
    if row["verdict"] == "FAIL":
        print(f"dry: НЕ ПРИНЯТО; {row['detail']}")
        return 1
    print("dry: ПРИНЯТО. Журнал: results/week6_runs.csv. Цветовые остатки не считались.")
    return 0


def control_curve():
    """Вход модели: только четыре узла (G, BP−RP, label)."""
    nodes = ((15, "0.70"), (16, "0.90"), (17, "1.10"), (18, "1.30"))
    return check_coverage((Fraction(g), Fraction(c), 1) for g, c in nodes)


def reference_colour(g):
    """Эталон для генерации звезды, независимый от интерполяции модели."""
    return Fraction("0.70") + Fraction("0.20") * (g - 15)


def absolute_error(residual, shift):
    error = abs(residual - shift)
    if not isfinite(error):
        raise ValueError("получено неконечное отклонение от известного сдвига")
    return error


def smoke_examples():
    """Выдаёт (G, цвет звезды, известный сдвиг, вычисленный остаток)."""
    colour_at = control_curve()
    for g_text, shift_text in (("15", "0"), ("16", "-0.05"), ("16.50", "0.05"),
                               ("17.25", "-0.05"), ("18", "0")):
        g, shift = Fraction(g_text), Fraction(shift_text)
        observed = reference_colour(g) + shift
        yield g, observed, shift, colour_residual(g, observed, colour_at)


def smoke(root):
    versions = environment()
    row = dict(timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
               stage="smoke", n=0, verdict="FAIL", **versions)
    start = perf_counter()
    try:
        if sys.version_info < (3, 11) or "не установлен" in versions.values():
            raise ValueError("по METHOD.md нужен Python 3.11+ с NumPy, pytest и Hypothesis")
        errors = []
        print("    G     BP−RP звезды     ожидаем r     получили r  (mag)")
        for g, observed, shift, residual in smoke_examples():
            print(f"{float(g):5.2f} {float(observed):16.3f} "
                  f"{float(shift):13.3f} {float(residual):14.3f}")
            errors.append(absolute_error(residual, shift))
            row["n"] = len(errors)
        worst = max(errors)
        row["metric"] = f"{float(worst):.12g}"
        detail = f"max |r − s| = {row['metric']} mag; допуск 1e-8 mag"
        if worst > TOL:
            raise ValueError(detail)
        row.update(verdict="PASS", detail=detail)
    except (OSError, ValueError) as exc:
        row["detail"] = str(exc)
    row["elapsed_s"] = f"{perf_counter() - start:.6f}"
    try:
        append_log(root / "results" / "week6_runs.csv", row)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"smoke: НЕ ПРИНЯТО; {row['detail']}; не удалось записать журнал: {exc}")
        return 1
    accepted = row["verdict"] == "PASS"
    print(f"smoke: {'ПРИНЯТО' if accepted else 'НЕ ПРИНЯТО'}; {row['detail']}.")
    return 0 if accepted else 1


def synthetic_magnitudes(seed, per_bin=30):
    """Равномерная выборка внутри каждого интервала; границы проверяет smoke."""
    if per_bin < 1:
        raise ValueError("per_bin должно быть положительным")
    rng = Random(seed)
    grid = 2**53
    # Смещение строго между 0 и 1. Fraction сохраняет его без округления к границе.
    return tuple(start + Fraction(rng.randrange(1, grid), grid)
                 for start in (15, 16, 17) for _ in range(per_bin))


def sweep(root):
    versions = environment()
    try:
        if sys.version_info < (3, 11) or "не установлен" in versions.values():
            raise ValueError("по METHOD.md нужен Python 3.11+ с NumPy, pytest и Hypothesis")
        colour_at = control_curve()
    except ValueError as exc:
        print(f"sweep: НЕ ПРИНЯТО; перебор не начат: {exc}")
        return 1
    seeds = (11, 23, 37)
    shifts = ("-0.05", "0.00", "+0.05")
    passed = 0
    print("seed  сдвиг (mag)    n   max |r − s| (mag)  итог")
    for seed in seeds:
        magnitudes = synthetic_magnitudes(seed)  # один набор G для всех трёх сдвигов
        counts = [sum(lo < g < lo + 1 for g in magnitudes) for lo in (15, 16, 17)]
        if len(magnitudes) != 90 or counts != [30, 30, 30]:
            print(f"sweep: НЕ ПРИНЯТО; seed={seed}: ожидалось 30 G внутри каждого интервала, "
                  f"получено {counts} при общем числе {len(magnitudes)}.")
            return 1
        for shift_text in shifts:
            shift = Fraction(shift_text)
            row = dict(timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                       stage="sweep", seed=seed, shift=shift_text, n=0,
                       verdict="FAIL", metric="", **versions)
            start = perf_counter()
            try:
                worst = Fraction(0)
                for g in magnitudes:
                    observed = reference_colour(g) + shift
                    residual = colour_residual(g, observed, colour_at)
                    worst = max(worst, absolute_error(residual, shift))
                    row["n"] += 1
                row["metric"] = f"{float(worst):.12g}"
                row["detail"] = "max |r − s| в mag; допуск 1e-8; по 30 G в трёх интервалах"
                if worst <= TOL:
                    row["verdict"] = "PASS"
                else:
                    row["detail"] += "; допуск превышен"
            except (ArithmeticError, ValueError) as exc:
                row["detail"] = str(exc)
            row["elapsed_s"] = f"{perf_counter() - start:.6f}"
            try:
                append_log(root / "results" / "week6_runs.csv", row)
            except (OSError, csv.Error, ValueError) as exc:
                print(f"sweep: НЕ ПРИНЯТО; не удалось записать seed={seed}, s={shift_text}: {exc}")
                return 1
            print(f"{seed:4d}  {shift_text:>11}  {row['n']:3d}  "
                  f"{row['metric'] or '—':>17}  {row['verdict']}")
            if row["verdict"] == "PASS":
                passed += 1
            else:
                print(f"      {row['detail']}")
    total = len(seeds) * len(shifts)
    ok = passed == total
    print(f"sweep: {'ПРИНЯТО' if ok else 'НЕ ПРИНЯТО'}; "
          f"{passed}/{total} прогонов прошли допуск 1e-8 mag.")
    return 0 if ok else 1


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("dry", "smoke", "sweep"))
    stage = cli.parse_args().stage
    raise SystemExit({"dry": dry, "smoke": smoke, "sweep": sweep}[stage](ROOT))
