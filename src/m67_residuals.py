"""M67: dry, smoke, sweep, оценка времени (budget) и сравнение на наблюдениях (full)."""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from importlib import metadata
from itertools import pairwise
from math import isfinite, log
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Cavallo": "m67_isochrone_apparent.csv",
    "Bossini": "m67_isochrone_bossini2019_apparent.csv",
}
BINS = ("[15,16)", "[16,17)", "[17,18]")
MIN_N = 20
BOOT_SEEDS = (11, 23, 37)
BOOT_DRAWS = 2000  # на один seed в одном интервале G
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


def observation_bins(path):
    """Общие строки для обеих моделей: один GaiaDR3, одна G, один наблюдаемый цвет."""
    groups = [[] for _ in BINS]
    seen = set()
    for where, row in records(path, ("GaiaDR3", "Gmag", "BP_RP")):
        key = row["GaiaDR3"]
        if not (key.isascii() and key.isdigit()) or key in seen:
            raise ValueError(f"{where}: GaiaDR3 пуст, некорректен или повторён: {key!r}")
        seen.add(key)
        g = number(row["Gmag"], f"{where} Gmag")
        colour = number(row["BP_RP"], f"{where} BP_RP")
        i = bin_index(g)
        if i is not None:
            groups[i].append((key, g, colour))
    return len(seen), groups


def member_counts(path):
    total, groups = observation_bins(path)
    return total, [len(group) for group in groups]


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


def validated_inputs(root):
    """Общая подготовка для budget и full; без вычисления остатков."""
    results = root / "results"
    cmd = [sys.executable, "-B", str(root / "checks" / "check_comparison.py")]
    code = subprocess.run(cmd, cwd=root, check=False).returncode
    if code:
        raise ValueError(f"checks/check_comparison.py: код {code}")
    _, groups = observation_bins(results / "m67_members_selected.csv")
    curves = {}
    for name, filename in MODELS.items():
        try:
            curves[name] = check_coverage(model_points(results / filename))
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
    return groups, curves


def paired_interval(abs_c, abs_b):
    """Интервал для разности медиан |r|. Элементы с одним индексом — одна звезда."""
    import numpy as np

    abs_c = np.asarray(abs_c, dtype=float)
    abs_b = np.asarray(abs_b, dtype=float)
    if abs_c.ndim != 1 or not abs_c.size or abs_c.shape != abs_b.shape:
        raise ValueError("для парного бутстрэпа нужны две непустые колонки одной длины")
    if (not np.isfinite(abs_c).all() or not np.isfinite(abs_b).all()
            or np.any(abs_c < 0) or np.any(abs_b < 0)):
        raise ValueError("в колонки |r| можно передать только конечные неотрицательные значения")
    n = abs_c.size
    samples = np.empty((len(BOOT_SEEDS), BOOT_DRAWS), dtype=float)
    for j, seed in enumerate(BOOT_SEEDS):
        rng = np.random.default_rng(seed)
        for start in range(0, BOOT_DRAWS, 128):
            stop = min(start + 128, BOOT_DRAWS)
            idx = rng.integers(0, n, size=(stop - start, n))
            samples[j, start:stop] = (np.median(abs_c[idx], axis=1)
                                      - np.median(abs_b[idx], axis=1))
    tail = 0.05 / (2 * len(BINS))
    low, high = np.quantile(samples.ravel(), [tail, 1 - tail], method="linear")
    return float(low), float(high)


def summarise_bin(stars, curves):
    """Единая операция для замера и полного расчёта: остатки, медианы и интервал D."""
    import numpy as np

    n = len(stars)
    if n < MIN_N:
        return {"n": n, "status": "недостаточно данных"}
    cav = np.empty(n, dtype=float)
    bos = np.empty(n, dtype=float)
    for j, (_, g, colour) in enumerate(stars):
        cav[j] = float(colour_residual(g, colour, curves["Cavallo"]))
        bos[j] = float(colour_residual(g, colour, curves["Bossini"]))
    if not np.isfinite(cav).all() or not np.isfinite(bos).all():
        raise ValueError("неконечный остаток в паре моделей")
    abs_c, abs_b = np.abs(cav), np.abs(bos)
    med_c, med_b = float(np.median(abs_c)), float(np.median(abs_b))
    low, high = paired_interval(abs_c, abs_b)
    return dict(n=n, status="рассчитано", median_r_cav=float(np.median(cav)),
                median_r_bos=float(np.median(bos)), median_abs_r_cav=med_c,
                median_abs_r_bos=med_b, D=med_c - med_b, ci_low=low, ci_high=high)


def budget_stars(counts):
    """Искусственные звёзды для таймера. Никакой наблюдаемой фотометрии на входе."""
    rng = Random(6706)
    groups = []
    for lo, n in zip((15, 16, 17), counts):
        group = []
        for j in range(n):
            g = lo + Fraction(rng.randrange(1, 10**12), 10**12)
            colour = reference_colour(g) + Fraction(rng.randrange(-1500, 1501), 10000)
            group.append((f"synthetic-{lo}-{j}", g, colour))
        groups.append(group)
    return groups


def budget(root):
    """Два замера масштаба. Для каждого размера — все три интервала и все три seed."""
    try:
        versions = environment()
        if sys.version_info < (3, 11) or "не установлен" in versions.values():
            raise ValueError("по METHOD.md нужен Python 3.11+ с NumPy, pytest и Hypothesis")
        import numpy as np  # импорт и запуск интерпретатора вне таймера

        t0 = perf_counter()
        real_groups, curves = validated_inputs(root)
        counts = tuple(map(len, real_groups))
        t_setup = perf_counter() - t0
        del real_groups  # дальше используются только количества по G и две модельные кривые
        # Для маленькой и большой выборок сохраняем те же три интервала и ветку n >= 20.
        sizes = [tuple(n if n < MIN_N else max(MIN_N, (n + d // 2) // d)
                       for n in counts) for d in (4, 2)]
        n1, n2 = map(sum, sizes)
        target = sum(counts)
        if not 0 < n1 < n2 < target:
            raise ValueError(f"два размера для оценки масштаба не различимы: {n1}, {n2}, {target}")
        large = budget_stars(sizes[1])
        small = [group[:n] for group, n in zip(large, sizes[0])]
        inputs = [small, large]
        for group in small:  # прогрев той же функции; синтетика и прогрев не входят в замеры
            summarise_bin(group, curves)
        times = [[], []]
        seeds = "/".join(map(str, BOOT_SEEDS))
        repeats = 5
        journal = root / "results" / "week6_runs.csv"
        for iteration in range(repeats):
            for k in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                stamp = datetime.now().astimezone().isoformat(timespec="seconds")
                t0 = perf_counter()
                for group in inputs[k]:
                    summarise_bin(group, curves)
                dt = perf_counter() - t0
                if not isfinite(dt) or dt <= 0:
                    raise ValueError("не удалось измерить положительное время")
                times[k].append(dt)
                append_log(journal, dict(timestamp=stamp, stage="budget", seed=seeds,
                           n=sum(sizes[k]), elapsed_s=f"{dt:.9f}", verdict="PASS",
                           detail=f"синтетика seed=6706; G: {sizes[k]}; замер {iteration+1}/{repeats}; "
                                  f"B={BOOT_DRAWS} на каждый bootstrap seed и интервал", **versions))
        t1, t2 = (float(np.median(v)) for v in times)
        kappa = log(t2 / t1) / log(n2 / n1)
        predicted = t2 * (target / n2) ** kappa
        total = t_setup + predicted
        print(f"Подготовка (проверка CSV, чтение и построение ветвей): {t_setup:.6f} с")
        for ns, ts, mid in zip(sizes, times, (t1, t2)):
            print(f"N={sum(ns)}, по G={ns}: медиана {mid:.6f} с "
                  f"({len(ts)} замеров, от {min(ts):.6f} до {max(ts):.6f} с)")
        print(f"κ = {kappa:.3f}")
        if t2 <= t1:
            print("На большем размере медиана времени не выросла; "
                  "по таким замерам показатель роста ненадёжен.")
        elif min(times[1]) <= max(times[0]):
            print("Диапазоны замеров перекрылись; показатель роста может быть чувствителен к шуму.")
        print(f"Прогноз для N={target}, по G={counts}: счёт ≈ {predicted:.6f} с; "
              f"с подготовкой ≈ {total:.6f} с.")
        append_log(journal, dict(timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                   stage="budget_forecast", seed=seeds, n=target, metric=f"{total:.9f}",
                   verdict="PASS", detail=f"прогноз в с; подготовка={t_setup:.9f}; "
                   f"N1={n1}, T1={t1:.9f}; N2={n2}, T2={t2:.9f}; κ={kappa:.6f}; "
                   f"счёт={predicted:.9f}; без запуска Python, импорта и записи итоговой таблицы", **versions))
        print("budget: ПРИНЯТО. Времена записаны в results/week6_runs.csv.")
        return 0
    except (OSError, ImportError, ValueError, ArithmeticError, csv.Error) as exc:
        print(f"budget: НЕ ПРИНЯТО; {exc}")
        return 1


SCIENCE_FIELDS = ("g_bin", "n", "median_r_cav", "median_r_bos", "median_abs_r_cav",
                  "median_abs_r_bos", "D", "ci_low", "ci_high", "status")


def result_row(g_bin, summary):
    row = {"g_bin": g_bin, **summary}
    if row["n"] < MIN_N:
        return row  # численные поля в CSV останутся пустыми
    lo, hi = row["ci_low"], row["ci_high"]
    if not all(isfinite(row[k]) for k in SCIENCE_FIELDS[2:-1]) or hi < lo:
        raise ValueError(f"{g_bin}: некорректная статистика или границы интервала")
    if lo <= 0 <= hi:
        row["status"] = "устойчивого различия не обнаружено"
    else:
        row["status"] = "интервал D выше нуля" if lo > 0 else "интервал D ниже нуля"
    return row


def last_forecast(path, n, versions):
    """Последний прогноз для этого N, набора seed и версий Python/NumPy, если он есть."""
    latest = None
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames != list(LOG_FIELDS):
                raise ValueError(f"{path.name}: несовместимый заголовок журнала")
            for row in reader:
                if (row["stage"] == "budget_forecast" and row["verdict"] == "PASS"
                        and row["n"] == str(n) and row["seed"] == "/".join(map(str, BOOT_SEEDS))
                        and all(row[k] == versions[k] for k in ("python", "numpy"))):
                    try:
                        latest = float(row["metric"])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{path.name}:{reader.line_num}: неверное время прогноза") from exc
                    if not isfinite(latest) or latest <= 0:
                        raise ValueError(f"{path.name}:{reader.line_num}: неверное время прогноза")
    return latest


def write_science_csv(path, rows):
    """Заменяем результат только после записи всех трёх строк во временный файл."""
    tmp = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                prefix=".m67_full_", suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            writer = csv.DictWriter(stream, SCIENCE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def full(root):
    results = root / "results"
    journal = results / "week6_runs.csv"
    output = results / "m67_residuals_by_g.csv"
    versions = environment()
    record = dict(timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                  stage="full", seed="/".join(map(str, BOOT_SEEDS)), verdict="FAIL", **versions)
    try:
        if sys.version_info < (3, 11) or "не установлен" in versions.values():
            raise ValueError("по METHOD.md нужен Python 3.11+ с NumPy, pytest и Hypothesis")
        import numpy  # загрузка до таймера, как в budget

        tick = perf_counter()
        groups, curves = validated_inputs(root)
        counts = tuple(map(len, groups))
        prep = perf_counter() - tick
        record["n"] = sum(counts)
        tick = perf_counter()
        summaries = [summarise_bin(group, curves) for group in groups]
        calc = perf_counter() - tick
        elapsed = prep + calc
        record["elapsed_s"] = f"{elapsed:.9f}"
        rows = [result_row(name, s) for name, s in zip(BINS, summaries, strict=True)]
        prediction = last_forecast(journal, record["n"], versions)
        timing = f"подготовка {prep:.6f} с; счёт {calc:.6f} с; всего {elapsed:.6f} с"
        if prediction is not None:
            comparison = (f"Прогноз {prediction:.6f} с; факт − прогноз {elapsed-prediction:+.6f} с "
                          f"({100*(elapsed/prediction-1):+.1f}%).")
        else:
            comparison = "Сопоставимый прогноз в журнале не найден."
        record["metric"] = json.dumps({r["g_bin"]: r.get("D") for r in rows},
                                      allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        record["detail"] = (f"D в mag по интервалам; n={counts}; B={BOOT_DRAWS} на seed; "
                            f"results/{output.name}; {timing}. {comparison}")
        write_science_csv(output, rows)
        record["verdict"] = "PASS"
    except (ImportError, OSError, ValueError, ArithmeticError, csv.Error) as exc:
        record["detail"] = str(exc)
    try:
        append_log(journal, record)
    except (OSError, ValueError, csv.Error) as exc:
        published = "новая CSV записана" if record["verdict"] == "PASS" else "новой CSV нет"
        print(f"full: НЕ ПРИНЯТО; {published}; {record['detail']}; ошибка журнала: {exc}")
        return 1
    if record["verdict"] != "PASS":
        print(f"full: НЕ ПРИНЯТО; результат за этот запуск не записан: {record['detail']}")
        return 1

    print("G          n    med r_Cav    med r_Bos   med|r|_Cav   med|r|_Bos"
          "           D     интервал D (всё в mag)")
    for row in rows:
        left = f"{row['g_bin']:<8} {row['n']:3d}"
        if row["n"] < MIN_N:
            print(f"{left}  {row['status']}")
        else:
            print(f"{left}  {row['median_r_cav']:+.6f}     {row['median_r_bos']:+.6f}     "
                  f"{row['median_abs_r_cav']:.6f}     {row['median_abs_r_bos']:.6f}   "
                  f"{row['D']:+.6f}   [{row['ci_low']:+.6f}; {row['ci_high']:+.6f}]"
                  f"  {row['status']}")
    print("D = med|r_Cav| − med|r_Bos|. Парный процентильный интервал: "
          "α=0.05 с поправкой на три интервала G.")
    print(f"Факт: {timing}. {comparison}")
    print(f"full: ПРИНЯТО; results/{output.name}; журнал: results/week6_runs.csv.")
    return 0


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    runners = {"dry": dry, "smoke": smoke, "sweep": sweep, "budget": budget, "full": full}
    cli.add_argument("stage", choices=tuple(runners))
    stage = cli.parse_args().stage
    raise SystemExit(runners[stage](ROOT))
