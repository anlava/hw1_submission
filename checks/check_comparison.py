#!/usr/bin/env python3
"""Check the Cavallo/Bossini reconstruction using raw inputs, not src/.

Install beside the existing checks/check.py. Standard library only. The old
checker remains unchanged and is run as part of this check. PNG checks validate
headers only; a person must inspect the plotted data, axes and legend.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# Avoid creating __pycache__ in the deliberately read-only checks directory.
sys.dont_write_bytecode = True
import check as baseline


PARAM_FILE = "m67_parameters_bossini2019.tsv"
MODEL_FILE = "m67_parsec12s_bossini2019_av0115.dat"
OUTPUT_FILE = "m67_isochrone_bossini2019_apparent.csv"
PARAM_COLUMNS = ["Cluster", "logA", "Dist", "AV", "Fe/H"]


def read_bossini_parameters(raw):
    path = raw / PARAM_FILE
    matches = []
    for line_no, row in baseline.read_vizier(path, PARAM_COLUMNS):
        values = {
            key: baseline.number(row[key], f"{path.name}:{line_no}:{key}")
            for key in PARAM_COLUMNS[1:]
        }
        if row["Cluster"] == "NGC2682":
            matches.append(values)
    baseline.require(len(matches) == 1,
                     f"{path.name}: expected exactly one NGC2682 row, found {len(matches)}")
    params = matches[0]
    baseline.require(params["AV"] >= 0, f"{path.name}: negative AV")
    return params


def check_common_metadata(path):
    """Validate the settings visible in the two user-supplied CMD headers."""
    comments = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.lstrip().startswith("#")]
    checks = [
        (r"^#\s*isochrones based on PARSEC release v1\.2S\s*\+\s*"
         r"COLIBRI S_37\s*\+\s*S_35\s*\+\s*PR16\s*$", "PARSEC/COLIBRI"),
        (r"^#\s*Photometric system:\s*Gaia EDR3\s*\(", "Gaia EDR3"),
        (r"^#\s*Using YBC version of bolometric corrections\b", "YBC"),
        (r"^#\s*O-rich circumstellar dust ignored\s*$", "O-rich dust"),
        (r"^#\s*C-rich circumstellar dust ignored\s*$", "C-rich dust"),
        (r"^#.*photometry includes extinction of\s+Av\s*=.*"
         r"star-by-star.*Cardelli et al 89\s*\+\s*O'Donnell 94\s+"
         r"Rv\s*=\s*3\.1(?:\s|$)", "extinction prescription"),
        (r"^#\s*isochrone terminated\s*$", "termination marker"),
    ]
    for pattern, description in checks:
        count = sum(re.search(pattern, line) is not None for line in comments)
        baseline.require(count == 1,
                         f"{path.name}: expected one {description} metadata entry, got {count}")


def expected_bossini_model(raw, params):
    path = raw / MODEL_FILE
    header = None
    extinction = []
    records = []
    terminated = False
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            location = f"{path.name}:{line_no}"
            if text.startswith("#"):
                names = text[1:].split()
                if names and names[0] == "Zini":
                    baseline.require(header is None, f"{location}: repeated model header")
                    baseline.require(len(names) == 31 and len(set(names)) == 31,
                                     f"{location}: expected 31 unique column names")
                    baseline.require(
                        {"Mini", "MH", "logAge", "logL", "label", "Gmag", "G_BPmag", "G_RPmag"}
                        <= set(names), f"{location}: missing model columns")
                    header = names
                if text[1:].strip() == "isochrone terminated":
                    terminated = True
                match = re.search(
                    r"photometry includes extinction of\s+Av\s*=\s*"
                    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
                    text, flags=re.IGNORECASE)
                if match:
                    extinction.append(baseline.number(match.group(1), location))
                continue
            baseline.require(not terminated, f"{location}: data after termination marker")
            baseline.require(header is not None, f"{location}: data before header")
            fields = text.split()
            baseline.require(len(fields) == len(header),
                             f"{location}: {len(fields)} fields, expected {len(header)}")
            row = {key: baseline.number(value, f"{location}:{key}")
                   for key, value in zip(header, fields)}
            baseline.require(row["label"].is_integer(), f"{location}: non-integer label")
            records.append((line_no, row))
    baseline.require(records, f"{path.name}: empty model")
    baseline.require(len(extinction) == 1, f"{path.name}: expected one Av metadata entry")
    baseline.close(extinction[0], params["AV"], f"{path.name}: Av versus AV")
    baseline.require(len({row["logAge"] for _, row in records}) == 1,
                     f"{path.name}: multiple ages")
    baseline.require(len({row["MH"] for _, row in records}) == 1,
                     f"{path.name}: multiple metallicities")
    check_common_metadata(path)
    kept = []
    removed = 0
    for line_no, row in records:
        location = f"{path.name}:{line_no}"
        baseline.close(row["logAge"], params["logA"], f"{location}: logAge versus logA")
        baseline.close(row["MH"], params["Fe/H"], f"{location}: MH versus Fe/H")
        if row["label"] == 9 and row["logL"] == -9.999:
            removed += 1
            continue
        kept.append({
            **{key: row[key] for key in baseline.ISO_COLUMNS if key not in ("G_app", "BP_RP")},
            "G_app": row["Gmag"] + params["Dist"],
            "BP_RP": row["G_BPmag"] - row["G_RPmag"],
        })
    baseline.require(kept, f"{path.name}: no model points after terminal-point removal")
    return len(records), removed, kept


def validate(project, results):
    old_message = baseline.validate(project, results)
    raw = project / "data" / "raw"
    check_common_metadata(raw / "m67_parsec12s_av018.dat")
    params = read_bossini_parameters(raw)
    n_raw, n_removed, model = expected_bossini_model(raw, params)
    baseline.compare_csv(results / OUTPUT_FILE, baseline.ISO_COLUMNS, model)
    baseline.check_png(results / "m67_cmd_comparison.png")
    return (
        old_message + "\n"
        f"принято: Bossini — модельных точек {n_raw}, исключено {n_removed}, "
        f"сохранено {len(model)}; CSV сверена построчно. "
        f"logAge={params['logA']}, [M/H]={params['Fe/H']}, "
        f"Av={params['AV']}, (m-M)_0={params['Dist']}.\n"
        "Общая выборка звёзд и обе модельные CSV проверены. "
        "У PNG проверены только заголовки; графики осматривает человек."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    results = args.results_dir if args.results_dir is not None else args.project_dir / "results"
    try:
        message = validate(args.project_dir, results)
    except (baseline.InvalidResult, OSError, UnicodeError, csv.Error) as exc:
        print(f"НЕ ПРИНЯТО: {exc}", file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
