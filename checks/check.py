#!/usr/bin/env python3
"""Independent data/output check for the current M67 reconstruction.

Standard library only. Does not import or run src/plot_m67.py, fit parameters,
change files, or determine whether the physical model fits the cluster.
"""

import argparse
import csv
import math
import re
import struct
import sys
from pathlib import Path


CLUSTER = "NGC_2682"
TOL = 1e-8
MEMBER_COLUMNS = ["Name", "GaiaDR3", "Gmag", "BPmag", "RPmag", "BP_RP"]
ISO_COLUMNS = [
    "Gmag", "G_BPmag", "G_RPmag", "G_app", "BP_RP",
    "Mini", "logAge", "MH", "label",
]
PARAM_COLUMNS = ["Cluster", "logAge50", "[Fe/H]50", "AV50", "dMod50"]
RAW_MEMBER_COLUMNS = ["Name", "GaiaDR3", "Prob", "inrt", "Gmag", "BPmag", "RPmag"]


class InvalidResult(Exception):
    pass


def require(condition, message):
    if not condition:
        raise InvalidResult(message)


def number(text, location, allow_missing=False):
    """Empty photometry is a missing value; arbitrary text is an error."""
    if allow_missing and not text.strip():
        return math.nan
    try:
        value = float(text)
    except ValueError as exc:
        raise InvalidResult(f"{location}: not a number: {text!r}") from exc
    if not allow_missing:
        require(math.isfinite(value), f"{location}: non-finite value")
    return value


def close(actual, expected, location):
    require(
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=0, abs_tol=TOL),
        f"{location}: got {actual!r}, expected {expected!r}",
    )


def read_vizier(path, required_columns):
    """Read one TSV table. Preserve empty first/last fields before trimming."""
    header = None
    units_seen = False
    separator_seen = False
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = [part.strip() for part in line.rstrip("\r\n").split("\t")]
            location = f"{path.name}:{line_no}"
            if header is None:
                require(fields[0] == required_columns[0],
                        f"{location}: expected a TSV header")
                require(len(fields) == len(set(fields)),
                        f"{location}: duplicate column names")
                require(set(required_columns) <= set(fields),
                        f"{location}: missing required columns {required_columns}")
                header = fields
                continue
            require(len(fields) == len(header),
                    f"{location}: {len(fields)} fields, expected {len(header)}")
            if not separator_seen:
                if all(field and set(field) == {"-"} for field in fields):
                    separator_seen = True
                else:
                    require(not units_seen,
                            f"{location}: expected the dashed separator after units")
                    units_seen = True
                continue
            rows.append((line_no, dict(zip(header, fields))))
    require(header is not None and separator_seen and rows,
            f"{path.name}: missing header, separator, or data")
    return rows


def read_parameters(raw):
    path = raw / "m67_parameters_cavallo2024.tsv"
    rows = read_vizier(path, PARAM_COLUMNS)
    matches = []
    for line_no, row in rows:
        values = {
            field: number(row[field], f"{path.name}:{line_no}:{field}")
            for field in PARAM_COLUMNS[1:]
        }
        if row["Cluster"] == CLUSTER:
            matches.append(values)
    require(len(matches) == 1,
            f"{path.name}: expected exactly one {CLUSTER} row, found {len(matches)}")
    require(matches[0]["AV50"] >= 0, f"{path.name}: negative AV50")
    return matches[0]


def expected_members(raw):
    path = raw / "m67_members_hr2023.tsv"
    rows = read_vizier(path, RAW_MEMBER_COLUMNS)
    selected = []
    for line_no, row in rows:
        location = f"{path.name}:{line_no}"
        require(bool(row["Name"]), f"{location}: empty cluster name")
        require(re.fullmatch(r"[0-9]+", row["GaiaDR3"]) is not None,
                f"{location}: GaiaDR3 must contain decimal digits, without float conversion")
        if row["Prob"]:
            number(row["Prob"], f"{location}:Prob", allow_missing=True)
        require(row["inrt"] in ("", "True", "False"),
                f"{location}: unexpected inrt value {row['inrt']!r}")
        magnitudes = {
            field: number(row[field], f"{location}:{field}", allow_missing=True)
            for field in ("Gmag", "BPmag", "RPmag")
        }
        if (row["Name"] == CLUSTER
                and all(math.isfinite(value) for value in magnitudes.values())
                and magnitudes["Gmag"] <= 18.0):
            selected.append({
                "Name": row["Name"], "GaiaDR3": row["GaiaDR3"],
                **magnitudes, "BP_RP": magnitudes["BPmag"] - magnitudes["RPmag"],
            })
    require(selected, f"{path.name}: empty selected sample")
    ids = [row["GaiaDR3"] for row in selected]
    require(len(ids) == len(set(ids)), f"{path.name}: duplicate IDs in the selected sample")
    return len(rows), selected


def expected_isochrone(raw, params):
    path = raw / "m67_parsec12s_av018.dat"
    header = None
    extinction = []
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            location = f"{path.name}:{line_no}"
            if text.startswith("#"):
                names = text[1:].split()
                if names and names[0] == "Zini":
                    require(header is None, f"{location}: repeated isochrone header")
                    require(len(names) == 31 and len(set(names)) == 31,
                            f"{location}: expected 31 unique column names")
                    require(
                        {"Mini", "MH", "logAge", "logL", "label",
                         "Gmag", "G_BPmag", "G_RPmag"} <= set(names),
                        f"{location}: missing required model columns",
                    )
                    header = names
                match = re.search(
                    r"photometry includes extinction of\s+Av\s*=\s*"
                    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
                    text, flags=re.IGNORECASE,
                )
                if match:
                    extinction.append(number(match.group(1), location))
                continue
            require(header is not None, f"{location}: data before column names")
            fields = text.split()
            require(len(fields) == len(header),
                    f"{location}: {len(fields)} fields, expected {len(header)}")
            record = {
                name: number(value, f"{location}:{name}")
                for name, value in zip(header, fields)
            }
            require(record["label"].is_integer(), f"{location}: non-integer label")
            records.append((line_no, record))
    require(records, f"{path.name}: empty isochrone")
    require(len(extinction) == 1, f"{path.name}: expected one explicit Av metadata entry")
    close(extinction[0], params["AV50"], f"{path.name}:Av versus AV50")
    require(len({row["logAge"] for _, row in records}) == 1,
            f"{path.name}: multiple ages")
    require(len({row["MH"] for _, row in records}) == 1,
            f"{path.name}: multiple metallicities")
    kept = []
    excluded = 0
    for line_no, row in records:
        location = f"{path.name}:{line_no}"
        close(row["logAge"], params["logAge50"], f"{location}:logAge")
        close(row["MH"], params["[Fe/H]50"], f"{location}:MH")
        if row["label"] == 9 and row["logL"] == -9.999:
            excluded += 1
            continue
        kept.append({
            **{key: row[key] for key in ISO_COLUMNS if key not in ("G_app", "BP_RP")},
            "G_app": row["Gmag"] + params["dMod50"],
            "BP_RP": row["G_BPmag"] - row["G_RPmag"],
        })
    require(kept, f"{path.name}: no model points after terminal-point removal")
    return len(records), excluded, kept


def read_result(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        require(next(reader, None) == columns,
                f"{path.name}: expected columns in this order: {','.join(columns)}")
        rows = []
        for fields in reader:
            line_no = reader.line_num
            require(len(fields) == len(columns),
                    f"{path.name}:{line_no}: wrong number of fields")
            rows.append((line_no, dict(zip(columns, fields))))
    return rows


def compare_csv(path, columns, expected, text_columns=()):
    actual = read_result(path, columns)
    require(len(actual) == len(expected),
            f"{path.name}: {len(actual)} rows, expected {len(expected)}")
    for (line_no, result), source in zip(actual, expected):
        for field in columns:
            location = f"{path.name}:{line_no}:{field}"
            if field in text_columns:
                require(result[field] == source[field],
                        f"{location}: got {result[field]!r}, expected {source[field]!r}")
            else:
                close(number(result[field], location), source[field], location)


def check_png(path):
    # Only file/header validation; axis direction and plotted data need visual review.
    with path.open("rb") as stream:
        header = stream.read(24)
    require(len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[12:16] == b"IHDR", f"{path.name}: not a PNG header")
    width, height = struct.unpack(">II", header[16:24])
    require(width > 0 and height > 0, f"{path.name}: invalid image dimensions")


def validate(project, results):
    raw = project / "data" / "raw"
    params = read_parameters(raw)
    n_members, members = expected_members(raw)
    n_model, n_terminal, model = expected_isochrone(raw, params)
    compare_csv(results / "m67_isochrone_apparent.csv", ISO_COLUMNS, model)
    compare_csv(results / "m67_members_selected.csv", MEMBER_COLUMNS, members,
                text_columns=("Name", "GaiaDR3"))
    check_png(results / "m67_cmd.png")
    return (
        f"принято: звёзд {n_members}, отобрано {len(members)}; "
        f"модельных точек {n_model}, исключено {n_terminal}, сохранено {len(model)}. "
        "Обе CSV сверены построчно с исходными данными."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-dir", type=Path,
                        help="Optional alternate outputs, e.g. a temporary damaged copy")
    args = parser.parse_args()
    results = args.results_dir if args.results_dir is not None else args.project_dir / "results"
    try:
        message = validate(args.project_dir, results)
    except (InvalidResult, OSError, UnicodeError, csv.Error) as exc:
        print(f"НЕ ПРИНЯТО: {exc}", file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
