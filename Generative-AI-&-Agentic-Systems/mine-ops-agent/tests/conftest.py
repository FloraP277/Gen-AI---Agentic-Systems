import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def dirs(tmp_path_factory):
    """Small SYNTHETIC tables in the lakehouse's format, plus a tiny regulations store."""
    root = tmp_path_factory.mktemp("data")
    data, regs = root / "lakehouse", root / "regs"
    for d in ("silver_mines", "gold_mine_quarterly_safety", "silver_accidents"):
        (data / d).mkdir(parents=True)
    regs.mkdir()

    pd.DataFrame({
        "MINE_ID": ["0100003", "0200001"], "CURRENT_MINE_NAME": ["Red Hill Quarry", "Deep Seam No. 2"],
        "STATE": ["WV", "KY"], "CURRENT_MINE_TYPE": ["Surface", "Underground"],
        "CURRENT_MINE_STATUS": ["Active", "Active"], "PRIMARY_CANVASS": ["Stone", "Coal"],
        "CURRENT_OPERATOR_NAME": ["Operator A", "Operator B"], "NO_EMPLOYEES": [40, 210],
    }).to_parquet(data / "silver_mines" / "part-0.parquet")

    rows = []
    for mine, base in (("0100003", 1), ("0200001", 4)):
        for i, (y, q) in enumerate([(2024, 1), (2024, 2), (2024, 3), (2024, 4), (2025, 1), (2025, 2),
                                    (2025, 3), (2025, 4), (2026, 1), (2026, 2)]):
            rows.append({"MINE_ID": mine, "year": y, "quarter": q, "inspections": 2, "violations": base + i,
                         "ss_violations": (base + i) // 3, "proposed_penalty": 100.0 * (base + i),
                         "accidents": 0, "injuries": 0, "days_lost": 0})
    pd.DataFrame(rows).to_parquet(data / "gold_mine_quarterly_safety" / "part-0.parquet")

    pd.DataFrame({
        "MINE_ID": ["0200001"] * 4 + ["0100003"],
        "ACCIDENT_DT": pd.to_datetime(["2025-02-01", "2025-06-10", "2026-03-03", "2019-01-01", "2026-01-15"]),
        "ACCIDENT_TYPE": ["Struck by", "Struck by", "Fall from machine", "Struck by", "Over-exertion"],
        "MINING_EQUIP": ["Scoop", "Scoop", "Roof bolter", "Scoop", None],
        "DEGREE_INJURY": ["DAYS AWAY FROM WORK ONLY", "FATALITY", "DAYS AWAY FROM WORK ONLY", "X", "NO DYS AWY"],
        "NO_INJURIES": [1, 1, 1, 1, 1], "DAYS_LOST": [10, 0, 5, 3, 0],
    }).to_parquet(data / "silver_accidents" / "part-0.parquet")

    sections = {"75.1725": {"title": "Machinery and equipment; operation and maintenance", "url": "u1", "part": "75"},
                "56.14130": {"title": "Roll-over protective structures and seat belts", "url": "u2", "part": "56"}}
    pd.DataFrame({
        "chunk_id": ["section:75.1725:0", "section:56.14130:0"], "section_id": list(sections),
        "part": ["75", "56"], "strategy": ["section", "section"], "position": [0, 0],
        "text": ["§ 75.1725 Machinery shall be maintained; scoop and mobile equipment struck by hazards.",
                 "§ 56.14130 Roll-over protective structures; seat belts; fall from machine."],
        "n_words": [12, 10],
    }).to_parquet(regs / "chunks.parquet")
    json.dump(sections, open(regs / "sections.json", "w"))
    return str(data), str(regs)


@pytest.fixture
def mine_data(dirs):
    from mineagent.data import MineData
    return MineData(*dirs)
