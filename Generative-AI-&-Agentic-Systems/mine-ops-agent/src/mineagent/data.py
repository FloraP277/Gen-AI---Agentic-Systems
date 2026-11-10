"""Read-only access to the Mine Safety Lakehouse tables and the regulations corpus.

DATA_DIR holds Parquet tables from the lakehouse project (local mode writes them
directly; on Databricks, export them as shown in the README):
    silver_mines/  silver_accidents/  gold_mine_quarterly_safety/
REGS_DIR holds the Regulations-to-Vectors local store (chunks.parquet, sections.json).
"""
import json
import os
import re
from functools import cached_property

import pandas as pd

MINE_ID = re.compile(r"^\d{7}$")


class ToolInputError(ValueError):
    """Bad arguments from the model; returned to it as an error message."""


def check_mine_id(mine_id: str) -> str:
    mine_id = str(mine_id).strip()
    if mine_id.isdigit() and len(mine_id) < 7:
        mine_id = mine_id.zfill(7)
    if not MINE_ID.match(mine_id):
        raise ToolInputError(f"mine_id must be 7 digits, got {mine_id!r}")
    return mine_id


class MineData:
    def __init__(self, data_dir: str, regs_dir: str | None = None):
        self.data_dir, self.regs_dir = data_dir, regs_dir

    def _table(self, name):
        return pd.read_parquet(os.path.join(self.data_dir, name))

    @cached_property
    def mines(self):
        return self._table("silver_mines").set_index("MINE_ID")

    @cached_property
    def quarterly(self):
        q = self._table("gold_mine_quarterly_safety")
        return q.assign(q_index=q["year"] * 4 + q["quarter"] - 1)

    @cached_property
    def accidents(self):
        a = self._table("silver_accidents")
        return a.assign(ACCIDENT_DT=pd.to_datetime(a["ACCIDENT_DT"]))

    @cached_property
    def latest_q_index(self) -> int:
        return int(self.quarterly["q_index"].max())

    # ---- tools ---------------------------------------------------------------

    def find_mines(self, state: str | None = None, name_contains: str | None = None, limit: int = 10) -> list[dict]:
        if not 1 <= limit <= 50:
            raise ToolInputError("limit must be between 1 and 50")
        m = self.mines.reset_index()
        if state:
            m = m[m["STATE"] == state.upper()]
        if name_contains:
            m = m[m["CURRENT_MINE_NAME"].str.contains(name_contains, case=False, na=False)]
        cols = ["MINE_ID", "CURRENT_MINE_NAME", "STATE", "CURRENT_MINE_TYPE", "CURRENT_MINE_STATUS"]
        return m[cols].head(limit).rename(columns=str.lower).to_dict("records")

    def get_mine_profile(self, mine_id: str) -> dict:
        mine_id = check_mine_id(mine_id)
        if mine_id not in self.mines.index:
            raise ToolInputError(f"no mine with id {mine_id}")
        r = self.mines.loc[mine_id]
        return {"mine_id": mine_id, "name": r["CURRENT_MINE_NAME"], "state": r["STATE"],
                "mine_type": r["CURRENT_MINE_TYPE"], "status": r["CURRENT_MINE_STATUS"],
                "commodity": r.get("PRIMARY_CANVASS"), "operator": r["CURRENT_OPERATOR_NAME"],
                "employees": None if pd.isna(r["NO_EMPLOYEES"]) else int(r["NO_EMPLOYEES"])}

    def get_violation_history(self, mine_id: str, quarters: int = 8) -> dict:
        mine_id = check_mine_id(mine_id)
        if not 1 <= quarters <= 40:
            raise ToolInputError("quarters must be between 1 and 40")
        q = self.quarterly
        q = q[(q["MINE_ID"] == mine_id) & (q["q_index"] > self.latest_q_index - quarters)].sort_values("q_index")
        rows = [{"quarter": f"{int(r.year)}-Q{int(r.quarter)}", "inspections": int(r.inspections),
                 "violations": int(r.violations), "ss_violations": int(r.ss_violations),
                 "proposed_penalty": round(float(r.proposed_penalty), 2)} for r in q.itertuples()]
        return {"mine_id": mine_id, "quarters": quarters, "by_quarter": rows,
                "total_violations": sum(r["violations"] for r in rows),
                "total_ss_violations": sum(r["ss_violations"] for r in rows),
                "total_proposed_penalty": round(sum(r["proposed_penalty"] for r in rows), 2)}

    def get_accident_summary(self, mine_id: str, years: int = 5) -> dict:
        mine_id = check_mine_id(mine_id)
        if not 1 <= years <= 20:
            raise ToolInputError("years must be between 1 and 20")
        a = self.accidents
        latest_year = int(self.quarterly["year"].max())
        a = a[(a["MINE_ID"] == mine_id) & (a["ACCIDENT_DT"].dt.year > latest_year - years)]
        top = lambda col: a[col].fillna("NOT REPORTED").value_counts().head(3).to_dict()   # noqa: E731
        return {"mine_id": mine_id, "years": years, "accidents": int(len(a)),
                "injuries": int(a["NO_INJURIES"].fillna(0).sum()), "days_lost": int(a["DAYS_LOST"].fillna(0).sum()),
                "top_accident_types": top("ACCIDENT_TYPE"), "top_equipment": top("MINING_EQUIP"),
                "serious": int(a["DEGREE_INJURY"].fillna("").str.upper().str.contains("FATAL|PERM").sum())}

    @cached_property
    def _regs(self):
        from rank_bm25 import BM25Okapi
        chunks = pd.read_parquet(os.path.join(self.regs_dir, "chunks.parquet"))
        chunks = chunks[chunks["strategy"] == "section"].reset_index(drop=True)
        sections = json.load(open(os.path.join(self.regs_dir, "sections.json")))
        return chunks, sections, BM25Okapi([_tok(t) for t in chunks["text"]])

    def search_regulations(self, query: str, part: str | None = None, k: int = 3) -> list[dict]:
        if not self.regs_dir:
            raise ToolInputError("regulations corpus not configured")
        if not query.strip() or not 1 <= k <= 10:
            raise ToolInputError("query must be non-empty and k between 1 and 10")
        chunks, sections, bm25 = self._regs
        q = _tok(query)
        scores = bm25.get_scores(q)
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        out = []
        for i in order:
            c = chunks.iloc[i]
            if len(out) == k:
                break
            if part and str(c["part"]) != str(part):
                continue
            if not set(q) & set(_tok(c["text"])):   # no shared words: not a match
                continue
            meta = sections.get(c["section_id"], {})
            out.append({"section": c["section_id"], "title": meta.get("title", ""),
                        "url": meta.get("url", ""), "text": c["text"][:800]})
        return out


def _tok(text: str) -> list[str]:
    return re.findall(r"\d+\.\d+|[a-z0-9]+", text.lower())
