"""Download and parse federal regulations from eCFR (ecfr.gov).

eCFR serves each title as XML. Inside it, parts are <DIV5 TYPE="PART">,
subparts are <DIV6 TYPE="SUBPART"> and sections are <DIV8 TYPE="SECTION">,
each with a <HEAD> heading and <P> paragraphs.
"""
import hashlib
import json
import urllib.request
from dataclasses import dataclass, field

from lxml import etree

ECFR = "https://www.ecfr.gov/api/versioner/v1"


@dataclass
class Section:
    section_id: str          # e.g. "75.400"
    part: str                # e.g. "75"
    subpart: str             # e.g. "Subpart E"
    title: str               # heading without the section number
    paragraphs: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)

    @property
    def content_hash(self) -> str:
        """Fingerprint of the section's content; changes when the rule changes."""
        return hashlib.sha256(f"{self.title}\n{self.text}".encode()).hexdigest()

    @property
    def url(self) -> str:
        return f"https://www.ecfr.gov/current/title-30/section-{self.section_id}"


def latest_date(title: int = 30) -> str:
    """Date of the most recent eCFR update for a title."""
    with urllib.request.urlopen(f"{ECFR}/titles.json", timeout=60) as r:
        titles = json.load(r)["titles"]
    return next(t["up_to_date_as_of"] for t in titles if t["number"] == title)


def download_part(part: str, date: str, title: int = 30) -> bytes:
    url = f"{ECFR}/full/{date}/title-{title}.xml?part={part}"
    req = urllib.request.Request(url, headers={"User-Agent": "regs-to-vectors"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _clean(text: str) -> str:
    return " ".join(text.split())


def _heading(div) -> str:
    head = div.find("HEAD")
    return _clean("".join(head.itertext())) if head is not None else ""


def parse_sections(xml: bytes) -> list[Section]:
    """Every section in an eCFR XML document, in document order."""
    root = etree.fromstring(xml, parser=etree.XMLParser(recover=True, huge_tree=True))
    sections = []
    for div in root.iter("DIV8"):
        if div.get("TYPE") != "SECTION":
            continue
        section_id = div.get("N", "").replace("§", "").strip()
        part_div = next((a for a in div.iterancestors("DIV5") if a.get("TYPE") == "PART"), None)
        subpart_div = next((a for a in div.iterancestors("DIV6") if a.get("TYPE") == "SUBPART"), None)
        heading = _heading(div)
        title = heading.replace("§", "").replace(section_id, "", 1).strip(" .")
        paragraphs = [_clean("".join(p.itertext())) for p in div.iter("P")]
        paragraphs = [p for p in paragraphs if p]
        if not section_id or not paragraphs:
            continue   # reserved or empty sections carry no content
        sections.append(Section(
            section_id=section_id,
            part=part_div.get("N") if part_div is not None else section_id.split(".")[0],
            subpart=_heading(subpart_div).split("—")[0].strip() if subpart_div is not None else "",
            title=title,
            paragraphs=paragraphs,
        ))
    return sections
