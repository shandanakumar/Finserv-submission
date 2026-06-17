"""
src/ingestion/version_registry.py

Version-aware document registry.
Tracks every regulation, every version, effective dates.
Enables:
  - "What changed between PSL 2020 and PSL 2025?"
  - "Show me the rule as it was in 2022"
  - "Which chunks are superseded?"
"""

from dataclasses import dataclass
from typing import Optional
from datetime import date


@dataclass
class RegulationVersion:
    """
    One version of one regulation.
    Every PDF you ingest maps to exactly one RegulationVersion.
    """
    regulation_id:   str          # e.g. "RBI_KYC"
    regulation_name: str          # e.g. "RBI Master Direction - Know Your Customer"
    jurisdiction:    str          # "IN" / "EU" / "GLOBAL"
    version:         str          # e.g. "v1", "v2", "2025-11-28"
    effective_from:  str          # ISO date "2025-11-28"
    effective_to:    Optional[str]# None means currently active
    source_url:      str          # direct PDF URL
    status:          str          # "active" | "superseded" | "withdrawn"
    filename:        str          # local filename in real_docs/


# ── Master registry of all documents ─────────────────────────
# Add each PDF you download here before ingesting
REGULATION_REGISTRY: list[RegulationVersion] = [

    # ── KYC ──────────────────────────────────────────────────
    RegulationVersion(
        regulation_id   = "RBI_KYC",
        regulation_name = "RBI Master Direction - Know Your Customer (KYC)",
        jurisdiction    = "IN",
        version         = "v1_2016",
        effective_from  = "2016-02-25",
        effective_to    = "2025-11-28",
        source_url      = "https://www.rbi.org.in/commonman/Upload/English/Notification/PDFs/MD18KYCF6E92C82E1E1419D87323E3869BC9F13.pdf",
        status          = "superseded",
        filename        = "RBI_KYC_2016_v1.pdf",
    ),
    RegulationVersion(
        regulation_id   = "RBI_KYC",
        regulation_name = "RBI Master Direction - Know Your Customer (KYC) 2025",
        jurisdiction    = "IN",
        version         = "v2_2025",
        effective_from  = "2025-11-28",
        effective_to    = None,       # currently active
        source_url      = "https://website.rbi.org.in/web/rbi/-/notifications/reserve-bank-of-india-commercial-banks-know-your-customer-directions-2025",
        status          = "active",
        filename        = "RBI_KYC_2025_v2.pdf",
    ),

    # ── PSL ──────────────────────────────────────────────────
    RegulationVersion(
        regulation_id   = "RBI_PSL",
        regulation_name = "RBI Master Direction - Priority Sector Lending",
        jurisdiction    = "IN",
        version         = "v1_2020",
        effective_from  = "2020-09-04",
        effective_to    = "2025-04-01",
        source_url      = "https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=11959&Mode=0",
        status          = "superseded",
        filename        = "RBI_PSL_2020_v1.pdf",
    ),
    RegulationVersion(
        regulation_id   = "RBI_PSL",
        regulation_name = "RBI Master Directions - Priority Sector Lending 2025",
        jurisdiction    = "IN",
        version         = "v2_2025",
        effective_from  = "2025-04-01",
        effective_to    = None,
        source_url      = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12799",
        status          = "active",
        filename        = "RBI_PSL_2025_v2.pdf",
    ),

    # ── Basel III ────────────────────────────────────────────
    RegulationVersion(
        regulation_id   = "BASEL_III",
        regulation_name = "Basel III: A Global Regulatory Framework (Original)",
        jurisdiction    = "GLOBAL",
        version         = "v1_2010",
        effective_from  = "2010-12-01",
        effective_to    = "2017-12-07",
        source_url      = "https://www.bis.org/publ/bcbs189.pdf",
        status          = "superseded",
        filename        = "BIS_Basel3_2010_v1.pdf",
    ),
    RegulationVersion(
        regulation_id   = "BASEL_III",
        regulation_name = "Basel III: Finalising Post-Crisis Reforms (d424)",
        jurisdiction    = "GLOBAL",
        version         = "v2_2017",
        effective_from  = "2017-12-07",
        effective_to    = None,          # still active — no replacement yet
        source_url      = "https://www.bis.org/bcbs/publ/d424.pdf",
        status          = "active",
        filename        = "BIS_Basel3_d424_v2.pdf",
    ),

    # ── MiFID II ─────────────────────────────────────────────
    RegulationVersion(
        regulation_id   = "MIFID_II",
        regulation_name = "MiFID II Directive 2014/65/EU",
        jurisdiction    = "EU",
        version         = "v1_2014",
        effective_from  = "2014-06-12",
        effective_to    = None,
        source_url      = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32014L0065",
        status          = "active",
        filename        = "MiFID CELEX_32014L0065_EN_TXT.pdf",
    ),

    # ── FATF ─────────────────────────────────────────────────
  
    RegulationVersion(
        regulation_id   = "FATF_AML",
        regulation_name = "FATF Recommendations - AML/CFT Updated 2025",
        jurisdiction    = "GLOBAL",
        version         = "v2_2025",
        effective_from  = "2023-10-01",
        effective_to    = None,
        source_url      = "https://www.fatf-gafi.org",
        status          = "active",
        filename        = "FATF Recommendations 2012.pdf",  # same file, update when new PDF available
    ),

    RegulationVersion(
        regulation_id   = "RBI_KYC",
        regulation_name = "RBI Commercial Banks KYC Directions 2025",
        jurisdiction    = "IN",
        version         = "v2_2025",
        effective_from  = "2025-11-28",
        effective_to    = None,
        source_url      = "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/169MD.pdf",
        status          = "active",
        filename        = "RBI_KYC_CommercialBanks_2025_v2.pdf",
        ),
        RegulationVersion(
            regulation_id   = "RBI_KYC",
            regulation_name = "RBI Master Direction KYC (Clean Reference 2024)",
            jurisdiction    = "IN",
            version         = "v1_5_2024",
            effective_from  = "2024-04-01",
            effective_to    = None,
            source_url      = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566",
            status          = "active",
            filename        = "RBI_KYC_2024_003_v5.txt",
        ),
]


def get_version(filename: str) -> Optional[RegulationVersion]:
    """Look up registry entry by filename."""
    for r in REGULATION_REGISTRY:
        if r.filename == filename:
            return r
    return None


def get_active_versions() -> list[RegulationVersion]:
    """Return only currently active regulations."""
    return [r for r in REGULATION_REGISTRY if r.status == "active"]


def get_all_versions(regulation_id: str) -> list[RegulationVersion]:
    """Return all versions of a regulation sorted by date."""
    versions = [r for r in REGULATION_REGISTRY if r.regulation_id == regulation_id]
    return sorted(versions, key=lambda r: r.effective_from)


def get_superseded_by(regulation_id: str, version: str) -> Optional[RegulationVersion]:
    """Find what replaced a given version."""
    all_v = get_all_versions(regulation_id)
    for i, v in enumerate(all_v):
        if v.version == version and i + 1 < len(all_v):
            return all_v[i + 1]
    return None