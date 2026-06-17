"""
src/ingestion/table_chunker.py

Hybrid table extractor for regulatory PDFs.
Uses Camelot (lattice) for grid tables + pdfplumber fallback.
Converts table rows to natural language sentences for better RAG retrieval.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TableChunker:

    def __init__(self):
        pass

    def extract_tables(self, file_path: str) -> list[dict]:
        """
        Hybrid extractor:
        1. Camelot lattice — best for RBI/BIS grid tables with visible borders
        2. pdfplumber fallback — for borderless tables
        Returns list of {table: rows, page: page_num}
        """
        tables = []

        # ── Camelot lattice (primary) ─────────────────────────
        try:
            import camelot
            camelot_tables = camelot.read_pdf(
                file_path,
                flavor="lattice",
                pages="all",
                strip_text="\n",
            )
            for t in camelot_tables:
                rows = t.df.fillna("").values.tolist()
                page = int(t.page)
                tables.append({"table": rows, "page": page, "source": "camelot"})
            logger.info(f"Camelot found {len(tables)} tables in {file_path}")
        except Exception as e:
            logger.warning(f"Camelot failed: {e} — trying pdfplumber")

        # ── pdfplumber fallback ───────────────────────────────
        if not tables:
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        extracted = page.extract_tables()
                        if extracted:
                            for t in extracted:
                                tables.append({
                                    "table": t,
                                    "page": page_num,
                                    "source": "pdfplumber"
                                })
                logger.info(f"pdfplumber found {len(tables)} tables in {file_path}")
            except Exception as e:
                logger.warning(f"pdfplumber also failed: {e}")

        return tables

    def clean_cell(self, cell) -> str:
        if not cell:
            return ""
        return re.sub(r"\s+", " ", str(cell)).strip()

    def _find_data_start(self, cleaned: list[list]) -> int:
        """
        Find where data rows start — handles multi-row headers.
        Data rows have a non-empty first column that looks like a category label.
        """
        skip_labels = {
            "", "categories", "category", "sr no", "sr. no",
            "s.no", "particulars", "para no", "para", "item",
        }
        for i, row in enumerate(cleaned):
            first_cell = row[0].lower().strip() if row else ""
            if first_cell and first_cell not in skip_labels:
                return i
        return 1

    def normalize_table(self, raw_table: list) -> Optional[dict]:
        """
        Convert raw table rows → structured dict with merged headers.
        Handles multi-row headers common in RBI/BIS documents.
        """
        if not raw_table:
            return None

        cleaned = [
            [self.clean_cell(c) for c in row]
            for row in raw_table
        ]
        cleaned = [row for row in cleaned if any(row)]

        if len(cleaned) < 2:
            return None

        data_start = self._find_data_start(cleaned)
        header_rows = cleaned[:data_start]
        body        = cleaned[data_start:]

        if not body:
            return None

        # Merge multi-row headers column by column
        num_cols = max(len(row) for row in cleaned)
        headers  = []
        for col_idx in range(num_cols):
            parts = [
                r[col_idx]
                for r in header_rows
                if col_idx < len(r) and r[col_idx]
            ]
            header_text = " ".join(parts).strip()
            headers.append(header_text if header_text else f"col_{col_idx}")

        # Build row dicts
        rows = []
        for row in body:
            row_dict = {}
            for col_idx, col_name in enumerate(headers):
                if col_idx < len(row) and row[col_idx]:
                    row_dict[col_name] = row[col_idx]
            if row_dict:
                rows.append(row_dict)

        return {"columns": headers, "rows": rows}

    def _row_to_sentence(self, row: dict, file_path: str) -> str:
        """
        Convert a table row dict to a natural language sentence.
        Uses document context to generate meaningful sentences.
        """
        fname = file_path.upper() if file_path else ""

        # Detect document type for context
        if "PSL" in fname:
            context = "Priority Sector Lending target"
        elif "KYC" in fname:
            context = "KYC requirement"
        elif "BASEL" in fname or "BIS" in fname or "D424" in fname:
            context = "Basel III capital requirement"
        elif "MIFID" in fname or "CELEX" in fname:
            context = "MiFID II requirement"
        elif "FATF" in fname:
            context = "FATF AML/CFT requirement"
        else:
            context = "regulatory requirement"

        # Get label (first non-empty value)
        values = list(row.values())
        keys   = list(row.keys())

        if not values:
            return ""

        label = values[0] if values else ""
        # Use second column as primary value (domestic banks / main requirement)
        value = values[1] if len(values) > 1 else values[0]

        if not label or not value:
            return ""

        # Skip rows that are sub-headers or notes
        skip_patterns = [
            "not applicable", "n/a", "nil", "—", "-",
            "see para", "as per", "refer to",
        ]
        if value.lower() in skip_patterns:
            return ""

        # Generate natural language sentence
        return f"The {label} {context} is {value}."

    def chunk_table(
        self,
        table_json: dict,
        table_name: str,
        page_number: int,
        file_path: str = "",
    ) -> list[dict]:
        """
        Convert table JSON → row-level RAG chunks with natural language text.
        """
        chunks = []
        for row in table_json["rows"]:
            # Natural language sentence — embeds and retrieves much better
            sentence = self._row_to_sentence(row, file_path)

            # Also keep structured text as fallback
            structured = " | ".join(
                f"{k}: {v}" for k, v in row.items() if v
            )

            text = sentence if sentence else structured
            if not text.strip():
                continue

            chunks.append({
                "table_name":  table_name,
                "page":        page_number,
                "columns":     table_json["columns"],
                "row":         row,
                "text":        text,
                "structured":  structured,
            })

        return chunks

    def process_pdf(self, file_path: str) -> list[dict]:
        """
        Full pipeline: PDF → tables → normalize → row chunks
        Returns list of chunk dicts with 'text' ready for embedding.
        """
        raw_tables = self.extract_tables(file_path)
        if not raw_tables:
            return []

        all_chunks = []
        for idx, raw in enumerate(raw_tables):
            table_json = self.normalize_table(raw["table"])
            if not table_json:
                continue

            chunks = self.chunk_table(
                table_json=table_json,
                table_name=f"table_{idx + 1}_page_{raw['page']}",
                page_number=raw["page"],
                file_path=file_path,
            )
            all_chunks.extend(chunks)

        logger.info(
            f"TableChunker: {len(raw_tables)} tables → {len(all_chunks)} row chunks"
        )
        return all_chunks