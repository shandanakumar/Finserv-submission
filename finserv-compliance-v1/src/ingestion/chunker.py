"""
src/ingestion/chunker.py

Hierarchical semantic chunking for regulatory documents.

Strategy:
1. Structural split: respect section/article/paragraph headingstic
2. Semantic refinement: split oversized sections at semantic boundaries
3. Overlap injection: 10% overlap between adjacent chunks from same section

This preserves regulatory document structure (critical for accurate citation)
and avoids splitting mid-clause, which degrades retrieval accuracy.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from src.ingestion.document_loader import RegulatoryDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    A chunk of a regulatory document with full provenance metadata.
    All metadata fields are indexed in Qdrant for filtered retrieval.
    """
    chunk_id: str           # {doc_id}_chunk_{n:04d}
    text: str
    doc_id: str
    source: str
    regulation_family: str
    jurisdiction: str
    version: str
    status: str
    title: str
    section_number: str     # e.g. "4.2.1", "Article 12", "Para 5"
    section_title: str
    chunk_index: int        # Position within document
    total_chunks: int       # Set after all chunks generated
    parent_section: str     # For parent-document retrieval
    char_start: int         # Character offset in original document
    char_end: int
    effective_date: Optional[str] = None
    supersedes: Optional[str] = None
    overlap_with_prev: bool = False
    overlap_with_next: bool = False


# ─── Heading patterns for common regulatory document formats ─────────────────
# Matches: "Article 12", "Section 4.2.1", "Para 5", "Chapter III", "4.2 Capital Requirements"
HEADING_PATTERNS = [
    # Basel III style: "4.2 Capital Requirements" or "4.2.1 Sub-section"
    re.compile(r'^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{3,80})', re.MULTILINE),
    # Regulatory article style: "Article 12: ..." or "Article 12 —"
    re.compile(r'^(Article\s+\d+[A-Z]?)[\s:\—\-]+([^\n]{0,80})', re.MULTILINE | re.IGNORECASE),
    # Section style: "Section 4" or "Section 4.2"
    re.compile(r'^(Section\s+\d+(?:\.\d+)*)[\s:\—\-]*([^\n]{0,80})', re.MULTILINE | re.IGNORECASE),
    # Paragraph style: "Para 5" or "5."
    re.compile(r'^(Para(?:graph)?\s+\d+|^\d+\.)[\s:\—\-]+([^\n]{0,80})', re.MULTILINE | re.IGNORECASE),
    # Roman numeral parts: "Part III", "Chapter IV"
    re.compile(r'^(Part\s+[IVXLC]+|Chapter\s+[IVXLC]+)[\s:\—\-]*([^\n]{0,80})', re.MULTILINE | re.IGNORECASE),
    # RBI circular style: "A. Applicability" or "I. Background"
    re.compile(r'^([A-Z]\.|[IVX]+\.)[\s]+([A-Z][^\n]{3,80})', re.MULTILINE),
]


class SemanticChunker:
    """
    Chunks regulatory documents while respecting structure and semantic boundaries.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 400,
        min_chunk_tokens: int = 50,
        overlap_tokens: int = 50,
        chars_per_token: float = 4.0,  # Approximate for English regulatory text
    ):
        self.max_chunk_chars = int(max_chunk_tokens * chars_per_token)
        self.min_chunk_chars = int(min_chunk_tokens * chars_per_token)
        self.overlap_chars = int(overlap_tokens * chars_per_token)

    def chunk_document(self, doc: RegulatoryDocument) -> list[DocumentChunk]:
        """
        Main entry point: chunk a RegulatoryDocument into DocumentChunks.
        """
        logger.info(f"Chunking document: {doc.doc_id} ({len(doc.content)} chars)")

        # Stage 1: Structural split by headings
        sections = self._split_by_structure(doc.content)

        # Stage 2: Semantic refinement (split large sections)
        refined_sections = []
        for section in sections:
            if len(section["text"]) > self.max_chunk_chars:
                sub_sections = self._split_large_section(section)
                refined_sections.extend(sub_sections)
            elif len(section["text"]) < self.min_chunk_chars and refined_sections:
                # Merge tiny sections with the previous one
                refined_sections[-1]["text"] += "\n" + section["text"]
            else:
                refined_sections.append(section)

        # Stage 3: Create chunks with overlap
        chunks = self._create_chunks_with_overlap(refined_sections, doc)

        logger.info(f"Chunked {doc.doc_id} into {len(chunks)} chunks")
        return chunks

    def _split_by_structure(self, content: str) -> list[dict]:
        """
        Identify section boundaries from heading patterns.
        Returns list of {section_number, section_title, text, char_start, char_end}.
        """
        # Find all heading positions
        heading_positions = []
        for pattern in HEADING_PATTERNS:
            for match in pattern.finditer(content):
                heading_positions.append({
                    "start": match.start(),
                    "section_number": match.group(1).strip(),
                    "section_title": match.group(2).strip() if match.lastindex >= 2 else "",
                })

        # Sort by position and deduplicate nearby matches (same heading matched by multiple patterns)
        heading_positions.sort(key=lambda x: x["start"])
        heading_positions = self._deduplicate_headings(heading_positions)

        if not heading_positions:
            # No structural headings found — treat entire document as one section
            logger.warning("No structural headings found — using paragraph-based fallback")
            return self._split_by_paragraphs(content)

        # Build sections from heading positions
        sections = []
        for i, heading in enumerate(heading_positions):
            start = heading["start"]
            end = heading_positions[i + 1]["start"] if i + 1 < len(heading_positions) else len(content)
            text = content[start:end].strip()

            if text:
                sections.append({
                    "section_number": heading["section_number"],
                    "section_title": heading["section_title"],
                    "text": text,
                    "char_start": start,
                    "char_end": end,
                    "parent_section": heading["section_number"].split(".")[0] if "." in heading["section_number"] else heading["section_number"],
                })

        # Prepend any content before the first heading as a preamble section
        if heading_positions and heading_positions[0]["start"] > 100:
            preamble = content[:heading_positions[0]["start"]].strip()
            if preamble:
                sections.insert(0, {
                    "section_number": "0",
                    "section_title": "Preamble",
                    "text": preamble,
                    "char_start": 0,
                    "char_end": heading_positions[0]["start"],
                    "parent_section": "0",
                })

        return sections

    def _deduplicate_headings(self, headings: list[dict], min_gap: int = 50) -> list[dict]:
        """Remove duplicate heading detections (same position matched by multiple patterns)."""
        if not headings:
            return []
        deduped = [headings[0]]
        for h in headings[1:]:
            if h["start"] - deduped[-1]["start"] > min_gap:
                deduped.append(h)
        return deduped

    def _split_by_paragraphs(self, content: str) -> list[dict]:
        """Fallback: split by double newlines (paragraph boundaries)."""
        paragraphs = re.split(r'\n\n+', content)
        sections = []
        char_offset = 0
        for i, para in enumerate(paragraphs):
            if para.strip():
                sections.append({
                    "section_number": str(i + 1),
                    "section_title": "",
                    "text": para.strip(),
                    "char_start": char_offset,
                    "char_end": char_offset + len(para),
                    "parent_section": str(i + 1),
                })
            char_offset += len(para) + 2  # +2 for the \n\n separator
        return sections

    def _split_large_section(self, section: dict) -> list[dict]:
        """
        Split a section that exceeds max_chunk_chars.
        Strategy: split at sentence boundaries, grouping sentences until max is reached.
        """
        text = section["text"]
        # Split at sentence boundaries (period + capital letter, or newline)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        sub_sections = []
        current_text = ""
        sub_idx = 0

        for sentence in sentences:
            if len(current_text) + len(sentence) > self.max_chunk_chars and current_text:
                sub_sections.append({
                    **section,
                    "section_number": f"{section['section_number']}.{sub_idx + 1}",
                    "text": current_text.strip(),
                    "char_start": section["char_start"],
                    "char_end": section["char_end"],
                })
                sub_idx += 1
                current_text = sentence
            else:
                current_text = current_text + " " + sentence if current_text else sentence

        if current_text.strip():
            sub_sections.append({
                **section,
                "section_number": f"{section['section_number']}.{sub_idx + 1}" if sub_idx > 0 else section["section_number"],
                "text": current_text.strip(),
                "char_start": section["char_start"],
                "char_end": section["char_end"],
            })

        return sub_sections

    def _create_chunks_with_overlap(
        self, sections: list[dict], doc: RegulatoryDocument
    ) -> list[DocumentChunk]:
        """
        Create final DocumentChunk objects, adding overlap text between adjacent chunks.
        Overlap only added between chunks from the same parent section.
        """
        chunks = []

        for i, section in enumerate(sections):
            # Add overlap from previous chunk (if same parent section)
            overlap_prefix = ""
            if i > 0 and sections[i - 1].get("parent_section") == section.get("parent_section"):
                prev_text = sections[i - 1]["text"]
                overlap_prefix = prev_text[-self.overlap_chars:] + "\n"

            chunk_text = overlap_prefix + section["text"]

            chunk = DocumentChunk(
                chunk_id=f"{doc.doc_id}_chunk_{i:04d}",
                text=chunk_text,
                doc_id=doc.doc_id,
                source=doc.source,
                regulation_family=doc.regulation_family,
                jurisdiction=doc.jurisdiction,
                version=doc.version,
                status=doc.status,
                title=doc.title,
                section_number=section["section_number"],
                section_title=section.get("section_title", ""),
                chunk_index=i,
                total_chunks=len(sections),  # Updated below
                parent_section=section.get("parent_section", section["section_number"]),
                char_start=section["char_start"],
                char_end=section["char_end"],
                effective_date=doc.effective_date,
                supersedes=doc.supersedes,
                overlap_with_prev=bool(overlap_prefix),
                overlap_with_next=i < len(sections) - 1 and sections[i + 1].get("parent_section") == section.get("parent_section"),
            )
            chunks.append(chunk)

        # Update total_chunks now that we know the final count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.ingestion.document_loader import RegulatoryDocument

    # ── Test document — simulates real Basel III PDF text ──────────
    TEST_TEXT = """BASEL III CAPITAL ADEQUACY FRAMEWORK
Bank for International Settlements — 2024

Part I Introduction
This framework establishes minimum capital requirements for internationally
active banks. All requirements are expressed as a percentage of risk-weighted
assets unless otherwise stated.

4.1 Minimum Capital Requirements
Banks must maintain a minimum Common Equity Tier 1 (CET1) ratio of 4.5% of
risk-weighted assets at all times. This requirement applies to all
internationally active banks without exception.

4.2 Capital Conservation Buffer
The capital distribution constraints imposed on G-SIBs will depend on the
G-SIB CET1 risk-weighted ratio and its leverage ratio. A G-SIB which meets
both its CET1 risk-weighted requirements will not be subject to minimum capital
conservation standards. In addition, a Common Equity Tier 1 capital
conservation buffer is set at 2.5% of risk-weighted assets for all banks.
Banks may also be subject to a countercyclical capital buffer requirement.

4.2.1 Distribution Restrictions
Breaching the conservation buffer triggers automatic restrictions on dividends,
share buybacks, and discretionary bonus payments to staff. The restrictions
increase as the buffer is depleted further below the 2.5% threshold.

4.3 Countercyclical Capital Buffer
National authorities may impose an additional buffer ranging from 0% to 2.5%
of risk-weighted assets depending on credit cycle conditions in their
jurisdiction. Banks must meet this requirement with CET1 capital.

Article 12 Large Exposure Limits
No single counterparty exposure may exceed 25% of Tier 1 capital at any time.
For G-SIB to G-SIB exposures the limit is tightened to 15% of Tier 1 capital.

Section 5 Liquidity Requirements
Banks must maintain adequate liquidity buffers at all times to survive a
30-day stress scenario as defined in the LCR framework.

Para 5.1 — Liquidity Coverage Ratio
The minimum LCR is set at 100%. Banks must hold sufficient High Quality
Liquid Assets to cover total net cash outflows over a 30-day stress period.

Chapter IV Operational Risk
The standardised approach for operational risk calculates capital requirements
based on the Business Indicator Component multiplied by the Internal Loss
Multiplier where applicable.

A. Applicability
These requirements apply to all banks that are subject to the Basel III
framework as adopted by their national supervisory authority.

B. Reporting
Banks must report their capital ratios to their supervisory authority on at
least a quarterly basis using the prescribed reporting templates.
"""

    # ── Create a RegulatoryDocument ────────────────────────────────
    doc = RegulatoryDocument(
        content=TEST_TEXT,
        doc_id="TEST_BASEL_001",
        source="BIS",
        regulation_family="Basel_III",
        jurisdiction="GLOBAL",
        version="2024-01-01",
        title="Basel III Capital Adequacy Framework Test",
    )

    # ── Run chunker ────────────────────────────────────────────────
    chunker = SemanticChunker(
        max_chunk_tokens=400,
        min_chunk_tokens=50,
        overlap_tokens=50,
    )

    print("\n" + "="*60)
    print("  CHUNKER DEBUG — Step by Step")
    print("="*60)

    # ── Step 1: show what headings the regex finds ─────────────────
    print("\n--- Step 1: Headings found by regex patterns ---")
    for i, pattern in enumerate(HEADING_PATTERNS, 1):
        matches = list(pattern.finditer(TEST_TEXT))
        if matches:
            print(f"\n  Pattern {i}:")
            for m in matches:
                print(f"    char {m.start():4d} | G1='{m.group(1)}'  G2='{m.group(2)[:40]}'")

    # ── Step 2: show structural sections before size enforcement ───
    print("\n--- Step 2: Sections after structural split ---")
    sections = chunker._split_by_structure(TEST_TEXT)
    print(f"  Found {len(sections)} sections")
    for s in sections:
        print(f"\n  [{s['section_number']}] {s['section_title']}")
        print(f"    chars: {s['char_start']}–{s['char_end']}  "
              f"length: {len(s['text'])}  "
              f"tokens(approx): {len(s['text'])//4}")
        print(f"    preview: {s['text'][:80].strip()}...")

    # ── Step 3: show final chunks with overlap ─────────────────────
    print("\n--- Step 3: Final chunks with overlap injected ---")
    chunks = chunker.chunk_document(doc)
    print(f"  Total chunks: {len(chunks)}")

    for chunk in chunks:
        print(f"\n  Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}")
        print(f"    chunk_id:      {chunk.chunk_id}")
        print(f"    section:       [{chunk.section_number}] {chunk.section_title}")
        print(f"    regulation:    {chunk.regulation_family}")
        print(f"    chars:         {len(chunk.text)}")
        print(f"    tokens(approx):{len(chunk.text)//4}")
        print(f"    overlap_prev:  {chunk.overlap_with_prev}")
        print(f"    overlap_next:  {chunk.overlap_with_next}")
        print(f"    text preview:  {chunk.text[:120].strip()}...")

    # ── Step 4: test specific retrieval scenario ───────────────────
    print("\n--- Step 4: Which chunk answers 'What triggers dividend restrictions?' ---")
    query_terms = ["triggers", "dividend", "restrictions", "buffer"]
    for chunk in chunks:
        hits = [t for t in query_terms if t.lower() in chunk.text.lower()]
        if len(hits) >= 2:
            print(f"\n  MATCH: Chunk {chunk.chunk_index+1} [{chunk.section_number}]")
            print(f"  Terms found: {hits}")
            print(f"  Text: {chunk.text[:200].strip()}...")

    # ── Step 5: show overlap working ──────────────────────────────
    print("\n--- Step 5: Overlap between adjacent chunks ---")
    for i in range(len(chunks) - 1):
        curr = chunks[i]
        nxt  = chunks[i + 1]
        if curr.overlap_with_next:
            shared = curr.text[-200:].strip()
            print(f"\n  Between chunk {i+1} and chunk {i+2}:")
            print(f"  Last 200 chars of chunk {i+1}: '...{shared[-80:]}'")
            print(f"  First 200 chars of chunk {i+2}: '{nxt.text[:80].strip()}...'")
            if shared[-50:].strip() in nxt.text[:200]:
                print(f"  Overlap confirmed ✓")

    print("\n" + "="*60)
    print(f"  Done — {len(chunks)} chunks created from test document")
    print("="*60)
    print("""
To run on a real PDF:
    python src/ingestion/chunker.py

To change test text:
    Edit TEST_TEXT in the if __name__ == '__main__' block

To test with a real file:
    doc = loader.load_document('real_docs/your_file.pdf')
    chunks = chunker.chunk_document(doc)
""")