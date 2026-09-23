"""Structured knowledge base for football analytics agents.

Loads the curated `knowledge_store/football.md` and provides keyword/section
search so each agent can pull in the relevant method context without loading
the entire file into its prompt window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.analytics.agents.base import KNOWLEDGE_DIR


@dataclass
class KnowledgeEntry:
    """One section of the knowledge base."""
    section_id: int
    title: str
    content: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class KnowledgeStore:
    """Searchable index over the football analytics knowledge base."""
    entries: list[KnowledgeEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "KnowledgeStore":
        """Load and index the knowledge base markdown file."""
        if path is None:
            path = KNOWLEDGE_DIR / "football.md"
        if not path.exists():
            return cls()
        text = path.read_text(encoding="utf-8")
        entries: list[KnowledgeEntry] = []
        sections = re.split(r"^## (\d+)\. (.+)$", text, flags=re.MULTILINE)
        # sections[0] is preamble (before first ##), then triples of (id, title, content)
        i = 1
        while i + 2 < len(sections):
            sid = int(sections[i])
            title = sections[i + 1].strip()
            body = sections[i + 2].strip()
            # extract keywords: first bold phrase, model names, metric names
            kw = set()
            for m in re.finditer(r"\*\*([^*]+)\*\*", body):
                kw.add(m.group(1).lower())
            kw.add(title.lower())
            entries.append(KnowledgeEntry(sid, title, body, sorted(kw)))
            i += 3
        return cls(entries=entries)

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeEntry]:
        """Return the most relevant knowledge sections for a query string."""
        query_lower = query.lower()
        terms = re.findall(r"[a-z]{3,}", query_lower)
        scored: list[tuple[float, KnowledgeEntry]] = []
        for e in self.entries:
            text = (e.title + " " + " ".join(e.keywords) + " " + e.content[:500]).lower()
            score = sum(1.0 for t in terms if t in text)
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]

    def section(self, section_id: int) -> KnowledgeEntry | None:
        for e in self.entries:
            if e.section_id == section_id:
                return e
        return None

    def titles(self) -> list[str]:
        return [e.title for e in self.entries]
