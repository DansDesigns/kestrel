"""Long-term memory.

A transcript is not memory. Compaction throws it away, and a new session starts
blank. This is the part that survives: a SQLite store the agent writes durable
facts into and searches on the way into every turn.

Design notes:

  * SQLite with FTS5 for retrieval, falling back to LIKE when a Python build
    lacks FTS5. No embedding model — that would mean a second model resident in
    memory, which is exactly the wrong trade on hardware that is already tight.
  * Recall is scored, not just matched: text relevance, stated importance,
    recency, and how often a memory has actually proved useful.
  * Writes are deduplicated. Agents restate the same fact endlessly, and a
    memory store that grows without bound is worse than none.
  * Retrieval is budgeted like everything else. Memories are injected as a
    compact block sized to its own slice of the context.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

KINDS = ["fact", "preference", "project", "person", "procedure", "decision"]
_WORD = re.compile(r"[A-Za-z0-9_]+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'fact',
    scope      TEXT NOT NULL DEFAULT '',
    tier       TEXT NOT NULL DEFAULT 'project',
    source     TEXT NOT NULL DEFAULT '',
    importance INTEGER NOT NULL DEFAULT 3,
    pinned     INTEGER NOT NULL DEFAULT 0,
    uses       INTEGER NOT NULL DEFAULT 0,
    created    REAL NOT NULL,
    last_used  REAL NOT NULL DEFAULT 0,
    norm       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_scope ON memories(scope);
CREATE UNIQUE INDEX IF NOT EXISTS idx_norm ON memories(scope, norm);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(
    text, content='memories', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
    INSERT INTO mem_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
    INSERT INTO mem_fts(mem_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memories BEGIN
    INSERT INTO mem_fts(mem_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO mem_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


PROJECT, GLOBAL, PERSONAL = "project", "global", "personal"


def _fts_probe(db) -> str:
    """A word that certainly appears, to test whether the index has been built."""
    row = db.execute("SELECT text FROM memories ORDER BY id LIMIT 1").fetchone()
    words = re.findall(r"[A-Za-z]{4,}", row["text"] if row else "")
    return words[0] if words else "the"


def classify(text: str) -> str:
    """Guess which tier a remembered fact belongs to.

    Automatic capture cannot ask, and filing everything under the project is
    how "the user is called Dan" ends up invisible in the next folder. The
    signals are blunt on purpose: a wrong guess is correctable in the Memory
    tab, an unaskable question is not.
    """
    low = " ".join(str(text or "").lower().split())
    personal = ("the user", "user's", "user is", "user prefers", "user likes",
                "user works", "user has", "their name", "he prefers",
                "she prefers", "they prefer", "my name", "i prefer", "call me")
    if any(term in low for term in personal):
        return PERSONAL
    universal = ("kestrel", "llama.cpp", "llama-server", "installed at",
                 "lives in c:", "lives at", "is located at", "on this machine",
                 "python is", "the shell tool", "gguf files", "models are kept",
                 "workspace root", "by default")
    if any(term in low for term in universal):
        return GLOBAL
    return PROJECT


TIERS = (PROJECT, GLOBAL, PERSONAL)
TIER_LABELS = {
    PROJECT: "this project only",
    GLOBAL: "how things work anywhere",
    PERSONAL: "about you",
}

# What each tier is for, in the model's words as well as the reader's.
TIER_HELP = {
    PROJECT: "facts about the project in this folder — its layout, its "
             "decisions, what has been tried",
    GLOBAL: "things true wherever Kestrel runs — where tools live, how a "
            "project is laid out, what Kestrel itself can do",
    PERSONAL: "things about the person — their name, how they like to work, "
              "what they have told you about themselves",
}


@dataclass
class Memory:
    id: int
    text: str
    kind: str = "fact"
    scope: str = ""
    tier: str = PROJECT
    source: str = ""
    importance: int = 3
    pinned: bool = False
    uses: int = 0
    created: float = 0.0
    last_used: float = 0.0
    score: float = 0.0

    def line(self) -> str:
        mark = "*" if self.pinned else "-"
        return f"{mark} [{self.kind}] {self.text}"

    def age_days(self) -> float:
        return max(0.0, (time.time() - self.created) / 86400)


def normalise(text: str) -> str:
    return " ".join(w.lower() for w in _WORD.findall(text))[:400]


class MemoryStore:
    # A recalled memory scoring below this share of the best match is dropped
    # rather than injected: relevance, not merely ranking, is what earns space
    # in a prompt that is re-sent every turn.
    MIN_SHARE = 0.45

    def __init__(self, path: str | Path, scope: str = ""):
        self.path = Path(path).expanduser()
        if not self.path.parent or str(self.path.parent) in ("", "."):
            self.path = Path.cwd() / self.path.name
        self.scope = scope
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"cannot create {self.path.parent}: {e}") from e
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.fts = True
        with self._lock:
            self.db.executescript(SCHEMA)
            # Before anything indexes or queries the column: an existing store
            # has no `tier`, and CREATE INDEX on a missing column fails outright
            # rather than being skipped.
            self._add_tier_column()
            # Filed before the full-text triggers exist: updating rows once they
            # do fires them against an index that has not been built yet.
            self._file_untiered()
            try:
                self.db.executescript(FTS_SCHEMA)
            except sqlite3.OperationalError:
                self.fts = False       # build without FTS5; LIKE search instead
            self._backfill_fts()
            self.db.commit()

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass

    # -- writing --------------------------------------------------------------
    def remember(self, text: str, kind: str = "fact", importance: int = 3,
                 source: str = "", pinned: bool = False, scope: str | None = None,
                 tier: str = "",
                 enforce: bool = True) -> tuple[int, bool]:
        """Store a memory. Returns (id, created_new). Re-stating an existing
        memory bumps its importance rather than adding a duplicate.

        `enforce` applies the durability filter; only a human editing the store
        by hand bypasses it.
        """
        text = " ".join(text.split()).strip()
        if not text:
            raise ValueError("empty memory")
        if enforce:
            ok, reason = is_durable(text)
            if not ok:
                raise ValueError(reason)
        if len(text) > 2000:
            text = text[:2000]
        kind = kind if kind in KINDS else "fact"
        importance = max(1, min(5, int(importance)))
        tier = tier if tier in TIERS else classify(text)
        # The tier decides the scope, whatever the caller passed. A global or
        # personal memory is stored without one: tying it to the folder it
        # happened to be learnt in is exactly what makes it invisible from
        # everywhere else, which is the problem tiers exist to solve.
        sc = "" if tier in (GLOBAL, PERSONAL) else (
            self.scope if scope is None else scope)
        norm = normalise(text)
        now = time.time()
        with self._lock:
            row = self.db.execute(
                "SELECT id, importance FROM memories WHERE scope=? AND norm=?",
                (sc, norm)).fetchone()
            if row:
                self.db.execute(
                    "UPDATE memories SET importance=MAX(importance,?), last_used=?, "
                    "pinned=MAX(pinned,?) WHERE id=?",
                    (importance, now, int(pinned), row["id"]))
                self.db.commit()
                return int(row["id"]), False
            cur = self.db.execute(
                "INSERT INTO memories(text,kind,scope,tier,source,importance,"
                "pinned,created,last_used,norm) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (text, kind, sc, tier, source, importance, int(pinned), now,
                 now, norm))
            self.db.commit()
            return int(cur.lastrowid), True

    def forget(self, memory_id: int) -> bool:
        with self._lock:
            cur = self.db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self.db.commit()
            return cur.rowcount > 0

    def update(self, memory_id: int, **fields) -> bool:
        allowed = {"text", "kind", "importance", "pinned", "source"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return False
        if "text" in sets:
            sets["norm"] = normalise(str(sets["text"]))
        clause = ", ".join(f"{k}=?" for k in sets)
        with self._lock:
            cur = self.db.execute(f"UPDATE memories SET {clause} WHERE id=?",
                                  (*sets.values(), memory_id))
            self.db.commit()
            return cur.rowcount > 0

    def set_pinned(self, memory_id: int, pinned: bool) -> bool:
        return self.update(memory_id, pinned=int(pinned))

    def clear_tier(self, tier: str, scope: str | None = None) -> int:
        """Forget one tier. Used to start a project over without losing what
        Kestrel knows about the machine or about the person."""
        sc = self.scope if scope is None else scope
        with self._lock:
            if tier == PROJECT:
                cur = self.db.execute(
                    "DELETE FROM memories WHERE tier=? AND scope=?", (tier, sc))
            else:
                cur = self.db.execute("DELETE FROM memories WHERE tier=?", (tier,))
            self.db.commit()
            return cur.rowcount

    def clear(self, scope: str | None = None) -> int:
        sc = self.scope if scope is None else scope
        with self._lock:
            cur = self.db.execute("DELETE FROM memories WHERE scope=?", (sc,))
            self.db.commit()
            return cur.rowcount

    # -- reading --------------------------------------------------------------
    def all(self, scope: str | None = None, limit: int = 500,
            tier: str = "") -> list[Memory]:
        sc = self.scope if scope is None else scope
        clause = "(scope=? OR scope='')"
        params: list = [sc]
        if tier in TIERS:
            clause += " AND tier=?"
            params.append(tier)
        params.append(limit)
        with self._lock:
            rows = self.db.execute(
                f"SELECT * FROM memories WHERE {clause} "
                "ORDER BY pinned DESC, importance DESC, created DESC LIMIT ?",
                params).fetchall()
        return [_row(r) for r in rows]

    def count(self, scope: str | None = None, tier: str = "") -> int:
        sc = self.scope if scope is None else scope
        clause = "(scope=? OR scope='')"
        params: list = [sc]
        if tier in TIERS:
            clause += " AND tier=?"
            params.append(tier)
        with self._lock:
            return int(self.db.execute(
                f"SELECT COUNT(*) c FROM memories WHERE {clause}",
                params).fetchone()["c"])

    def _backfill_fts(self) -> None:
        """Index rows that predate the full-text table. Lock held.

        The triggers only fire on writes, so a store that existed before the
        index was added has rows the index has never seen — they are stored,
        listed and counted, but never found by searching, which looks exactly
        like memory not working.
        """
        if not self.fts:
            return
        try:
            rows = self.db.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
            if not rows:
                return
            hit = self.db.execute(
                "SELECT COUNT(*) c FROM mem_fts WHERE mem_fts MATCH ?",
                (_fts_probe(self.db),)).fetchone()["c"]
            if hit:
                return
            # An external-content table cannot be filled by inserting rows: its
            # content lives in `memories`, and the index is rebuilt from there.
            self.db.execute("INSERT INTO mem_fts(mem_fts) VALUES('rebuild')")
        except sqlite3.Error:
            self.fts = False

    def _add_tier_column(self) -> None:
        """Called with the lock held, before the indexes."""
        columns = {r["name"] for r in
                   self.db.execute("PRAGMA table_info(memories)").fetchall()}
        if "tier" not in columns:
            self.db.execute("ALTER TABLE memories ADD COLUMN tier TEXT "
                            "NOT NULL DEFAULT 'project'")
            self._needs_filing = True
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)")

    def _file_untiered(self) -> None:
        """Sort a pre-tier store into tiers. Called with the lock held.

        Everything already there was learnt before the distinction existed, so
        it is filed by the same guess a new memory gets rather than dumped into
        one tier and left wrong.
        """
        if not getattr(self, "_needs_filing", False):
            return
        self._needs_filing = False
        for row in self.db.execute("SELECT id, text FROM memories").fetchall():
            guess = classify(row["text"])
            if guess != PROJECT:
                self.db.execute(
                    "UPDATE memories SET tier=?, scope='' WHERE id=?",
                    (guess, row["id"]))
        self.db.commit()

    def _tier_clause(self, scope: str) -> tuple[str, list]:
        """Project memories from this folder; global and personal from anywhere.

        That is the whole point of the tiers: what Kestrel knows about the
        machine, and what it knows about you, should not have to be relearnt in
        every new folder.
        """
        return ("(tier IN (?, ?) OR (tier = ? AND scope = ?))",
                [GLOBAL, PERSONAL, PROJECT, scope])

    def search(self, query: str, limit: int = 8, scope: str | None = None,
               include_pinned: bool = True) -> list[Memory]:
        """Relevance, importance, recency and proven usefulness, combined."""
        sc = self.scope if scope is None else scope
        terms = [w for w in _WORD.findall(query.lower()) if len(w) > 2]
        results: dict[int, Memory] = {}

        if terms:
            rows = self._match(terms, sc, limit * 4)
            for r, relevance in rows:
                m = _row(r)
                # A memory has to actually contain something that was asked
                # about. Ranking alone will happily return the least-bad match
                # in the store, which is how unrelated memories surface.
                haystack = normalise(m.text)
                hits = sum(1 for t in terms if t in haystack)
                if not hits:
                    continue
                # One incidental word in common is not relevance. A single hit
                # only counts when it is a distinctive word rather than one
                # that appears in half the store — otherwise a note about the
                # clock surfaces on every prompt that happens to say "time".
                if hits == 1 and len(terms) > 2 and self._common(terms, haystack):
                    continue
                m.score = self._score(m, max(relevance, float(hits)))
                m.hits = hits
                results[m.id] = m

        if include_pinned:
            with self._lock:
                pinned = self.db.execute(
                    "SELECT * FROM memories WHERE (scope=? OR scope='') AND pinned=1",
                    (sc,)).fetchall()
            for r in pinned:
                m = results.get(int(r["id"])) or _row(r)
                m.score = max(m.score, 100.0)   # pinned always makes the cut
                results[m.id] = m

        out = sorted(results.values(), key=lambda m: -m.score)[:limit]
        if out:
            self._touch([m.id for m in out])
        return out

    def _match(self, terms: list[str], scope: str, limit: int):
        if self.fts:
            expr = " OR ".join(f'"{t}"' for t in terms)
            try:
                with self._lock:
                    rows = self.db.execute(
                        "SELECT m.*, bm25(mem_fts) AS rank FROM mem_fts "
                        "JOIN memories m ON m.id = mem_fts.rowid "
                        "WHERE mem_fts MATCH ? AND (m.scope=? OR m.scope='') "
                        "ORDER BY rank LIMIT ?", (expr, scope, limit)).fetchall()
                # bm25 returns negative numbers, better is more negative
                return [(r, max(0.0, -float(r["rank"]))) for r in rows]
            except sqlite3.OperationalError:
                self.fts = False
        clause = " OR ".join("norm LIKE ?" for _ in terms)
        args = [f"%{t}%" for t in terms]
        with self._lock:
            rows = self.db.execute(
                f"SELECT * FROM memories WHERE (scope=? OR scope='') AND ({clause}) LIMIT ?",
                (scope, *args, limit)).fetchall()
        out = []
        for r in rows:
            hits = sum(1 for t in terms if t in r["norm"])
            out.append((r, float(hits)))
        return out

    def _common(self, terms: list[str], haystack: str) -> bool:
        """Is the only matching term a word that matches nearly everything?

        A store where "time" appears in half the entries will surface one of
        them for any prompt mentioning time, which is how an irrelevant note
        ends up in every prompt.
        """
        matched = [t for t in terms if t in haystack]
        if not matched:
            return True
        try:
            row = self.db.execute(
                "SELECT COUNT(*) FROM memories WHERE norm LIKE ?",
                (f"%{matched[0]}%",)).fetchone()
        except Exception:
            return False
        total = self.count() or 1
        return bool(row and row[0] / total > 0.4)

    @staticmethod
    def _score(m: Memory, relevance: float) -> float:
        recency = 1.0 / (1.0 + m.age_days() / 45.0)
        used = min(1.0, m.uses / 8.0)
        return (2.0 * min(relevance, 8.0)
                + 1.6 * m.importance
                + 2.5 * recency
                + 1.2 * used
                + (6.0 if m.pinned else 0.0))

    def _touch(self, ids: Iterable[int]) -> None:
        now = time.time()
        with self._lock:
            self.db.executemany(
                "UPDATE memories SET uses=uses+1, last_used=? WHERE id=?",
                [(now, i) for i in ids])
            self.db.commit()

    # -- prompt block ---------------------------------------------------------
    def block(self, query: str, counter, token_budget: int, limit: int = 8,
              scope: str | None = None) -> tuple[str, list[Memory]]:
        """Render recalled memories to fit a token allowance."""
        if token_budget < 24:
            return "", []
        found = self.search(query, limit=limit, scope=scope)
        if not found:
            return "", []
        # Keep the strong matches and anything close to them; drop the tail.
        # A weak match propped up by recency and importance is still weak, and
        # it costs context on every single turn.
        best = max(m.score for m in found)
        found = [m for m in found
                 if m.pinned or m.score >= best * self.MIN_SHARE]
        lines: list[str] = []
        used = counter.count("What you remember:") + 4
        kept: list[Memory] = []
        for m in found:
            line = m.line()
            cost = counter.count(line) + 1
            if used + cost > token_budget:
                break
            lines.append(line)
            used += cost
            kept.append(m)
        if not lines:
            return "", []
        return "What you remember:\n" + "\n".join(lines), kept

    # -- maintenance ----------------------------------------------------------
    def purge_ephemeral(self, scope: str | None = None) -> list[str]:
        """Delete stored memories that would not be accepted today.

        The durability filter arrived after some stores did, so a note about
        the current time can sit in one indefinitely, matching often and
        meaning nothing. Returns what was removed, for reporting.
        """
        removed: list[str] = []
        for m in self.all(scope=scope, limit=5000):
            if m.pinned:
                continue
            ok, _reason = is_durable(m.text)
            if not ok:
                removed.append(m.text)
                self.forget(m.id)
        return removed

    def prune(self, max_items: int = 2000, min_importance: int = 2) -> int:
        """Drop the least useful unpinned memories once the store gets large."""
        with self._lock:
            total = self.db.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
            if total <= max_items:
                return 0
            excess = total - max_items
            cur = self.db.execute(
                "DELETE FROM memories WHERE id IN ("
                "  SELECT id FROM memories WHERE pinned=0 AND importance<=?"
                "  ORDER BY uses ASC, importance ASC, last_used ASC LIMIT ?)",
                (min_importance, excess))
            self.db.commit()
            return cur.rowcount

    def export(self) -> str:
        return "\n".join(json.dumps({
            "text": m.text, "kind": m.kind, "importance": m.importance,
            "pinned": m.pinned, "scope": m.scope, "source": m.source,
        }) for m in self.all(limit=10000))

    def import_jsonl(self, blob: str) -> int:
        n = 0
        for line in blob.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not d.get("text"):
                continue
            _, created = self.remember(
                d["text"], d.get("kind", "fact"), int(d.get("importance", 3)),
                d.get("source", "import"), bool(d.get("pinned")), d.get("scope"))
            n += int(created)
        return n


def _row(r) -> Memory:
    return Memory(
        id=int(r["id"]), text=r["text"], kind=r["kind"], scope=r["scope"],
        source=r["source"], importance=int(r["importance"]),
        pinned=bool(r["pinned"]), uses=int(r["uses"]),
        created=float(r["created"]), last_used=float(r["last_used"]),
    )


CAPTURE_PROMPT = """From the exchange below, extract only what will still be true and
still be useful in a month.

Record ONLY:
- a preference the user stated about how they want things done
- a stable fact about their setup, hardware, or tooling
- a decision that was made, and why
- a procedure that worked and would be repeated

Record NOTHING about:
- the current date, time, or anything that changes on its own
- what a tool returned this turn, or any command output
- file contents, directory listings, or search results
- what happened in this conversation, unless it is a decision above
- anything you would have to check again before trusting

Most exchanges contain nothing worth keeping. Writing nothing is the correct and
expected answer. Never invent something to fill the space.

Reply with one JSON object per line and nothing else:
{"text": "...", "kind": "fact|preference|project|person|procedure|decision", "importance": 1-5}

At most %d lines.

--- exchange ---
%s"""

# Things that read like facts but expire. A memory store that fills with
# timestamps and command output is worse than no memory at all: it costs context
# every turn and recalls noise.
EPHEMERAL = [
    (re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b"), "contains a clock time"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "contains a date"),
    (re.compile(r"\b\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|"
                r"august|september|october|november|december)\b", re.I), "contains a date"),
    (re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
     "contains a weekday"),
    (re.compile(r"\b(today|tonight|yesterday|tomorrow|right now|just now|currently|"
                r"at the moment|this session|so far)\b", re.I), "describes the present moment"),
    (re.compile(r"\b(the )?(output|result|response|return value) (of|from|was|is)\b", re.I),
     "records tool output"),
    (re.compile(r"\bexit(ed with)? code\b|\bexit \d+\b", re.I), "records a command result"),
    (re.compile(r"\b(contains|listed|returned|printed|showed) \d+ (files?|lines?|results?|"
                r"matches|items?)\b", re.I), "records a listing"),
    (re.compile(r"\b(the )?(current|latest|newest|most recent) ", re.I),
     "describes a value that changes"),
    (re.compile(r"```|\|\s*-{3,}\s*\|"), "contains code or a table"),
]


def is_durable(text: str) -> tuple[bool, str]:
    """Would this still be true, and still useful, in a month?

    Applied to everything before it is stored. The model is asked to be
    selective and mostly is, but 'the current time is 14:32' is exactly the kind
    of thing that slips through a prompt instruction and never through a regex.
    """
    clean = " ".join(str(text or "").split())
    if len(clean.split()) < 4:
        return False, "too short to mean anything later"
    if len(clean) > 400:
        return False, "too long — memories should be single facts"
    for pattern, reason in EPHEMERAL:
        if pattern.search(clean):
            return False, reason
    letters = sum(c.isalpha() for c in clean)
    if letters < len(clean) * 0.5:
        return False, "mostly punctuation or numbers"
    return True, ""


def parse_capture(blob: str) -> list[dict]:
    """Pull memory objects out of a capture response, tolerating fences and prose."""
    out = []
    for line in blob.splitlines():
        line = line.strip().strip("`").strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            try:
                d = json.loads(re.sub(r",\s*}", "}", line))
            except ValueError:
                continue
        text = str(d.get("text") or "").strip()
        if len(text) < 8:
            continue
        out.append({
            "text": text,
            "kind": str(d.get("kind") or "fact"),
            "importance": int(d.get("importance") or 3),
        })
    return out
