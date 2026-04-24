"""
Dhātu Database — The Ontological Foundation of Brahman-1.

This module provides:
1. SQLite storage for all dhātu metadata
2. Fast lookup by root, semantic class, gaṇa
3. Derivation graph (dhātu → all valid forms)
4. Embedding table (initialized, trained later)
"""

import sqlite3
import json
import csv
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional, Dict

DB_PATH = Path("data/processed/dhatu.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

class SemanticClass(Enum):
    EXISTENCE    = "existence"      # as, bhū, vṛt
    MOTION       = "motion"         # gam, i, car, pat
    COGNITION    = "cognition"      # vid, jñā, man, budh, tark, ūh
    PERCEPTION   = "perception"     # śru, dṛś, spṛś
    COMMUNICATION = "communication" # vac, brū, vad, śiṣ
    CREATION     = "creation"       # kṛ, sṛj, jan, likhf
    DESTRUCTION  = "destruction"    # han, naś, mṛ
    TRANSFORMATION = "transformation" # pac, tap, vṛdh
    TRANSFER     = "transfer"       # dā, nī, vah
    ACQUISITION  = "acquisition"    # labh, āp, hṛ, pā
    RESTRICTION  = "restriction"    # rudh, bandh
    SEPARATION   = "separation"     # tyaj, muc, viyuj
    RELATION     = "relation"       # yuj, lag, mil
    CAUSATION    = "causation"      # nī, pravṛt, rakṣ
    EMOTION      = "emotion"        # prī, kup, śuc, bhī, hṛṣ
    DESIRE       = "desire"         # iṣ, kām, tṛp
    POSITION     = "position"       # sthā, vas, sad, dhā
    STATE        = "state"          # sam, jīv, svap, jāgṛ
    ABILITY      = "ability"        # śak
    CONFLICT     = "conflict"       # yudh, ji

# Vibhakti (Case) → Semantic Role mapping
# This is the CORE mapping for the reasoning engine
VIBHAKTI_TO_ROLE = {
    1: {"case": "nominative", "sanskrit": "kartā/prathamā", 
        "roles": ["AGENT", "EXPERIENCER", "THEME_SUBJECT"],
        "logic_op": "∃x: AGENT(x)",
        "propbank": ["ARG0", "ARG1_nom"]},
    2: {"case": "accusative", "sanskrit": "karma/dvitīyā",
        "roles": ["PATIENT", "THEME", "DESTINATION"],
        "logic_op": "PATIENT(y)",
        "propbank": ["ARG1", "ARG2_goal"]},
    3: {"case": "instrumental", "sanskrit": "karaṇa/tṛtīyā",
        "roles": ["INSTRUMENT", "MEANS", "ACCOMPANIMENT"],
        "logic_op": "BY_MEANS(z)",
        "propbank": ["ARG2_with", "ARGM-MNR"]},
    4: {"case": "dative", "sanskrit": "sampradāna/caturthī",
        "roles": ["GOAL", "BENEFICIARY", "PURPOSE"],
        "logic_op": "FOR(w)",
        "propbank": ["ARG2", "ARG3", "ARGM-PRP"]},
    5: {"case": "ablative", "sanskrit": "apādāna/pañcamī",
        "roles": ["SOURCE", "CAUSE", "SEPARATION_POINT"],
        "logic_op": "FROM(v) ∨ BECAUSE(v)",
        "propbank": ["ARG3_from", "ARGM-CAU"]},
    6: {"case": "genitive", "sanskrit": "sambandha/ṣaṣṭhī",
        "roles": ["POSSESSOR", "RELATION", "WHOLE"],
        "logic_op": "BELONGS_TO(u)",
        "propbank": ["ARG2_of"]},
    7: {"case": "locative", "sanskrit": "adhikaraṇa/saptamī",
        "roles": ["LOCATION", "TIME", "CONDITION"],
        "logic_op": "AT(t) ∨ WHEN(t)",
        "propbank": ["ARGM-LOC", "ARGM-TMP", "ARGM-ADV"]},
    8: {"case": "vocative", "sanskrit": "sambodha/sambodhana",
        "roles": ["ADDRESSEE"],
        "logic_op": "ADDRESS(s)",
        "propbank": ["DIS"]},
}

@dataclass
class Dhatu:
    id: int
    root_iast: str          # IAST transliteration: √gam
    root_devanagari: str    # Devanagari: √गम्
    meaning: str            # English gloss: "to go"
    gana: int               # Verb class 1-10
    semantic_class: str     # SemanticClass enum value
    transitivity: str       # transitive / intransitive / both
    pada: str               # parasmaipada / ātmanepada / ubhayapada
    common_forms: str       # JSON list of common derived forms
    logic_predicate: str    # FOL predicate name: GO, KNOW, etc.
    frequency_rank: int     # Frequency rank in corpus
    
@dataclass  
class DhatuDerivation:
    dhatu_id: int
    form: str               # The derived word (IAST)
    suffix_type: str        # pratyaya type
    grammatical_info: str   # case, number, person, tense etc.
    meaning: str

def create_database():
    """Initialize SQLite schema."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.executescript("""
        CREATE TABLE IF NOT EXISTS dhatus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            root_iast TEXT NOT NULL UNIQUE,
            root_devanagari TEXT,
            meaning TEXT NOT NULL,
            gana INTEGER CHECK(gana BETWEEN 1 AND 10),
            semantic_class TEXT NOT NULL,
            transitivity TEXT CHECK(transitivity IN ('transitive','intransitive','both')),
            pada TEXT DEFAULT 'parasmaipada',
            common_forms TEXT DEFAULT '[]',
            logic_predicate TEXT,
            frequency_rank INTEGER DEFAULT 9999
        );
        
        CREATE TABLE IF NOT EXISTS derivations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dhatu_id INTEGER REFERENCES dhatus(id),
            form TEXT NOT NULL,
            suffix_type TEXT,
            grammatical_info TEXT,
            meaning TEXT
        );
        
        CREATE TABLE IF NOT EXISTS vibhakti_roles (
            vibhakti_number INTEGER PRIMARY KEY,
            case_name TEXT,
            sanskrit_name TEXT,
            semantic_roles TEXT,
            logic_operator TEXT,
            propbank_args TEXT
        );
        
        CREATE TABLE IF NOT EXISTS sandhi_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id TEXT UNIQUE,
            phoneme_1 TEXT,
            phoneme_2 TEXT,
            result TEXT,
            panini_ref TEXT,
            rule_type TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_semantic_class ON dhatus(semantic_class);
        CREATE INDEX IF NOT EXISTS idx_gana ON dhatus(gana);
        CREATE INDEX IF NOT EXISTS idx_root_iast ON dhatus(root_iast);
    """)
    
    # Populate vibhakti roles
    for num, data in VIBHAKTI_TO_ROLE.items():
        c.execute("""
            INSERT OR REPLACE INTO vibhakti_roles 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (num, data["case"], data["sanskrit"],
              json.dumps(data["roles"]), data["logic_op"],
              json.dumps(data["propbank"])))
    
    conn.commit()
    return conn

def infer_logic_predicate(meaning: str, semantic_class: str) -> str:
    """
    Derive a clean FOL predicate name from dhātu meaning.
    e.g., "to go" → GO, "to know" → KNOW, "to be born" → BORN
    """
    # Remove "to " prefix, take first word, uppercase
    clean = meaning.strip().lower()
    clean = re.sub(r'^to\s+', '', clean)
    clean = re.sub(r'[^a-z_]', '_', clean.split('/')[0].split('(')[0].strip())
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean.upper()[:20]

def load_seed_data(conn: sqlite3.Connection):
    """Load from downloaded DCS data or seed list."""
    c = conn.cursor()
    seed_path = Path("data/raw/seed_dhatus.json")
    dcs_path = Path("data/raw/dhatupatha.csv")
    
    loaded = 0
    
    # Try DCS first
    if dcs_path.exists():
        print("Loading from DCS dhātupāṭha...")
        with open(dcs_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    root = row.get("root", row.get("dhatu", "")).strip()
                    meaning = row.get("meaning", row.get("english", "unknown")).strip()
                    gana = int(row.get("gana", row.get("class", 1)))
                    if not root or not meaning:
                        continue
                    pred = infer_logic_predicate(meaning, "existence")
                    c.execute("""INSERT OR IGNORE INTO dhatus 
                                 (root_iast, meaning, gana, semantic_class, 
                                  transitivity, logic_predicate)
                                 VALUES (?, ?, ?, ?, ?, ?)""",
                              (root, meaning, gana, "existence", "both", pred))
                    loaded += 1
                except Exception:
                    continue
        print(f"  ✓ Loaded {loaded} dhātus from DCS")
    
    # Always load seed list (has richer metadata)
    if seed_path.exists():
        print("Loading curated seed dhātus with metadata...")
        with open(seed_path, encoding="utf-8") as f:
            seeds = json.load(f)
        for entry in seeds:
            root_iast, root_dev, meaning, gana, sem_class, trans = entry
            pred = infer_logic_predicate(meaning, sem_class)
            c.execute("""INSERT OR REPLACE INTO dhatus
                         (root_iast, root_devanagari, meaning, gana,
                          semantic_class, transitivity, logic_predicate)
                         VALUES (?, ?, ?, ?, ?, ?, ?)""",
                      (root_iast, root_dev, meaning, gana, sem_class, trans, pred))
            loaded += 1
        print(f"  ✓ Loaded {len(seeds)} curated dhātus")
    
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM dhatus").fetchone()[0]
    if count == 0:
        print("ERROR: No dhātus inserted. Check seed data format.")
    else:
        print(f"✓ {count} dhātus committed to database")

class DhatuDB:
    """High-level interface to the dhātu database."""
    
    def __init__(self):
        self.conn = create_database()
        if self._is_empty():
            load_seed_data(self.conn)
    
    def _is_empty(self) -> bool:
        return self.conn.execute("SELECT COUNT(*) FROM dhatus").fetchone()[0] == 0
    
    def lookup(self, root: str) -> Optional[Dhatu]:
        row = self.conn.execute(
            "SELECT * FROM dhatus WHERE root_iast = ?", (root,)
        ).fetchone()
        if row:
            return Dhatu(*row)
        return None
    
    def by_semantic_class(self, cls: str) -> List[Dhatu]:
        rows = self.conn.execute(
            "SELECT * FROM dhatus WHERE semantic_class = ?", (cls,)
        ).fetchall()
        return [Dhatu(*r) for r in rows]
    
    def search(self, query: str) -> List[Dhatu]:
        rows = self.conn.execute(
            "SELECT * FROM dhatus WHERE meaning LIKE ?", (f"%{query}%",)
        ).fetchall()
        return [Dhatu(*r) for r in rows]
    
    def get_vibhakti_role(self, vibhakti_number: int) -> dict:
        row = self.conn.execute(
            "SELECT * FROM vibhakti_roles WHERE vibhakti_number = ?",
            (vibhakti_number,)
        ).fetchone()
        if row:
            return {
                "vibhakti": row[0], "case": row[1], "sanskrit": row[2],
                "roles": json.loads(row[3]), "logic_op": row[4],
                "propbank": json.loads(row[5])
            }
        return {}
    
    def stats(self) -> Dict:
        c = self.conn
        return {
            "total": c.execute("SELECT COUNT(*) FROM dhatus").fetchone()[0],
            "by_class": dict(c.execute(
                "SELECT semantic_class, COUNT(*) FROM dhatus GROUP BY semantic_class"
            ).fetchall()),
            "by_gana": dict(c.execute(
                "SELECT gana, COUNT(*) FROM dhatus GROUP BY gana"
            ).fetchall()),
        }

if __name__ == "__main__":
    db = DhatuDB()
    stats = db.stats()
    print(f"\n✓ DhatuDB initialized")
    print(f"  Total dhātus: {stats['total']}")
    print(f"  By semantic class: {stats['by_class']}")
    
    # Test lookup
    test = db.lookup("√gam")
    if test:
        print(f"\nTest lookup '√gam': {test.meaning} (class: {test.semantic_class})")
    
    # Test vibhakti
    nom = db.get_vibhakti_role(1)
    print(f"\nVibhakti 1 (Nominative): roles={nom['roles']}, logic={nom['logic_op']}")
