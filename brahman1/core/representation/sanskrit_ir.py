"""
The Sanskrit Intermediate Representation (SIR).

This is NOT a Sanskrit sentence. It is a typed semantic graph
derived from Sanskrit's grammatical structure.

The SIR is the bridge between:
- Surface language (English input)
- Formal logic (FOL reasoning)
- Neural representation (transformer embeddings)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import re

# ─── Type Definitions ───────────────────────────────────────────

class Vibhakti(Enum):
    """8 Sanskrit grammatical cases as semantic role types."""
    NOMINATIVE   = (1, "kartā",     "AGENT")        # subject/doer
    ACCUSATIVE   = (2, "karma",     "PATIENT")       # direct object
    INSTRUMENTAL = (3, "karaṇa",   "INSTRUMENT")    # by means of
    DATIVE       = (4, "sampradāna","GOAL")          # for/to whom
    ABLATIVE     = (5, "apādāna",  "SOURCE")         # from/because
    GENITIVE     = (6, "sambandha", "POSSESSOR")     # of/belonging
    LOCATIVE     = (7, "adhikaraṇa","LOCATION")      # in/at/when
    VOCATIVE     = (8, "sambodha",  "ADDRESSEE")     # O! addressed
    
    def __init__(self, number, sanskrit_name, primary_role):
        self.number = number
        self.sanskrit_name = sanskrit_name
        self.primary_role = primary_role

class Lakara(Enum):
    """Tense-Aspect-Mood (TAM) from Sanskrit lakāra system."""
    LAT   = "present"           # laṭ
    LIT   = "perfect"           # liṭ (non-eyewitness past)
    LUT   = "periphrastic_future"  # luṭ
    LRT   = "simple_future"    # lṛṭ
    LOT   = "imperative"        # loṭ
    LAN   = "imperfect"         # laṅ (eyewitness past)
    LIN   = "optative"          # liṅ (potential/wish)
    LUN   = "aorist"            # luṅ (immediate past)
    LRN   = "conditional"       # lṛṅ

class LogicOp(Enum):
    """Logical operators extracted from Sanskrit particles."""
    AND         = ("ca",     "∧")     # and
    OR          = ("vā",     "∨")     # or
    NOT         = ("na",     "¬")     # not
    IF_THEN     = ("yadi",   "→")     # if (conditional)
    THEREFORE   = ("ataḥ",   "∴")     # therefore
    BECAUSE     = ("yasmāt", "←")    # because (reverse implication)
    ALWAYS      = ("sadā",   "□")     # always (necessity modal)
    SOMETIMES   = ("kadācit","◇")    # sometimes (possibility modal)
    ALL         = ("sarve",  "∀")    # all (universal quantifier)
    SOME        = ("kecit",  "∃")    # some (existential quantifier)
    NONE        = ("na kaścit","¬∃") # none
    
    def __init__(self, sanskrit_particle, fol_symbol):
        self.particle = sanskrit_particle
        self.symbol = fol_symbol

# ─── Core Node Types ────────────────────────────────────────────

@dataclass
class DhatuNode:
    """A verbal root — the predicate of a proposition."""
    root: str                    # √gam
    logic_predicate: str         # GO
    semantic_class: str          # motion
    lakara: Optional[Lakara]     # tense/aspect/mood
    person: Optional[int]        # 1, 2, or 3
    number: Optional[str]        # singular, dual, plural
    voice: str = "active"        # active / passive / middle
    
    def to_fol_predicate(self) -> str:
        tense_marker = f"_{self.lakara.value.upper()}" if self.lakara else ""
        return f"{self.logic_predicate}{tense_marker}"

@dataclass
class ArgumentNode:
    """A nominal argument filling a case role."""
    surface_form: str            # The word as it appears
    lemma: str                   # Dictionary form
    vibhakti: Vibhakti           # Grammatical case
    semantic_role: str           # AGENT, PATIENT, etc.
    gender: Optional[str]        # masculine, feminine, neuter
    number: Optional[str]        # singular, dual, plural
    is_compound: bool = False    # Is this a samāsa?
    compound_parts: List[str] = field(default_factory=list)
    
    def to_fol_term(self) -> str:
        # Variables: agents→x, patients→y, instruments→z, etc.
        var_map = {
            "AGENT": "x", "PATIENT": "y", "INSTRUMENT": "z",
            "GOAL": "w", "SOURCE": "v", "POSSESSOR": "u",
            "LOCATION": "l", "ADDRESSEE": "a"
        }
        var = var_map.get(self.semantic_role, "q")
        return f"{var}_{self.lemma.replace('√','').replace(' ','_')}"

@dataclass
class CompoundNode:
    """A samāsa (Sanskrit compound) with decomposition."""
    surface: str                          # Full compound
    compound_type: str                    # tatpuruṣa, bahuvrīhi, dvandva, avyayībhāva, karmadhāraya
    parts: List['ArgumentNode']           # Component nodes
    head: Optional['ArgumentNode'] = None # Head of compound
    
    def to_english(self) -> str:
        """Generate English gloss of compound."""
        type_templates = {
            "tatpuruṣa": "{modifier} of/for/by/from {head}",
            "karmadhāraya": "{modifier} {head}",
            "bahuvrīhi": "having {modifier} {head}",
            "dvandva": "{modifier} and {head}",
            "avyayībhāva": "according to {head}",
        }
        template = type_templates.get(self.compound_type, "{modifier} {head}")
        parts_en = [p.lemma for p in self.parts]
        if len(parts_en) >= 2:
            return template.format(modifier=parts_en[0], head=parts_en[-1])
        return " ".join(parts_en)

# ─── The Main SIR Dataclass ─────────────────────────────────────

@dataclass
class SanskritIR:
    """
    Sanskrit Intermediate Representation.
    
    A typed semantic graph capturing:
    - Predicate-argument structure (from vibhakti)
    - Logical operators (from particles)  
    - Compound decomposition (from samāsa)
    - Tense/Aspect/Mood (from lakāra)
    
    Can be converted to:
    - First-Order Logic (for symbolic reasoning)
    - NetworkX graph (for graph neural networks)
    - JSON (for serialization/training)
    """
    
    # Core predicate-argument structure
    predicate: Optional[DhatuNode]
    arguments: Dict[str, ArgumentNode] = field(default_factory=dict)
    # Key is semantic role: "AGENT", "PATIENT", etc.
    
    # Logical operators found in sentence
    operators: List[LogicOp] = field(default_factory=list)
    
    # Compound nodes
    compounds: List[CompoundNode] = field(default_factory=list)
    
    # Multiple propositions (for complex sentences)
    sub_propositions: List['SanskritIR'] = field(default_factory=list)
    
    # Source sentence
    source_text: str = ""
    
    # Confidence scores (from neural SRL)
    confidence: Dict[str, float] = field(default_factory=dict)
    
    # Validation result
    is_valid: bool = True
    validation_notes: List[str] = field(default_factory=list)

    def to_fol(self) -> str:
        """
        Convert SIR to First-Order Logic string.
        
        Example:
        "Rāma goes to the forest" →
        SIR: predicate=GO, AGENT=rāma, GOAL=forest
        FOL: ∃x∃w(AGENT(x, rāma) ∧ GOAL(w, forest) ∧ GO(x, w))
        """
        if not self.predicate:
            return "UNKNOWN_PREDICATE"
        
        pred = self.predicate.to_fol_predicate()
        parts = []
        variables = []
        
        for role, arg in self.arguments.items():
            var = arg.to_fol_term()
            variables.append(var)
            parts.append(f"{role}({var}, {arg.lemma})")
        
        # Build quantifiers
        quantifiers = ""
        for op in self.operators:
            if op in (LogicOp.ALL,):
                quantifiers = "∀x "
            elif op in (LogicOp.SOME,):
                quantifiers = "∃x "
        
        if not quantifiers:
            quantifiers = "∃" + "∃".join(set(v[0] for v in variables)) + " " if variables else ""
        
        # Combine predicate with arguments
        arg_str = ", ".join(v for v in variables)
        predicate_str = f"{pred}({arg_str})" if arg_str else f"{pred}()"
        
        # Add logical operators
        op_prefix = ""
        for op in self.operators:
            if op == LogicOp.NOT:
                op_prefix = "¬"
            elif op == LogicOp.IF_THEN:
                op_prefix = "IF "
        
        # Join all parts
        if parts:
            conditions = " ∧ ".join(parts)
            fol = f"{quantifiers}({conditions} ∧ {predicate_str})"
        else:
            fol = f"{quantifiers}{op_prefix}{predicate_str}"
        
        # Handle sub-propositions
        for sub in self.sub_propositions:
            sub_fol = sub.to_fol()
            for op in self.operators:
                if op == LogicOp.IF_THEN:
                    fol = f"({fol} → {sub_fol})"
                elif op == LogicOp.AND:
                    fol = f"({fol} ∧ {sub_fol})"
                elif op == LogicOp.OR:
                    fol = f"({fol} ∨ {sub_fol})"
                elif op == LogicOp.THEREFORE:
                    fol = f"({fol} ∴ {sub_fol})"
        
        return fol
    
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        def serialize_node(obj):
            if obj is None: return None
            d = asdict(obj)
            # Handle Enums in asdict output
            if "lakara" in d and d["lakara"]:
                d["lakara"] = d["lakara"].name
            if "vibhakti" in d and d["vibhakti"]:
                d["vibhakti"] = d["vibhakti"].name
            return d

        return {
            "source_text": self.source_text,
            "predicate": serialize_node(self.predicate),
            "arguments": {k: serialize_node(v) for k, v in self.arguments.items()},
            "operators": [op.name for op in self.operators],
            "fol": self.to_fol(),
            "compounds": [asdict(c) for c in self.compounds],
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "validation_notes": self.validation_notes,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def explain(self) -> str:
        """Human-readable explanation of the SIR structure."""
        lines = [f"Source: '{self.source_text}'"]
        if self.predicate:
            lines.append(f"Predicate: {self.predicate.root} ({self.predicate.logic_predicate})"
                         f" — {self.predicate.semantic_class}")
        for role, arg in self.arguments.items():
            lines.append(f"  {role} ({arg.vibhakti.sanskrit_name}): "
                         f"'{arg.surface_form}' → {arg.lemma}")
        if self.operators:
            lines.append(f"Logical operators: {[op.name for op in self.operators]}")
        lines.append(f"FOL: {self.to_fol()}")
        return "\n".join(lines)

# ─── SIR Builder (helper) ────────────────────────────────────────

class SIRBuilder:
    """Fluent builder for constructing SIR objects."""
    
    def __init__(self, source: str = ""):
        self._sir = SanskritIR(predicate=None, source_text=source)
    
    def set_predicate(self, root: str, logic_pred: str, sem_class: str,
                      lakara: Lakara = None) -> 'SIRBuilder':
        self._sir.predicate = DhatuNode(
            root=root, logic_predicate=logic_pred,
            semantic_class=sem_class, lakara=lakara,
            person=3, number="singular"
        )
        return self
    
    def add_argument(self, surface: str, lemma: str, 
                     vibhakti: Vibhakti) -> 'SIRBuilder':
        role = vibhakti.primary_role
        self._sir.arguments[role] = ArgumentNode(
            surface_form=surface, lemma=lemma,
            vibhakti=vibhakti, semantic_role=role,
            gender=None, number="singular"
        )
        return self
    
    def add_operator(self, op: LogicOp) -> 'SIRBuilder':
        self._sir.operators.append(op)
        return self
    
    def build(self) -> SanskritIR:
        return self._sir

# ─── Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test: "Rāma goes to the forest"
    sir = (SIRBuilder("Rāma goes to the forest")
           .set_predicate("√gam", "GO", "motion", Lakara.LAT)
           .add_argument("rāmaḥ", "rāma", Vibhakti.NOMINATIVE)
           .add_argument("vanam", "vana", Vibhakti.ACCUSATIVE)
           .build())
    
    print("=== SIR Test: 'Rāma goes to the forest' ===")
    print(sir.explain())
    print()
    
    # Test: "If it rains, the ground gets wet"
    rain_sir = (SIRBuilder("If it rains")
                .set_predicate("√vṛṣ", "RAIN", "transformation", Lakara.LAT)
                .add_operator(LogicOp.IF_THEN)
                .build())
    
    wet_sir = (SIRBuilder("the ground gets wet")
               .set_predicate("√ārdra-bhū", "BECOME_WET", "transformation", Lakara.LAT)
               .add_argument("bhūmiḥ", "bhūmi", Vibhakti.NOMINATIVE)
               .build())
    
    rain_sir.sub_propositions.append(wet_sir)
    
    print("=== SIR Test: 'If it rains, the ground gets wet' ===")
    print(rain_sir.explain())
    print(f"\nFOL: {rain_sir.to_fol()}")
    print()
    
    # Test: Syllogism
    print("=== SIR Test: Syllogism ===")
    major = (SIRBuilder("All humans are mortal")
             .set_predicate("√as", "IS", "existence", Lakara.LAT)
             .add_argument("manuṣyāḥ", "manuṣya", Vibhakti.NOMINATIVE)
             .add_argument("martyāḥ", "martya", Vibhakti.NOMINATIVE)
             .add_operator(LogicOp.ALL)
             .build())
    print(f"Major premise FOL: {major.to_fol()}")
    
    minor = (SIRBuilder("Socrates is human")
             .set_predicate("√as", "IS", "existence", Lakara.LAT)
             .add_argument("sōkrāṭaḥ", "sōkrāṭa", Vibhakti.NOMINATIVE)
             .add_argument("manuṣyaḥ", "manuṣya", Vibhakti.ACCUSATIVE)
             .build())
    print(f"Minor premise FOL: {minor.to_fol()}")
