from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field

class Vibhakti(Enum):
    """
    The 8 Pāṇinian Grammatical Cases (Vibhaktis) mapped to computational logic.
    These define the relationship between a token and the root operator (Dhātu).
    """
    NOMINATIVE = auto()    # Kartā: The independent agent/subject (e.g., Variable A)
    ACCUSATIVE = auto()    # Karma: The direct object/target of the action
    INSTRUMENTAL = auto()  # Karaṇa: The most effective means/instrument
    DATIVE = auto()        # Sampradāna: The recipient or purpose of the action
    ABLATIVE = auto()      # Apādāna: The fixed point from which movement/comparison occurs
    GENITIVE = auto()      # Sambandha: Relational pointer/ownership
    LOCATIVE = auto()      # Adhikaraṇa: The locus/context (state/memory address)
    VOCATIVE = auto()      # Sambodhana: Addressing/Interrupt signal

@dataclass(frozen=True)
class DhatuNode:
    """
    A strictly typed representation of a root operator (Dhātu) and its grammatical arguments.
    Acts as the core unit of the Pāṇinian AST.
    """
    root_operator: str
    arguments: Dict[Vibhakti, str] = field(default_factory=dict)

    def __post_init__(self):
        # Validation to ensure all keys are strictly Vibhakti Enums
        for key in self.arguments.keys():
            if not isinstance(key, Vibhakti):
                raise TypeError(f"Argument key {key} must be a Vibhakti Enum member.")

    def __repr__(self) -> str:
        args_str = ", ".join([f"{k.name}={v}" for k, v in self.arguments.items()])
        return f"DhātuNode(Op='{self.root_operator}', Args={{ {args_str} }})"

class PaninianTokenizer:
    """
    A deterministic AST Lexer for Cloud-Scale Neuro-Symbolic Logic.
    Bypasses statistical BPE in favor of strictly typed grammatical parsing.
    """
    
    def __init__(self):
        # Mapping simple logical operators to Sanskrit root concepts
        self.operator_map = {
            ">": "COMPARE_GREATER",
            "<": "COMPARE_LESSER",
            "=": "EQUATE",
            "!=": "DIFFERENTIATE",
            "+": "AUGMENT",
            "-": "REDUCE",
        }

    def tokenize(self, sentence: str) -> DhatuNode:
        """
        Parses a synthetic logical sentence into a Pāṇinian AST DhatuNode.
        Example: "If A > B" -> DhātuNode(Op='COMPARE_GREATER', Args={NOMINATIVE=A, ABLATIVE=B})
        """
        # Cleanup and normalization
        sentence = sentence.strip()
        if sentence.lower().startswith("if "):
            sentence = sentence[3:].strip()

        # Token identification (Space-based, no BPE)
        tokens = sentence.split()
        
        # Logic for simple binary operations: [Variable] [Operator] [Variable]
        if len(tokens) >= 3:
            var_left = tokens[0]
            op_symbol = tokens[1]
            var_right = tokens[2]

            if op_symbol in self.operator_map:
                root_op = self.operator_map[op_symbol]
                
                # In Pāṇinian logic:
                # The first variable is the Kartā (Nominative)
                # The second variable (in comparison) is the Apādāna (Ablative)
                args = {
                    Vibhakti.NOMINATIVE: var_left,
                    Vibhakti.ABLATIVE: var_right
                }
                return DhatuNode(root_operator=root_op, arguments=args)

        # Fallback for more descriptive English-like logic
        if "greater than" in sentence.lower():
            # Extract variables assuming format "X is greater than Y"
            parts = sentence.lower().split("is greater than")
            if len(parts) == 2:
                return DhatuNode(
                    root_operator="COMPARE_GREATER",
                    arguments={
                        Vibhakti.NOMINATIVE: parts[0].strip().upper(),
                        Vibhakti.ABLATIVE: parts[1].strip().upper()
                    }
                )

        return DhatuNode(root_operator="NULL_OP", arguments={})

    def get_vocabulary_size(self) -> int:
        """Return the count of fixed Vibhaktis + known operators."""
        return len(Vibhakti) + len(self.operator_map)

if __name__ == "__main__":
    # Internal Unit Test for Cloud Lexer
    tokenizer = PaninianTokenizer()
    
    test_cases = [
        "If A > B",
        "X = Y",
        "P is greater than Q"
    ]
    
    print("--- SanskritCore AST Tokenizer (Cloud-v1) ---")
    for test in test_cases:
        ast = tokenizer.tokenize(test)
        print(f"Input:  {test}")
        print(f"Output: {ast}\n")
