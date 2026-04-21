import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict, Tuple, Any, Optional
from tokenizer import Vibhakti, DhatuNode, PaninianTokenizer
from model import generate_vibhakti_mask

class LogicSynthesizer:
    """
    Programmatic generator of mathematically perfect logical syllogisms.
    Generates (Premise_String, Conclusion_String, List[AST_Nodes]).
    """
    
    def __init__(self):
        self.variables = ["X", "Y", "Z", "P", "Q", "R", "A", "B", "C", "Node_1", "Node_2"]
        self.entities = ["Alice", "Bob", "Charlie", "Socrates", "The_King", "The_Sage"]
        self.tokenizer = PaninianTokenizer()

    def generate_transitive(self) -> Tuple[str, str, List[DhatuNode]]:
        """A > B, B > C |- A > C"""
        v1, v2, v3 = random.sample(self.variables, 3)
        premise = f"{v1} > {v2} {v2} > {v3}"
        conclusion = f"{v1} > {v3}"
        
        nodes = [
            self.tokenizer.tokenize(f"{v1} > {v2}"),
            self.tokenizer.tokenize(f"{v2} > {v3}")
        ]
        return premise, conclusion, nodes

    def generate_modus_ponens(self) -> Tuple[str, str, List[DhatuNode]]:
        """If P = Q and P, then Q"""
        p, q = random.sample(self.variables, 2)
        premise = f"{p} = {q} {p}"
        conclusion = f"{q}"
        
        nodes = [
            self.tokenizer.tokenize(f"{p} = {q}"),
            DhatuNode(root_operator="ASSERT", arguments={Vibhakti.NOMINATIVE: p})
        ]
        return premise, conclusion, nodes

    def generate_structural_ambiguity(self) -> Tuple[str, str, List[DhatuNode]]:
        """A = B maps to EQUATE(A, B)"""
        e1, e2 = random.sample(self.entities, 2)
        premise = f"{e1} = {e2}"
        conclusion = f"{e1} is {e2}"
        
        nodes = [self.tokenizer.tokenize(f"{e1} = {e2}")]
        return premise, conclusion, nodes

    def create_batch(self, num_samples: int) -> List[Tuple[str, str, List[DhatuNode]]]:
        data = []
        generators = [self.generate_transitive, self.generate_modus_ponens, self.generate_structural_ambiguity]
        for _ in range(num_samples):
            gen = random.choice(generators)
            data.append(gen())
        return data

class SanskritLogicDataset(Dataset):
    """
    PyTorch Dataset for Pāṇinian Neuro-Symbolic training.
    Optimized for high-throughput cloud data loading.
    """
    def __init__(self, num_samples: int = 10000):
        self.synthesizer = LogicSynthesizer()
        self.data = self.synthesizer.create_batch(num_samples)
        
        # Fixed vocabulary for the reasoning prototype
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2}
        all_tokens = set()
        for p, c, _ in self.data:
            all_tokens.update(p.split())
            all_tokens.update(c.split())
        
        for i, tok in enumerate(sorted(list(all_tokens))):
            self.vocab[tok] = i + 3
            
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        premise, conclusion, nodes = self.data[idx]
        
        input_tokens = premise.split()
        input_ids = [self.vocab.get(t, 1) for t in input_tokens]
        
        # Simple mapping: first word is often Kartā (0), second is Op, etc.
        # In a real system, this would come from the AST/Lexer mapping.
        case_ids = [0] * len(input_ids) 
        if len(case_ids) > 2:
            case_ids[2] = 4 # Assign Ablative (4) to the comparison limit
        
        label_tokens = conclusion.split()
        label_ids = [self.vocab.get(t, 1) for t in label_tokens]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "case_ids": torch.tensor(case_ids, dtype=torch.long),
            "tokens": input_tokens,
            "nodes": nodes
        }

def collate_fn(batch: List[Dict[str, Any]], op_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """
    Kaggle-Optimized Collation: 
    Pads sequences and batches the 2D Sparse Grammatical Mask.
    """
    input_ids = [item["input_ids"] for item in batch]
    label_ids = [item["label_ids"] for item in batch]
    case_ids = [item["case_ids"] for item in batch]
    batch_tokens = [item["tokens"] for item in batch]
    batch_nodes = [item["nodes"] for item in batch]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    label_ids_padded = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=0)
    case_ids_padded = torch.nn.utils.rnn.pad_sequence(case_ids, batch_first=True, padding_value=0)
    
    # Generate the Grammatical Constraint Mask M
    mask = generate_vibhakti_mask(batch_tokens, batch_nodes, op_map)
    
    return {
        "input_ids": input_ids_padded,
        "label_ids": label_ids_padded,
        "case_ids": case_ids_padded,
        "vibhakti_mask": mask
    }

if __name__ == "__main__":
    # Dataset and Collation Verification
    print("--- SanskritCore Dataset (Cloud-v1) ---")
    dataset = SanskritLogicDataset(num_samples=100)
    tokenizer = PaninianTokenizer()
    
    loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer.operator_map)
    )
    
    batch = next(iter(loader))
    print(f"Batch Input Shape:  {batch['input_ids'].shape}")
    print(f"Batch Label Shape:  {batch['label_ids'].shape}")
    print(f"Vibhakti Mask Shape: {batch['vibhakti_mask'].shape}")
    print(f"\n✓ Dataset verification complete.")
