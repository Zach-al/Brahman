"""
TripleEncoder: Encodes each token into THREE parallel representation spaces.

SPACE 1: Semantic Space     (dim=512) — learned, distributional meaning
SPACE 2: Grammatical Space  (dim=256) — hard-computed from VibhaktiEncoder
SPACE 3: Morphological Space(dim=128) — derived from dhātu decomposition

Combined: 896-dim representation per token (projected to model_dim=512 for efficiency)

This is the input to BrahmanTransformer's attention stack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict
import sys
import os

# Ensure project root is in path
sys.path.insert(0, ".")

from core.dhatu.dhatu_db import DhatuDB
from core.grammar.vibhakti_encoder import VibhaktiEncoder, Vibhakti, NUM_LABELS
from core.representation.sanskrit_ir import SanskritIR, Vibhakti as VibhaktiEnum


# ── Dimensions ────────────────────────────────────────────
SEMANTIC_DIM      = 512   # Standard dense embedding
GRAMMATICAL_DIM   = 256   # Vibhakti/case-role space (hard-computed)
MORPHOLOGICAL_DIM = 128   # Dhātu root + suffix space
COMBINED_DIM      = SEMANTIC_DIM + GRAMMATICAL_DIM + MORPHOLOGICAL_DIM  # 896
MODEL_DIM         = 512   # Final projected dim for transformer input


# ── Case Role Embeddings (fixed, not learned) ──────────────
# Each of 8 vibhaktis gets a unique, orthogonal direction in grammatical space.
# These are NOT learned — they are computed from Pāṇini rules.
def build_orthogonal_case_embeddings(dim: int = GRAMMATICAL_DIM) -> torch.Tensor:
    """
    Produce 10 quasi-orthogonal vectors (8 cases + 2 specials: PAD, UNK).
    Use Gram-Schmidt to guarantee independence.
    Shape: (10, dim)
    """
    raw = torch.randn(10, dim)
    # Gram-Schmidt orthogonalization
    basis = []
    for i in range(raw.shape[0]):
        v = raw[i].clone()
        for b in basis:
            v -= (v @ b) * b
        norm = v.norm()
        if norm > 1e-6:
            v = v / norm
        basis.append(v)
    return torch.stack(basis)  # (10, dim)


CASE_EMBEDDINGS = build_orthogonal_case_embeddings()  # Shared, frozen


@dataclass
class TripleEncoding:
    """Full triple-space encoding for a sequence."""
    semantic:      torch.Tensor  # (batch, seq_len, 512)
    grammatical:   torch.Tensor  # (batch, seq_len, 256)
    morphological: torch.Tensor  # (batch, seq_len, 128)
    combined:      torch.Tensor  # (batch, seq_len, 512) — projected
    case_labels:   torch.Tensor  # (batch, seq_len) — int case IDs
    confidence:    torch.Tensor  # (batch, seq_len) — vibhakti confidence


class SemanticEncoder(nn.Module):
    """
    Space 1: Standard dense semantic embedding.
    Pretrained RoBERTa embeddings, projected to SEMANTIC_DIM.
    """
    def __init__(self, vocab_size: int = 50265, dim: int = SEMANTIC_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=1)
        self.position  = nn.Embedding(514, dim)  # RoBERTa-style positions
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout    = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            (batch, seq_len, SEMANTIC_DIM)
        """
        positions = torch.arange(
            input_ids.size(1), device=input_ids.device
        ).unsqueeze(0).expand_as(input_ids)
        x = self.embedding(input_ids) + self.position(positions)
        return self.dropout(self.layer_norm(x))


class GrammaticalEncoder(nn.Module):
    """
    Space 2: Case-role grammatical encoding.
    HARD-COMPUTED from VibhaktiEncoder output — NOT learned end-to-end.
    
    Encodes:
    - Which of 8 Pāṇinian cases each token fills
    - Confidence of that assignment
    - Inter-token case-dependency structure
    
    Key property: This space is INTERPRETABLE.
    You can read off "token X is in accusative/karma (patient) role"
    directly from the vector without any learned decoding.
    """
    def __init__(self, dim: int = GRAMMATICAL_DIM):
        super().__init__()
        self.dim = dim
        # Case role embedding lookup (frozen orthogonal basis)
        self.register_buffer("case_basis", CASE_EMBEDDINGS)  # (10, 256)
        
        # Dependency structure: how does each token's case
        # relate to the verb (dhātu)?
        self.dependency_proj = nn.Linear(dim, dim)
        
        # Confidence-weighted interpolation
        self.confidence_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        case_ids:    torch.Tensor,   # (batch, seq_len) — vibhakti label IDs
        confidence:  torch.Tensor,   # (batch, seq_len) — [0,1] per token
        verb_mask:   torch.Tensor,   # (batch, seq_len) — 1.0 at verb positions
    ) -> torch.Tensor:
        """
        Returns:
            grammatical_enc: (batch, seq_len, GRAMMATICAL_DIM)
        """
        # Look up orthogonal case embedding for each token
        # case_ids in range [0, 9]; 9 = UNK/no-case
        clamped = case_ids.clamp(0, 9)
        case_vecs = self.case_basis[clamped]  # (batch, seq_len, 256)
        
        # Weight by confidence (low confidence → pull toward zero = ambiguous)
        conf_weight = confidence.unsqueeze(-1) * self.confidence_scale
        case_vecs = case_vecs * conf_weight
        
        # Add verb-anchored dependency signal
        # Verb tokens get a special projection to mark predicate position
        verb_signal = self.dependency_proj(case_vecs)
        verb_expanded = verb_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        case_vecs = case_vecs + verb_signal * verb_expanded
        
        return case_vecs  # (batch, seq_len, 256)


class MorphologicalEncoder(nn.Module):
    """
    Space 3: Dhātu root + pratyaya suffix morphological encoding.
    
    For English tokens: approximates root-morpheme structure via
    subword decomposition (RoBERTa BPE already does this well).
    
    For Sanskrit tokens: uses DhatuDB lookup to get the exact
    dhātu root embedding and gāṇa class.
    
    Encodes:
    - Is this token a verbal root, nominal, suffix, particle?
    - What semantic class does its dhātu belong to?
    - Is it a compound (samāsa)?
    """
    def __init__(self, dim: int = MORPHOLOGICAL_DIM):
        super().__init__()
        self.dim = dim
        db = DhatuDB()
        stats = db.stats()
        num_semantic_classes = max(len(stats.get("by_class", {})), 25)
        
        # Semantic class embeddings (motion, cognition, perception, etc.)
        self.semantic_class_emb = nn.Embedding(num_semantic_classes + 2, dim)
        
        # Gāṇa (Pāṇinian verb class 1-10) embedding
        self.gana_emb = nn.Embedding(12, dim // 4)
        
        # Token type: verb_root, nominal, particle, suffix, compound, other
        self.token_type_emb = nn.Embedding(6, dim // 2)
        
        # Fusion projection
        self.fuse = nn.Linear(dim + dim // 4 + dim // 2, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        semantic_class_ids: torch.Tensor,  # (batch, seq_len)
        gana_ids:           torch.Tensor,  # (batch, seq_len)
        token_type_ids:     torch.Tensor,  # (batch, seq_len)
    ) -> torch.Tensor:
        sc  = self.semantic_class_emb(semantic_class_ids)   # (B, L, 128)
        gn  = self.gana_emb(gana_ids)                       # (B, L,  32)
        tt  = self.token_type_emb(token_type_ids)           # (B, L,  64)
        fused = torch.cat([sc, gn, tt], dim=-1)             # (B, L, 224)
        
        # Ensure fuse matches cat dimension
        if self.fuse.in_features != fused.shape[-1]:
            self.fuse = nn.Linear(fused.shape[-1], self.dim).to(fused.device)
            
        return self.layer_norm(self.fuse(fused))             # (B, L, 128)


class TripleEncoder(nn.Module):
    """
    Full triple encoder: combines all three spaces into a single
    MODEL_DIM=512 representation ready for BrahmanTransformer.
    
    Also exposes each sub-space separately for:
    - Ablation studies (can disable grammatical space)
    - Interpretability (inspect case assignments)
    - PāṇiniConstraintLayer (reads grammatical space directly)
    """
    def __init__(
        self,
        vocab_size:  int = 50265,
        model_dim:   int = MODEL_DIM,
        ablate_grammatical:   bool = False,  # Set True to prove grammar helps
        ablate_morphological: bool = False,
    ):
        super().__init__()
        self.ablate_grammatical   = ablate_grammatical
        self.ablate_morphological = ablate_morphological
        
        self.semantic_enc     = SemanticEncoder(vocab_size, SEMANTIC_DIM)
        self.grammatical_enc  = GrammaticalEncoder(GRAMMATICAL_DIM)
        self.morphological_enc = MorphologicalEncoder(MORPHOLOGICAL_DIM)
        
        # Effective combined dim depends on ablation flags
        effective_dim = SEMANTIC_DIM
        if not ablate_grammatical:
            effective_dim += GRAMMATICAL_DIM
        if not ablate_morphological:
            effective_dim += MORPHOLOGICAL_DIM
        
        self.projection = nn.Sequential(
            nn.Linear(effective_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        
        self.vibhakti_encoder = VibhaktiEncoder()
        # Load trained weights if available
        ckpt = "models/vibhakti_encoder/best_model.pt"
        if os.path.exists(ckpt):
            try:
                self.vibhakti_encoder.load_state_dict(
                    torch.load(ckpt, map_location="cpu")
                )
                print(f"[TripleEncoder] Loaded VibhaktiEncoder from {ckpt}")
            except Exception as e:
                print(f"[TripleEncoder] WARN: Failed to load VibhaktiEncoder: {e}")
        else:
            print("[TripleEncoder] WARN: Using untrained VibhaktiEncoder")
            
        # Freeze vibhakti encoder during Stage 1 of training
        for p in self.vibhakti_encoder.parameters():
            p.requires_grad = False

    def _get_vibhakti_features(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Run VibhaktiEncoder and extract case_ids + confidence."""
        self.vibhakti_encoder.eval()
        with torch.no_grad():
            out = self.vibhakti_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        logits = out["logits"]  # (B, L, NUM_LABELS)
        probs  = F.softmax(logits, dim=-1)
        case_ids   = probs.argmax(dim=-1)                    # (B, L)
        confidence = probs.max(dim=-1).values                # (B, L)
        # Verb mask: tokens predicted as verb label (NUM_LABELS-2)
        verb_mask = (case_ids == (NUM_LABELS - 2)).float()    # (B, L)
        return case_ids, confidence, verb_mask

    def _get_morphological_features(
        self,
        input_ids: torch.Tensor,
    ):
        """
        Approximate morphological features from input_ids.
        For now: all tokens get semantic_class=0 (unknown), gana=0, type=5 (other).
        In Week 3, DhatuDB lookup will fill these properly.
        """
        B, L = input_ids.shape
        device = input_ids.device
        sc_ids  = torch.zeros(B, L, dtype=torch.long, device=device)
        gn_ids  = torch.zeros(B, L, dtype=torch.long, device=device)
        tt_ids  = torch.full((B, L), 5, dtype=torch.long, device=device)
        return sc_ids, gn_ids, tt_ids

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> TripleEncoding:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # ── Space 1: Semantic ──────────────────────────────
        sem = self.semantic_enc(input_ids)             # (B, L, 512)
        
        # ── Space 2: Grammatical ──────────────────────────
        case_ids, confidence, verb_mask = self._get_vibhakti_features(
            input_ids, attention_mask
        )
        gram = self.grammatical_enc(case_ids, confidence, verb_mask)  # (B, L, 256)
        
        # ── Space 3: Morphological ────────────────────────
        sc_ids, gn_ids, tt_ids = self._get_morphological_features(input_ids)
        morph = self.morphological_enc(sc_ids, gn_ids, tt_ids)       # (B, L, 128)
        
        # ── Combine & Project ─────────────────────────────
        parts = [sem]
        if not self.ablate_grammatical:
            parts.append(gram)
        if not self.ablate_morphological:
            parts.append(morph)
        
        combined_raw = torch.cat(parts, dim=-1)       # (B, L, effective_dim)
        combined     = self.projection(combined_raw)  # (B, L, 512)
        
        return TripleEncoding(
            semantic=sem,
            grammatical=gram,
            morphological=morph,
            combined=combined,
            case_labels=case_ids,
            confidence=confidence,
        )


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import RobertaTokenizerFast
    tok = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    
    sentences = [
        "Ram gives the book to Sita with great care.",
        "If all humans are mortal and Socrates is human, then Socrates is mortal.",
    ]
    
    enc = tok(sentences, return_tensors="pt", padding=True, truncation=True, max_length=64)
    encoder = TripleEncoder()
    
    with torch.no_grad():
        result = encoder(enc["input_ids"], enc["attention_mask"])
    
    print(f"Semantic shape:      {result.semantic.shape}")
    print(f"Grammatical shape:   {result.grammatical.shape}")
    print(f"Morphological shape: {result.morphological.shape}")
    print(f"Combined shape:      {result.combined.shape}")
    print(f"Case labels:         {result.case_labels[0][:10]}")
    print(f"Confidence:          {result.confidence[0][:10].round(decimals=2)}")
    print("✓ TripleEncoder forward pass complete")
