"""
All 5 multi-task training objectives for Brahman-1.

Weights (from Master Prompt):
  Objective 1: Language Modeling          weight=1.0
  Objective 2: Semantic Role Prediction   weight=2.5
  Objective 3: Logical Entailment         weight=3.0  ← highest
  Objective 4: Compound Decomposition     weight=1.5
  Objective 5: Cross-lingual Alignment    weight=2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ObjectiveLosses:
    lm:         torch.Tensor
    srl:        torch.Tensor
    entailment: torch.Tensor
    compound:   torch.Tensor
    alignment:  torch.Tensor
    total:      torch.Tensor

    def to_dict(self) -> Dict[str, float]:
        return {
            "lm":         self.lm.item(),
            "srl":        self.srl.item(),
            "entailment": self.entailment.item(),
            "compound":   self.compound.item(),
            "alignment":  self.alignment.item(),
            "total":      self.total.item(),
        }


# ─── Objective 1: Language Modeling ─────────────────────────

class LanguageModelingLoss(nn.Module):
    """Standard causal next-token prediction loss."""
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits:  torch.Tensor,   # (B, L, V)
        labels:  torch.Tensor,   # (B, L)  — shifted input_ids
    ) -> torch.Tensor:
        B, L, V = logits.shape
        # Shift: predict token t from tokens 0..t-1
        shift_logits = logits[:, :-1, :].reshape(-1, V)
        shift_labels = labels[:, 1:].reshape(-1)
        return self.ce(shift_logits, shift_labels)


# ─── Objective 2: Semantic Role Prediction ──────────────────

class SemanticRoleLoss(nn.Module):
    """
    Token-level cross-entropy over 10-class vibhakti labels.
    Directly trains the grammatical encoding space.
    Uses class weights to handle imbalance (most tokens = CASE_OTHER).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Upweight grammatically meaningful cases (0-7) vs. other (9)
        weights = torch.ones(num_classes)
        weights[9] = 0.2   # CASE_OTHER — very common, downweight
        weights[8] = 1.5   # CASE_VERB  — important anchor
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        self.num_classes = num_classes

    def forward(
        self,
        srl_logits: torch.Tensor,  # (B, L, 10) from VibhaktiEncoder head
        srl_labels: torch.Tensor,  # (B, L) vibhakti case IDs
    ) -> torch.Tensor:
        B, L, C = srl_logits.shape
        return self.ce(
            srl_logits.reshape(-1, C),
            srl_labels.reshape(-1).clamp(0, self.num_classes - 1)
        )


# ─── Objective 3: Logical Entailment ────────────────────────

class LogicalEntailmentLoss(nn.Module):
    """
    Binary classification: does hypothesis follow from premise?
    Uses CLS token representation for classification.
    
    This is the highest-weight objective (3.0) because it directly
    trains the neuro-symbolic reasoning capability — the core hypothesis.
    
    Also includes auxiliary FOL consistency loss:
    If Z3 proves a conclusion, the model's output probability for
    the entailment label should be > 0.9 (soft supervision from theorem prover).
    """
    def __init__(self, model_dim: int = 512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(model_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2),
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        premise_cls:    torch.Tensor,   # (B, D) CLS of premise
        hypothesis_cls: torch.Tensor,   # (B, D) CLS of hypothesis
        labels:         torch.Tensor,   # (B,) 0=contradiction, 1=entailment
        z3_proofs:      Optional[torch.Tensor] = None,  # (B,) soft targets from Z3
    ) -> torch.Tensor:
        combined = torch.cat([premise_cls, hypothesis_cls], dim=-1)  # (B, 2D)
        logits   = self.classifier(combined)                          # (B, 2)
        hard_loss = self.ce(logits, labels)

        # Z3 consistency: if Z3 proved entailment, push prob toward 1.0
        soft_loss = torch.tensor(0.0, device=logits.device)
        if z3_proofs is not None:
            probs = F.softmax(logits, dim=-1)[:, 1]  # prob of entailment
            soft_loss = F.mse_loss(probs, z3_proofs.float())

        return hard_loss + 0.3 * soft_loss


# ─── Objective 4: Compound Decomposition ────────────────────

class CompoundDecompositionLoss(nn.Module):
    """
    Trains morphological encoding space to decompose Sanskrit compounds.
    
    Given: compound token embedding
    Predict: head morpheme class + modifier morpheme class
    
    For Phase 2 POC: simplified to binary classification
    "Is this token a compound head?" (trains morphological attention).
    
    Full compound decomposition in Week 4 with DCS compound annotations.
    """
    def __init__(self, model_dim: int = 512):
        super().__init__()
        self.head_detector = nn.Linear(model_dim, 2)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        morphological_enc: torch.Tensor,  # (B, L, 128) from TripleEncoder
        compound_labels:   torch.Tensor,  # (B, L) 1=compound head, 0=non-head, -100=ignore
    ) -> torch.Tensor:
        # Project morphological space to head prediction
        # Handle dim mismatch if any
        if morphological_enc.shape[-1] < 512:
            morphological_enc = F.pad(morphological_enc, (0, 512 - morphological_enc.shape[-1]))
        elif morphological_enc.shape[-1] > 512:
            morphological_enc = morphological_enc[:, :, :512]
            
        logits = self.head_detector(morphological_enc)
        
        # Check if there are any valid labels
        mask = (compound_labels != -100)
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        return self.ce(
            logits.reshape(-1, 2),
            compound_labels.reshape(-1)
        )


# ─── Objective 5: Cross-lingual Semantic Alignment ──────────

class CrosslingualAlignmentLoss(nn.Module):
    """
    English sentence and its Sanskrit SIR representation must map
    to the same point in the shared semantic space.
    
    Implements InfoNCE (NT-Xent) contrastive loss:
    - Positive pair: (English CLS, SIR CLS from same sentence)
    - Negative pairs: all other English-SIR combinations in batch
    
    This ensures the translation layer (English→SIR) is lossless.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(
        self,
        english_cls: torch.Tensor,  # (B, D) CLS of English sentence
        sir_cls:     torch.Tensor,  # (B, D) CLS of corresponding SIR
    ) -> torch.Tensor:
        B = english_cls.shape[0]

        # L2 normalize
        en  = F.normalize(english_cls, dim=-1)  # (B, D)
        sir = F.normalize(sir_cls,     dim=-1)  # (B, D)

        # Similarity matrix: (B, B)
        sim = torch.mm(en, sir.T) / self.temp

        # Targets: diagonal is the positive pair
        targets = torch.arange(B, device=sim.device)
        loss_en  = F.cross_entropy(sim,   targets)
        loss_sir = F.cross_entropy(sim.T, targets)
        return (loss_en + loss_sir) / 2.0


# ─── Combined Multi-Task Loss ─────────────────────────────────

class BrahmanMultiTaskLoss(nn.Module):
    """
    Combines all 5 objectives with specified weights.
    Implements loss balancing via uncertainty weighting (Kendall et al. 2018)
    to prevent one objective from dominating.
    """
    WEIGHTS = {
        "lm":         1.0,
        "srl":        2.5,
        "entailment": 3.0,
        "compound":   1.5,
        "alignment":  2.0,
    }

    def __init__(self, model_dim: int = 512, use_uncertainty_weighting: bool = True):
        super().__init__()
        self.lm_loss        = LanguageModelingLoss()
        self.srl_loss       = SemanticRoleLoss()
        self.entailment_loss = LogicalEntailmentLoss(model_dim)
        self.compound_loss  = CompoundDecompositionLoss(model_dim)
        self.alignment_loss = CrosslingualAlignmentLoss()

        # Learnable log variance for uncertainty weighting
        if use_uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                k: nn.Parameter(torch.zeros(1)) for k in self.WEIGHTS
            })
        else:
            self.log_vars = None

    def _weight(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        base_weight = self.WEIGHTS[name]
        if self.log_vars is not None:
            log_var = self.log_vars[name]
            precision = torch.exp(-log_var)
            return base_weight * precision * loss + 0.5 * log_var
        return base_weight * loss

    def forward(
        self,
        # LM
        lm_logits:   torch.Tensor, lm_labels: torch.Tensor,
        # SRL
        srl_logits:  torch.Tensor, srl_labels: torch.Tensor,
        # Entailment
        premise_cls: torch.Tensor, hypothesis_cls: torch.Tensor, entailment_labels: torch.Tensor,
        # Compound
        morph_enc:   torch.Tensor, compound_labels: torch.Tensor,
        # Alignment
        english_cls: torch.Tensor, sir_cls: torch.Tensor,
        # Optional
        z3_proofs:   Optional[torch.Tensor] = None,
    ) -> ObjectiveLosses:

        lm   = self.lm_loss(lm_logits, lm_labels)
        srl  = self.srl_loss(srl_logits, srl_labels)
        ent  = self.entailment_loss(premise_cls, hypothesis_cls, entailment_labels, z3_proofs)
        comp = self.compound_loss(morph_enc, compound_labels)
        aln  = self.alignment_loss(english_cls, sir_cls)

        total = (self._weight("lm",         lm)
               + self._weight("srl",        srl)
               + self._weight("entailment", ent)
               + self._weight("compound",   comp)
               + self._weight("alignment",  aln))

        return ObjectiveLosses(lm=lm, srl=srl, entailment=ent,
                               compound=comp, alignment=aln, total=total)
