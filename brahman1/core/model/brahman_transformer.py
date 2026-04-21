"""
BrahmanTransformer: Full model integrating all Week 2 components.

Architecture:
  Input tokens
       ↓
  TripleEncoder      ← Semantic + Grammatical + Morphological spaces
       ↓
  N × BrahmanBlock:
    MultiHeadAttention
         ↓
    PāṇiniConstraintLayer  ← Biases attention by case-role rules
         ↓
    FFN + LayerNorm
       ↓
  NeuroSymbolicFusion  ← Blends neural + Z3 symbolic outputs
       ↓
  Output logits (vocab distribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import sys

# Ensure project root is in path
sys.path.insert(0, ".")

from core.representation.triple_encoder import TripleEncoder, MODEL_DIM
from core.model.panini_constraint_layer import PāṇiniConstraintLayer
from core.model.neurosymbolic_fusion    import NeuroSymbolicFusion, FusionOutput


class BrahmanAttention(nn.Module):
    """
    Multi-head attention with PāṇiniConstraintLayer injected into scores.
    Standard scaled dot-product attention, biased by grammatical rules.
    """
    def __init__(
        self,
        model_dim:   int  = MODEL_DIM,
        num_heads:   int  = 8,
        dropout:     float = 0.1,
        ablate_panini: bool = False,
    ):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = model_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        
        self.q = nn.Linear(model_dim, model_dim)
        self.k = nn.Linear(model_dim, model_dim)
        self.v = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.panini = PāṇiniConstraintLayer(num_heads=num_heads, disabled=ablate_panini)

    def forward(
        self,
        x:           torch.Tensor,            # (B, L, D)
        case_ids:    torch.Tensor,             # (B, L)
        verb_mask:   torch.Tensor,             # (B, L)
        attn_mask:   Optional[torch.Tensor] = None,  # (B, 1, L, L)
    ) -> torch.Tensor:
        B, L, D = x.shape
        
        def split_heads(t):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        Q, K, V = split_heads(self.q(x)), split_heads(self.k(x)), split_heads(self.v(x))
        
        # Raw attention scores
        scores = (Q @ K.transpose(-2, -1)) / self.scale  # (B, H, L, L)
        
        # Apply PāṇiniConstraintLayer bias
        scores = self.panini(scores, case_ids, verb_mask)
        
        if attn_mask is not None:
            # Broadcast mask: (B, 1, L, L) or (1, 1, L, L)
            scores = scores + attn_mask
        
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = (weights @ V).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class BrahmanBlock(nn.Module):
    """Single transformer block with Pāṇini-biased attention."""
    def __init__(
        self,
        model_dim:     int   = MODEL_DIM,
        num_heads:     int   = 8,
        ffn_dim:       int   = 2048,
        dropout:       float = 0.1,
        ablate_panini: bool  = False,
    ):
        super().__init__()
        self.attn    = BrahmanAttention(model_dim, num_heads, dropout, ablate_panini)
        self.norm1   = nn.LayerNorm(model_dim)
        self.norm2   = nn.LayerNorm(model_dim)
        self.ffn     = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, case_ids, verb_mask, attn_mask=None):
        x = x + self.attn(self.norm1(x), case_ids, verb_mask, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class BrahmanTransformer(nn.Module):
    """
    Full Brahman-1 model.
    
    Configuration for Phase 2 POC (~125M params — fast iteration):
      model_dim = 512, num_heads = 8, num_layers = 6, ffn_dim = 2048
    """
    def __init__(
        self,
        vocab_size:    int   = 50265,
        model_dim:     int   = MODEL_DIM,   # 512
        num_heads:     int   = 8,
        num_layers:    int   = 6,           # 6 for fast POC
        ffn_dim:       int   = 2048,
        max_seq_len:   int   = 512,
        dropout:       float = 0.1,
        ablate_panini: bool  = False,       # Ablation experiment switch
        ablate_grammatical:   bool = False, # Disable grammatical space
        ablate_morphological: bool = False, # Disable morphological space
    ):
        super().__init__()
        self.model_dim = model_dim
        
        # ── Encoding ──────────────────────────────────────
        self.triple_encoder = TripleEncoder(
            vocab_size=vocab_size,
            model_dim=model_dim,
            ablate_grammatical=ablate_grammatical,
            ablate_morphological=ablate_morphological,
        )
        
        # ── Transformer Blocks ────────────────────────────
        self.blocks = nn.ModuleList([
            BrahmanBlock(model_dim, num_heads, ffn_dim, dropout, ablate_panini)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(model_dim)
        
        # ── Output ────────────────────────────────────────
        self.fusion = NeuroSymbolicFusion(model_dim, vocab_size)
        
        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sir_list:       Optional[list] = None,
    ) -> FusionOutput:
        B, L = input_ids.shape
        
        # ── Triple Encoding ───────────────────────────────
        enc = self.triple_encoder(input_ids, attention_mask)
        x         = enc.combined      # (B, L, 512)
        case_ids  = enc.case_labels   # (B, L)
        verb_mask = (case_ids == 8).float()  # crude verb mask
        
        # Build causal attention mask for autoregressive generation
        causal = torch.triu(
            torch.full((L, L), -1e4, device=input_ids.device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        
        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
            pad_mask = pad_mask * -1e4
            combined_mask = causal + pad_mask
        else:
            combined_mask = causal
        
        # ── Transformer Blocks ────────────────────────────
        for block in self.blocks:
            x = block(x, case_ids, verb_mask, combined_mask)
        
        x = self.final_norm(x)
        
        # ── CLS feature for task type classification ──────
        cls_features = x[:, 0, :]   # (B, model_dim)
        
        # ── Neuro-Symbolic Fusion ─────────────────────────
        return self.fusion(x, cls_features, sir_list)

    def count_parameters(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        triple = sum(p.numel() for p in self.triple_encoder.parameters())
        blocks = sum(p.numel() for b in self.blocks for p in b.parameters())
        fusion = sum(p.numel() for p in self.fusion.parameters())
        return {
            "total_M":           round(total / 1e6, 1),
            "triple_encoder_M":  round(triple / 1e6, 1),
            "transformer_M":     round(blocks / 1e6, 1),
            "fusion_M":          round(fusion / 1e6, 1),
        }


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    
    from transformers import RobertaTokenizerFast
    tok = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    
    sentences = [
        "All humans are mortal. Socrates is human. Therefore Socrates is mortal.",
        "The teacher gives a book to the student with great care.",
        "If it rains then the ground becomes wet.",
    ]
    
    enc = tok(sentences, return_tensors="pt", padding=True,
              truncation=True, max_length=64)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    
    # ── Test 1: Full model (all components active) ─────
    model = BrahmanTransformer(num_layers=4).to(DEVICE)
    params = model.count_parameters()
    print(f"\nModel parameters: {params}")
    
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    
    print(f"\nOutput logits shape: {out.logits.shape}")
    print(f"Alpha (neural trust): {out.alpha.round(decimals=3)}")
    
    # ── Test 2: Ablation — no Pāṇini constraints ───────
    ablated_model = BrahmanTransformer(
        num_layers=4, ablate_panini=True
    ).to(DEVICE)
    
    with torch.no_grad():
        ablated_out = ablated_model(enc["input_ids"], enc["attention_mask"])
    
    print(f"\nAblated model logits shape: {ablated_out.logits.shape}")
    print("✓ Ablation test passed — model runs without Pāṇini constraints")
    
    # ── Test 3: Parameter efficiency check ─────────────
    print(f"\nParam efficiency check:")
    print(f"  Full model:    {params['total_M']}M parameters")
    for size in [6, 12]:
        m = BrahmanTransformer(num_layers=size)
        p = m.count_parameters()
        print(f"  {size}-layer model: {p['total_M']}M params")
    
    print("\n✓ ALL WEEK 2 TESTS PASSED")
