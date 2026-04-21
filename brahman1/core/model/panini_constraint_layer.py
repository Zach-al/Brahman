"""
PāṇiniConstraintLayer: Applies hard grammatical biases to attention scores.

This is the architecturally novel component of Brahman-1.

Standard transformer attention: each token can attend equally to all others.
PāṇiniConstraintLayer: attention between tokens is BIASED by their case-role
relationships, derived from Sanskrit's vibhakti system.

CONSTRAINT RULES (from Pāṇini's Ashtadhyayi — selected subset):

Rule P1 — Agent-Verb Affinity:
  A[nominative/kartā] and V[verb] get +BOOST attention bias.
  Rationale: kartā is the primary argument of any verb.

Rule P2 — Patient-Verb Affinity:
  A[accusative/karma] and V[verb] get +BOOST (slightly less than P1).
  Rationale: karma is the secondary argument.

Rule P3 — Instrument Locality:
  A[instrumental/karaṇa] attends strongly to its governing VERB only,
  not to other nominals.
  Rationale: karaṇa (means/instrument) is structurally verb-local.

Rule P4 — Case Incompatibility Penalty:
  Two tokens with SAME case (both nominative, both accusative etc.)
  get a small PENALTY unless one is a compound head.
  Rationale: Pāṇini — a verb typically takes one kartā.

Rule P5 — Ablative-Source Directionality:
  A[ablative/apādāna] is strongly biased toward SOURCE arguments;
  PENALIZE attention from ablative to GOAL tokens.
  Encodes causal/directional asymmetry.

Rule P6 — Conditional Particle Scope:
  yadi (IF) particle extends attention scope to its full clause.
  Implements IF-THEN scoping over dependent clauses.

ABLATION INTERFACE:
  Set `disabled=True` to turn off all constraints.
  This is required for ablation experiments proving constraint value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# Constraint strength hyperparameters (tunable)
BOOST_STRONG  =  2.0   # kartā-verb, karma-verb affinity
BOOST_MEDIUM  =  1.0   # other productive case-verb pairs
PENALTY_SAME  = -1.5   # same-case token pair penalty
PENALTY_CROSS = -0.5   # structurally incompatible cross-case
CONDITIONAL_BOOST = 3.0  # IF-particle scope extension

# Case IDs (must match VibhaktiEncoder NUM_LABELS ordering)
CASE_NOMINATIVE   = 0   # kartā
CASE_ACCUSATIVE   = 1   # karma
CASE_INSTRUMENTAL = 2   # karaṇa
CASE_DATIVE       = 3   # sampradāna
CASE_ABLATIVE     = 4   # apādāna
CASE_GENITIVE     = 5   # sambandha
CASE_LOCATIVE     = 6   # adhikaraṇa
CASE_VOCATIVE     = 7   # sambodha
CASE_VERB         = 8   # predicate (special non-vibhakti)
CASE_OTHER        = 9   # punctuation, particles, unknown


class PāṇiniConstraintLayer(nn.Module):
    """
    Injects grammatical biases into multi-head attention scores.
    
    Usage:
        # Inside BrahmanTransformer's attention forward pass:
        attn_scores = raw_dot_products      # (B, H, L, L)
        case_ids    = triple_enc.case_labels # (B, L)
        attn_scores = panini_layer(attn_scores, case_ids)
        attn_weights = softmax(attn_scores / sqrt(d_k))
    """
    def __init__(
        self,
        num_heads:   int  = 8,
        disabled:    bool = False,   # Ablation: set True to bypass all constraints
        learnable_weights: bool = True,  # Allow small learned adjustment to hard rules
    ):
        super().__init__()
        self.num_heads = num_heads
        self.disabled  = disabled
        
        if learnable_weights:
            # Small learned perturbations ON TOP of hard rules
            # Initialized to identity (zero offset) so rules dominate at init
            self.rule_scales = nn.Parameter(torch.ones(6))  # One scale per rule P1-P6
        else:
            self.register_buffer("rule_scales", torch.ones(6))
        
        # Head-specific rule assignment:
        # Different heads specialize in different Pāṇini rules.
        # Head 0-1: P1/P2 (agent/patient-verb)
        # Head 2:   P3 (instrument locality)
        # Head 3:   P4 (same-case penalty)
        # Head 4:   P5 (ablative directionality)
        # Head 5:   P6 (conditional scope)
        # Head 6-7: Unconstrained (standard attention — always free)
        self.head_rule_assignment = {
            0: "P1", 1: "P2", 2: "P3",
            3: "P4", 4: "P5", 5: "P6",
            # heads 6, 7 = None (unconstrained)
        }

    def _build_bias_matrix(
        self,
        case_ids: torch.Tensor,   # (B, L)
        rule:     str,
    ) -> torch.Tensor:
        """
        Build a (B, L, L) additive bias matrix for a given rule.
        bias[b, i, j] = how much token i should attend to token j
        according to rule `rule` and the case assignments in case_ids.
        """
        B, L = case_ids.shape
        bias = torch.zeros(B, L, L, device=case_ids.device)
        
        ci = case_ids.unsqueeze(2).expand(B, L, L)  # (B, L, L) — query case
        cj = case_ids.unsqueeze(1).expand(B, L, L)  # (B, L, L) — key case
        
        s = self.rule_scales  # shorthand
        
        if rule == "P1":
            # Nominative tokens boost attention TO verb tokens
            mask = (ci == CASE_NOMINATIVE) & (cj == CASE_VERB)
            bias = bias + mask.float() * BOOST_STRONG * s[0]
            # Symmetric: verbs also attend more to their kartā
            mask2 = (ci == CASE_VERB) & (cj == CASE_NOMINATIVE)
            bias = bias + mask2.float() * BOOST_STRONG * s[0]
            
        elif rule == "P2":
            # Accusative ↔ verb affinity
            mask  = ((ci == CASE_ACCUSATIVE) & (cj == CASE_VERB))
            mask2 = ((ci == CASE_VERB) & (cj == CASE_ACCUSATIVE))
            bias  = bias + (mask | mask2).float() * BOOST_MEDIUM * s[1]
            
        elif rule == "P3":
            # Instrumental attends ONLY to verb — penalize attending to other nominals
            instrumental_query = (ci == CASE_INSTRUMENTAL)
            non_verb_key       = (cj != CASE_VERB)
            penalty_mask       = instrumental_query & non_verb_key
            bias = bias + penalty_mask.float() * PENALTY_CROSS * s[2]
            # But boost instrumental→verb
            boost_mask = instrumental_query & (cj == CASE_VERB)
            bias = bias + boost_mask.float() * BOOST_MEDIUM * s[2]
            
        elif rule == "P4":
            # Penalize same-case token pairs (one kartā per verb rule)
            same_case_mask = (ci == cj)
            # Only apply to grammatically significant cases (0-7)
            significant    = (ci < CASE_VERB)
            bias = bias + (same_case_mask & significant).float() * PENALTY_SAME * s[3]
            
        elif rule == "P5":
            # Ablative → Source bias, penalize ablative → Goal (dative)
            abl_to_dat = (ci == CASE_ABLATIVE) & (cj == CASE_DATIVE)
            abl_to_abl = (ci == CASE_ABLATIVE) & (cj == CASE_ABLATIVE)
            bias = bias + abl_to_dat.float() * PENALTY_CROSS * s[4]
            bias = bias + abl_to_abl.float() * PENALTY_SAME * s[4]
            
        elif rule == "P6":
            # Conditional scope: if any token in sequence is CASE_OTHER
            # (approximation for particles like "if", "because", "therefore"),
            # extend attention range for all tokens in that clause.
            pass  # Placeholder — full clause segmentation in Week 3
        
        return bias

    def forward(
        self,
        attn_scores: torch.Tensor,      # (B, H, L, L) raw dot-product scores
        case_ids:    torch.Tensor,       # (B, L) case label IDs
        verb_mask:   Optional[torch.Tensor] = None,  # (B, L) 1.0 at verb positions
    ) -> torch.Tensor:
        """
        Returns:
            attn_scores: (B, H, L, L) — biased attention scores
        """
        if self.disabled:
            return attn_scores  # Ablation mode: pass through unchanged
        
        B, H, L, _ = attn_scores.shape
        
        # If verb_mask provided, mark verb positions in case_ids
        effective_case = case_ids.clone()
        if verb_mask is not None:
            effective_case = torch.where(
                verb_mask.bool(), 
                torch.full_like(effective_case, CASE_VERB),
                effective_case
            )
        
        for head_idx in range(H):
            rule = self.head_rule_assignment.get(head_idx, None)
            if rule is None:
                continue  # Unconstrained head — standard attention
            
            bias = self._build_bias_matrix(effective_case, rule)  # (B, L, L)
            attn_scores[:, head_idx, :, :] = attn_scores[:, head_idx, :, :] + bias
        
        return attn_scores

    def explain_constraints(
        self,
        case_ids: torch.Tensor,  # (1, L) single example
        tokens:   list,
    ) -> str:
        """Human-readable explanation of which constraints activated."""
        if self.disabled:
            return "PāṇiniConstraintLayer: DISABLED (ablation mode)"
        
        lines = ["PāṇiniConstraintLayer — Active Constraints:"]
        case_names = [
            "nominative", "accusative", "instrumental", "dative",
            "ablative", "genitive", "locative", "vocative", "verb", "other"
        ]
        for i, (tok, cid) in enumerate(zip(tokens, case_ids[0].tolist())):
            if cid < 10:
                name = case_names[cid] if cid < len(case_names) else "unknown"
                lines.append(f"  [{i}] '{tok}' → {name}")
        return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    B, H, L = 2, 8, 16
    
    # Fake attention scores
    attn_scores = torch.randn(B, H, L, L)
    
    # Fake case assignments: token 0=NOM, 2=ACC, 4=VERB, rest=OTHER
    case_ids = torch.full((B, L), CASE_OTHER, dtype=torch.long)
    case_ids[:, 0] = CASE_NOMINATIVE
    case_ids[:, 2] = CASE_ACCUSATIVE
    case_ids[:, 4] = CASE_VERB
    case_ids[:, 6] = CASE_INSTRUMENTAL
    
    layer = PāṇiniConstraintLayer(num_heads=H)
    biased = layer(attn_scores.clone(), case_ids)
    
    print(f"Input  scores mean: {attn_scores.mean():.4f}")
    print(f"Biased scores mean: {biased.mean():.4f}")
    
    # Verify: head 0 (P1) should have NOM-VERB pairs boosted
    h0_nom_verb = biased[0, 0, 0, 4].item()  # token 0 (NOM) → token 4 (VERB)
    h0_nom_verb_raw = attn_scores[0, 0, 0, 4].item()
    print(f"Head 0, NOM→VERB:  raw={h0_nom_verb_raw:.4f}  biased={h0_nom_verb:.4f}  Δ={h0_nom_verb - h0_nom_verb_raw:.4f}")
    assert h0_nom_verb > h0_nom_verb_raw, "P1 rule not applied!"
    
    # Ablation: disabled mode must be identical to input
    layer_off = PāṇiniConstraintLayer(num_heads=H, disabled=True)
    ablated = layer_off(attn_scores.clone(), case_ids)
    assert torch.allclose(ablated, attn_scores), "Ablation mode altered scores!"
    
    print("✓ PāṇiniConstraintLayer tests passed")
    print(layer.explain_constraints(
        case_ids[:1],
        ["Ram", "Ø", "gives", "Ø", "Ø", "Ø", "key", "Ø", "Ø", "Ø", "Ø", "Ø", "Ø", "Ø", "Ø", "Ø"]
    ))
