import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional
from tokenizer import Vibhakti, DhatuNode

class VibhaktiAttention(nn.Module):
    """
    Mathematical Self-Attention with Pāṇinian Grammatical Constraints.
    Optimized for Nvidia Tensor Cores (Cloud T4).
    Implements: Softmax((QK^T / sqrt(dk)) + M)V
    """
    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Scratch implementation of projections for maximum control
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # 1. Project to Q, K, V
        # Q, K, V shape: [batch, n_heads, seq_len, d_k]
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # Scores: [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Apply Grammatical Constraint Mask (M)
        if mask is not None:
            # Mask value 0.0 allows attention, -inf blocks it.
            scores = scores + mask
            
        # 4. Softmax and Value aggregation
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # 5. Concatenate heads and final projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)

class SanskritCoreBlock(nn.Module):
    """Transformer Block with enforced Pāṇinian structural compositionality."""
    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.attention = VibhaktiAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Residual connection 1: Attention
        x = x + self.attention(self.norm1(x), mask)
        # Residual connection 2: FeedForward
        x = x + self.ff(self.norm2(x))
        return x

class SanskritCoreModel(nn.Module):
    """
    The Pāṇinian Neuro-Symbolic Transformer.
    Bypasses statistical guessing by forcing attention through a grammatical AST.
    """
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 128, d_model)) # seq length up to 128
        
        self.layers = nn.ModuleList([
            SanskritCoreBlock(d_model, n_heads) for _ in range(num_layers)
        ])
        
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)
        x = self.embedding(tokens) + self.pos_encoding[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.output_head(x)

def generate_vibhakti_mask(batch_tokens: List[List[str]], batch_ast: List[List[DhatuNode]], op_map: Dict[str, str]) -> torch.Tensor:
    """
    Dynamically generates the 2D Sparse Grammatical Attention Mask (M).
    Returns a tensor of shape [batch, 1, seq_len, seq_len].
    """
    batch_size = len(batch_tokens)
    seq_len = max(len(t) for t in batch_tokens)
    
    # Initialize with -inf (mathematical zero after Softmax)
    mask = torch.full((batch_size, 1, seq_len, seq_len), float('-inf'))
    
    # Reverse map for operator identification
    rev_op_map = {v: k for k, v in op_map.items()}
    
    for b in range(batch_size):
        tokens = batch_tokens[b]
        nodes = batch_ast[b]
        
        # Map tokens to indices (handles multiple occurrences)
        token_to_indices = {}
        for i, tok in enumerate(tokens):
            if tok not in token_to_indices:
                token_to_indices[tok] = []
            token_to_indices[tok].append(i)
        
        # Self-attention is always allowed
        for i in range(len(tokens)):
            mask[b, 0, i, i] = 0.0
            
        # Enforce AST-defined relationships
        for node in nodes:
            target_op_symbol = rev_op_map.get(node.root_operator, node.root_operator)
            op_indices = token_to_indices.get(target_op_symbol, [])
            
            for op_idx in op_indices:
                for vib, var_name in node.arguments.items():
                    if var_name in token_to_indices:
                        arg_indices = token_to_indices[var_name]
                        for arg_idx in arg_indices:
                            # Direct Dhātu-Routing
                            mask[b, 0, op_idx, arg_idx] = 0.0
                            mask[b, 0, arg_idx, op_idx] = 0.0
                            
    return mask

if __name__ == "__main__":
    # Internal Unit Test for Neural Forward Pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- SanskritCore Neural Architecture (Cloud-v1) ---")
    print(f"Targeting Device: {device}")
    
    vocab_size = 500
    model = SanskritCoreModel(vocab_size=vocab_size).to(device)
    
    # Mock data: "X > Y"
    tokens_text = ["X", ">", "Y"]
    tokens_tensor = torch.randint(0, vocab_size, (1, 3)).to(device)
    
    # Mock AST from tokenizer
    from tokenizer import PaninianTokenizer
    tokenizer = PaninianTokenizer()
    ast = [tokenizer.tokenize("X > Y")]
    
    mask = generate_vibhakti_mask([tokens_text], [ast], tokenizer.operator_map).to(device)
    
    output = model(tokens_tensor, mask)
    print(f"Forward Pass Status: SUCCESS")
    print(f"Output Matrix Shape: {output.shape}")
