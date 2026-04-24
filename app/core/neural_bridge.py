"""
Kāraka Neural Bridge — Production Module.
Device-agnostic inference (MPS for local, CPU for Railway).
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional
from app.core.panini_engine import PaniniEngine


KARAKA_LABELS = ['Kartr', 'Karman', 'Karana', 'Sampradana', 'Apadana', 'Adhikarana']
IDX_TO_KARAKA = {i: k for i, k in enumerate(KARAKA_LABELS)}
KARAKA_TO_IDX = {k: i for i, k in enumerate(KARAKA_LABELS)}

# Kāraka → legally allowed Vibhakti (The Law)
KARAKA_TO_VIBHAKTI = {
    'Kartr': ['nominative', 'instrumental'],
    'Karman': ['accusative'],
    'Karana': ['instrumental'],
    'Sampradana': ['dative'],
    'Apadana': ['ablative'],
    'Adhikarana': ['locative']
}


class KarakaBridge(nn.Module):
    """
    Siamese network with root and Kāraka prediction heads.
    Loaded once at startup, runs inference on every request.
    """

    def __init__(self, model_name: str = "distilbert-base-multilingual-cased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size
        self.root_head = nn.Linear(self.hidden_dim, 2000)
        self.karaka_head = nn.Linear(self.hidden_dim, 6)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        root_logits = self.root_head(last_hidden_state)
        karaka_logits = self.karaka_head(last_hidden_state)
        return root_logits, karaka_logits


def get_device() -> torch.device:
    """Auto-detect: MPS for local dev, CPU for Railway/cloud."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        # Optimize CPU threads for cloud deployment
        import os
        num_threads = int(os.environ.get("TORCH_NUM_THREADS", "4"))
        torch.set_num_threads(num_threads)
        return torch.device("cpu")


def load_bridge(weights_path: str, device: torch.device) -> KarakaBridge:
    """Load the trained KarakaBridge model weights."""
    model = KarakaBridge()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def verify_karaka_prediction(
    word: str,
    predicted_karaka: str,
    engine: PaniniEngine
) -> Tuple[bool, List[str], List[str]]:
    """
    Check neural prediction against Pāṇinian morphological law.
    Returns (is_valid, detected_vibhaktis, allowed_vibhaktis).
    """
    analysis = engine.segment_word(word)

    if analysis['pos'] != 'noun':
        return True, [], []

    detected_vibhaktis = analysis.get('vibhakti', [])
    if isinstance(detected_vibhaktis, str):
        detected_vibhaktis = [detected_vibhaktis]

    allowed_vibhaktis = KARAKA_TO_VIBHAKTI.get(predicted_karaka, [])

    for v in detected_vibhaktis:
        if v in allowed_vibhaktis:
            return True, detected_vibhaktis, allowed_vibhaktis

    return False, detected_vibhaktis, allowed_vibhaktis
