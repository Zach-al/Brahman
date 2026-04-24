import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from brahman2 import PaniniRuleEngine

class KarakaBridge(nn.Module):
    def __init__(self, model_name="distilbert-base-multilingual-cased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size
        
        # Head 1: Predicts the Root (Dhatu/Pratipadika)
        self.root_head = nn.Linear(self.hidden_dim, 2000) # Targeted for the Dhatupatha
        
        # Head 2: Predicts the Karaka (Semantic Role)
        # Roles: Kartr, Karman, Karana, Sampradana, Apadana, Adhikarana
        self.karaka_head = nn.Linear(self.hidden_dim, 6) 

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # [batch, seq, hidden]
        
        root_logits = self.root_head(last_hidden_state)
        karaka_logits = self.karaka_head(last_hidden_state)
        
        return root_logits, karaka_logits

def verify_karaka_prediction(word: str, predicted_karaka: str, engine: PaniniRuleEngine) -> bool:
    """
    Checks the neural prediction against Pāṇinian morphological law.
    Returns True if the prediction is morphologically possible, else False.
    """
    # 1. Map Kāraka to legally allowed Vibhakti (Cases)
    karaka_to_vibhakti = {
        'Kartr': ['nominative', 'instrumental'],
        'Karman': ['accusative'],
        'Karana': ['instrumental'],
        'Sampradana': ['dative'],
        'Apadana': ['ablative'],
        'Adhikarana': ['locative']
    }
    
    # 2. Run the word through the Symbolic Engine (The Law)
    analysis = engine.segment_word(word)
    
    # If the word isn't a noun, Karaka doesn't directly apply in this simple loop
    if analysis['pos'] != 'noun':
        return True 
        
    detected_vibhaktis = analysis.get('vibhakti', [])
    if isinstance(detected_vibhaktis, str):
        detected_vibhaktis = [detected_vibhaktis]
        
    allowed_vibhaktis = karaka_to_vibhakti.get(predicted_karaka, [])
    
    # Superposition collapse: is ANY of the detected vibhaktis allowed for this Karaka?
    for v in detected_vibhaktis:
        if v in allowed_vibhaktis:
            return True
            
    return False

if __name__ == "__main__":
    # Setup for Bhupen's MacBook Pro
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = KarakaBridge().to(device)
    print(f"✓ Neural Bridge initialized on {device}")
