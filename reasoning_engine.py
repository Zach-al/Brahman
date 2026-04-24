import torch
from transformers import AutoTokenizer
from brahman2 import PaniniRuleEngine
from neural_bridge import KarakaBridge, verify_karaka_prediction
import json

class AnvayaBodhaEngine:
    def __init__(self):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        self.bridge = KarakaBridge().to(self.device)
        self.bridge.eval()
        
        # We need the rule engine to test morphology
        # Using dummy sutras; in a full implementation we would query brahman_v2.db
        self.symbolic_engine = PaniniRuleEngine(dhatus=[{'root': 'गम्'}], sutras=[])
        
        self.karaka_labels = ['Kartr', 'Karman', 'Karana', 'Sampradana', 'Apadana', 'Adhikarana']
        
    def evaluate(self, sentence: str) -> dict:
        words = sentence.strip().split()
        
        results = {
            "sentence": sentence,
            "status": "Valid Anvaya",
            "violations": [],
            "roles": {},
            "symbolic_parse": {}
        }
        
        has_verb = False
        has_karman = False
        
        for i, word in enumerate(words):
            word_inputs = self.tokenizer(word, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _, w_logits = self.bridge(word_inputs.input_ids, word_inputs.attention_mask)
            
            # Simulated Neural Prediction:
            # An untrained network will produce random logits. 
            # We'll map the argmax to our 6 Karakas to simulate the "guess".
            pred_idx = w_logits[0][1].argmax().item() if w_logits.size(1) > 1 else 0
            predicted_karaka = self.karaka_labels[pred_idx % len(self.karaka_labels)]
            
            # The Law: Symbolic Parse
            analysis = self.symbolic_engine.segment_word(word)
            results['symbolic_parse'][word] = analysis
            
            if analysis['pos'] == 'verb':
                has_verb = True
                results['roles'][word] = "Verb"
            elif analysis['pos'] == 'noun':
                # The Gatekeeper: Verify Neural Guess vs Symbolic Law
                is_valid = verify_karaka_prediction(word, predicted_karaka, self.symbolic_engine)
                
                if not is_valid:
                    # Segfault the prediction!
                    results['violations'].append(
                        f"Linguistic Violation: '{word}' cannot be '{predicted_karaka}' due to incompatible Vibhakti {analysis.get('vibhakti')}."
                    )
                    results['status'] = "Linguistic Violation (Equivocation Fallacy)"
                else:
                    results['roles'][word] = predicted_karaka
                    if predicted_karaka == 'Karman':
                        has_karman = True
            else:
                results['roles'][word] = "Unknown"
                        
        # 2. Check Ākāṅkṣā (Expectancy)
        # If there's a transitive verb like गच्छति (to go), it typically expects a destination/object (Karman)
        if has_verb and not has_karman and results['status'] != "Linguistic Violation (Equivocation Fallacy)":
            results['violations'].append("Ākāṅkṣā Violation: Verb expects an object (Karman) but none was legally resolved.")
            results['status'] = "Linguistic Violation (Expectancy Fallacy)"
            
        return results

if __name__ == "__main__":
    print("="*60)
    print("BRAHMAN-2.0 REASONING ENGINE: ANVAYA-BODHA")
    print("="*60)
    
    engine = AnvayaBodhaEngine()
    
    # Test 1: The standard sentence
    test_sentence_1 = "रामः वनं गच्छति"
    print(f"\n[Test 1] Analyzing: '{test_sentence_1}'")
    res_1 = engine.evaluate(test_sentence_1)
    print(json.dumps(res_1, indent=2, ensure_ascii=False))
