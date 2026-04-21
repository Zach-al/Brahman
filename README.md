# Brahman (formerly SanskritCore)
**Neuro-Symbolic Reasoning via Pāṇinian Intermediate Representation**

Standard Large Language Models (LLMs) rely on statistical Byte-Pair Encoding (BPE) and dense all-to-all attention. They require billions of parameters simply to "memorize" logic and resolve linguistic ambiguity in languages like English.

**Brahman** fundamentally alters this architecture. Instead of guessing logical relationships, Brahman utilizes a 2,500-year-old formal generative grammar—Pāṇini’s *Aṣṭādhyāyī*—as a strictly typed Intermediate Representation (IR). 

By mapping English logic into a Pāṇinian Abstract Syntax Tree (AST), Brahman forces mathematical compositionality directly into the neural network's forward pass using custom **Vibhakti Attention Masking**.

## 🧠 Core Architecture

The pipeline consists of four integrated modules:

1. **The Pāṇinian Lexer (`tokenizer.py`)**
   Bypasses subword tokenization. Parses input into a strict AST where root operators (*Dhātus*) accept variables strictly bound by 8 Sanskrit computational cases (*Vibhaktis*).

2. **Vibhakti Attention (`model.py`)**
   A custom transformer block that generates a dynamic, sparse grammar mask. Variables (*Kartā/Agent*) are mathematically forced to route their attention through their assigned operators (*Kriyā/Verb*), preventing hallucinated cross-attention. Unrelated tokens are masked with `-inf`.

3. **Synthetic Logic Generator (`dataset.py`)**
   A programmatic "clean room" data generator that creates mathematically perfect logical syllogisms (Transitive, Modus Ponens) and explicitly pairs them with their Pāṇinian AST counterparts for training.

4. **The Execution Engine (`train.py`)**
   A highly optimized PyTorch training loop engineered for seamless execution on both Apple Silicon (`mps`) and Nvidia Data Centers (`cuda` with AMP).

## 🚀 Getting Started (Cloud / Kaggle)

Brahman is optimized for headless execution on cloud GPUs. 

1. Clone this repository into your Kaggle Notebook:
   ```bash
   !git clone https://github.com/YOUR_USERNAME/Brahman.git
   %cd Brahman
   ```
2. Install mathematical dependencies:
   ```bash
   !pip install torch sympy networkx
   ```
3. Run the Kaggle-Optimized training loop:
   ```bash
   !python3 train.py --full
   ```

## 🔬 Scientific Evaluation (Ablation)
To prove the efficiency of the Pāṇinian IR over standard transformer architectures, the training script includes an `--ablation` flag. Running `python3 train.py --ablation` strips the Vibhakti mask, replacing it with standard causal padding to establish a statistical baseline for logical reasoning tasks.

---
*Language shapes thought. Formal language enforces formal reasoning.*
