# Transformers-from-scratch  

**Author:** Sankalp Bahad  
**Language:** Python (PyTorch)  
**Description:** A ground-up implementation of the Transformer architecture for sequence-to-sequence tasks — covering data processing, training, evaluation and output logging with BLEU scores.

---

## Table of Contents  
1. [Project Motivation](#project-motivation)  
2. [Features & Scope](#features-&-scope)  
3. [Repository Structure](#repository-structure)  
4. [Getting Started](#getting-started)  
   1. [Prerequisites](#prerequisites)  
   2. [Installation](#installation)  
   3. [Running the Code](#running-the-code)  
5. [Usage](#usage)  
6. [Results & Evaluation](#results-&-evaluation)  
7. [Analysis](#analysis)  
8. [Future Work](#future-work)  
9. [License & Credits](#license-&-credits)  

---

## Project Motivation  
While many frameworks provide high-level Transformer implementations, this project builds the architecture **from scratch** in PyTorch, to deepen understanding of the internals: multi-head self-attention, positional encodings, masking, encoder-decoder interactions and decoding. Ideal for learning, experimentation and adapting to custom sequence modelling tasks.

---

## Features & Scope  
- Fully custom implementation of the Transformer (no reliance on `torch.nn.Transformer` or similar abstractions).  
- End-to-end pipeline: data loading → tokenisation (via `nltk`) → training loop → evaluation (via `sacrebleu`).  
- Outputs saved: `train_outputs.txt` with predictions & BLEU scores, `test_outputs.txt` with evaluation results.  
- Includes a `Report.pdf` summarising graphs, analysis and performance metrics of the assignment.  
- Simple but extensible: you can plug in new datasets, modify hyper-parameters, or extend to new languages/tasks.

---

## Repository Structure  
├─ final_code.py # main training & evaluation script
├─ train_outputs.txt # logged training outputs: [prediction]\t[true]\t[BLEU_Score]
├─ test_outputs.txt # logged test outputs: [prediction]\t[true]\t[BLEU_Score]
├─ Report.pdf # project report with graphs & analysis
└─ README.md # this file

---

## Getting Started  

### Prerequisites  
Ensure you have installed the following:  
- Python 3.x  
- PyTorch  
- NLTK (`nltk.download('punkt')` and `nltk.download('stopwords')`)  
- NumPy  
- sacrebleu  
- tqdm  
- Additional standard libraries: `string`, `math`  

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/SankalpBahad/Transformers-from-scratch.git  
   cd Transformers-from-scratch
   ```
Install dependencies (if using pip):

pip install torch numpy nltk sacrebleu tqdm  
Download or link dataset & trained model (as referenced in Report) before running.

### Running the Code
To train the model:
  ```
  python final_code.py  
  ```
This will run the training loop, log intermediate results to train_outputs.txt, and save the final model files (for which links are provided in the Report).

To evaluate on the test set:
Execute the same script (assuming it triggers evaluation at end or separate evaluation section) — results will appear in test_outputs.txt.

### Usage
You can swap in your own dataset by modifying the data-loading section in final_code.py.

You can adjust model hyper-parameters (number of heads, layers, embedding dimension, learning rate) by editing the script.

Use the output files (train_outputs.txt, test_outputs.txt) to analyse model performance — each line contains predicted sentence, true sentence and BLEU score.

### Results & Evaluation
The project logs BLEU scores per sentence for both training and test sets, enabling fine-grained performance tracking.

The Report.pdf provides visualisations of loss curves, BLEU score distributions, and qualitative examples of correct/incorrect predictions.

Based on these results, you can interpret how model depth, heads or other hyper-parameters affect translation/reconstruction quality.

### Analysis
Key learnings from this implementation include:

Multi-head self-attention implementation and how different heads may attend to distinct linguistic patterns.

Importance of positional encoding when sequence order matters.

Role of masking in encoder-decoder architecture for autoregressive generation.

Relation between training hyper-parameters and BLEU score stability.

Challenges in low-resource or custom domains: overfitting, generalisation, and interpreting transformer internals.
