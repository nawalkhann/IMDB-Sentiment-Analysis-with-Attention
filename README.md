# IMDB Sentiment Analysis with Attention 🎬

A sentiment classifier for IMDB movie reviews using **Bidirectional LSTMs with Additive (Bahdanau) and Multiplicative (Luong) Attention**, built in PyTorch.

**Course:** MGT488 – Advanced Deep Learning
**Dataset:** IMDB Movie Reviews (50,000 labeled reviews)

## 🎯 Project Overview
This project classifies movie reviews as **positive** or **negative** by:
1. Preprocessing raw review text (cleaning, negation tagging, elongation normalization)
2. Encoding text into padded integer sequences
3. Training a BiLSTM + Attention model to classify sentiment
4. Comparing attention mechanisms, loss functions, and optimizers across 4 experiments
5. Visualizing attention weights to interpret model decisions

## 🧰 Tech Stack
- Python
- PyTorch
- Pandas, NumPy
- Scikit-learn (metrics, train/test split)
- Matplotlib (visualization)

## 📂 Project Structure
```
├── IMDB_Sentiment_Attention.ipynb   # Main notebook
├── requirements.txt                 # Dependencies
└── README.md
```

## 🔬 Workflow
1. **Preprocessing** – Lowercase, expand contractions, remove HTML/URLs, tag negation (`_NEG` suffix), normalize elongated words (e.g. "gooood" → "good")
2. **Tokenization & Vocabulary** – Build a 20,000-word vocabulary, encode/pad reviews to length 300
3. **Model Architecture** – BiLSTM encoder + Additive or Multiplicative attention → classification head
4. **Experiments:**
   - Exp 1: Additive Attention + BCELoss + Adam
   - Exp 2: Multiplicative Attention + BCELoss + Adam
   - Exp 3: Additive Attention + BCEWithLogitsLoss + Adam
   - Exp 4: Additive Attention + BCELoss + SGD (momentum)
5. **Evaluation** – Accuracy, F1-score, confusion matrix, classification report per experiment
6. **Inference & Interpretability** – Predict sentiment on new reviews with attention-weight visualization; compare attention maps between mechanisms

## 🚀 How to Run
This notebook was built for **Google Colab** with Google Drive integration.
1. Open in Google Colab
2. Mount Google Drive when prompted
3. Ensure `IMDB Dataset.csv` is available at the path referenced in the notebook (update the path if running elsewhere)
4. Run cells sequentially (GPU recommended)

## 📊 Key Findings
- Additive attention produces more focused weights on sentiment-bearing words; multiplicative attention is faster but spreads weight more broadly
- BCELoss and BCEWithLogitsLoss yield similar results; the latter is numerically more stable
- Adam converges faster and reaches higher accuracy than SGD with momentum for this task
- Negation tagging (`_NEG` suffix) improves the model's ability to distinguish phrases like "good" from "not_NEG good", boosting F1-score

The best-performing model (by accuracy) is saved as a pickle file for reuse in inference.
