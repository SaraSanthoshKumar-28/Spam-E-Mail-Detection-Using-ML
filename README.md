# Spam-E-Mail-Detection-Using-ML
# 📧 Hybrid Spam Detector (Numeric + NLP)

A robust spam detection system that combines traditional **numeric feature-based classification** with **text-based machine learning (TF-IDF)** to accurately classify emails/SMS messages as **Spam** or **Ham** (not spam).

## 🧠 Key Features

- ✅ Dual-model hybrid:
  - **Model 1**: Uses numeric Spambase features (UCI dataset) and Genetic Algorithm-optimized Naive Bayes.
  - **Model 2**: Uses TF-IDF vectors from labeled messages and Multinomial Naive Bayes.
- 📊 Confusion matrix visualization for Spambase classifier.
- 🎛️ Adjustable threshold for spam classification.
- 🌐 Gradio-powered web interface for real-time prediction.
- 🧬 Genetic Algorithm used to optimize the Naive Bayes smoothing parameter (alpha).
- 🔍 Interpretable results with confidence probabilities.

---

## 🗂 Dataset Sources

- **Numeric features**: [UCI Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase) (fetched via `ucimlrepo`)
- **Text dataset**: Synthetic `spam_ham_synthetic_dataset.csv` with labeled SMS/email messages.

---

## 🚀 How It Works

1. **Numeric Model**:
   - Converts input text into a 57-dimensional Spambase-like feature vector.
   - Trains a `MultinomialNB` using Genetic Algorithm to find the best smoothing parameter (`alpha`).

2. **Text Model**:
   - Applies `TfidfVectorizer` to SMS/email data.
   - Trains a second `MultinomialNB` classifier.

3. **Hybrid Prediction**:
   - Both models produce a spam probability.
   - If **either model exceeds the threshold**, the message is classified as **Spam**.

---

## 💻 Gradio Web App

Run the project and interact through a friendly UI:

```bash
python your_script_name.py
.
├── spam_ham_synthetic_dataset.csv     # Synthetic text dataset
├── spambase_cm.png                    # Confusion matrix image
├── hybrid_spam_detector.ipynb         # Full source code (Jupyter)
├── README.md                          # This file
└── your_script.py                     # Python script version (optional)
 ## Installation
Install the required Python packages using pip:
pip install pandas numpy matplotlib seaborn scikit-learn gradio ucimlrepo
Or use:
pip install -r requirements.txt

##🛠 How to Run
Clone this repository.
Make sure spam_ham_synthetic_dataset.csv is present.
Run the .ipynb notebook or Python file.
Test messages via the Gradio interface.

📈 Sample Input

Congratulations! You’ve been selected for a $1000 Amazon gift card.
##Output:
Final Prediction: Spam
Model Metrics: Spam probabilities, accuracies, and reports shown


