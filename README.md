# 🧠 Disaster Tweet Classification using Deep Learning (NLP)

This project uses Natural Language Processing and Deep Learning to classify tweets as disaster-related or not. It addresses the challenge of semantic ambiguity where disaster-related words are used in non-disaster contexts.

---

## 📌 Problem Statement

Tweets may contain words like *“fire”*, *“crash”*, or *“explosion”* that can either indicate a real disaster or be used metaphorically. The task is to **build a binary classifier** that distinguishes actual disaster tweets from figurative or unrelated ones.

---

## 🗂️ Dataset Overview

- **Train Set**: 7,613 tweets  
- **Test Set**: 3,263 tweets  
- **Max Length**: 34 tokens per tweet  
- **Vocabulary Size**: ~14,000  
- **Class Distribution**: Slight imbalance  

---

## 🧹 Data Preprocessing

- Removed **URLs**, **mentions**, **numbers**, and **punctuation**
- Retained key stop words (e.g., *no*, *not*) for sentiment
- Appended keyword tokens to help model focus on disaster context
- Applied **tokenization** and **padding**
- Conducted EDA with **word clouds** and **VADER sentiment scores**

---

## 🔍 Exploratory Analysis

- **Word Clouds**: Highlighted contrast in language between disaster and non-disaster tweets  
- **Sentiment Analysis**: Used VADER to observe distribution of positive vs. negative tweets  
- **Class Balance**: Slight imbalance handled with class weights

---

## 🧠 Model Architecture

Implemented in **TensorFlow / Keras**, all models used **GloVe Twitter-200** embeddings. Explored multiple RNN variants:

- ✅ **Final Model**: Stacked Bi-directional GRU + Attention
  - Embedding size: 200 (GloVe)
  - RNN layers: Bi-GRU (128 → 64 units)
  - Attention layer to focus on contextually important words
  - Dense layers: 32 → 16 with **LeakyReLU**, Dropout (0.5)
  - Optimizer: Adam, Loss: Binary Crossentropy
  - Techniques: BatchNorm, EarlyStopping, ReduceLROnPlateau
  - Evaluation: F1 Score, Confusion Matrix

---

## 📈 Model Performance

| Model Variant                      | Validation F1 Score | Kaggle F1 Score |
|-----------------------------------|----------------------|------------------|
| Simple RNN                        | 0.56                 | -                |
| LSTM                              | 0.74                 | -                |
| Bi-Directional LSTM               | 0.76                 | -                |
| Stacked LSTM                      | 0.77                 | -                |
| Stacked Bi-GRU with Attention 🔥 | 0.76                 | **0.8127** ✅    |

> The final model showed excellent generalization with minimal overfitting.

---

## 🛠️ Libraries Used

- `tensorflow`, `keras`
- `gensim`, `nltk`, `sklearn`
- `matplotlib`, `seaborn`, `wordcloud`

---

## 🧪 Training Strategy

- **Stratified K-Fold** Cross Validation for better generalization  
- **Model Averaging** across 5 runs to stabilize predictions  
- **Class Weights** used to handle imbalance  
- **EarlyStopping & ReduceLROnPlateau** callbacks  

---

## 🎓 Key Takeaways

- Domain-specific embeddings (GloVe Twitter) outperform general ones
- Attention layers help improve both accuracy and interpretability
- Complex models require a balance between performance and training time
- Averaging predictions across runs mitigates randomness

---

## 🤝 Acknowledgements

- [Kaggle Dataset: Real or Not - NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- [GloVe Twitter Embeddings](https://nlp.stanford.edu/projects/glove/)

---

## 🚀 Future Enhancements

- Deploy as a web app using **Streamlit or Flask**
- Integrate **BERTweet** or **DistilBERT** for performance comparison
- Add **explainability** using LIME or SHAP

---
