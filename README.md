# Probabilistic SMS Spam Detection

**Course:** BAMAT207 – Probability and Statistics
**Topic:** Probability and Random Variables
**Student:** Srivathsan Kuppusamy Saravanan (25BCE2363)

A Python project that classifies an SMS message as **Spam** or **Genuine (Ham)** using conditional probability, Bayes' theorem, random variables and a Multinomial Naive Bayes classifier. For every new message it prints the prediction probability.

## Dataset
[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) from the UCI Machine Learning Repository (5,574 labelled messages: spam and ham). The notebook downloads it automatically with `ucimlrepo`.

## What the notebook covers
1. **Dataset handling:** load data, check missing values, remove duplicates, encode Spam = 1 and Ham = 0
2. **Dataset analysis:** P(Spam), P(Ham) and a class-distribution bar chart
3. **Word-based probability:** P(Spam | word) for *free, prize, win, urgent, offer*, verified with Bayes' theorem
4. **Random variable analysis:** E(Y) = p and Var(Y) = p(1 − p)
5. **Data preparation:** 80/20 stratified train-test split and `CountVectorizer`
6. **Model building:** `MultinomialNB`
7. **Evaluation:** accuracy, precision, recall, F1-score, confusion matrix, and the misclassified spam/ham messages
8. **Real-time testing:** enter any SMS and get `P(Spam | Message)` and `P(Ham | Message)`

## Files
| File | Description |
|---|---|
| `Probabilistic_SMS_Spam_Detection.ipynb` | Main notebook (code, charts, results) |
| `sms_spam_cleaned.csv` | Cleaned dataset |
| `spam_naive_bayes_model.joblib` | Trained Naive Bayes model |
| `spam_count_vectorizer.joblib` | Fitted CountVectorizer |
| `requirements.txt` | Python dependencies |

## How to run
**Google Colab:** upload the notebook, then Runtime → Run all.

**Locally:**
```bash
pip install -r requirements.txt
jupyter notebook Probabilistic_SMS_Spam_Detection.ipynb
```

## Use the saved model
```python
import joblib, re

model = joblib.load("spam_naive_bayes_model.joblib")
vectorizer = joblib.load("spam_count_vectorizer.joblib")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " url ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

msg = "Congratulations! You have won a free prize. Claim now."
p_ham, p_spam = model.predict_proba(vectorizer.transform([clean_text(msg)]))[0]
print("Spam" if p_spam >= 0.5 else "Ham", round(p_spam, 2), round(p_ham, 2))
```

## Results
Accuracy, precision, recall, F1-score and the confusion matrix are shown in the notebook (Section 7).

## Libraries
pandas, numpy, matplotlib, scikit-learn, ucimlrepo, re, joblib
