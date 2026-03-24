# 🐦 Twitter Hate Speech Detection — Hybrid Bi-LSTM & Ensemble Framework

> A hybrid machine learning system for ternary hate speech classification on Twitter, combining Bi-Directional LSTM with XGBoost and Random Forest ensemble classifiers, achieving a **95.50% weighted F1-score**.

---

## 📄 Project Report

The full academic paper for this project is available here:

📥 **[Project Report (PDF)]([./CSE424_Project_Report_24141071_A_HYBRID_BI-LSTM_AND_ENSEMBLE_FRAMEWORK_FOR_ENHANCED_HATE_SPEECH_DETECTION.pdf](https://drive.google.com/file/d/1iy0WujGsOSP81xqBMY4Gbr0eYW_KU8To/view?usp=drive_link))**

> *A Hybrid Bi-LSTM and Ensemble Framework for Enhanced Hate Speech Detection* — Nawroz Haseen Tumul, Brac University

---

## 📁 Repository Structure
```
├── _FINAL_Twitter_Hate_Speech_Detection_24141071.ipynb       # ✅ Final notebook (recommended)
├── _OLD_Twitter_Hate_Speech_Detection_24141071.ipynb          # Earlier version
├── _Updated__After_Final_Submission_Twitter_Hate_Speech_Detection_24141071.ipynb  # Post-submission updates
├── labeled_data.csv                                           # Dataset (Crowdflower/Davidson et al., 2017)
├── CSE424_Project_Report_24141071_...pdf                      # Full project report
├── Task1_LiteratureReview_24141071_...pdf                     # Literature review
└── README.md
```

---

## 🚀 How to Run the Notebooks

The notebooks are designed to be run on **Google Colab**. Follow the steps below to get started.

### Step 1 — Download the Notebook

Download the notebook file you want to run from this repository:

- **Recommended:** `_FINAL_Twitter_Hate_Speech_Detection_24141071.ipynb`

You can download it by clicking the file in GitHub, then clicking the **⬇ Download raw file** button (top-right).

---

### Step 2 — Upload the Dataset to Google Drive

The notebook reads the dataset (`labeled_data.csv`) from your Google Drive. You need to upload it there before running.

1. Download `labeled_data.csv` from this repository
2. Go to [Google Drive](https://drive.google.com)
3. Upload `labeled_data.csv` to your Drive (you can place it anywhere, but the **root of My Drive** is easiest)

> **Note:** The notebook will prompt you to mount your Google Drive and will look for the CSV file. Make sure the file path in the notebook matches where you uploaded it. The default expected path is typically:
> ```
> /content/drive/MyDrive/labeled_data.csv
> ```
> If you placed it in a subfolder, update this path in the notebook accordingly.

---

### Step 3 — Upload the Notebook to Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Select the `.ipynb` file you downloaded in Step 1

---

### Step 4 — Mount Google Drive & Run

Once the notebook is open in Colab:

1. Run the first cell — it will ask you to **mount Google Drive** and prompt for authorization
2. Follow the authorization steps
3. After mounting, run the remaining cells **in order** from top to bottom

> ⚠️ **Runtime Recommendation:** Go to **Runtime → Change runtime type** and select **GPU** (T4 GPU) for significantly faster training, especially for the Bi-LSTM and hybrid model cells.

---

## 📊 Dataset

This project uses the **Crowdflower Twitter Hate Speech Dataset** introduced by Davidson et al. (2017).

| Class | Label | Distribution |
|-------|-------|-------------|
| Hate Speech | 0 | ~5.8% |
| Offensive Language | 1 | ~77.4% |
| Neither | 2 | ~16.8% |

**Total samples:** 24,783 tweets  
**Source:** Davidson, T., Warmsley, D., Macy, M. and Weber, I. (2017). *Automated hate speech detection and the problem of offensive language.* [arXiv:1703.04009](https://doi.org/10.48550/arxiv.1703.04009)

---

## 🧠 Model Overview

The project trains and evaluates the following models:

| Model | Weighted F1-Score |
|-------|-------------------|
| Naïve Bayes | 0.8322 |
| Logistic Regression | 0.9344 |
| XGBoost | 0.9340 |
| Random Forest | 0.9413 |
| Bi-LSTM | 0.9293 |
| **Hybrid Bi-LSTM + XGBoost** | **0.9550** ✅ |
| Hybrid Bi-LSTM + Random Forest | 0.9516 |

### Feature Groups Used
- **BERTweet embeddings** (768-dim contextual embeddings)
- **TF-IDF vectors** (5,000 features)
- **N-gram matrix** (2,000 bigram/trigram features)
- **Linguistic features** — sentiment score, lexical density, readability scores (Flesch-Kincaid, Gunning Fog), slang markers, NER (race/religion/figure counts), aggressive verb-noun combos, and more

---

## 📦 Dependencies

The notebook installs all required packages automatically. Key libraries include:

- `transformers`, `torch` — BERTweet embeddings
- `tensorflow` / `keras` — Bi-LSTM model
- `scikit-learn` — classical ML models, preprocessing
- `xgboost` — XGBoost classifier
- `nltk`, `vaderSentiment`, `textstat` — NLP & linguistic features
- `shap` — Explainable AI (SHAP analysis)
- `imbalanced-learn` — SMOTE oversampling

---

## 📌 Notes

- The notebook may take **30–60 minutes** to run fully end-to-end on a GPU runtime, primarily due to BERTweet embedding generation
- SHAP analysis cells at the end of the notebook are optional and can be skipped if you only need the model results
- All random seeds are fixed for reproducibility

---

## 👤 Author

**Nawroz Haseen Tumul**  
Brac University  
Course: CSE424 — Machine Learning  
Student ID: 24141071
