# 🛍️ Mall Customer K-Means Clustering App

Interactive Streamlit app to segment mall customers using K-Means clustering.

---

## 📁 Project Structure
```
mall_clustering/
├── app.py              ← main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 🔧 Features

| Tab | What it does |
|-----|-------------|
| 📄 Raw Data   | Preview uploaded CSV, row/col/null counts |
| 🧹 Cleaning   | Step-by-step cleaning log + clean data preview |
| 📊 EDA        | Histograms, gender pie chart, correlation heatmap |
| 📈 Elbow      | Elbow curve + silhouette score across K = 2–10 |
| 🎯 Clusters   | Scatter plot, PCA 2D view, cluster profiles, box plots |
| 💾 Export     | Download labelled CSV |

---

## 🧹 Cleaning Pipeline (automatic)

1. Normalise column names to lower-snake-case
2. Remove duplicate rows
3. Drop all-NaN rows
4. Fill numeric NaNs with column **median**
5. Fill categorical NaNs with column **mode**
6. Remove outliers using **3 × IQR** rule
7. Label-encode `gender` → `gender_enc`

---

## ⚙️ Sidebar Controls

| Control | Description |
|---------|-------------|
| Upload CSV | Accepts `Mall_Customers.csv` or any compatible CSV |
| Feature preset | Choose which columns to cluster on |
| K slider | Number of clusters (2–10) |
| Init method | `k-means++` (recommended) or `random` |
| Max iterations | KMeans convergence budget |
| n_init | Number of random restarts |

---
