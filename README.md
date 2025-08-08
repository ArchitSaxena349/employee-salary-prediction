# 🚀 Employee Salary Prediction

*Predicting paychecks from profiles with Python magic.*

[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20Here-brightgreen?style=for-the-badge&logo=streamlit)](https://architsaxena349-employee-salary-prediction-app-k8nmmd.streamlit.app/)

---

## ✨ Overview

This repo is my no‑nonsense dive into predicting employee salaries using machine learning. We take features like education, job type, experience, and distance from a city—and spit out salary estimates. Useful for hiring teams, consulting gigs, or just flexing your ML skills on medium‑sized HR datasets.

---

## 🧠 What’s Inside

- **Data Exploration** – visuals, correlation heatmaps, outlier detection, trends across features  
- **Feature Engineering** – label/ordinal encoding, group stats, new derived features like `years_experience`, `miles_from_metropolis` grouping  
- **Modeling** – baseline comparison: Linear Regression, Random Forest, Gradient Boosting (GBR wins)  
- **Model Evaluation** – error metrics: MSE, MAE, RMSE, R² score  
- **Streamlit UI** – [Live App 🌐](https://architsaxena349-employee-salary-prediction-app-k8nmmd.streamlit.app/) to test real-time predictions  
- **Notebooks or Scripts** – step‑by‑step pipelines: EDA → modeling → deployment-ready

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8+
- Git
- pip

### Installation

```bash
git clone https://github.com/ArchitSaxena349/employee-salary-prediction.git
cd employee-salary-prediction
pip install -r requirements.txt
````

---

## 🛠️ Usage

### Run Locally

```bash
streamlit run app.py
# Opens on http://localhost:8501
```

### Or... just try it here:

👉 [Live Streamlit App](https://architsaxena349-employee-salary-prediction-app-k8nmmd.streamlit.app/)

---

## 📊 Model Performance Summary

| Model                       | MSE          | MAE | RMSE | R² Score |
| --------------------------- | ------------ | --- | ---- | -------- |
| Baseline (mean salary)      | \~644.26     | —   | —    | —        |
| Linear Regression           | \~358.15     | —   | —    | —        |
| Random Forest Regressor     | \~313.27     | —   | —    | —        |
| Gradient Boosting Regressor | **\~313.06** | —   | —    | Highest  |

GBR edges out the rest for clean, reliable performance. Still open to hyper-tuning if you're feeling spicy. 🌶️

---

## 🗂️ Project Structure

```
employee-salary-prediction/
├── data/                        # raw + cleaned datasets
├── notebooks/                   # EDA, modeling, pipeline demos
├── model/                       # trained models (e.g. `.joblib`, `.pkl`)
├── app.py / streamlit_app.py    # Streamlit UI
├── requirements.txt             # Python dependencies
├── README.md                    # this file
└── LICENSE                      # project license (e.g. MIT)
```

---

## 🚧 Future Work & Improvements

* Add richer features like polynomial terms for experience or distance
* Test other algorithms: SVR, XGBoost, Lasso/Ridge
* Robust cross-validation and hyperparameter tuning
* Feature importance visualizations to explain predictions
* Dockerfile + CI/CD for smoother deployments

---

## 🎓 Who Should Try This

* HR/Recruiting teams looking to estimate fair salary offers
* Students building machine learning portfolios
* Anyone questioning how experience, education, and location impact paychecks

---

## 📞 Feedback & Contact

Spotted bugs? Wanna collab? Got better predictions than me?
Hit the GitHub issues or slide into my DMs.
I’m Archit—skeptical coder, ML enthusiast, and an explorer of truth through data.

---

## 📝 License

Distributed under the **MIT License** — copy, remix, reuse — just give credit where due.

---

**Prediction isn’t prophecy—but it’s the next best thing.**

Cheers,
**Archit Saxena** 🚀
