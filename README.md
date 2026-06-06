# 🏠 HouseWise — ML House Price Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**🚀 Live Deployment:** [https://tanmay-housewise.streamlit.app/](https://tanmay-housewise.streamlit.app/)

## Overview

**HouseWise** is a production-ready machine learning web application built with Streamlit. It predicts property prices based on housing metrics (like age, rooms, and location) using California Housing data. It features an opulent **Art Deco** aesthetic with a luxurious Gatsby-inspired UI. The application provides robust Exploratory Data Analysis (EDA), trains and compares 5 different regression models—including a powerful Stacking Ensemble—and gives users an interactive interface to predict and understand house prices.

## Features

- 🏠 **Predict Price**: Interactive sliders for inputting housing metrics, returning an estimated property value with confidence intervals and a category badge (Budget/Mid-range/Premium/Luxury).
- 📊 **EDA Dashboard**: Extensive data visualizations including price distributions, age group analysis, MedInc vs Price trends, a feature correlation matrix, and geographical maps of property values.
- 🤖 **Model Comparison**: Side-by-side performance metrics of 5 regression models, showing actual vs predicted scatter plots, residual plots, and learning curves for the best-performing model.
- 🔍 **Feature Importance**: See exactly which features impact property values the most, accompanied by plain English explanations and their correlation to the final price.
- ⚙️ **One-Click Retraining**: Re-run the ML pipeline straight from the app to train Linear, Ridge, Lasso, Random Forest, and Gradient Boosting models if missing.

## ML Model Performance

The application trains five distinct regression models on the dataset (80/20 train-test split, standard scaling, and log-transformed target). 

| Model | Expected R² | Expected RMSE |
|---|---|---|
| **Stacking Ensemble** | **~0.85** | **~$45,000** |
| Random Forest | ~0.83 | ~$47,000 |
| Gradient Boosting | ~0.80 | ~$51,000 |
| Ridge Regression | ~0.61 | ~$72,000 |
| Linear Regression | ~0.61 | ~$72,000 |

*(Note: The Stacking Ensemble combines HistGradientBoosting, Random Forest, and SVR, using Ridge as the final estimator to achieve the best performance.)*

## Tech Stack

| Component | Technology |
|---|---|
| Frontend / App | Streamlit 1.32 |
| Machine Learning | scikit-learn 1.4 |
| Data Processing | pandas, numpy, scipy |
| Visualization | plotly, seaborn, matplotlib |
| Persistence | joblib |

## Project Structure

```
house_price_predictor/
├── app.py                     # Streamlit frontend & interactive dashboards
├── train.py                   # ML training pipeline & feature engineering
├── eda.py                     # Exploratory Data Analysis (EDA) functions
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .streamlit/
│   └── config.toml            # Custom Streamlit UI theme
├── scaler.pkl                 # (Generated) Fitted StandardScaler
├── best_model.pkl             # (Generated) Best performing ML model
├── model_*.pkl                # (Generated) Other individual models
├── feature_names.json         # (Generated) List of used features
├── metrics.json               # (Generated) Model metrics & EDA stats
└── feature_importance.json    # (Generated) Importance scores per model
```

## Key Insights from EDA

- **Income Dictates Price:** `MedInc` (Median Income) is by far the strongest predictor and most correlated feature to housing prices.
- **Geographical Hotspots:** Proximity to the coast and major cities (visualized in the latitude/longitude scatter and map) drives prices up dramatically.
- **Age is Nuanced:** `HouseAge` has a moderate effect. Very new houses and very old (historic) houses tend to carry slight premiums depending on the area.

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/housewise.git
   cd housewise/house_price_predictor
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models** (Requires ~10 seconds)
   ```bash
   python train.py
   ```

5. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   The app will automatically open in your browser at `http://localhost:8501`.

## Streamlit Cloud Deployment

1. Push your code to a public GitHub repository. Ensure `requirements.txt` is in the root (or specified folder).
2. Create an account on [Streamlit Community Cloud](https://share.streamlit.io).
3. Click **New app** and connect your GitHub account.
4. Select your repository, branch, and set the **Main file path** to `app.py` (or `house_price_predictor/app.py`).
5. (Optional) You can commit the generated `.pkl` and `.json` files to save the app from retraining on start, or let the app prompt the user to train on first load.
6. Click **Deploy!**

## License

This project is licensed under the MIT License. See the LICENSE file for details.
