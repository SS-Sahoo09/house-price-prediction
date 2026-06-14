## Live Demo

🔗 Streamlit App:
https://house-price-prediction-ss-sahoo.streamlit.app/

## GitHub Repository

🔗 GitHub:
https://github.com/SS-Sahoo09/house-price-prediction

# 🏠 House Price Prediction System

## Overview

The House Price Prediction System is a Machine Learning-based web application designed to estimate residential property prices accurately using various housing and market-related features. The system utilizes advanced machine learning techniques and data analytics to assist buyers, sellers, investors, and real estate agencies in making informed property valuation decisions.

The project implements multiple regression algorithms, including Decision Tree Regression, Random Forest Regression, and XGBoost Regression. After performance evaluation, XGBoost was selected as the final model due to its superior prediction accuracy and overall performance.

The application is deployed using Streamlit, providing an interactive and user-friendly interface for real-time house price prediction.

---

## Features

### Real-Time House Price Prediction

* Predicts house prices instantly based on user inputs.
* Uses a trained XGBoost Regression model.

### Interactive Web Application

* Built using Streamlit.
* Easy-to-use and responsive interface.

### Machine Learning Model Comparison

* Decision Tree Regression
* Random Forest Regression
* XGBoost Regression

### Data Analytics and Visualization

* House Price Distribution Analysis
* Area vs Price Analysis
* Correlation Heatmaps
* Demand Index Analysis
* Feature Importance Analysis

### Market Intelligence Features

* Infrastructure Score
* Demand Index
* Builder Reputation
* Future Development Score
* Crime Rate Analysis
* Metro Connectivity Analysis

---

## Dataset Information

### Dataset Size

* Total Records: 100,000
* Total Features: 27
* Target Variable: House Price

### Important Features

* State
* District/City
* Locality Type
* Area (Square Feet)
* Number of Bedrooms (BHK)
* Bathrooms
* Floors
* Garden Area
* Garage Capacity
* Property Type
* Furnishing Status
* Road Connectivity
* Crime Rate
* Demand Index
* Infrastructure Score
* Builder Reputation
* Future Development Score

---

## Technologies Used

### Programming Language

* Python

### Machine Learning Libraries

* Scikit-Learn
* XGBoost
* NumPy
* Pandas

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Web Application

* Streamlit

### Model Serialization

* Joblib
* Pickle

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Encoding
6. Model Training
7. Model Evaluation
8. Model Deployment

---

## Model Performance

| Model                    | R² Score |
| ------------------------ | -------- |
| Decision Tree Regression | 0.87     |
| Random Forest Regression | 0.91     |
| XGBoost Regression       | 0.93     |

### Selected Model

XGBoost Regression

Reason:

* Highest R² Score
* Lowest Prediction Error
* Better Generalization
* Improved Handling of Non-Linear Relationships

---

## Project Structure

```text
House_Price_Prediction_2_0/
│
├── app.py
├── advanced_house_price_dataset_100000.csv
├── model_columns.pkl
├── xgboost_house_price_model.pkl
├── requirements.txt
├── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Future Enhancements

* Integration with live real estate APIs
* Advanced investment analytics
* Price trend forecasting
* Property recommendation system
* Mobile application support
* GIS and map-based property analysis

---

## Learning Outcomes

* Machine Learning Model Development
* Data Preprocessing and Feature Engineering
* Exploratory Data Analysis
* Model Evaluation Techniques
* Streamlit Web Application Development
* Deployment of Machine Learning Applications

---

## Author

**Sadashiva Sahoo**
A MCA 1st Year Student
of Odisha University of Technology and Research (OUTR), Bhubaneswar

---

## License

This project is developed for educational and learning purposes.
