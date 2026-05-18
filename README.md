# House_Price_Predictor_Model
An end-to-end Machine Learning regression project built to predict house prices using a structured, production-ready pipeline. The system handles raw house data featuring a mix of numerical and categorical variables, processes them seamlessly, evaluates multiple machine learning models, and exports the highest-performing model for deployment.
**House Price Prediction – Machine Learning Project**
This project predicts house prices using multiple machine-learning models.
It includes data preprocessing, feature engineering, pipelines, model training, evaluation, and a Streamlit web app for real-time price prediction.
**📊 Project Features**

-Handles missing values
-Converts categorical → numeric
-Uses Pipelines + ColumnTransformer
-Models Trained:

Linear Regression
Random Forest Regressor
Gradient Boosting Regressor

-Evaluation Metrics:

Accuracy (R² Score)
MAE
MSE
RMSE

-Interactive Streamlit Web App
-User-friendly interface
-Uses joblib for model saving
**
How It Works
**
1 Data is loaded and cleaned
2 Pipeline transforms numeric & categorical features
3 Models are trained & evaluated
4 Best model saved using joblib
5 Streamlit app loads the trained model
6 User enters values → model predicts price
**
How to Run Locally**
pip install -r requirements.txt
streamlit run app.py
