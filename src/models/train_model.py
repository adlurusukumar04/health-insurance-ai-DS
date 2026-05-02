import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_fraud_model():
    print("--- Phase 3: Model Training Started ---")
    
    # 1. Load the Gold Layer Features
    if not os.path.exists("data/processed/final_features.csv"):
        print("Error: Gold Layer data not found. Please run feature engineering first.")
        return
        
    df = pd.read_csv("data/processed/final_features.csv")
    
    # 2. Define Target and Features (Assuming 'is_fraud' is your label)
    # Adjust 'is_fraud' if your column name is different
    # Line 18: Tell the model to ignore IDs and the label during training
    X = df.drop(columns=['fraud_label', 'claim_id', 'member_id', 'provider_npi'], errors='ignore')

    # Line 21: Tell the model THIS is what we are predicting
    y = df['fraud_label']
    
    # 3. Split Data (Stratified to maintain 1-3% fraud ratio in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 4. Initialize XGBoost with Scale_Pos_Weight to handle Imbalance
    # This addresses the 1-3% fraud ratio requirement
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=ratio, 
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # 5. Train Model
    print("Training XGBoost Ensemble...")
    model.fit(X_train, y_train)
    
    # 6. Evaluation against Project Success Metrics
    y_pred = model.predict(X_test)
    print("\n--- Model Performance Audit ---")
    print(classification_report(y_test, y_pred))
    
    # 7. Save the Model (The 'Inference Engine')
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_detection_model.pkl")
    print("--- Model Saved to models/fraud_detection_model.pkl ---")

if __name__ == "__main__":
    train_fraud_model()