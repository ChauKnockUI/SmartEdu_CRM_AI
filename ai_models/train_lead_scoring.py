import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_clean_data(file_path):
    """
    Load data from Kaggle dataset and perform initial cleaning.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Replace "Select" with NaN since it means the user did not choose an option
    df = df.replace('Select', np.nan)
    
    # --- ĐÚNG NHƯ BẠN NÓI: Sinh viên năm 3/4 chỉ nên là 1 điểm cộng (nhích xác suất lên chút xíu) ---
    # Ta sẽ trộn cột 'Student_Year' vào File Data Gốc
    np.random.seed(42)
    def generate_student_year(row):
        if row['What is your current occupation'] == 'Student':
            return np.random.choice(['Year 1', 'Year 2', 'Year 3', 'Year 4'])
        return 'Not Applicable'
        
    df['Student_Year'] = df.apply(generate_student_year, axis=1)
    
    # Logic cộng nhẹ điểm: Nếu là năm 3/4 màng đang Rớt (0), cho họ thêm 20% cơ hội lật kèo thành Chốt (1).
    def adjust_conversion(row):
        if row['Student_Year'] in ['Year 3', 'Year 4'] and row['Converted'] == 0:
            return 1 if np.random.rand() < 0.20 else 0
        return row['Converted']
        
    df['Converted'] = df.apply(adjust_conversion, axis=1)
    
    # (Tùy chọn xuất hẳn ra file Dataset vật lý mới cho bạn)
    enhanced_csv_path = file_path.replace(".csv", "_Enhanced.csv")
    # Ghi đè file nếu đã tồn tại
    df.to_csv(enhanced_csv_path, index=False)
    
    # Define features to use in the model (based on what Sales can practically input)
    selected_features = [
        'Lead Origin', 
        'Lead Source', 
        'Total Time Spent on Website', 
        'What is your current occupation',
        'Specialization',
        'TotalVisits',
        'Page Views Per Visit',
        'Student_Year'
    ]
    target = 'Converted'
    
    # Drop rows where the target variable is missing
    df = df[selected_features + [target]].copy()
    df = df.dropna(subset=[target])
    
    return df[selected_features], df[target]

def build_model_pipeline():
    """
    Build the pipeline for data preprocessing and Machine Learning model.
    """
    numeric_features = ['Total Time Spent on Website', 'TotalVisits', 'Page Views Per Visit']
    categorical_features = ['Lead Origin', 'Lead Source', 'What is your current occupation', 'Specialization', 'Student_Year']
    
    # Pipeline for filling missing numeric values and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for filling missing categorical values and one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Final Random Forest model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    return model_pipeline

def main():
    print("--- Lead Scoring Model Training Data Pipeline ---")
    
    data_path = r"d:\DA\DACN\DACN3\SmartEdu_CRM\dataset\archive\Lead Scoring.csv"
    
    try:
        X, y = load_and_clean_data(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    print(f"Data loaded successfully. Total records: {len(X)}")
    
    # 1. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Build and Train Model
    print("\nTraining Random Forest model...")
    rf_pipeline = build_model_pipeline()
    rf_pipeline.fit(X_train, y_train)
    print("Training complete.")
    
    # 3. Evaluate Model
    y_pred = rf_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 4. Export Model
    model_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(model_dir, exist_ok=True)
    
    model_filepath = os.path.join(model_dir, 'lead_scoring_pipeline.pkl')
    joblib.dump(rf_pipeline, model_filepath)
    print(f"\nModel successfully exported to: {model_filepath}")
    
    # 5. Sanity Check / Simulation
    print("\n--- Running Prediction Simulation ---")
    mock_lead = pd.DataFrame([{
        'Lead Origin': 'Landing Page Submission',
        'Lead Source': 'Google',
        'Total Time Spent on Website': 500, # Not much time spent
        'What is your current occupation': 'Student', 
        'Specialization': 'Finance Management',
        'TotalVisits': 2,
        'Page Views Per Visit': 2,
        'Student_Year': 'Year 4' # 4th year student needing TOEIC
    }])
    
    prediction = rf_pipeline.predict(mock_lead)[0]
    probability = rf_pipeline.predict_proba(mock_lead)[0][1]
    
    print(f"Simulation Data: {mock_lead.to_dict('records')[0]}")
    print(f"Prediction Result: {'Converted' if prediction == 1 else 'Not Converted'}")
    print(f"Conversion Probability: {probability * 100:.1f}%")

if __name__ == "__main__":
    main()
