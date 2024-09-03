import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def load_and_prepare_data(file_path):
    # Load data from Excel
    df = pd.read_excel(file_path)

    # Convert relevant columns to numeric, coercing errors to NaN
    df['Targeted Productivity'] = pd.to_numeric(df['Targeted Productivity'], errors='coerce')
    df['Overtime'] = pd.to_numeric(df['Overtime'], errors='coerce')
    df['No. of Workers'] = pd.to_numeric(df['No. of Workers'], errors='coerce')
    df['Actual Productivity'] = pd.to_numeric(df['Actual Productivity'], errors='coerce')

    # Impute missing values in 'Actual Productivity' with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    df['Actual Productivity'] = imputer.fit_transform(df[['Actual Productivity']])

    # Check for missing values after imputation
    print("Missing values after imputation:")
    print(df.isnull().sum())

    # Continue with the rest of your data preparation...
    numeric_features = df.select_dtypes(include=['number'])

    if numeric_features.empty:
        raise ValueError("No numeric columns available for PCA. Check the input data.")

    # Impute remaining missing values with the mean of each column
    imputed_features = imputer.fit_transform(numeric_features)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(imputed_features)
    df['PC1'] = pca_components[:, 0]
    df['PC2'] = pca_components[:, 1]

    return df
def train_anomaly_detection_model(df):
    # Train Isolation Forest
    X = df[['PC1', 'PC2']]
    iso_forest = IsolationForest(contamination=0.001, random_state=0)
    y_pred = iso_forest.fit_predict(X)
    y_pred = [1 if label == -1 else 0 for label in y_pred]

    # Train Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001)
    y_pred_lof = lof.fit_predict(X)
    y_pred_lof = [1 if label == -1 else 0 for label in y_pred_lof]

    # Add predictions to DataFrame
    df['Anomaly'] = y_pred

    return df, y_pred, y_pred_lof

def classify_fraudulent_transactions(new_data, trained_model):
    # Convert relevant columns to numeric, coercing errors to NaN
    new_data['Targeted Productivity'] = pd.to_numeric(new_data['Targeted Productivity'], errors='coerce')
    new_data['Overtime'] = pd.to_numeric(new_data['Overtime'], errors='coerce')
    new_data['No. of Workers'] = pd.to_numeric(new_data['No. of Workers'], errors='coerce')
    new_data['Actual Productivity'] = pd.to_numeric(new_data['Actual Productivity'], errors='coerce')

    # Impute missing values with the mean of each column
    numeric_features = new_data.select_dtypes(include=['number'])
    imputer = SimpleImputer(strategy='mean')
    imputed_features = imputer.fit_transform(numeric_features)

    # Apply PCA using the trained PCA model (assuming PCA was used)
    pca_components = trained_model['pca'].transform(imputed_features)
    new_data['PC1'] = pca_components[:, 0]
    new_data['PC2'] = pca_components[:, 1]

    # Predict anomalies using the trained Isolation Forest model
    X = new_data[['PC1', 'PC2']]
    y_pred = trained_model['iso_forest'].predict(X)
    fraudulent_transactions = new_data[y_pred == -1]  # -1 indicates anomalies

    return fraudulent_transactions
