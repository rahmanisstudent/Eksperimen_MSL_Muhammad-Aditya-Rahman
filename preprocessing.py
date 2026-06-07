import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# pyrefly: ignore [missing-import]
import category_encoders as ce

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class HousePricePreprocessor:
    """
    A class to automate the preprocessing pipeline for the Jakarta House Price dataset.
    This replicates the exact steps from the notebook but in a modular and structured way.
    """
    def __init__(self, target_col='price', categorical_cols=None, test_size=0.2, random_state=42, smoothing=10):
        self.target_col = target_col
        self.categorical_cols = categorical_cols if categorical_cols is not None else ['district', 'city']
        self.test_size = test_size
        self.random_state = random_state
        self.smoothing = smoothing
        
        # Saved states of estimators for future inference
        self.encoder = None
        self.scaler = None
        self.numeric_cols_for_outliers = None

    def load_data(self, file_path):
        """Step 1: Load the dataset from CSV"""
        logger.info(f"Loading dataset from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} tidak ditemukan!")
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    def handle_missing_values(self, df):
        """Step 2: Drop rows with missing values"""
        initial_shape = df.shape
        missing_count = df.isnull().sum().sum()
        logger.info(f"Checking for missing values. Found: {missing_count} missing entries.")
        
        # Copy to avoid modifying the original dataframe in-place outside the function
        df_cleaned = df.copy()
        df_cleaned.dropna(inplace=True)
        
        final_shape = df_cleaned.shape
        logger.info(f"Missing values handled. Rows dropped: {initial_shape[0] - final_shape[0]}")
        return df_cleaned

    def check_duplicates(self, df):
        """Step 3: Check and report duplicated data"""
        duplicate_count = df.duplicated().sum()
        logger.info(f"Checking for duplicate rows. Found: {duplicate_count} duplicates.")
        return duplicate_count

    def handle_outliers(self, df):
        """Step 4: Remove outliers using the IQR method on all numerical columns"""
        logger.info("Handling outliers using IQR method...")
        df_cleaned = df.copy()
        
        # Select numeric columns (including target if it is in df at this stage)
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        self.numeric_cols_for_outliers = list(numeric_cols)
        
        initial_rows = len(df_cleaned)
        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filtering outliers for current column
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
            
        final_rows = len(df_cleaned)
        logger.info(f"Outlier handling complete. Rows removed: {initial_rows - final_rows}. Current shape: {df_cleaned.shape}")
        return df_cleaned

    def split_features_target(self, df):
        """Step 5: Separate features and target variable"""
        logger.info(f"Splitting features and target column: {self.target_col}")
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in DataFrame!")
        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]
        return X, y

    def split_train_test(self, X, y):
        """Step 6: Split dataset into training and testing sets"""
        logger.info(f"Splitting dataset with test_size={self.test_size}, random_state={self.random_state}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def encode_categorical(self, X_train, X_test, y_train):
        """Step 7: Encode categorical columns using Target Encoding with smoothing"""
        logger.info(f"Encoding categorical columns: {self.categorical_cols} using TargetEncoder")
        self.encoder = ce.TargetEncoder(cols=self.categorical_cols, smoothing=self.smoothing)
        
        # Fit on training data and transform both
        X_train_encoded = self.encoder.fit_transform(X_train, y_train)
        X_test_encoded = self.encoder.transform(X_test)
        
        return X_train_encoded, X_test_encoded

    def scale_features(self, X_train_encoded, X_test_encoded):
        """Step 8: Scale features using StandardScaler"""
        logger.info("Scaling features using StandardScaler")
        self.scaler = StandardScaler()
        
        # Fit on encoded training data and transform both
        X_train_scaled = self.scaler.fit_transform(X_train_encoded)
        X_test_scaled = self.scaler.transform(X_test_encoded)
        
        # Convert back to DataFrame for readability
        X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns, index=X_train_encoded.index)
        X_test_final = pd.DataFrame(X_test_scaled, columns=X_test_encoded.columns, index=X_test_encoded.index)
        
        return X_train_final, X_test_final

    def transform_target(self, y_train, y_test):
        """Step 9: Apply log1p transformation to reduce price target skewness"""
        logger.info("Applying log1p transformation to target variable (price)")
        y_train_trans = np.log1p(y_train)
        y_test_trans = np.log1p(y_test)
        return y_train_trans, y_test_trans

    def fit_transform(self, file_path):
        """
        Executes the entire preprocessing pipeline sequentially.
        Returns:
            X_train_final, X_test_final, y_train_final, y_test_final
        """
        logger.info("=== Starting Preprocessing Pipeline (fit_transform) ===")
        
        # 1. Load data
        df = self.load_data(file_path)
        
        # 2. Handle missing values
        df = self.handle_missing_values(df)
        
        # 3. Check duplicates (log check)
        self.check_duplicates(df)
        
        # 4. Handle outliers
        df = self.handle_outliers(df)
        
        # 5. Split features and target
        X, y = self.split_features_target(df)
        
        # 6. Train-test split
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        
        # 7. Target Encoding
        X_train_encoded, X_test_encoded = self.encode_categorical(X_train, X_test, y_train)
        
        # 8. Feature Scaling
        X_train_final, X_test_final = self.scale_features(X_train_encoded, X_test_encoded)
        
        # 9. Target Transformation
        y_train_final, y_test_final = self.transform_target(y_train, y_test)
        
        logger.info("=== Preprocessing Pipeline Completed Successfully ===")
        return X_train_final, X_test_final, y_train_final, y_test_final

    def transform_new_data(self, df_new):
        """
        Helper function to transform new unseen data using the fitted encoder and scaler.
        Note: This assumes the input data has already been cleaned.
        """
        if self.encoder is None or self.scaler is None:
            raise ValueError("Preprocessor has not been fitted yet! Run fit_transform first.")
            
        logger.info("Transforming new unseen data...")
        df_copy = df_new.copy()
        
        # If target column is present, split it
        if self.target_col in df_copy.columns:
            X_new = df_copy.drop(self.target_col, axis=1)
            y_new = df_copy[self.target_col]
            y_new_trans = np.log1p(y_new)
        else:
            X_new = df_copy
            y_new_trans = None
            
        # Transform features
        X_new_encoded = self.encoder.transform(X_new)
        X_new_scaled = self.scaler.transform(X_new_encoded)
        X_new_final = pd.DataFrame(X_new_scaled, columns=X_new_encoded.columns, index=X_new_encoded.index)
        
        return X_new_final, y_new_trans


# Functional interface (as requested: "berisikan fungsi untuk melakukan preprocessing secara otomatis")
def preprocess_jakarta_house_data(file_path, target_col='price'):
    """
    Functional wrapper that instantiates the preprocessor class, runs the entire pipeline,
    and returns preprocessed training/testing sets ready for training.
    
    Args:
        file_path (str): Path to the Jakarta House Price dataset.
        target_col (str): Name of the target variable.
        
    Returns:
        tuple: (X_train_final, X_test_final, y_train, y_test, preprocessor_instance)
    """
    preprocessor = HousePricePreprocessor(target_col=target_col)
    X_train_final, X_test_final, y_train, y_test = preprocessor.fit_transform(file_path)
    return X_train_final, X_test_final, y_train, y_test, preprocessor


def main():
    # File path of the dataset
    data_file = "jakarta_house.csv"
    
    print(f"=== Menjalankan Preprocessing pada {data_file} ===")
    
    # Run the automated preprocessing function
    try:
        X_train_final, X_test_final, y_train, y_test, preprocessor = preprocess_jakarta_house_data(data_file)
        
        print("\n=== Preprocessing Selesai! ===")
        print(f"Shape X_train (features): {X_train_final.shape}")
        print(f"Shape X_test (features) : {X_test_final.shape}")
        print(f"Shape y_train (target)   : {y_train.shape}")
        print(f"Shape y_test (target)    : {y_test.shape}")
        
        # Menggabungkan kembali fitur dan target untuk disimpan ke CSV
        X_train_final_copy = X_train_final.copy()
        X_test_final_copy = X_test_final.copy()
        
        X_train_final_copy[preprocessor.target_col] = y_train
        X_test_final_copy[preprocessor.target_col] = y_test
        
        df_preprocessed = pd.concat([X_train_final_copy, X_test_final_copy], axis=0)
        
        # Simpan ke file CSV sesuai permintaan
        output_file = 'jakarta_hose_preprocessing.csv'
        df_preprocessed.to_csv(output_file, index=False)
        print(f"\n[OK] Hasil preprocessing disimpan ke: {output_file}")
        
        print("\n=== 5 Baris Pertama X_train_final ===")
        print(X_train_final.head())
        
        print("\n=== Statistik Deskriptif X_train_final ===")
        print(X_train_final.describe().round(2))
        
        print("\n=== Statistik Deskriptif y_train ===")
        print(y_train.describe().round(4))
        
        print("\nSemua data siap digunakan untuk melatih model!")
        
    except Exception as e:
        print(f"Terjadi kesalahan saat preprocessing: {e}")

if __name__ == "__main__":
    main()
