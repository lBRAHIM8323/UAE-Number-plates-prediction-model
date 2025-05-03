import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def extract_features(df):
    """Extract and engineer features from the raw dataset."""
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Clean and convert price to numeric
    def clean_price(price):
        if isinstance(price, str):
            if 'AED' in price:
                # Extract number and remove commas
                cleaned = re.sub(r'[^\d.]', '', price)
                return float(cleaned) if cleaned else np.nan
        return np.nan
    
    df_processed['price_numeric'] = df_processed['price'].apply(clean_price)
    
    # Filter out rows with missing prices for training
    df_clean = df_processed[df_processed['price_numeric'].notna()].copy()
    
    # Feature: City prestige (based on your insight that Dubai > Abu Dhabi > others)
    city_prestige = {
        'dubai': 3,
        'abu-dhabi': 2, 
        'fujairah': 1,
        # Add other cities with appropriate values
    }
    df_clean['city_prestige'] = df_clean['city'].map(city_prestige).fillna(0)
    
    # Feature: Convert plate number to integer for processing
    df_clean['plate_number'] = df_clean['plate_number'].astype(str).str.replace(',', '')
    df_clean['plate_number_int'] = pd.to_numeric(df_clean['plate_number'], errors='coerce')
    
    # Feature: Number of digits in plate number
    df_clean['num_digits'] = df_clean['plate_number'].apply(lambda x: len(str(x)))
    
    # Feature: Repeating digits
    def count_repeating_digits(number):
        number_str = str(number)
        max_repeats = 1
        current_repeats = 1
        
        for i in range(1, len(number_str)):
            if number_str[i] == number_str[i-1]:
                current_repeats += 1
                max_repeats = max(max_repeats, current_repeats)
            else:
                current_repeats = 1
        
        return max_repeats
    
    df_clean['max_repeating_digits'] = df_clean['plate_number'].apply(count_repeating_digits)
    
    # Feature: Is palindrome
    def is_palindrome(number):
        number_str = str(number)
        return 1 if number_str == number_str[::-1] else 0
    
    df_clean['is_palindrome'] = df_clean['plate_number'].apply(is_palindrome)
    
    # Feature: Contains special patterns
    def has_pattern_123(number):
        return 1 if '123' in str(number) else 0
    
    def has_pattern_786(number):
        return 1 if '786' in str(number) else 0
    
    # Add more patterns as needed
    df_clean['has_pattern_123'] = df_clean['plate_number'].apply(has_pattern_123)
    df_clean['has_pattern_786'] = df_clean['plate_number'].apply(has_pattern_786)
    
    # Feature: Digit sum (some people value specific sums)
    df_clean['digit_sum'] = df_clean['plate_number'].apply(
        lambda x: sum(int(digit) for digit in str(x) if digit.isdigit())
    )
    
    # Feature: Number of unique digits
    df_clean['unique_digits'] = df_clean['plate_number'].apply(
        lambda x: len(set(digit for digit in str(x) if digit.isdigit()))
    )
    
    # Feature: Is sequential (e.g., 1234, 9876)
    def is_sequential(number):
        number_str = str(number)
        if len(number_str) < 3:
            return 0
        
        # Check for ascending sequence
        asc_seq = all(int(number_str[i]) + 1 == int(number_str[i+1]) for i in range(len(number_str)-1))
        
        # Check for descending sequence
        desc_seq = all(int(number_str[i]) - 1 == int(number_str[i+1]) for i in range(len(number_str)-1))
        
        return 1 if asc_seq or desc_seq else 0
    
    df_clean['is_sequential'] = df_clean['plate_number'].apply(is_sequential)
    
    # Feature: Code prestige (some codes might be more prestigious)
    # This would require domain knowledge or analysis of your full dataset
    
    # One-hot encode city and code
    df_clean = pd.get_dummies(df_clean, columns=['city'], prefix='city')
    df_clean = pd.get_dummies(df_clean, columns=['code'], prefix='code')
    
    # For very large datasets with many codes, you might want to use frequency encoding instead
    
    return df_clean

def prepare_features_for_model(df_clean):
    """Prepare features for neural network training."""
    
    # Select features and target
    feature_cols = [col for col in df_clean.columns if col not in 
                    ['price', 'price_numeric', 'timestamp', 'plate_number']]
    
    X = df_clean[feature_cols]
    y = df_clean['price_numeric']
    
    # Create preprocessing pipeline with scalers
    numeric_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    
    preprocessor = StandardScaler()
    X_processed = preprocessor.fit_transform(X[numeric_features])
    
    return X_processed, y, preprocessor, numeric_features

# Example usage:
# df = pd.read_csv('number_plates.csv')
# df_processed = extract_features(df)
# X, y, preprocessor, feature_names = prepare_features_for_model(df_processed)