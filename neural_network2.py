import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load data
print("Loading data...")
df = pd.read_csv("processed_data.csv")

# Display data info
print(f"Dataset shape: {df.shape}")
print("\nMissing values:")
print(df.isnull().sum().sum())

# Keep a reference dataframe for later analysis
reference_df = df[['plate_number', 'price', 'price_numeric']].copy()

# Feature engineering
print("\nPerforming feature engineering...")
# Add relevant features if they don't exist already
if 'digits_ratio' not in df.columns:
    df['digits_ratio'] = df['num_digits'] / 7  # Assuming max plate length is 7
if 'repeating_ratio' not in df.columns and 'max_repeating_digits' in df.columns and 'num_digits' in df.columns:
    df['repeating_ratio'] = df['max_repeating_digits'] / df['num_digits'].replace(0, 1)
if 'uniqueness_ratio' not in df.columns and 'unique_digits' in df.columns and 'num_digits' in df.columns:
    df['uniqueness_ratio'] = df['unique_digits'] / df['num_digits'].replace(0, 1)

# Remove extreme outliers based on percentile
upper_limit = df['price_numeric'].quantile(0.995)
df_filtered = df[df['price_numeric'] <= upper_limit].copy()
print(f"\nRemoved {df.shape[0] - df_filtered.shape[0]} extreme outliers (>{upper_limit})")

# Prepare features and target
excluded_columns = ["plate_number", "price", "timestamp", "price_numeric"]
feature_columns = [col for col in df_filtered.columns if col not in excluded_columns]
X = df_filtered[feature_columns].copy()
y = df_filtered["price_numeric"].copy()

# Check for any remaining categorical columns and remove them
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    print(f"Removing {len(categorical_cols)} categorical columns: {categorical_cols}")
    X = X.drop(columns=categorical_cols)

# Log transform the target for better modeling
y_log = np.log1p(y)

# Standardize features
print("\nPreprocessing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data with stratification by price range to ensure balanced representation
price_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
    X_scaled, y_log, price_bins, test_size=0.2, random_state=42, stratify=price_bins)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Function to evaluate model performance
def evaluate_model(y_true, y_pred_log, model_name):
    # Convert log predictions back to original scale
    y_pred = np.expm1(y_pred_log)
    y_true_orig = np.expm1(y_true)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred))
    r2 = r2_score(y_true_orig, y_pred)
    
    # Calculate MAPE manually to handle zeros
    abs_perc_errors = []
    for true, pred in zip(y_true_orig, y_pred):
        if true != 0:  # Avoid division by zero
            abs_perc_errors.append(abs((pred - true) / true))
    mape = np.mean(abs_perc_errors) if abs_perc_errors else 0
    
    print(f"\n===== {model_name} Performance =====")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2%}")
    
    # Calculate percentage of predictions within error ranges
    errors = np.array(abs_perc_errors)
    within_10pct = np.mean(errors <= 0.10)
    within_20pct = np.mean(errors <= 0.20)
    within_50pct = np.mean(errors <= 0.50)
    
    print(f"Predictions within 10% error: {within_10pct:.2%}")
    print(f"Predictions within 20% error: {within_20pct:.2%}")
    print(f"Predictions within 50% error: {within_50pct:.2%}")
    
    # Show sample predictions
    indices = np.random.choice(len(y_true), 10, replace=False)
    
    print("\nSample predictions:")
    for i in indices:
        true_val = y_true_orig.iloc[i] if hasattr(y_true_orig, 'iloc') else y_true_orig[i]
        pred_val = y_pred[i]
        pct_error = (pred_val - true_val) / true_val if true_val != 0 else float('inf')
        print(f"Predicted: {pred_val:,.2f} | Actual: {true_val:,.2f} | Error: {pct_error:.2%}")
    
    return mape, within_20pct, y_pred

# 1. Gradient Boosting Regressor
print("\nTraining Gradient Boosting model...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42,
    verbose=1
)
gb_model.fit(X_train, y_train)

# Evaluate Gradient Boosting
gb_preds = gb_model.predict(X_test)
gb_mape, gb_within_20, gb_preds_orig = evaluate_model(y_test, gb_preds, "Gradient Boosting")

# 2. Random Forest
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_preds = rf_model.predict(X_test)
rf_mape, rf_within_20, rf_preds_orig = evaluate_model(y_test, rf_preds, "Random Forest")

# 3. Neural Network
print("\nTraining Neural Network model...")
tf.random.set_seed(42)
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dense(1)
])

nn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

checkpoint = ModelCheckpoint('best_nn_model.h5', save_best_only=True, monitor='val_loss')
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    checkpoint
]

history = nn_model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=100,  # Reduced from 200 to speed up training
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.tight_layout()
plt.savefig('nn_training_history.png')
plt.close()

# Evaluate Neural Network
nn_preds = nn_model.predict(X_test).flatten()
nn_mape, nn_within_20, nn_preds_orig = evaluate_model(y_test, nn_preds, "Neural Network")

# 4. Ensemble of models (weighted average)
print("\nEvaluating Ensemble model...")
# Use weighted average with weights determined by model performance
total_within_20 = gb_within_20 + rf_within_20 + nn_within_20
weights = [
    gb_within_20 / total_within_20, 
    rf_within_20 / total_within_20, 
    nn_within_20 / total_within_20
]
print(f"Ensemble weights: GB={weights[0]:.2f}, RF={weights[1]:.2f}, NN={weights[2]:.2f}")

ensemble_preds = weights[0] * gb_preds + weights[1] * rf_preds + weights[2] * nn_preds
ensemble_mape, ensemble_within_20, ensemble_preds_orig = evaluate_model(y_test, ensemble_preds, "Ensemble")

# Summary of all models
print("\n===== Model Comparison =====")
models = ["Gradient Boosting", "Random Forest", "Neural Network", "Ensemble"]
mapes = [gb_mape, rf_mape, nn_mape, ensemble_mape]
within_20pcts = [gb_within_20, rf_within_20, nn_within_20, ensemble_within_20]

summary_df = pd.DataFrame({
    'Model': models,
    'MAPE': mapes,
    'Within 20% Error': within_20pcts
})
print(summary_df.sort_values('MAPE'))

# Feature importance analysis for tree-based models
feature_names = X.columns.tolist()
if hasattr(rf_model, 'feature_importances_'):
    # Plot Random Forest feature importance
    rf_importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=imp_df.head(20))
    plt.title('Random Forest Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    
    print("\nTop 10 Important Features:")
    print(imp_df.head(10))

# Create a prediction function for new data
def predict_price(plate_features, use_ensemble=True):
    """
    Predict price for new plate features
    
    Parameters:
    plate_features: DataFrame or dict - Features of the plate
    use_ensemble: bool - Whether to use ensemble model or best single model
    
    Returns:
    float - Predicted price
    """
    # Convert to DataFrame if dict
    if isinstance(plate_features, dict):
        plate_features = pd.DataFrame([plate_features])
    
    # Ensure we have the right columns in the right order
    if not all(col in plate_features.columns for col in feature_names):
        missing = [col for col in feature_names if col not in plate_features.columns]
        print(f"Warning: Missing features: {missing}")
        return None
    
    # Select only the features used in training
    plate_features = plate_features[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(plate_features)
    
    if use_ensemble:
        # Get predictions from all models
        gb_pred = gb_model.predict(X_scaled)
        rf_pred = rf_model.predict(X_scaled)
        nn_pred = nn_model.predict(X_scaled).flatten()
        
        # Apply weighted average
        log_pred = weights[0] * gb_pred + weights[1] * rf_pred + weights[2] * nn_pred
    else:
        # Use best single model (based on previous results)
        best_model_idx = np.argmin(mapes[:3])  # Only consider individual models
        if best_model_idx == 0:
            log_pred = gb_model.predict(X_scaled)
        elif best_model_idx == 1:
            log_pred = rf_model.predict(X_scaled)
        else:
            log_pred = nn_model.predict(X_scaled).flatten()
    
    # Convert from log scale back to original
    return np.expm1(log_pred)

# Example of using the prediction function
if len(X) > 0:
    print("\nSample prediction for a new plate:")
    sample_plate = X.iloc[0].to_dict()
    print(f"Features: {sample_plate}")
    predicted_price = predict_price(pd.DataFrame([sample_plate]))
    print(f"Predicted price: {predicted_price[0]:,.2f} AED")

    # Save evaluation graphs
    # Create actual vs predicted scatter plot for ensemble
    plt.figure(figsize=(10, 7))
    plt.scatter(np.expm1(y_test), ensemble_preds_orig, alpha=0.3)
    plt.plot([np.expm1(y_test).min(), np.expm1(y_test).max()], 
            [np.expm1(y_test).min(), np.expm1(y_test).max()], 'r--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Ensemble Model - Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig('ensemble_predictions.png')
    plt.close()

# Save the best model
import joblib
print("\nSaving models...")

try:
    # Save preprocessor
    joblib.dump(scaler, 'plate_price_scaler.pkl')
    print("Saved scaler")
    
    # Save feature names
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    print("Saved feature names")
    
    # Find best model
    best_model_name = models[np.argmin(mapes)]
    print(f"Best model: {best_model_name}")
    
    # Save Gradient Boosting model
    joblib.dump(gb_model, 'plate_price_gb_model.pkl')
    
    # Save Random Forest model
    joblib.dump(rf_model, 'plate_price_rf_model.pkl')
    
    # NN model already saved via callback
    
    print("All models saved successfully!")
except Exception as e:
    print(f"Error saving models: {e}")

print("\nAnalysis completed successfully.")