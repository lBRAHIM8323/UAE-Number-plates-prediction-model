import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from keras_tuner import Hyperband

# Load the dataset and preprocess (from previous artifacts)
df = pd.read_csv('number_plates.csv')
df_processed = extract_features(df)
X, y, preprocessor, feature_names = prepare_features_for_model(df_processed)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Define a model-building function with hyperparameters for tuning
def model_builder(hp):
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Dense(
        units=hp.Int('units_input', min_value=32, max_value=256, step=32),
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Float('l2_input', min_value=1e-5, max_value=1e-2, sampling='log')),
        input_shape=(X_train.shape[1],)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hp.Float('dropout_input', min_value=0.0, max_value=0.5, step=0.1)))
    
    # Hidden layers
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation=hp.Choice(f'activation_{i}', values=['relu', 'elu', 'selu']),
            kernel_regularizer=regularizers.l2(hp.Float(f'l2_{i}', min_value=1e-5, max_value=1e-2, sampling='log'))
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(layers.Dense(1))
    
    # Compile
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

# 2. Setup and run hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    tuner = Hyperband(
        model_builder,
        objective='val_loss',
        max_epochs=100,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='number_plates'
    )
    
    # Define early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # Search for best hyperparameters
    tuner.search(
        X_train, y_train,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Get best hyperparameters and build model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    
    return best_model, best_hps

# 3. Ensemble approach: Combine neural network with boosting models
def build_ensemble_model(X_train, y_train, X_test, y_test):
    # Train neural network (use the tuned model)
    best_nn_model, best_hps = tune_hyperparameters(X_train, y_train)
    
    # Train the tuned model
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    history = best_nn_model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Train Gradient Boosting model
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # Make predictions with each model
    nn_preds = best_nn_model.predict(X_test).flatten()
    xgb_preds = xgb_model.predict(X_test)
    gb_preds = gb_model.predict(X_test)
    
    # Simple ensemble: Average the predictions (weighted)
    # We could tune these weights through cross-validation
    ensemble_preds = 0.5 * nn_preds + 0.3 * xgb_preds + 0.2 * gb_preds
    
    # Evaluate ensemble
    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    ensemble_r2 = r2_score(y_test, ensemble_preds)
    
    print("Ensemble Model Performance:")
    print(f"MAE: AED {ensemble_mae:,.2f}")
    print(f"RMSE: AED {ensemble_rmse:,.2f}")
    print(f"R²: {ensemble_r2:.4f}")
    
    # Return all models for the prediction pipeline
    return {
        'neural_network': best_nn_model,
        'xgboost': xgb_model,
        'gradient_boosting': gb_model,
        'hyperparameters': best_hps,
        'history': history,
        'ensemble_metrics': {
            'mae': ensemble_mae,
            'rmse': ensemble_rmse,
            'r2': ensemble_r2
        }
    }

# 4. Cross-validation for more robust evaluation
def cross_validate_model(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = {
        'mae': [],
        'rmse': [],
        'r2': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold+1}/{n_splits}")
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create a simpler model for cross-validation (faster)
        model = models.Sequential([
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(X.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        model.fit(
            X_fold_train, y_fold_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_fold_val, y_fold_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_fold_val).flatten()
        mae = mean_absolute_error(y_fold_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        r2 = r2_score(y_fold_val, y_pred)
        
        fold_scores['mae'].append(mae)
        fold_scores['rmse'].append(rmse)
        fold_scores['r2'].append(r2)
        
        print(f"Fold {fold+1} - MAE: AED {mae:,.2f}, RMSE: AED {rmse:,.2f}, R²: {r2:.4f}")
    
    # Average scores
    avg_mae = np.mean(fold_scores['mae'])
    avg_rmse = np.mean(fold_scores['rmse'])
    avg_r2 = np.mean(fold_scores['r2'])
    
    print("\nCross-validation average scores:")
    print(f"MAE: AED {avg_mae:,.2f}")
    print(f"RMSE: AED {avg_rmse:,.2f}")
    print(f"R²: {avg_r2:.4f}")
    
    return fold_scores

# 5. Feature importance analysis
def analyze_feature_importance(X, y, feature_names):
    # Use XGBoost for feature importance
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        random_state=42
    )
    
    xgb_model.fit(X, y)
    
    # Get feature importance
    importance = xgb_model.feature_importances_
    
    # Create DataFrame for visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    return feature_importance

# 6. Comprehensive prediction function
def predict_plate_price_ensemble(city, code, plate_number, models, preprocessor, feature_names):
    """
    Predict the price of a new number plate using the ensemble model
    
    Args:
        city: City name (e.g., 'dubai')
        code: Code letter/number (e.g., 'b')
        plate_number: The plate number (e.g., '6901')
        models: Dictionary of trained models
        preprocessor: Fitted preprocessor
        feature_names: List of feature names used in training
    
    Returns:
        Predicted price in AED and confidence interval
    """
    # Create a sample dataframe with the new plate info
    new_plate = pd.DataFrame({
        'city': [city],
        'code': [code],
        'plate_number': [str(plate_number)],
        'price': ['Unknown']  # Placeholder
    })
    
    # Apply feature engineering
    processed_plate = extract_features(new_plate)
    
    # Match the columns with training data
    missing_cols = set(feature_names) - set(processed_plate.columns)
    for col in missing_cols:
        processed_plate[col] = 0
    
    # Select only the columns used during training
    processed_plate = processed_plate[feature_names]
    
    # Preprocess
    X_new = preprocessor.transform(processed_plate)
    
    # Get predictions from each model
    nn_pred = models['neural_network'].predict(X_new)[0][0]
    xgb_pred = models['xgboost'].predict(X_new)[0]
    gb_pred = models['gradient_boosting'].predict(X_new)[0]
    
    # Ensemble prediction (weighted average)
    ensemble_pred = 0.5 * nn_pred + 0.3 * xgb_pred + 0.2 * gb_pred
    
    # Calculate a simple confidence interval based on the variance of predictions
    predictions = [nn_pred, xgb_pred, gb_pred]
    std_dev = np.std(predictions)
    confidence_interval = (ensemble_pred - 1.96 * std_dev, ensemble_pred + 1.96 * std_dev)
    
    return {
        'predicted_price': ensemble_pred,
        'confidence_interval': confidence_interval,
        'individual_predictions': {
            'neural_network': nn_pred,
            'xgboost': xgb_pred,
            'gradient_boosting': gb_pred
        }
    }

# Main execution
if __name__ == "__main__":
    # Run cross-validation
    print("Running cross-validation...")
    cv_scores = cross_validate_model(X, y)
    
    # Build ensemble model
    print("\nBuilding ensemble model...")
    models = build_ensemble_model(X_train, y_train, X_test, y_test)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(X, y, feature_names)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Example prediction
    test_plates = [
        {'city': 'dubai', 'code': 'b', 'plate_number': '1234'},
        {'city': 'dubai', 'code': 'o', 'plate_number': '7777'},
        {'city': 'abu-dhabi', 'code': '18', 'plate_number': '123'},
        {'city': 'fujairah', 'code': 'j', 'plate_number': '786'}
    ]
    
    print("\nExample predictions:")
    for plate in test_plates:
        result = predict_plate_price_ensemble(
            city=plate['city'],
            code=plate['code'],
            plate_number=plate['plate_number'],
            models=models,
            preprocessor=preprocessor,
            feature_names=feature_names
        )
        
        print(f"\n{plate['city'].title()} {plate['code'].upper()} {plate['plate_number']}:")
        print(f"Predicted price: AED {result['predicted_price']:,.2f}")
        print(f"95% Confidence interval: AED {result['confidence_interval'][0]:,.2f} - AED {result['confidence_interval'][1]:,.2f}")
    
    # Save the models
    tf.keras.models.save_model(models['neural_network'], 'number_plate_nn_model.h5')
    models['xgboost'].save_model('number_plate_xgb_model.json')
    
    # Visualize the neural network training history
    history = models['history']
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'].head(15), feature_importance['Importance'].head(15))
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()  # Display the most important at the top
    plt.tight_layout()
    plt.savefig('feature_importance.png')