import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('number_plates.csv')

# Feature engineering (from previous artifact)
df_processed = extract_features(df)
X, y, preprocessor, feature_names = prepare_features_for_model(df_processed)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network
def build_model(input_shape, learning_rate=0.001):
    model = models.Sequential([
        # Input layer
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                    input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layers
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer (regression)
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model

# Create and train the model
input_shape = X_train.shape[1]
model = build_model(input_shape)

# Use early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# Reduce learning rate when plateauing
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,
    patience=5, 
    min_lr=1e-6
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model
y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: AED {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Visualize training history
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
plt.show()

# Prediction function for new plates
def predict_plate_price(city, code, plate_number, model, preprocessor, feature_names):
    """
    Predict the price of a new number plate
    
    Args:
        city: City name (e.g., 'dubai')
        code: Code letter/number (e.g., 'b')
        plate_number: The plate number (e.g., '6901')
        model: Trained neural network model
        preprocessor: Fitted preprocessor
        feature_names: List of feature names used in training
    
    Returns:
        Predicted price in AED
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
    
    # Select and order the columns to match the training data
    processed_plate = processed_plate[feature_names]
    
    # Preprocess
    X_new = preprocessor.transform(processed_plate)
    
    # Predict
    predicted_price = model.predict(X_new)[0][0]
    
    return predicted_price

# Example usage
if __name__ == "__main__":
    # Save the model
    model.save('number_plate_price_model.h5')
    
    # Example prediction
    predicted_price = predict_plate_price(
        city='dubai',
        code='b',
        plate_number='1234',
        model=model,
        preprocessor=preprocessor,
        feature_names=feature_names
    )
    
    print(f"Predicted price for Dubai B 1234: AED {predicted_price:,.2f}")