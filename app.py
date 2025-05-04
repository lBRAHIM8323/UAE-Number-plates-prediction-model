import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import xgboost as xgb
import json
from pathlib import Path

class NumberPlatePricerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Number Plate Price Predictor")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")
        
        # Load models and preprocessor
        self.load_models()
        
        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Number Plate Price Predictor", 
            font=("Arial", 18, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=10, sticky="w")
        
        # Description
        desc_label = ttk.Label(
            main_frame,
            text="Enter the details of the number plate to get a price estimate",
            font=("Arial", 10)
        )
        desc_label.grid(row=1, column=0, columnspan=3, pady=(0, 20), sticky="w")
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Plate Details", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky="ew")
        
        # City selection
        ttk.Label(input_frame, text="City:").grid(row=0, column=0, pady=5, padx=5, sticky="w")
        self.city_var = tk.StringVar()
        city_combo = ttk.Combobox(
            input_frame, 
            textvariable=self.city_var, 
            state="readonly",
            width=15
        )
        city_combo['values'] = ('dubai', 'abu-dhabi', 'fujairah')
        city_combo.current(0)
        city_combo.grid(row=0, column=1, pady=5, padx=5, sticky="w")
        
        # Code selection
        ttk.Label(input_frame, text="Code:").grid(row=1, column=0, pady=5, padx=5, sticky="w")
        self.code_var = tk.StringVar()
        self.code_entry = ttk.Entry(input_frame, textvariable=self.code_var, width=10)
        self.code_entry.grid(row=1, column=1, pady=5, padx=5, sticky="w")
        ttk.Label(input_frame, text="(e.g., 'b', 'o', '18')").grid(row=1, column=2, pady=5, padx=5, sticky="w")
        
        # Plate number input
        ttk.Label(input_frame, text="Plate Number:").grid(row=2, column=0, pady=5, padx=5, sticky="w")
        self.plate_var = tk.StringVar()
        self.plate_entry = ttk.Entry(input_frame, textvariable=self.plate_var, width=10)
        self.plate_entry.grid(row=2, column=1, pady=5, padx=5, sticky="w")
        ttk.Label(input_frame, text="(e.g., '1234', '7777')").grid(row=2, column=2, pady=5, padx=5, sticky="w")
        
        # Predict button
        predict_btn = ttk.Button(main_frame, text="Predict Price", command=self.predict_price)
        predict_btn.grid(row=3, column=0, pady=20)
        
        # Reset button
        reset_btn = ttk.Button(main_frame, text="Reset", command=self.reset_form)
        reset_btn.grid(row=3, column=1, pady=20)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        self.results_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky="ew")
        
        # Price prediction label
        self.price_label = ttk.Label(
            self.results_frame, 
            text="Enter plate details and click 'Predict Price'",
            font=("Arial", 12)
        )
        self.price_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="w")
        
        # Confidence interval label
        self.confidence_label = ttk.Label(
            self.results_frame,
            text="",
            font=("Arial", 10)
        )
        self.confidence_label.grid(row=1, column=0, columnspan=2, pady=5, sticky="w")
        
        # Feature importance frame
        self.feature_frame = ttk.LabelFrame(main_frame, text="Plate Value Factors", padding="10")
        self.feature_frame.grid(row=5, column=0, columnspan=3, pady=10, sticky="ew")
        
        # Create a canvas for the matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.feature_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tips frame
        tips_frame = ttk.LabelFrame(main_frame, text="Tips for Valuable Plates", padding="10")
        tips_frame.grid(row=6, column=0, columnspan=3, pady=10, sticky="ew")
        
        tips_text = """
• Dubai plates are generally more valuable than other cities
• Repeating digits significantly increase value (e.g., 7777, 8888)
• Palindromes (same forward and backward) are desirable
• Special patterns like 123, 786, 1000 command higher prices
• Lower numbers generally have higher value
• Single-digit plates are extremely valuable
• Code 'AA' in Dubai is considered prestigious
        """
        
        tips_label = ttk.Label(tips_frame, text=tips_text, justify=tk.LEFT)
        tips_label.pack(anchor="w")
        
        # Set initial state
        self.reset_form()
    
    def load_models(self):
        """Load the trained models and preprocessor"""
        try:
            # Check if models exist, if not create dummy models for demo
            nn_path = Path("number_plate_nn_model.h5")
            xgb_path = Path("number_plate_xgb_model.json")
            
            if not nn_path.exists() or not xgb_path.exists():
                print("Models not found. Creating demo models...")
                self.create_demo_models()
            
            # Load neural network model
            self.nn_model = tf.keras.models.load_model("number_plate_nn_model.h5")
            
            # Load XGBoost model
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model("number_plate_xgb_model.json")
            
            # Load preprocessor and feature names
            with open("preprocessor.pkl", "rb") as f:
                self.preprocessor = pickle.load(f)
            
            with open("feature_names.json", "r") as f:
                self.feature_names = json.load(f)
            
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            messagebox.showerror("Model Loading Error", 
                                 "Could not load prediction models. Using demo mode.")
            self.create_demo_models()
    
    def create_demo_models(self):
        """Create placeholder models for demonstration"""
        # Create a simple neural network model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Save the model
        model.save("number_plate_nn_model.h5")
        self.nn_model = model
        
        # Create a simple XGBoost model
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.save_model("number_plate_xgb_model.json")
        
        # Create dummy preprocessor and feature names
        self.preprocessor = StandardScaler()
        self.feature_names = [
            'city_dubai', 'city_abu-dhabi', 'city_fujairah',
            'max_repeating_digits', 'is_palindrome', 'num_digits',
            'has_pattern_123', 'has_pattern_786', 'unique_digits', 'is_sequential'
        ]
        
        # Save preprocessor and feature names
        with open("preprocessor.pkl", "wb") as f:
            pickle.dump(self.preprocessor, f)
        
        with open("feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
    
    def extract_features_manual(self, city, code, plate_number):
        """Extract features manually for demonstration without requiring the full pipeline"""
        features = {
            'city_dubai': 1 if city == 'dubai' else 0,
            'city_abu-dhabi': 1 if city == 'abu-dhabi' else 0,
            'city_fujairah': 1 if city == 'fujairah' else 0,
            'code_' + code: 1,
            'num_digits': len(str(plate_number)),
            'plate_number_int': int(plate_number) if plate_number.isdigit() else 0
        }
        
        # Calculate repeat digits
        number_str = str(plate_number)
        max_repeats = 1
        current_repeats = 1
        
        for i in range(1, len(number_str)):
            if number_str[i] == number_str[i-1]:
                current_repeats += 1
                max_repeats = max(max_repeats, current_repeats)
            else:
                current_repeats = 1
        
        features['max_repeating_digits'] = max_repeats
        
        # Check if palindrome
        features['is_palindrome'] = 1 if number_str == number_str[::-1] else 0
        
        # Check for patterns
        features['has_pattern_123'] = 1 if '123' in number_str else 0
        features['has_pattern_786'] = 1 if '786' in number_str else 0
        
        # Count unique digits
        features['unique_digits'] = len(set(digit for digit in number_str if digit.isdigit()))
        
        # Check if sequential
        is_seq = 0
        if len(number_str) >= 3:
            # Check for ascending sequence
            asc_seq = all(int(number_str[i]) + 1 == int(number_str[i+1]) for i in range(len(number_str)-1))
            
            # Check for descending sequence
            desc_seq = all(int(number_str[i]) - 1 == int(number_str[i+1]) for i in range(len(number_str)-1))
            
            is_seq = 1 if asc_seq or desc_seq else 0
        
        features['is_sequential'] = is_seq
        
        # City prestige
        city_prestige = {'dubai': 3, 'abu-dhabi': 2, 'fujairah': 1}
        features['city_prestige'] = city_prestige.get(city, 0)
        
        # Create feature array in the right order
        X = np.zeros(len(self.feature_names))
        for i, feature in enumerate(self.feature_names):
            if feature in features:
                X[i] = features[feature]
        
        # In real app, we would use the preprocessor
        # X = self.preprocessor.transform(X.reshape(1, -1))
        
        return X.reshape(1, -1), features
    
    def predict_price(self):
        """Predict the price based on user input"""
        try:
            # Get user input
            city = self.city_var.get()
            code = self.code_var.get()
            plate_number = self.plate_var.get()
            
            # Validate input
            if not city or not code or not plate_number:
                messagebox.showwarning("Incomplete Input", "Please fill all fields.")
                return
            
            # Check if plate number is numeric
            if not plate_number.isdigit():
                messagebox.showwarning("Invalid Input", "Plate number must be numeric.")
                return
            
            # Extract features
            X, features = self.extract_features_manual(city, code, plate_number)
            
            # Get predictions from models
            nn_pred = self.nn_model.predict(X)[0][0]
            xgb_pred = self.xgb_model.predict(X)[0]
            
            # For demo purposes, add some randomness based on the features
            price_multiplier = 1.0
            
            # Apply city factors
            if city == 'dubai':
                price_multiplier *= 2.5
            elif city == 'abu-dhabi':
                price_multiplier *= 1.5
            
            # Apply special pattern factors
            if features['is_palindrome']:
                price_multiplier *= 1.8
            
            if features['max_repeating_digits'] > 2:
                price_multiplier *= 1.5 * features['max_repeating_digits']
            
            if features['has_pattern_123'] or features['has_pattern_786']:
                price_multiplier *= 1.3
            
            if features['is_sequential']:
                price_multiplier *= 1.4
            
            # Adjust for number of digits (lower is better)
            digit_factor = max(1, 5 - features['num_digits']) * 1.2
            price_multiplier *= digit_factor
            
            # Base price
            base_price = 10000
            
            # Calculate final price
            final_price = base_price * price_multiplier
            
            # Add model predictions for real implementation
            # In demo mode, we'll use a weighted ensemble
            ensemble_price = (final_price + nn_pred + xgb_pred) / 3
            
            # For demo, create some variability
            ensemble_price = max(5000, ensemble_price)
            
            # Create confidence interval
            std_dev = ensemble_price * 0.15  # 15% standard deviation
            lower_bound = max(1000, ensemble_price - 1.96 * std_dev)
            upper_bound = ensemble_price + 1.96 * std_dev
            
            # Display results
            self.price_label.configure(
                text=f"Estimated Price: AED {ensemble_price:,.2f}",
                font=("Arial", 14, "bold")
            )
            
            self.confidence_label.configure(
                text=f"95% Confidence Interval: AED {lower_bound:,.2f} - AED {upper_bound:,.2f}"
            )
            
            # Create feature importance visualization
            self.visualize_factors(features)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error details: {e}")
    
    def visualize_factors(self, features):
        """Visualize the factors affecting the plate value"""
        self.ax.clear()
        
        # Select the most relevant features
        factor_names = [
            'City Prestige', 
            'Repeating Digits', 
            'Palindrome', 
            'Special Pattern',
            'Low Number',
            'Sequential'
        ]
        
        # Calculate factor values
        factor_values = [
            features['city_prestige'] / 3,  # Normalize to 0-1
            (features['max_repeating_digits'] - 1) / 3,  # Normalize
            features['is_palindrome'],
            max(features['has_pattern_123'], features['has_pattern_786']),
            1.0 / (1.0 + int(str(features['plate_number_int']))/10000),  # Lower number = higher value
            features['is_sequential']
        ]
        
        # Scale to percentages
        factor_percentages = [val * 100 for val in factor_values]
        
        # Create horizontal bar chart
        bars = self.ax.barh(factor_names, factor_percentages, color='skyblue')
        
        # Add value labels
        for i, v in enumerate(factor_percentages):
            self.ax.text(v + 1, i, f"{v:.1f}%", va='center')
        
        self.ax.set_title('Factors Affecting Plate Value')
        self.ax.set_xlabel('Impact (%)')
        self.ax.set_xlim(0, 110)  # Leave room for labels
        
        # Update canvas
        self.canvas.draw()
    
    def reset_form(self):
        """Reset the form to default values"""
        self.city_var.set('dubai')
        self.code_var.set('')
        self.plate_var.set('')
        self.price_label.configure(text="Enter plate details and click 'Predict Price'")
        self.confidence_label.configure(text="")
        
        # Clear the plot
        self.ax.clear()
        self.ax.set_title('Enter details to see factors')
        self.canvas.draw()

if __name__ == "__main__":
    # Add missing imports for demo mode
    from sklearn.preprocessing import StandardScaler
    
    root = tk.Tk()
    app = NumberPlatePricerApp(root)
    root.mainloop()