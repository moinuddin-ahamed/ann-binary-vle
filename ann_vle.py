#!/usr/bin/env python3
"""
Production-Quality Artificial Neural Network for Binary VLE Prediction

This script implements a TensorFlow/Keras ANN to predict vapor composition (y1)
in binary azeotropic systems using liquid composition (x1), temperature (T),
and pressure (P) as inputs.

Author: Moinuddin Ahamed
Date: September 15, 2025
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import fsolve

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ann_vle.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VLEDataGenerator:
    """
    Generate synthetic VLE data for binary systems using Wilson activity coefficient model.
    
    This class implements the Wilson equation for activity coefficients and generates
    vapor-liquid equilibrium data for binary azeotropic systems.
    """
    
    def __init__(self, system_type: str = "ethanol_water"):
        """
        Initialize the VLE data generator.
        
        Args:
            system_type: Type of binary system ("ethanol_water" or "acetone_chloroform")
        """
        self.system_type = system_type
        self._setup_system_parameters()
        
    def _setup_system_parameters(self):
        """Set up system-specific parameters for Wilson equation."""
        if self.system_type == "ethanol_water":
            # Ethanol (1) - Water (2) system parameters
            self.component_names = ["Ethanol", "Water"]
            # Antoine equation parameters [A, B, C] for P_sat in mmHg, T in K
            self.antoine_params = {
                1: [8.20417, 1642.89, -42.85],  # Ethanol
                2: [8.07131, 1730.63, -39.724]  # Water
            }
            # Wilson parameters (dimensionless)
            self.lambda_12 = 0.1649  # λ12
            self.lambda_21 = 0.2937  # λ21
            # Azeotrope properties (experimental)
            self.x_az_exp = 0.894  # Azeotropic composition (x1)
            self.T_az_exp = 351.15  # Azeotropic temperature (K) at 1 atm
            
        elif self.system_type == "acetone_chloroform":
            # Acetone (1) - Chloroform (2) system parameters
            self.component_names = ["Acetone", "Chloroform"]
            self.antoine_params = {
                1: [7.11714, 1210.595, -229.664],  # Acetone
                2: [6.95465, 1170.966, -226.232]   # Chloroform
            }
            self.lambda_12 = 0.8404
            self.lambda_21 = 1.2175
            self.x_az_exp = 0.340
            self.T_az_exp = 337.58
            
        else:
            raise ValueError(f"Unsupported system type: {self.system_type}")
            
    def antoine_vapor_pressure(self, T: float, component: int) -> float:
        """
        Calculate vapor pressure using Antoine equation.
        
        Args:
            T: Temperature in Kelvin
            component: Component number (1 or 2)
            
        Returns:
            Vapor pressure in atm
        """
        A, B, C = self.antoine_params[component]
        # Antoine equation: log10(P_mmHg) = A - B/(T + C)
        log_p_mmhg = A - B / (T + C)
        p_mmhg = 10 ** log_p_mmhg
        return p_mmhg / 760.0  # Convert mmHg to atm
        
    def wilson_activity_coefficients(self, x1: float, T: float) -> Tuple[float, float]:
        """
        Calculate activity coefficients using Wilson equation.
        
        Args:
            x1: Liquid mole fraction of component 1
            T: Temperature in Kelvin
            
        Returns:
            Tuple of (gamma1, gamma2) activity coefficients
        """
        x2 = 1.0 - x1
        
        # Wilson equation
        ln_gamma1 = -np.log(x1 + self.lambda_12 * x2) + x2 * (
            self.lambda_12 / (x1 + self.lambda_12 * x2) - 
            self.lambda_21 / (self.lambda_21 * x1 + x2)
        )
        
        ln_gamma2 = -np.log(x2 + self.lambda_21 * x1) - x1 * (
            self.lambda_12 / (x1 + self.lambda_12 * x2) - 
            self.lambda_21 / (self.lambda_21 * x1 + x2)
        )
        
        return np.exp(ln_gamma1), np.exp(ln_gamma2)
        
    def calculate_vle_point(self, x1: float, T: float, P: float) -> float:
        """
        Calculate vapor composition for given liquid composition, temperature, and pressure.
        
        Args:
            x1: Liquid mole fraction of component 1
            T: Temperature in Kelvin
            P: Total pressure in atm
            
        Returns:
            Vapor mole fraction of component 1 (y1)
        """
        if x1 <= 0 or x1 >= 1:
            return x1  # Boundary conditions
            
        # Get activity coefficients
        gamma1, gamma2 = self.wilson_activity_coefficients(x1, T)
        
        # Get vapor pressures
        P1_sat = self.antoine_vapor_pressure(T, 1)
        P2_sat = self.antoine_vapor_pressure(T, 2)
        
        # Calculate vapor composition using modified Raoult's law
        x2 = 1.0 - x1
        y1 = (gamma1 * x1 * P1_sat) / P
        
        # Ensure physical bounds
        return np.clip(y1, 0.0, 1.0)
        
    def generate_dataset(self, n_points: int = 500) -> pd.DataFrame:
        """
        Generate synthetic VLE dataset with dense sampling near azeotrope.
        
        Args:
            n_points: Total number of data points to generate
            
        Returns:
            DataFrame with columns [x1, T, P, y1]
        """
        logger.info(f"Generating {n_points} VLE data points for {self.system_type}")
        
        data_points = []
        
        # Generate temperature range around azeotropic temperature
        T_min = self.T_az_exp - 20  # ±20K around azeotrope
        T_max = self.T_az_exp + 20
        
        # Pressure range (typically 0.5 to 2 atm for better generalization)
        P_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        
        for i in range(n_points):
            # Dense sampling near azeotrope (60% of points)
            if i < n_points * 0.6:
                # Sample around azeotropic composition
                if np.random.random() < 0.5:
                    x1 = np.random.beta(8, 2)  # Skewed towards high x1
                else:
                    x1 = np.random.uniform(0.7, 1.0)  # Dense near azeotrope
            else:
                # Uniform sampling across full range (40% of points)
                x1 = np.random.uniform(0.01, 0.99)
                
            # Sample temperature
            T = np.random.uniform(T_min, T_max)
            
            # Sample pressure
            P = np.random.choice(P_values)
            
            # Calculate corresponding y1
            y1 = self.calculate_vle_point(x1, T, P)
            
            data_points.append([x1, T, P, y1])
            
        # Create DataFrame
        df = pd.DataFrame(data_points, columns=['x1', 'T', 'P', 'y1'])
        
        # Remove any invalid points
        df = df[(df['y1'] >= 0) & (df['y1'] <= 1)].reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} valid VLE data points")
        logger.info(f"Data ranges - x1: [{df['x1'].min():.3f}, {df['x1'].max():.3f}], "
                   f"T: [{df['T'].min():.1f}, {df['T'].max():.1f}] K, "
                   f"y1: [{df['y1'].min():.3f}, {df['y1'].max():.3f}]")
        
        return df


class ANNVLEModel:
    """
    Artificial Neural Network for VLE prediction with physics constraints.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ANN model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Build the neural network architecture.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building ANN architecture")
        
        # Input layer (3 features: x1, T, P)
        inputs = layers.Input(shape=(3,), name='input_layer')
        
        # Hidden layers with ReLU activation
        x = inputs
        for i, units in enumerate(self.config['hidden_layers']):
            x = layers.Dense(
                units, 
                activation='relu',
                kernel_initializer='he_normal',
                name=f'hidden_{i+1}'
            )(x)
            
            # Add dropout for regularization
            if self.config['dropout_rate'] > 0:
                x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Output layer with sigmoid activation (physics constraint: 0 ≤ y1 ≤ 1)
        outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='ANN_VLE')
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Model architecture:\n{model.summary()}")
        return model
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """
        Prepare and split data for training.
        
        Args:
            df: DataFrame with VLE data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Preparing data for training")
        
        # Features and target
        X = df[['x1', 'T', 'P']].values
        y = df['y1'].values.reshape(-1, 1)
        
        # Train-validation-test split (70-15-15)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=RANDOM_SEED  # 0.176 ≈ 0.15/0.85
        )
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Starting model training")
        
        # Build model
        self.model = self.build_model()
        
        # Setup callbacks
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate scheduler
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['patience']//2,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath='models/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=f'logs/tensorboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info("Model training completed")
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"Test Metrics - MSE: {mse:.6f}, RMSE: {rmse:.6f}, "
                   f"MAE: {mae:.6f}, R²: {r2:.4f}")
        
        return metrics
        
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")


class VLEAnalyzer:
    """
    Analysis and visualization tools for VLE predictions.
    """
    
    def __init__(self, model: ANNVLEModel, data_generator: VLEDataGenerator):
        """
        Initialize the analyzer.
        
        Args:
            model: Trained ANN model
            data_generator: VLE data generator for reference calculations
        """
        self.model = model
        self.data_generator = data_generator
        
    def plot_training_history(self, save_path: str = 'plots/training_history.png') -> None:
        """Plot training and validation loss curves."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.model.history.history['loss'], label='Training Loss')
        plt.plot(self.model.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History - Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.model.history.history['mae'], label='Training MAE')
        plt.plot(self.model.history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Training History - MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training history plot saved to {save_path}")
        
    def plot_parity_plot(self, X_test: np.ndarray, y_test: np.ndarray, 
                        save_path: str = 'plots/parity_plot.png') -> None:
        """Create parity plot of experimental vs predicted values."""
        y_pred = self.model.model.predict(X_test, verbose=0)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # ±10% error bands
        plt.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'g--', alpha=0.5, label='±10% Error')
        plt.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'g--', alpha=0.5)
        
        plt.xlabel('Experimental y₁')
        plt.ylabel('Predicted y₁')
        plt.title('Parity Plot: ANN Predictions vs Experimental Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Add R² annotation
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Parity plot saved to {save_path}")
        
    def detect_azeotrope(self, T: float = None, P: float = 1.0) -> Tuple[float, float]:
        """
        Detect azeotrope composition by finding where y1 ≈ x1.
        
        Args:
            T: Temperature in Kelvin (if None, use experimental azeotropic T)
            P: Pressure in atm
            
        Returns:
            Tuple of (x_azeotrope, T_azeotrope)
        """
        if T is None:
            T = self.data_generator.T_az_exp
            
        def azeotrope_condition(x1):
            """Azeotrope condition: y1 - x1 = 0"""
            X_scaled = self.model.scaler_X.transform([[x1, T, P]])
            y1_pred = self.model.model.predict(X_scaled, verbose=0)[0, 0]
            return y1_pred - x1
            
        # Solve for azeotropic composition
        try:
            # Try multiple initial guesses
            initial_guesses = [0.5, 0.8, 0.9, self.data_generator.x_az_exp]
            solutions = []
            
            for guess in initial_guesses:
                try:
                    sol = fsolve(azeotrope_condition, guess)[0]
                    if 0 < sol < 1:  # Valid mole fraction
                        solutions.append(sol)
                except:
                    continue
            
            if solutions:
                x_az_pred = np.mean(solutions)  # Average of valid solutions
            else:
                x_az_pred = np.nan
            
        except:
            x_az_pred = np.nan
            
        logger.info(f"Azeotrope detection - Predicted: x1 = {x_az_pred:.4f} at T = {T:.1f} K")
        logger.info(f"Experimental azeotrope: x1 = {self.data_generator.x_az_exp:.4f} at T = {self.data_generator.T_az_exp:.1f} K")
        
        return x_az_pred, T
        
    def compare_with_raoult_law(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Compare ANN performance with Raoult's Law baseline.
        
        Args:
            X_test: Test features [x1, T, P]
            y_test: Test targets [y1]
            
        Returns:
            Dictionary comparing ANN and Raoult's Law metrics
        """
        logger.info("Comparing with Raoult's Law baseline")
        
        # Raoult's Law predictions (ideal solution)
        y_raoult = []
        for i in range(len(X_test)):
            x1, T, P = self.model.scaler_X.inverse_transform(X_test[i:i+1])[0]
            
            # Get vapor pressures
            P1_sat = self.data_generator.antoine_vapor_pressure(T, 1)
            P2_sat = self.data_generator.antoine_vapor_pressure(T, 2)
            
            # Raoult's Law: y1 = (x1 * P1_sat) / P
            y1_raoult = (x1 * P1_sat) / P
            y_raoult.append(np.clip(y1_raoult, 0, 1))
            
        y_raoult = np.array(y_raoult)
        
        # ANN predictions
        y_ann = self.model.model.predict(X_test, verbose=0).flatten()
        
        # Calculate metrics for both methods
        metrics_ann = {
            'mse': mean_squared_error(y_test, y_ann),
            'mae': mean_absolute_error(y_test, y_ann),
            'r2': r2_score(y_test, y_ann)
        }
        
        metrics_raoult = {
            'mse': mean_squared_error(y_test, y_raoult),
            'mae': mean_absolute_error(y_test, y_raoult),
            'r2': r2_score(y_test, y_raoult)
        }
        
        logger.info(f"ANN Performance - MSE: {metrics_ann['mse']:.6f}, MAE: {metrics_ann['mae']:.6f}, R²: {metrics_ann['r2']:.4f}")
        logger.info(f"Raoult's Law Performance - MSE: {metrics_raoult['mse']:.6f}, MAE: {metrics_raoult['mae']:.6f}, R²: {metrics_raoult['r2']:.4f}")
        
        return {'ann': metrics_ann, 'raoult': metrics_raoult}


def main(args):
    """Main function to run the complete ANN VLE pipeline."""
    logger.info("Starting ANN VLE Prediction Pipeline")
    
    # Create directories if they don't exist
    for directory in ['data', 'models', 'plots', 'logs']:
        Path(directory).mkdir(exist_ok=True)
    
    # Configuration
    config = {
        'hidden_layers': [args.hidden_units] * args.hidden_layers,
        'dropout_rate': args.dropout,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience
    }
    
    # Step 1: Generate VLE dataset
    data_generator = VLEDataGenerator(system_type=args.system)
    df = data_generator.generate_dataset(n_points=args.n_points)
    df.to_csv('data/vle_dataset.csv', index=False)
    
    # Step 2: Initialize and prepare model
    model = ANNVLEModel(config)
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(df)
    
    # Step 3: Train model
    model.train(X_train, y_train, X_val, y_val)
    
    # Step 4: Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Step 5: Save model
    model.save_model('models/ann_vle_model.h5')
    
    # Step 6: Analysis and visualization
    analyzer = VLEAnalyzer(model, data_generator)
    analyzer.plot_training_history()
    analyzer.plot_parity_plot(X_test, y_test)
    
    # Azeotrope detection
    x_az_pred, T_az = analyzer.detect_azeotrope()
    
    # Comparison with Raoult's Law
    comparison_metrics = analyzer.compare_with_raoult_law(X_test, y_test)
    
    # Save results summary
    results = {
        'model_config': config,
        'test_metrics': metrics,
        'azeotrope_detection': {
            'predicted': {'x1': x_az_pred, 'T': T_az},
            'experimental': {'x1': data_generator.x_az_exp, 'T': data_generator.T_az_exp}
        },
        'comparison_with_raoult': comparison_metrics
    }
    
    import json
    with open('models/results_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Final test R² score: {metrics['r2']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANN for Binary VLE Prediction")
    
    # Data parameters
    parser.add_argument('--system', type=str, default='ethanol_water',
                       choices=['ethanol_water', 'acetone_chloroform'],
                       help='Binary system type')
    parser.add_argument('--n_points', type=int, default=500,
                       help='Number of data points to generate')
    
    # Model architecture
    parser.add_argument('--hidden_layers', type=int, default=3,
                       help='Number of hidden layers')
    parser.add_argument('--hidden_units', type=int, default=64,
                       help='Number of neurons in each hidden layer')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for regularization')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)