"""
Training Pipeline for Hybrid Power Demand Forecasting Model
Implements training for Trend + Fourier + LSTM models
"""

import numpy as np
import pandas as pd
import pickle
import json
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from itertools import product

from src.models.seq2seq_lstm import UnivariateSeq2SeqLSTM, train_model


class PowerDemandTrainer:
    """
    Trainer for hybrid power demand forecasting model
    
    Three-stage model:
    1. Trend: Log-linear regression
    2. Fourier Seasonality: Grid search for optimal harmonics
    3. LSTM Residual: Random search for optimal hyperparameters
    """
    
    def __init__(self, model_dir: str = "models/production", 
                 window_size: int = 168, horizon: int = 24):
        """
        Initialize trainer
        
        Args:
            model_dir: Directory to save trained models
            window_size: LSTM window size (hours)
            horizon: Forecast horizon (hours)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.window_size = window_size
        self.horizon = horizon
        
        # Models
        self.trend_model = None
        self.fourier_model = None
        self.lstm_model = None
        self.residual_scaler = None
        self.exog_scaler = None
        
        # Best parameters
        self.best_fourier_params = {}
        self.best_lstm_params = {}
        
        # For global index calculation (test_data.ipynb style)
        self.all_data_length = None
    
    def train_trend(self, train_data: pd.DataFrame, 
                    target_col: str = "power demand(MW)") -> np.ndarray:
        """
        Train trend model using log-linear regression
        
        Args:
            train_data: Training dataframe with timestamp index
            target_col: Target column name
            
        Returns:
            trend_predictions: Trend predictions for training data
        """
        print("\n" + "="*80)
        print("TRAINING TREND MODEL (Log-Linear Regression)")
        print("="*80)
        
        # Time index
        t = np.arange(len(train_data))
        X = sm.add_constant(t)
        
        # Log transform target
        y = np.log(train_data[target_col].values)
        
        # Fit OLS
        self.trend_model = sm.OLS(y, X).fit()
        
        # Predict
        trend_pred_log = self.trend_model.predict(X)
        trend_pred = np.exp(trend_pred_log)
        
        print(f"\n✓ Trend model trained")
        print(f"  - R²: {self.trend_model.rsquared:.4f}")
        print(f"  - Coefficients: {self.trend_model.params}")
        
        return trend_pred
    
    def _build_fourier_matrix(self, t_idx: np.ndarray, 
                              Kd: int, Kw: int, Ky: int) -> np.ndarray:
        """
        Build Fourier feature matrix
        
        Args:
            t_idx: Time indices
            Kd: Daily harmonics
            Kw: Weekly harmonics
            Ky: Yearly harmonics
            
        Returns:
            Fourier feature matrix (N, 2*(Kd+Kw+Ky))
        """
        features = []
        
        # Daily (period = 24)
        for k in range(1, Kd + 1):
            features.append(np.sin(2 * np.pi * k * t_idx / 24))
            features.append(np.cos(2 * np.pi * k * t_idx / 24))
        
        # Weekly (period = 24*7)
        for k in range(1, Kw + 1):
            features.append(np.sin(2 * np.pi * k * t_idx / (24 * 7)))
            features.append(np.cos(2 * np.pi * k * t_idx / (24 * 7)))
        
        # Yearly (period = 24*365.25)
        for k in range(1, Ky + 1):
            features.append(np.sin(2 * np.pi * k * t_idx / (24 * 365.25)))
            features.append(np.cos(2 * np.pi * k * t_idx / (24 * 365.25)))
        
        return np.column_stack(features)
    
    def _select_harmonics_from_cevr(self, fourier_coefs: np.ndarray, 
                                    max_K: int, threshold: float = 0.95) -> int:
        """
        Select number of harmonics based on Cumulative Explained Variance Ratio (CEVR)
        
        Args:
            fourier_coefs: Fourier coefficients [a1, b1, a2, b2, ...]
            max_K: Maximum number of harmonics
            threshold: CEVR threshold (default: 0.95)
            
        Returns:
            Selected number of harmonics
        """
        # Calculate energy for each harmonic pair (ak, bk)
        harmonics = max_K
        if len(fourier_coefs) < 2 * harmonics:
            return harmonics
        
        energies = []
        for k in range(harmonics):
            a_k = fourier_coefs[2 * k]
            b_k = fourier_coefs[2 * k + 1]
            energy = a_k**2 + b_k**2
            energies.append(energy)
        
        energies = np.array(energies)
        total_energy = energies.sum()
        
        if total_energy == 0:
            return harmonics
        
        cumulative = np.cumsum(energies) / total_energy
        for idx, ratio in enumerate(cumulative, start=1):
            if ratio >= threshold:
                return idx
        
        return harmonics
    
    def train_fourier(self,
                     train_data: pd.DataFrame,
                     val_data: pd.DataFrame,
                     detrended_col: str = "detrend",
                     exog_features: Optional[List[str]] = None,
                     Kd: Optional[int] = None,
                     Kw: Optional[int] = None,
                     Ky: Optional[int] = None,
                     grid_search: bool = True,
                     Kd_range: Optional[np.ndarray] = None,
                     Kw_range: Optional[np.ndarray] = None,
                     Ky_range: Optional[np.ndarray] = None,
                     use_cevr: bool = True,
                     cevr_threshold: float = 0.95,
                     all_data_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train Fourier seasonality model with Grid Search or fixed harmonics
        
        Like test_data.ipynb: Generate Fourier terms using global index (0 to all_data_length)
        
        Args:
            train_data: Training dataframe
            val_data: Validation dataframe
            detrended_col: Detrended target column
            exog_features: List of exogenous feature names
            Kd: Daily harmonics (if provided, used as fixed value)
            Kw: Weekly harmonics (if provided, used as fixed value)
            Ky: Yearly harmonics (if provided, used as fixed value)
            grid_search: If True, perform grid search over harmonics
            Kd_range: Range for Kd grid search (default: np.arange(1, 15))
            Kw_range: Range for Kw grid search (default: np.arange(1, 15))
            Ky_range: Range for Ky grid search (default: np.arange(1, 15))
            use_cevr: If True, apply CEVR-based harmonic selection after grid search
            cevr_threshold: CEVR threshold (default: 0.95)
            all_data_length: Total length of ALL data (train+val+test) for global index
            
        Returns:
            (train_seasonality, val_seasonality): Fitted seasonality for train and val
        """
        # Prepare data
        y_train = train_data[detrended_col].values
        y_val = val_data[detrended_col].values
        
        # Use global index if all_data_length is provided (like test_data.ipynb)
        if all_data_length is not None:
            print(f"  ✅ Using global index (test_data.ipynb style): t=0 to t={all_data_length-1}")
            t_train = np.arange(len(y_train))
            t_val = np.arange(len(y_train), len(y_train) + len(y_val))
            # Store all_data_length for later use in inference
            self.all_data_length = all_data_length
        else:
            t_train = np.arange(len(y_train))
            t_val = np.arange(len(y_train), len(y_train) + len(y_val))
        
        # Exogenous features
        X_exog_train = None
        X_exog_val = None
        if exog_features is not None:
            self.exog_scaler = StandardScaler()
            X_exog_train = self.exog_scaler.fit_transform(train_data[exog_features].values)
            X_exog_val = self.exog_scaler.transform(val_data[exog_features].values)
        
        # Determine if we should do grid search
        if grid_search and (Kd is None or Kw is None or Ky is None):
            # Set default ranges
            if Kd_range is None:
                Kd_range = np.arange(1, 15)
            if Kw_range is None:
                Kw_range = np.arange(1, 15)
            if Ky_range is None:
                Ky_range = np.arange(1, 15)
            
            print("\n" + "="*80)
            print("TRAINING FOURIER SEASONALITY MODEL (Grid Search)")
            print("="*80)
            print(f"Grid search ranges:")
            print(f"  - Kd (daily): {Kd_range.min()} to {Kd_range.max()}")
            print(f"  - Kw (weekly): {Kw_range.min()} to {Kw_range.max()}")
            print(f"  - Ky (yearly): {Ky_range.min()} to {Ky_range.max()}")
            print(f"  - Total combinations: {len(Kd_range) * len(Kw_range) * len(Ky_range)}")
            
            best_val_mse = float('inf')
            best_Kd = None
            best_Kw = None
            best_Ky = None
            best_model = None
            best_train_seasonality = None
            best_val_seasonality = None
            
            total_combinations = len(Kd_range) * len(Kw_range) * len(Ky_range)
            current_combination = 0
            
            print(f"\nStarting grid search...")
            for Kd_val in Kd_range:
                for Kw_val in Kw_range:
                    for Ky_val in Ky_range:
                        current_combination += 1
                        if current_combination % 100 == 0:
                            print(f"  Progress: {current_combination}/{total_combinations} combinations tested...")
                        
                        fourier_train = self._build_fourier_matrix(t_train, Kd_val, Kw_val, Ky_val)
                        fourier_val = self._build_fourier_matrix(t_val, Kd_val, Kw_val, Ky_val)
                        
                        # Combine with exogenous features
                        if X_exog_train is not None:
                            X_train = np.hstack([fourier_train, X_exog_train])
                            X_val = np.hstack([fourier_val, X_exog_val])
                        else:
                            X_train = fourier_train
                            X_val = fourier_val
                        
                        # Train and evaluate
                        model = LinearRegression().fit(X_train, y_train)
                        val_pred = model.predict(X_val)
                        val_mse = mean_squared_error(y_val, val_pred)
                        
                        # Update best if better
                        if val_mse < best_val_mse:
                            best_val_mse = val_mse
                            best_Kd = Kd_val
                            best_Kw = Kw_val
                            best_Ky = Ky_val
                            best_model = model
                            best_train_seasonality = model.predict(X_train)
                            best_val_seasonality = val_pred
            
            # Store best parameters and model
            self.best_fourier_params = {
                'Kd': int(best_Kd),
                'Kw': int(best_Kw),
                'Ky': int(best_Ky)
            }
            final_model = best_model
            final_train_pred = best_train_seasonality
            final_val_pred = best_val_seasonality

            if use_cevr:
                coef = best_model.coef_
                daily_len = 2 * best_Kd
                weekly_len = 2 * best_Kw
                yearly_len = 2 * best_Ky
                daily_cevr = self._select_harmonics_from_cevr(coef[:daily_len], best_Kd, cevr_threshold)
                weekly_cevr = self._select_harmonics_from_cevr(
                    coef[daily_len:daily_len + weekly_len], best_Kw, cevr_threshold)
                yearly_cevr = self._select_harmonics_from_cevr(
                    coef[daily_len + weekly_len:daily_len + weekly_len + yearly_len], best_Ky, cevr_threshold)

                if (daily_cevr != best_Kd) or (weekly_cevr != best_Kw) or (yearly_cevr != best_Ky):
                    print("\nApplying CEVR-based harmonic selection:")
                    print(f"  - Original (Kd, Kw, Ky): ({best_Kd}, {best_Kw}, {best_Ky})")
                    print(f"  - Selected (Kd, Kw, Ky): ({daily_cevr}, {weekly_cevr}, {yearly_cevr})")
                    best_Kd, best_Kw, best_Ky = daily_cevr, weekly_cevr, yearly_cevr
                    fourier_train_final = self._build_fourier_matrix(t_train, best_Kd, best_Kw, best_Ky)
                    fourier_val_final = self._build_fourier_matrix(t_val, best_Kd, best_Kw, best_Ky)

                    if X_exog_train is not None:
                        X_train_final = np.hstack([fourier_train_final, X_exog_train])
                        X_val_final = np.hstack([fourier_val_final, X_exog_val])
                    else:
                        X_train_final = fourier_train_final
                        X_val_final = fourier_val_final

                    final_model = LinearRegression().fit(X_train_final, y_train)
                    final_train_pred = final_model.predict(X_train_final)
                    final_val_pred = final_model.predict(X_val_final)
                    self.best_fourier_params = {
                        'Kd': int(best_Kd),
                        'Kw': int(best_Kw),
                        'Ky': int(best_Ky)
                    }

            self.fourier_model = final_model

            print(f"\n✓ Grid search complete!")
            print(f"  - Best Kd (daily): {self.best_fourier_params['Kd']}")
            print(f"  - Best Kw (weekly): {self.best_fourier_params['Kw']}")
            print(f"  - Best Ky (yearly): {self.best_fourier_params['Ky']}")
            print(f"  - Best Validation MSE: {best_val_mse:.4f}")
            
            return final_train_pred, final_val_pred
        
        else:
            # Use fixed harmonics
            if Kd is None:
                Kd = 3
            if Kw is None:
                Kw = 13
            if Ky is None:
                Ky = 3
            
            print("\n" + "="*80)
            print("TRAINING FOURIER SEASONALITY MODEL (Fixed Harmonics)")
            print("="*80)
            print(f"Using fixed harmonics: Kd={Kd}, Kw={Kw}, Ky={Ky}")
            
            fourier_train = self._build_fourier_matrix(t_train, Kd, Kw, Ky)
            fourier_val = self._build_fourier_matrix(t_val, Kd, Kw, Ky)
            
            # Combine with exogenous features
            if X_exog_train is not None:
                X_train = np.hstack([fourier_train, X_exog_train])
                X_val = np.hstack([fourier_val, X_exog_val])
            else:
                X_train = fourier_train
                X_val = fourier_val
            
            print("\nTraining Fourier regression...")
            self.fourier_model = LinearRegression().fit(X_train, y_train)
            
            train_pred = self.fourier_model.predict(X_train)
            val_pred = self.fourier_model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            
            self.best_fourier_params = {
                'Kd': int(Kd),
                'Kw': int(Kw),
                'Ky': int(Ky)
            }
            
            print(f"\n✓ Fourier model trained")
            print(f"  - Kd (daily): {Kd}")
            print(f"  - Kw (weekly): {Kw}")
            print(f"  - Ky (yearly): {Ky}")
            print(f"  - Validation MSE: {val_mse:.4f}")
            
            return train_pred, val_pred
    
    def train_lstm(self,
                   train_residuals: np.ndarray,
                   val_residuals: np.ndarray,
                   n_iter: int = 50,
                   epochs: int = 100,
                   device: str = 'cuda') -> float:
        """
        Train LSTM residual model with random search
        
        Args:
            train_residuals: Training residuals
            val_residuals: Validation residuals
            n_iter: Number of random search iterations
            epochs: Maximum epochs per trial
            device: Device to use ('cuda' or 'cpu')
            
        Returns:
            best_val_loss: Best validation loss achieved
        """
        print("\n" + "="*80)
        print("TRAINING LSTM RESIDUAL MODEL (Random Search)")
        print("="*80)
        
        # Scale residuals
        self.residual_scaler = StandardScaler()
        train_residuals_scaled = self.residual_scaler.fit_transform(
            train_residuals.reshape(-1, 1)
        ).flatten()
        val_residuals_scaled = self.residual_scaler.transform(
            val_residuals.reshape(-1, 1)
        ).flatten()
        
        # Create non-overlapping windows (stride = horizon)
        def create_windows(data, window_size, horizon):
            X, y = [], []
            for i in range(0, len(data) - window_size - horizon + 1, horizon):
                X.append(data[i:i+window_size])
                y.append(data[i+window_size:i+window_size+horizon])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_windows(train_residuals_scaled, self.window_size, self.horizon)
        X_val, y_val = create_windows(val_residuals_scaled, self.window_size, self.horizon)
        
        print(f"\nCreating non-overlapping windows (stride={self.horizon})...")
        print(f"  - Train windows: {len(X_train)}")
        print(f"  - Val windows: {len(X_val)}")
        
        # Random search parameter distributions
        from sklearn.model_selection import ParameterSampler
        
        param_distributions = {
            'hidden_size': [64, 128, 256],
            'num_layers': [2, 3],
            'dropout': [0.1, 0.2, 0.3],
            'bidirectional': [False, True],
            'use_attention': [False, True],
            'batch_size': [32, 64],
            'learning_rate': [0.0005, 0.001, 0.005],
            'optimizer': ['adamw'],
            'weight_decay': [0.0, 1e-4],
            'grad_clip': [0.5, 1.0],
            'teacher_forcing_ratio': [0.0, 0.3, 0.5, 0.7],
            'scheduler': ['cosine', 'reduce'],
            'early_stopping_patience': [10, 15]
        }
        
        sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=42)
        
        best_val_loss = float('inf')
        best_model_state = None
        best_params = None
        
        print(f"\nStarting random search ({n_iter} iterations)...")
        
        for i, params in enumerate(sampler):
            print(f"\n{'='*80}")
            print(f"Trial {i+1}/{n_iter}")
            print(f"{'='*80}")
            print(f"Model parameters: {sum(p.numel() for p in UnivariateSeq2SeqLSTM(params['hidden_size'], params['num_layers'], params['dropout'], self.horizon, params['bidirectional'], params['use_attention']).parameters()):,}")
            
            try:
                # Create model
                model = UnivariateSeq2SeqLSTM(
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout'],
                    output_size=self.horizon,
                    bidirectional=params['bidirectional'],
                    use_attention=params['use_attention']
                )
                
                # Prepare data loaders
                from torch.utils.data import TensorDataset, DataLoader
                
                X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
                y_train_t = torch.FloatTensor(y_train).unsqueeze(-1)
                X_val_t = torch.FloatTensor(X_val).unsqueeze(-1)
                y_val_t = torch.FloatTensor(y_val).unsqueeze(-1)
                
                train_dataset = TensorDataset(X_train_t, y_train_t)
                val_dataset = TensorDataset(X_val_t, y_val_t)
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=params['batch_size'],
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=params['batch_size'],
                    shuffle=False
                )
                
                # Train
                config = {**params, 'epochs': epochs}
                val_loss = train_model(model, train_loader, val_loader, config, device=device, verbose=False)
                
                print(f"\nValidation Loss: {val_loss:.6f}")
                
                # Update best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    best_params = params.copy()
                    print("✨ New best model!")
                
            except Exception as e:
                print(f"❌ Trial failed: {e}")
                continue
        
        # Load best model
        if best_model_state is not None:
            self.lstm_model = UnivariateSeq2SeqLSTM(
                hidden_size=best_params['hidden_size'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout'],
                output_size=self.horizon,
                bidirectional=best_params['bidirectional'],
                use_attention=best_params['use_attention']
            )
            self.lstm_model.load_state_dict(best_model_state)
            self.best_lstm_params = best_params
            
            print(f"\n{'='*80}")
            print("BEST LSTM PARAMETERS:")
            print(f"{'='*80}")
            for key, value in sorted(best_params.items()):
                print(f"  {key:25s}: {value}")
            print(f"\nBest Validation Loss: {best_val_loss:.6f}")
            print(f"{'='*80}")
        else:
            raise ValueError("All LSTM training trials failed!")
        
        return best_val_loss
    
    def save_models(self, train_data_length: int):
        """
        Save all trained models
        
        Args:
            train_data_length: Length of training data (for correct index calculation)
        """
        print("\n" + "="*80)
        print("SAVING MODELS")
        print("="*80)
        
        # Save trend model
        with open(self.model_dir / "trend_model.pkl", "wb") as f:
            pickle.dump(self.trend_model, f)
        print("✓ Saved trend model")
        
        # Save Fourier model
        with open(self.model_dir / "fourier_model.pkl", "wb") as f:
            pickle.dump(self.fourier_model, f)
        print("✓ Saved Fourier model")
        
        # Save LSTM model
        if self.lstm_model is not None:
            torch.save(self.lstm_model.state_dict(), self.model_dir / "lstm_model.pth")
            print("✓ Saved LSTM model")
        
        # Save scalers
        if self.residual_scaler is not None:
            with open(self.model_dir / "residual_scaler.pkl", "wb") as f:
                pickle.dump(self.residual_scaler, f)
            print("✓ Saved residual scaler")
        
        if self.exog_scaler is not None:
            with open(self.model_dir / "exog_scaler.pkl", "wb") as f:
                pickle.dump(self.exog_scaler, f)
            print("✓ Saved exog scaler")
        
        # Save config
        config = {
            'window_size': self.window_size,
            'horizon': self.horizon,
            'Kd': self.best_fourier_params.get('Kd'),
            'Kw': self.best_fourier_params.get('Kw'),
            'Ky': self.best_fourier_params.get('Ky'),
            'train_data_length': train_data_length,
            'lstm_params': self.best_lstm_params
        }
        
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("✓ Saved config")
        
        print(f"\n✅ All models saved to: {self.model_dir}")

