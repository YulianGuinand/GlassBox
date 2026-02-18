import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional
import io

class DataManager:
    """Handles data loading, preprocessing, and splitting for GlassBox."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.target_column: str = ""
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_encoders = {}
        self.feature_types = {} 
        self.feature_metadata = {} # Stores min, max, mean, etc.
        self.is_classification = True
        
    def load_csv(self, file_buffer: io.BytesIO) -> pd.DataFrame:
        """Loads a CSV file into a Pandas DataFrame."""
        self.df = pd.read_csv(file_buffer)
        return self.df
    
    def preprocess(self, target_column: str, feature_columns: Optional[List[str]] = None, test_size: float = 0.2, random_state: int = 42):
        """
        Preprocesses the data: 
        - Select features
        - Handle Dates (to timestamp)
        - Handle Categorical (LabelEncode)
        - Scale Numeric (StandardScaler)
        - Encode Target
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv() first.")
            
        self.target_column = target_column
        self.feature_encoders = {} # Reset encoders
        self.feature_types = {}
        self.feature_metadata = {} # Reset metadata
        
        # 1. Filter Data
        df_clean = self.df.copy()
        if feature_columns:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = [col for col in df_clean.columns if col != target_column]
            
        # Keep only relevant columns to avoid issues with unused dirty columns
        df_clean = df_clean[self.feature_columns + [target_column]]
        
        # 2. Drop missing values
        df_clean = df_clean.dropna()
        if len(df_clean) == 0:
            raise ValueError("Dataset is empty after dropping NaN values.")

        # 3. Process Features
        X_processed = pd.DataFrame()
        
        for col in self.feature_columns:
            series = df_clean[col]
            meta = {}
            
            # Check for Date
            if pd.api.types.is_datetime64_any_dtype(series):
                dt_series = series
                X_processed[col] = dt_series.astype('int64') // 10**9
                self.feature_types[col] = 'date'
                meta = {
                    'min': dt_series.min(),
                    'max': dt_series.max(), 
                    'default': dt_series.max() # Default to latest date
                }
            elif series.dtype == 'object':
                # Try to convert to datetime first if it looks like a date
                try:
                    # simplistic check: is first element a date string?
                    pd.to_datetime(series.iloc[0]) 
                    # If success, convert whole col
                    dt_series = pd.to_datetime(series, errors='coerce')
                    if dt_series.notna().all():
                        X_processed[col] = dt_series.astype('int64') // 10**9
                        self.feature_types[col] = 'date'
                        meta = {
                            'min': dt_series.min(),
                            'max': dt_series.max(), 
                            'default': dt_series.max()
                        }
                        self.feature_metadata[col] = meta
                        continue
                except:
                    pass
                
                # If not date, treat as categorical -> Label Encode
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(series.astype(str))
                self.feature_encoders[col] = le # Store encoder
                self.feature_types[col] = 'categorical'
                meta = {
                    'classes': le.classes_.tolist(),
                    'default': le.classes_[0]
                }
            else:
                # Numeric
                X_processed[col] = series
                self.feature_types[col] = 'numeric'
                meta = {
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'mean': float(series.mean()),
                    'default': float(series.mean())
                }
            
            self.feature_metadata[col] = meta
        
        X = X_processed.values
        y = df_clean[target_column].values
        
        # 4. Encode Target
        # Force classification if few unique values, else Regression
        unique_targets = len(set(y))
        if df_clean[target_column].dtype == 'object' or isinstance(y[0], str) or unique_targets < 20:
             self.label_encoder.fit(y)
             y = self.label_encoder.transform(y)
             self.is_classification = True
        else:
             self.is_classification = False
             y = y.astype(float) # Ensure float for regression
        
        # 5. Split Data
        # Stratify only if classification
        stratify = y if (self.is_classification and unique_targets > 1) else None
        
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        # 6. Scale Features
        self.X_train = self.scaler.fit_transform(X_train_np)
        self.X_test = self.scaler.transform(X_test_np)
        
        # 7. Convert to PyTorch Tensors
        self.y_train = y_train_np
        self.y_test = y_test_np
        
    def transform_input(self, input_dict: dict) -> torch.Tensor:
        """
        Transforms a single input dictionary into a tensor using fitted scalers/encoders.
        """
        if not self.feature_columns:
            raise ValueError("Model not trained (no feature columns).")
            
        # Ensure feature_types/metadata exists for legacy objects
        if not hasattr(self, 'feature_types'):
            self.feature_types = {}
        if not hasattr(self, 'feature_metadata'):
            self.feature_metadata = {}

        # Create DF
        df = pd.DataFrame([input_dict])
        
        # Process cols
        X_processed = pd.DataFrame()
        
        for col in self.feature_columns:
            val = df[col]
            dtype = self.feature_types.get(col, 'numeric')
            
            if dtype == 'categorical':
                if col in self.feature_encoders:
                    le = self.feature_encoders[col]
                    try:
                        X_processed[col] = le.transform(val.astype(str))
                    except ValueError:
                        # Fallback for unseen
                        X_processed[col] = 0 
            
            elif dtype == 'date':
                # Expecting datetime object or string or number from UI
                try:
                    dt_val = pd.to_datetime(val)
                    X_processed[col] = dt_val.astype('int64') // 10**9
                except:
                     X_processed[col] = 0
            
            else: # Numeric
                 X_processed[col] = pd.to_numeric(val, errors='coerce').fillna(0)
                    
        # Scale
        X_np = self.scaler.transform(X_processed.values)
        return torch.FloatTensor(X_np)

    def get_dataloaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Creates PyTorch DataLoaders for training and testing."""
        if self.X_train is None:
             raise ValueError("Data not preprocessed. Call preprocess() first.")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        X_test_tensor = torch.FloatTensor(self.X_test)
        
        if self.is_classification:
            y_train_tensor = torch.LongTensor(self.y_train)
            y_test_tensor = torch.LongTensor(self.y_test)
        else:
            y_train_tensor = torch.FloatTensor(self.y_train).unsqueeze(1)
            y_test_tensor = torch.FloatTensor(self.y_test).unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
        
    @property
    def input_dim(self) -> int:
        return len(self.feature_columns)
    
    @property
    def output_dim(self) -> int:
        if self.is_classification:
            return len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 0
        else:
            return 1
