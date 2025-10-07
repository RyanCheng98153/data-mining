import re, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    return pd.read_csv(file_path)

def data_cleaning(data_df, value_columns, invalid_pattern):
    """Clean invalid values and convert to float"""
    for col in value_columns:
        data_df.iloc[:, col] = data_df.iloc[:, col].astype(str)
        data_df.iloc[:, col] = data_df.iloc[:, col].mask(
            data_df.iloc[:, col].str.contains(invalid_pattern, case=False, na=False),
            np.nan
        )
        data_df.iloc[:, col] = pd.to_numeric(data_df.iloc[:, col], errors='coerce')
    
    return data_df

def data_interpolation(data_df, value_columns, method='forward_fill'):
    """
    More conservative interpolation strategies:
    - 'forward_fill': Use previous valid value
    - 'mean': Use feature's mean value
    - 'linear': Linear interpolation (original method)
    """
    for i, row in data_df.iterrows():
        not_nan = row.iloc[value_columns].notna().to_numpy().nonzero()[0]
        
        if len(not_nan) == len(value_columns) or len(not_nan) == 0:
            continue
        
        np_row = row.iloc[value_columns].to_numpy(dtype=float)
        
        if method == 'linear':
            x = np.arange(len(np_row))
            data_df.iloc[i, value_columns] = np.interp(x, x[not_nan], np_row[not_nan])
        
        elif method == 'forward_fill':
            # Fill NaN with previous valid value
            filled = np_row.copy()
            for j in range(len(filled)):
                if np.isnan(filled[j]) and j > 0:
                    filled[j] = filled[j-1]
            data_df.iloc[i, value_columns] = filled
        
        elif method == 'mean':
            # Fill NaN with mean of valid values
            mean_val = np.nanmean(np_row)
            np_row[np.isnan(np_row)] = mean_val
            data_df.iloc[i, value_columns] = np_row
    
    return data_df

def data_dropdown(data_df, value_columns, drop_type='date', skip_for_test=False):
    """
    Drop rows with insufficient PM2.5 data
    """
    if skip_for_test:
        return data_df
    
    dropdown_items = []
    for i, row in data_df.iterrows():
        not_nan = row.iloc[value_columns].notna().to_numpy().nonzero()[0]

        if row.ItemName.strip() == 'PM2.5' and len(not_nan) < 2:
            dropdown_items.append({'date': row.Date, 'feature': row.ItemName})

    for item in dropdown_items:
        if drop_type == 'date':
            data_df = data_df[data_df.Date != item['date']]
        
    return data_df


# ========== Enhanced Linear Regression ==========

def train_linear_regression(X, y, w=None, b=None, lr=0.00001, iterations=1000, verbose=True):
    """
    Gradient Descent training with NaN handling
    """
    n_samples, n_features = X.shape

    if w is None:
        w = np.zeros(n_features)
    if b is None:
        b = 0.0

    best_rmse = float('inf')
    patience = 5
    no_improve = 0

    for iteration in range(iterations):
        y_pred = np.zeros(n_samples)
        
        for i in range(n_samples):
            valid_mask = ~np.isnan(X[i])
            if valid_mask.any():
                y_pred[i] = np.dot(X[i][valid_mask], w[valid_mask]) + b
            else:
                y_pred[i] = b
        
        error = y_pred - y
        
        dw = np.zeros(n_features)
        for j in range(n_features):
            valid_samples = ~np.isnan(X[:, j])
            if valid_samples.any():
                dw[j] = np.mean(error[valid_samples] * X[valid_samples, j])
        
        db = np.mean(error)
        
        w -= lr * dw
        b -= lr * db

        if verbose and iteration % 10000 == 0:
            rmse = np.sqrt(np.mean(error ** 2))
            print(f"Iteration {iteration:6d} | RMSE = {rmse:.4f}")
            
            # Early stopping check
            if rmse < best_rmse:
                best_rmse = rmse
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"Early stopping at iteration {iteration}")
                break

    return w, b


def predict(X, w, b):
    """Prediction with NaN handling"""
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    
    for i in range(n_samples):
        valid_mask = ~np.isnan(X[i])
        if valid_mask.any():
            predictions[i] = np.dot(X[i][valid_mask], w[valid_mask]) + b
        else:
            predictions[i] = b
    
    return predictions


def save_model(model_path, feature_names, w, b):
    model_data = {
        "feature_names": feature_names,
        "weights": w.tolist(),
        "bias": float(b)
    }
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, ensure_ascii=False, indent=4)
    print(f"✅ Model saved to {model_path}")


def load_model(model_path):
    if not os.path.exists(model_path):
        return None, None, None
    with open(model_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    feature_names = model_data["feature_names"]
    w = np.array(model_data["weights"], dtype=float)
    b = float(model_data["bias"])
    print(f"✅ Model loaded from {model_path}")
    return feature_names, w, b


# ========== FIXED: Temporal Feature Engineering ==========

def create_temporal_features(values, feature_name):
    """
    Create temporal features from time series:
    - Current value (hour N)
    - Previous value (hour N-1)
    - Change from previous hour (delta)
    - 2-hour average
    - 3-hour average
    """
    features = {}
    
    # Current value (last hour in sequence)
    features[f'{feature_name}_current'] = values[-1] if len(values) > 0 else np.nan
    
    # Previous value
    features[f'{feature_name}_prev'] = values[-2] if len(values) > 1 else np.nan
    
    # Change (delta)
    if len(values) > 1 and not np.isnan(values[-1]) and not np.isnan(values[-2]):
        features[f'{feature_name}_delta'] = values[-1] - values[-2]
    else:
        features[f'{feature_name}_delta'] = np.nan
    
    # Moving averages (only if we have enough data)
    if len(values) >= 2:
        features[f'{feature_name}_avg2'] = np.nanmean(values[-2:])
    else:
        features[f'{feature_name}_avg2'] = np.nan
    
    if len(values) >= 3:
        features[f'{feature_name}_avg3'] = np.nanmean(values[-3:])
    else:
        features[f'{feature_name}_avg3'] = np.nan
    
    return features


def extract_features_and_target_sequential(train_df, feature_items, target_item, input_hours, target_hour_idx):
    """
    FIXED: Extract features from input_hours to predict target_hour
    
    Args:
        train_df: DataFrame with all training data
        feature_items: List of feature names (e.g., ['PM2.5', 'PM10', ...])
        target_item: Target variable name (e.g., 'PM2.5')
        input_hours: Column indices for input hours (e.g., [2,3,4,...,10] for hours 1-9)
        target_hour_idx: Column index for target hour (e.g., 11 for hour 10)
    
    Returns:
        X: Feature matrix (n_dates, n_features)
        y: Target vector (n_dates,)
    """
    all_dates = train_df['Date'].unique()
    X_list = []
    y_list = []
    valid_dates = []
    
    for date in all_dates:
        date_subset = train_df[train_df['Date'] == date]
        
        # Get target PM2.5 value for this date
        target_row = date_subset[date_subset['ItemName'].str.strip() == target_item]
        if target_row.empty or target_hour_idx >= len(target_row.columns):
            continue
        
        y_val = target_row.iloc[0, target_hour_idx]
        if pd.isna(y_val):
            continue  # Skip if target is missing
        
        # Build feature vector with temporal features
        feature_vector = []
        
        for item in feature_items:
            item_row = date_subset[date_subset['ItemName'].str.strip() == item]
            
            if item_row.empty:
                # Feature not available - use NaN for all temporal features
                for _ in range(5):  # 5 temporal features per item
                    feature_vector.append(np.nan)
            else:
                # Extract values for input hours
                values = item_row.iloc[0, input_hours].to_numpy(dtype=float)
                
                # Create temporal features
                temp_features = create_temporal_features(values, item)
                feature_vector.extend([
                    temp_features[f'{item}_current'],
                    temp_features[f'{item}_prev'],
                    temp_features[f'{item}_delta'],
                    temp_features[f'{item}_avg2'],
                    temp_features[f'{item}_avg3']
                ])
        
        X_list.append(feature_vector)
        y_list.append(y_val)
        valid_dates.append(date)
    
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    
    print(f"✅ Extracted {len(y)} valid training samples from {len(all_dates)} dates")
    print(f"   Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y, valid_dates


# ========== Testing Function ==========

def predict_next_hour_pm25(test_path, model_path="./model.json", cur_hour=9):
    """
    Predict hour 10 PM2.5 using hours 1-9 data with temporal features
    """
    next_hour = cur_hour + 1
    
    columns = ["Date", "ItemName"] + [str(i) for i in range(1, 11)]
    test_df = pd.read_csv(test_path, header=None, names=columns)
    
    value_columns = [i+2 for i in range(cur_hour)]
    test_df = data_cleaning(test_df, value_columns=value_columns, invalid_pattern='#|A|x|X|\*')
    test_df = data_interpolation(test_df, value_columns=value_columns, method='forward_fill')
    test_df = data_dropdown(test_df, value_columns=value_columns, drop_type='date', skip_for_test=True)
    print("✅ Test data loaded and cleaned.")

    # Load model
    feature_names, w, b = load_model(model_path)
    if w is None or feature_names is None:
        raise ValueError("❌ Model not found.")

    # Extract base feature names (remove temporal suffixes)
    base_features = []
    for fname in feature_names:
        base = fname.split('_current')[0].split('_prev')[0].split('_delta')[0].split('_avg')[0]
        if base not in base_features:
            base_features.append(base)
    
    print(f"📊 Using {len(base_features)} base features")

    results = []
    y_true_list = []
    y_pred_list = []

    all_dates = test_df["Date"].unique()

    for date in all_dates:
        subset = test_df[test_df["Date"] == date]

        # Build feature vector with temporal features
        feature_vector = []
        
        for item in base_features:
            row = subset[subset["ItemName"].str.strip() == item]
            
            if row.empty:
                for _ in range(5):
                    feature_vector.append(np.nan)
            else:
                values = row.iloc[0, value_columns].to_numpy(dtype=float)
                temp_features = create_temporal_features(values, item)
                feature_vector.extend([
                    temp_features[f'{item}_current'],
                    temp_features[f'{item}_prev'],
                    temp_features[f'{item}_delta'],
                    temp_features[f'{item}_avg2'],
                    temp_features[f'{item}_avg3']
                ])
        
        X_input = np.array([feature_vector], dtype=float)
        y_pred_next_hour = predict(X_input, w, b)[0]
        
        results.append({"index": date, "answer": y_pred_next_hour})
        y_pred_list.append(y_pred_next_hour)

        # Get true value if available
        pm25_row = subset[subset["ItemName"].str.strip() == "PM2.5"]
        if not pm25_row.empty and str(next_hour) in pm25_row.columns:
            true_val = pm25_row.iloc[0][str(next_hour)]
            if not pd.isna(true_val):
                y_true_list.append(float(true_val))

    result_df = pd.DataFrame(results)
    print(f"✅ Prediction complete. Predicted {len(result_df)} dates.")

    if len(y_true_list) == len(y_pred_list) and len(y_true_list) > 0:
        rmse = np.sqrt(np.mean((np.array(y_true_list) - np.array(y_pred_list)) ** 2))
        print(f"✅ Prediction RMSE = {rmse:.4f}")
    else:
        rmse = None
        print("⚠️ RMSE cannot be calculated (missing true PM2.5 values).")

    return result_df, rmse


def unique_filename(file_path):
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{base}_v{counter}{ext}"
        counter += 1
    return new_file_path


def main_training(important_features=None, name=None, iterations=100000, model_path=None, interpolation_method='forward_fill'):
    train_df = read_csv("./data/train.csv")
    
    # Use hours 1-9 as input, hour 10 as target
    input_hours = [i+3 for i in range(9)]  # columns 3-11 (hours 1-9)
    target_hour_idx = 11  # column 11 (hour 10)
    
    invalid_pattern = '#|A|x|X|\*'
    
    print("✅ Data loaded.")

    train_df = data_cleaning(train_df, input_hours, invalid_pattern)
    train_df = data_interpolation(train_df, input_hours, method=interpolation_method)
    train_df = data_dropdown(train_df, input_hours, drop_type='date')

    print(f"✅ Data cleaned. Remaining rows: {len(train_df)}")

    if important_features is None:
        important_features = [
            'PM2.5',
            'PM10',
            'NO2',
            'NOx',
            'SO2',
            'CO',
            'O3',
            'AMB_TEMP',
            'RH',
            'WIND_SPEED'
        ]
    
    feature_items = important_features
    target_item = "PM2.5"

    # Build feature names with temporal suffixes
    feature_names_with_temporal = []
    for item in feature_items:
        feature_names_with_temporal.extend([
            f'{item}_current',
            f'{item}_prev',
            f'{item}_delta',
            f'{item}_avg2',
            f'{item}_avg3'
        ])

    # Setup output directory
    if name is not None:
        output_dir = f"./history_model/{name}"
    else:
        output_dir = f"./history_model/temporal_{'_'.join(feature_items[:3])}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_path is None:
        model_path = "./model.json"

    # Extract features with temporal engineering
    X, y, valid_dates = extract_features_and_target_sequential(
        train_df, feature_items, target_item, input_hours, target_hour_idx
    )
    
    print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
    print(f"NaN count in X: {np.isnan(X).sum()} out of {X.size} values ({np.isnan(X).sum()/X.size*100:.2f}%)")

    # Load or initialize model
    loaded_features, w, b = load_model(model_path)
    
    if w is None or loaded_features != feature_names_with_temporal:
        print("🔁 Starting new training with temporal features.")
        w, b = None, None
    
    # Train
    w, b = train_linear_regression(X, y, w=w, b=b, lr=0.00001, iterations=iterations, verbose=True)

    # Save
    save_model(model_path, feature_names_with_temporal, w, b)
    save_model(unique_filename(f"{output_dir}/model.json"), feature_names_with_temporal, w, b)

    # Validate
    y_pred = predict(X, w, b)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"✅ Final Training RMSE = {rmse:.4f}")


import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to model file')
    parser.add_argument('--interpolation', type=str, default='forward_fill', 
                       choices=['forward_fill', 'mean', 'linear'],
                       help='Interpolation method')

    args = parser.parse_args()
    model_path = args.model if args.model is not None else "./model.json"
    
    # Select important features
    important_features = [
        'PM2.5',
        'PM10',
        'NO2',
        'NOx',
        'SO2',
        'CO',
        'O3',
        'AMB_TEMP',
        'RH',
        'WIND_SPEED',
        'RAINFALL'
    ]
    for _ in range(20):  # Repeat to emphasize importance
        # Train with temporal features
        main_training(
            important_features=important_features, 
            name="temporal_v1", 
            iterations=10000, 
            model_path=model_path,
            interpolation_method=args.interpolation
        )

        # Test
        test_path = "./data/test.csv"
        cur_hour = 9
        
        pred_df, rmse = predict_next_hour_pm25(
            test_path, 
            model_path=model_path, 
            cur_hour=cur_hour
        )
        
        rmse_str = f"{rmse:.4f}" if rmse is not None else "NA"
        
        print(pred_df.head())
        output_file = f"./prediction_temporal_rmse_{rmse_str}.csv"
        pred_df.to_csv(output_file, index=False)
        print(f"✅ Saved prediction to {output_file}")