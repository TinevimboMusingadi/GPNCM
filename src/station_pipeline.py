import os
import glob
import pandas as pd
import yaml
import sys

# Ensure we run from project root context
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

def process_all_stations():
    # Load config
    config_path = os.path.join(PROJECT_DIR, 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    feature_cols = config['data']['feature_cols']
    target_col = config['data']['target_col']
    train_frac = config['data']['train_frac']
    val_frac = config['data']['val_frac']

    input_dir = os.path.join(PROJECT_DIR, 'data_set', 'Stations')
    output_base_dir = os.path.join(PROJECT_DIR, 'data', 'clean_stations')
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    print(f"Found {len(csv_files)} stations to process...")
    
    for file_path in csv_files:
        station_name = os.path.basename(file_path).replace('.csv', '')
        station_out_dir = os.path.join(output_base_dir, station_name)
        os.makedirs(station_out_dir, exist_ok=True)
        
        print(f"Processing station: {station_name}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        if 'date' not in df.columns:
            print(f"Skipping {station_name} - no 'date' column.")
            continue
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Ensure all feature columns exist
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            print(f"Warning: Station {station_name} missing columns {missing_cols}")
            for c in missing_cols:
                df[c] = 0.0 # fill with 0 if tracking is missing, though dropping is safer. We'll fill for now.
        
        # Cleaning: Fill small gaps
        df[feature_cols] = df[feature_cols].ffill(limit=3)
        df[feature_cols] = df[feature_cols].bfill(limit=3)
        
        # Drop rows with NaN target
        df = df.dropna(subset=[target_col])
        
        # Drop rows where ANY of the features are still NaN
        df = df.dropna(subset=feature_cols)
        
        # Outlier clip
        df[target_col] = df[target_col].clip(lower=0, upper=100)
        
        if len(df) < 1000:
            print(f"Station {station_name} has too few valid rows ({len(df)}). Skipping.")
            continue
            
        # Select columns to save
        save_df = df[['date'] + feature_cols]
        
        # Perform splits
        n = len(save_df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        
        df_train = save_df.iloc[:train_end]
        df_val = save_df.iloc[train_end:val_end]
        df_test = save_df.iloc[val_end:]
        
        # Save splits as parquet for fast loading
        df_train.to_parquet(os.path.join(station_out_dir, 'train.parquet'), index=False)
        df_val.to_parquet(os.path.join(station_out_dir, 'val.parquet'), index=False)
        df_test.to_parquet(os.path.join(station_out_dir, 'test.parquet'), index=False)
        
        # Optionally save the full clean
        save_df.to_parquet(os.path.join(station_out_dir, 'full_clean.parquet'), index=False)
        
        print(f"  -> Saved {n} rows (Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)})")

if __name__ == "__main__":
    process_all_stations()
    print("All stations processed successfully.")
