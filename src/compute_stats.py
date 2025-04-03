import os
import pandas as pd
import numpy as np
from data_loader import DataLoader
import json
from enums import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

def compute_numeric_stats(df):
    stats = {}
    available_numeric = [col for col in NUMERIC_COLUMNS if col in df.columns]
    for col in available_numeric:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'min': df[col].min(),
            'max': df[col].max(),
            'std': df[col].std(),
            '5th_percentile': df[col].quantile(0.05),
            '95th_percentile': df[col].quantile(0.95),
            'missing_values': df[col].isna().sum()
        }
    return stats

def compute_categorical_stats(df):
    stats = {}
    available_categorical = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    for col in available_categorical:
        value_counts = df[col].value_counts(normalize=True)
        stats[col] = {
            'unique_classes': df[col].nunique(),
            'missing_values': df[col].isna().sum(),
            'class_proportions': value_counts.to_dict()
        }
    return stats

def flatten_categorical_stats(cat_stats):
    flat_stats = {}
    for col, stats in cat_stats.items():
        flat_stats[col] = stats.copy()
        flat_stats[col]["class_proportions"] = json.dumps(stats["class_proportions"], ensure_ascii=False)
    return flat_stats

if __name__ == '__main__':
    loader = DataLoader("data.csv")
    df = loader.load_data()

    os.makedirs("stats", exist_ok=True)
    
    if df is not None:
        numeric_stats = compute_numeric_stats(df)
        categorical_stats = compute_categorical_stats(df)
        
        numeric_stats_df = pd.DataFrame(numeric_stats).transpose()
        numeric_stats_df.index.name = "name"
        numeric_stats_df.to_csv(os.path.join("stats", "numeric_stats.csv"), index=True)
        print("Statystyki numeryczne zapisane są w pliku stats/numeric_stats.csv")
        
        with open(os.path.join("stats", "categorical_stats.json"), "w") as f:
            json.dump(categorical_stats, f, indent=4, default=lambda x: x.item() if hasattr(x, 'item') else x)
        print("Statystyki kategoryczne zapisano w pliku stats/categorical_stats.json")