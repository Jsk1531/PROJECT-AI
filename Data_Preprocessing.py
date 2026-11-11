from pathlib import Path
from config import save_plot
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

def load_data():
    df = pd.read_csv(BASE_DIR / "Used_cars.csv")
    df.columns = df.columns.str.strip()
    return df
def clean_data(df):
        df['price'] = (
            df['price']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .astype(float)
        )

        df['milage'] = (
            df['milage']
            .astype(str)
            .str.lower()
            .str.replace('mi.', '', regex=False)
            .str.replace('miles', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
            .astype(float)
        )
        return df   

def explore(df):
    print("Dataset shape:", df.shape)
    print(df.head())
    

    print("Dataset dimensions:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nData types and missing-value overview:")
    df.info()

    print("\nSummary of numeric columns:")
    print(df.describe())
               
            
    print(df)
    import matplotlib.pyplot as plt
    

    import numpy as np
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    print("Numeric columns:", numeric_columns)
    column_to_plot = "price"

    plt.figure(figsize=(9,5))
    plt.hist(df[column_to_plot].dropna(), bins=20,range=(0,150000), edgecolor="black")
    plt.title(f"Distribution of {column_to_plot}")
    plt.xlabel(column_to_plot)
    plt.grid(alpha=0.25)
    save_plot("price_distribution.png")
    plt.show()
    feature_for_plot = "milage"
    target_for_plot = "price"  
    df_normal=df[df[target_for_plot]<=150000]
    plt.figure(figsize=(8, 5))
    plt.scatter(df_normal[feature_for_plot], df_normal[target_for_plot], alpha=0.65)
    plt.title(f"{target_for_plot} vs {feature_for_plot}")
    plt.xlabel(feature_for_plot)
    plt.ylabel(target_for_plot)
    plt.grid(alpha=0.25)
    save_plot("price_vs_milage.png")
    plt.show()

    print("Available columns:", df.columns.tolist())

    target_column = "price"

    feature_columns = ["model_year","milage"]


    missing_columns = [
        col for col in [target_column] + feature_columns
        if col not in df.columns
    ]
    if missing_columns:
        print("Error: These columns do not exist:", missing_columns)
    else:
        print("Target:", target_column)
        print("Features:", feature_columns)
        print(df[[target_column]+ feature_columns].describe())
if __name__ == "__main__":
    explore(clean_data(load_data()))
