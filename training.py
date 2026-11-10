import pandas as pd                       
import numpy as np                        
import matplotlib.pyplot as plt           
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
print("Imported all the libraries rtequired")



def add_features(df):
    df["horsepower"] = (
        df["engine"].str.extract(r"(\d+(?:\.\d+)?)\s*HP", expand=False).astype(float)
    )

    
    df["displacement_l"] = (
        df["engine"]
        .str.extract(r"(\d+(?:\.\d+)?)\s*(?:L\b|Liter)", expand=False)
        .astype(float)
    )

    
    cyl_match = df["engine"].str.extract(
        r"(?:[VIW](\d{1,2})|(\d{1,2})\s*[- ]?[Cc]ylinder)", expand=True
    )
    df["cylinders"] = cyl_match[0].combine_first(cyl_match[1]).astype(float)

    
    df["is_turbo"] = (
        df["engine"].str.contains(r"turbo", case=False, na=False).astype(int)
    )
    df["is_electric"] = (
        df["engine"].str.contains(r"electric", case=False, na=False).astype(int)
    )
    
    df["horsepower"] = df.groupby(["brand", "model"])["horsepower"].transform(
        lambda x: x.fillna(x.median())
    )
    df["displacement_l"] = df.groupby(["brand", "model"])[
        "displacement_l"
    ].transform(lambda x: x.fillna(x.median()))
    df["cylinders"] = df.groupby(["brand", "model"])["cylinders"].transform(
        lambda x: x.fillna(x.median())
    )

    
    df["displacement_l"] = np.where(
        df["is_electric"] == 1, 0.0, df["displacement_l"].fillna(df["displacement_l"].median())
    )
    df["cylinders"] = np.where(
        df["is_electric"] == 1, 0.0, df["cylinders"].fillna(df["cylinders"].median())
    )
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    brand_price_map = df.groupby("brand")["price"].median()
    df["brand_value"] = df["brand"].map(brand_price_map)

    clear_df = df[(df["price"] >= 1000) & (df["price"] <= 250000)].copy()


    features = [
        "brand_value",
        "model_year",
        "milage",
        "horsepower",
        "displacement_l",
        "cylinders",
        "is_turbo",
        "is_electric",
    ]

    x = clear_df[features]
    y = clear_df["price"]
    return df,x,y

def train_model(df,X,y):              
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("defining the features and Splitting data done ")

    model=LinearRegression()
    print("model training started")
    model.fit(X_train, y_train)
    print("Training is done ")
    print("Model is Ready to Predict")
    y_pred = model.predict(X_test)
    return df,model,y_test,y_pred

