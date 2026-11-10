from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd


def predict_model(df, X, y, model):
  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    comparison = pd.DataFrame({
        "Actual Price": y_test.to_numpy(),
        "Predicted Price": y_pred,
        "Error (Actual - Predicted)": y_test.to_numpy() - y_pred
    })
     
    print(comparison.head(10))
    new_car = pd.DataFrame({
    "brand_value": [20000],
    "model_year": [2018],
    "milage": [50000],
    "horsepower": [200],
    "displacement_l": [2.0],
    "cylinders": [4],
    "is_turbo": [0],
    "is_electric": [0]
    })

    predicted_price = model.predict(new_car)[0]

    print("Car details:")
    print(new_car)

    print(f"Predicted Used-Car Price: ${predicted_price:,.2f}")
    
    
    
    print("Enter the car details for prediction:")
    brand_value = float(input("Brand value: "))
    model_year = int(input("Model year: "))
    milage = float(input("Mileage: "))
    horsepower = float(input("Horsepower: "))
    displacement_l = float(input("Engine displacement in litres: "))
    cylinders = float(input("Number of cylinders: "))
    is_turbo = int(input("Turbo engine? Enter 1 for Yes, 0 for No: "))
    is_electric = int(input("Electric car? Enter 1 for Yes, 0 for No: "))

    new_car = pd.DataFrame({
        "brand_value": [brand_value],
        "model_year": [model_year],
        "milage": [milage],
        "horsepower": [horsepower],
        "displacement_l": [displacement_l],
        "cylinders": [cylinders],
        "is_turbo": [is_turbo],
        "is_electric": [is_electric]
    })


    predicted_price = model.predict(new_car)[0]

    print(f"\nPredicted Used-Car Price: ${predicted_price:,.2f}")
    return df, model, y_test, y_pred, comparison




