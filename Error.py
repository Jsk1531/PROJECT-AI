from config import save_plot
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
def evaluate_model(y_test, y_pred, X, model, comparison):                                           
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  : {mae:,.3f}")
    print(f"MSE  : {mse:,.3f}")
    print(f"RMSE : {rmse:,.3f}")
    print(f"R²   : {r2:.3f}")
    error_df = pd.DataFrame({
        "Actual Price": y_test,
        "Predicted Price": y_pred,
        "Prediction Error": y_test - y_pred,
        "Absolute Error": np.abs(y_test - y_pred)
    })
    error_df["Percentage Error"] = (
        np.abs(y_test - y_pred) / np.abs(y_test)
    ) * 100

    print("\nACTUAL VS PREDICTED PRICES")
    print(error_df.head(10))

    print("\nPERCENTAGE ERROR")
    print(error_df[["Actual Price", "Predicted Price", "Percentage Error"]].head(10))

    print("\nBEST PREDICTIONS")
    print(error_df.nsmallest(5, "Absolute Error"))

    print("\nWORST PREDICTIONS")
    print(error_df.nlargest(5, "Absolute Error"))
    print(error_df.head(10))
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Car Prices")
    plt.grid(alpha=0.3)
    save_plot("actual_vs_predicted.png")
    plt.show()
    plt.figure(figsize=(8, 5))
    plt.hist(error_df["Prediction Error"], bins=30, edgecolor="black")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.grid(alpha=0.3)
    save_plot("error_distribution.png")
    plt.show()
    importance = pd.DataFrame({
     'Features': X.columns,
        'Importance': model.coef_
    })

    importance['Absolute Importance'] = abs(importance['Importance'])

    importance = importance.sort_values(
        by='Absolute Importance',
        ascending=False
    )

    print(importance)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance['Features'],
        importance['Absolute Importance']
    )

    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Car Features")
    plt.title("Feature Importance Ranking")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_plot("feature_importance.png")
    plt.show()
    return mae, mse, rmse, r2,error_df,importance