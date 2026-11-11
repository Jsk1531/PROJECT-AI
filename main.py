import os
import sys
import pandas as pd

from config import OUTPUT_DIR, save_table
from Data_Preprocessing import load_data, clean_data, explore
from training import add_features, train_model
from predict import predict_model
from error import evaluate_model


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


log_file = open(os.path.join(OUTPUT_DIR, "run_log.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)


def main():
    df = load_data()
    df = clean_data(df)
    df, X, y = add_features(df)
    explore(df)
    df, model, y_test, y_pred = train_model(df, X, y)
    df, model, y_test, y_pred, comparison = predict_model(df, X, y, model)
    mae, mse, rmse, r2, error_df, importance = evaluate_model(y_test, y_pred, X, model, comparison)

    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R2"],
        "Value": [mae, mse, rmse, r2],
    })
    save_table(metrics_df, "model_metrics.csv")
    save_table(comparison, "predictions_comparison.csv")
    save_table(error_df, "actual_vs_predicted.csv")
    save_table(importance, "feature_importance.csv")

    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
    log_file.close()