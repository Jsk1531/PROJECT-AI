from Data_Preprocessing import load_data, clean_data, explore
from training import add_features,train_model
from Predict import predict_model
def main():
    df = load_data()
    df = clean_data(df)
    df, X, y = add_features(df)
    explore(df)
    df, model,y_test,y_pred = train_model(df, X, y)  
    df, model, y_test, y_pred, comparison=predict_model(df, X, y, model) 
if __name__ == "__main__":
    main()
    