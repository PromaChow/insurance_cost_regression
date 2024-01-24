from data_extraction import extract_data
from data_preprocessing import pre_processing_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algo import neural_net, randomforest_regression
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def main():
    df1, df2 = extract_data()
    df = pre_processing_data(df1, df2)

    X = df.drop(columns="charges")
    X = X.values
    y = df['charges'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def evaluate_model(model, X_test, y_test):
        
        
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'Root Mean Squared Error on Test Set: {rmse:.4f}')
        
        y_test, y_pred = y_test.ravel(), y_pred.ravel()
        residuals = y_pred - y_test
        std_dev = np.std(residuals)

        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)

        plt.fill_between([min(y_test), max(y_test)],
                        [min(y_test) - std_dev, max(y_test) - std_dev],
                        [min(y_test) + std_dev, max(y_test) + std_dev],
                        color='gray', alpha=0.2)
        
        
        
        
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')

        plt.show()
            
        outside_band = np.abs(residuals) > std_dev
        percent_pts = np.sum(outside_band)/np.shape(y_test)[0]*100

        print("Percentage of points outside the band:", percent_pts)

        return
    

    randf_model = randomforest_regression(X_train, y_train)
    evaluate_model(randf_model, X_test, y_test)

    nn_model = neural_net(X_train_scaled, y_train)
    evaluate_model(nn_model, X_test_scaled, y_test)
    
if __name__ == "__main__":
    main()