import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def neural_net(X_train, y_train, X_test, y_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = models.Sequential([
            layers.Dense(30, activation='softplus', input_dim=X_train.shape[1]),
            layers.Dropout(0.2),  
            layers.Dense(50, activation='softplus'), 
            layers.Dropout(0.2),
            layers.Dense(30, activation='softplus'), 
            layers.Dropout(0.2),
            # layers.Dense(60, activation='softplus'), 
            # layers.Dropout(0.2),
            # layers.Dense(30, activation='softplus'), 
            # layers.Dropout(0.2),
            layers.Dense(1, activation='linear')  
            ])
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=20, verbose=2, validation_split=0.2)

    rsme = np.sqrt(model.evaluate(X_test, y_test, verbose=0))

    print(f'RMSE on Test Set: {rsme:.4f}')

    
    return 

def randomforest_regression(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=100, random_state=42, criterion='squared_error')
    y_train = y_train.ravel()
    model.fit(X_train, y_train)
    
    
    return model




