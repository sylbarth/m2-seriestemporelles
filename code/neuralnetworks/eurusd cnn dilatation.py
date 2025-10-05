import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import tcn


SEQ_LENGTH = 5
FORECAST_HORIZON = 1


np.random.seed(1)
tf.random.set_seed(1)


eurusd = pd.read_csv("EURUSD.csv",index_col=[0])
eurusd.dropna(inplace=True)
data = eurusd["Close"].to_list()


def build_data_seq(data, seq_length: int, forecast_horizon: int = 1):
    X = []
    y = []
    for t in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[t:t+seq_length])
        y.append(data[t+seq_length:t+seq_length+forecast_horizon])
    return X, y

X, y = build_data_seq(data, seq_length=SEQ_LENGTH, forecast_horizon=FORECAST_HORIZON)
print(pd.DataFrame(data))
print(pd.DataFrame(X))
print(pd.DataFrame(y))


# check
X = np.array(X).reshape(-1, SEQ_LENGTH, 1)
y = np.array(y)
print("X.shape", X.shape, "y.shape", y.shape)
print(X)
print(y)

# train test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# model
def build_tcn_model(input_shape, forecast_horizon: int):
    model = tf.keras.Sequential([
        tcn.TCN(
            input_shape=input_shape,
            kernel_size=2,
            use_skip_connections=False,
            use_batch_norm=False,
            use_weight_norm=False,
            use_layer_norm=False
        ),
        Dense(forecast_horizon, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

# # train du modèle sur 20 epochs
model = build_tcn_model((SEQ_LENGTH, X.shape[2]), FORECAST_HORIZON)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

r = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)


plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

# One-step forecast using true targets
yhat = model.predict(X_test)
yhat = yhat[:,0]

print("MSE", mean_squared_error(y_test, yhat))

plt.plot(y_test, label="y")
plt.plot(yhat, label="yhat")
plt.title("TCN Predictions")
plt.legend()
plt.show()
