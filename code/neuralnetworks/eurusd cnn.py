import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


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

# check
X = np.array(X).reshape(-1, SEQ_LENGTH, 1)
y = np.array(y)
print("X.shape", X.shape, "y.shape", y.shape)

# train test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# model
def build_cnn_model(input_shape, forecast_horizon: int):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(forecast_horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# # train du modèle sur 20 epochs
model = build_cnn_model((SEQ_LENGTH, X.shape[2]), FORECAST_HORIZON)
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
plt.title("CNN Predictions")
plt.legend()
plt.show()

