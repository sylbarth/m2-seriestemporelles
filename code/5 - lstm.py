import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


np.random.seed(1)
tf.random.set_seed(1)


eurusd = pd.read_csv("../data/eurusd.csv",index_col=[0])
eurusd.dropna(inplace=True)
data = eurusd["Close"].to_list()

T = 20
X = []
y = []
for t in range(len(data) - T):
    X.append(data[t:t+T])
    y.append(data[t+T])

X = np.array(X).reshape(-1, T, 1)
y = np.array(y)
N = len(X)
N_TRAIN = N/2
print(N_TRAIN)
exit(1)

print("X.shape", X.shape, "y.shape", y.shape)
print(X)
print(y)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(T, 1)),
    tf.keras.layers.Dense(1)
])

# # Compilation du modèle
model.compile(loss='mse', optimizer=Adam(learning_rate=0.05))


# # train du modèle sur 20 epochs
r = model.fit(X[:-N//2], y[:-N//2], batch_size=32, epochs=20, validation_data=(X[-N//2:], y[-N//2:]), verbose=1)

plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

# One-step forecast using true targets
yhat = model.predict(X)
yhat = yhat[:,0]

print("MSE", mean_squared_error(y, yhat))

plt.plot(y, label="y")
plt.plot(yhat, label="yhat")
plt.title("LSTM Predictions")
plt.legend()
plt.show()
