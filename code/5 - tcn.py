import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error

import tcn

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

print("X.shape", X.shape, "y.shape", y.shape)
print(X)
print(y)

# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(64, input_shape=(T, 1)),
#     tf.keras.layers.Dense(1)
# ])

model = tf.keras.Sequential([
    tcn.TCN(input_shape=(T, 1),
        kernel_size=2,
        use_skip_connections=False,
        use_batch_norm=False,
        use_weight_norm=False,
        use_layer_norm=False
        ),
    tf.keras.layers.Dense(1, activation='linear')
])

# # Compilation du modèle
model.summary()
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))


# # train du modèle sur 20 epochs
r = model.fit(X[:-N//2], y[:-N//2], batch_size=32, epochs=30, validation_data=(X[-N//2:], y[-N//2:]), verbose=1)

# One-step forecast using true targets
p = model.predict(X)

print("MSE", mean_squared_error(y, p))

plt.plot(y, label="y")
plt.plot(p, label="p")
plt.title("TCN Predictions")
plt.legend(['actual', 'predicted'])
plt.show()
