import numpy as np
import tensorflow as tf

x = np.arange(0, 20)
y = np.zeros_like(x)
for i in range(len(x)):
    if x[i] % 2 == 0:
        y[i] = 0  # Even number
    else:
        y[i] = 1  # Odd number
n_classes = len(np.unique(y))


model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10, input_shape=(1,)),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ]
)

print("Train without precision metric")
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
)
model.fit(x, y, epochs=2)


print("Train with precision metric")
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[tf.keras.metrics.Precision()],
)
model.fit(x, y, epochs=2)
