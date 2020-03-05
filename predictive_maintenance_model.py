import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming X_train, X_test, y_train, y_test are already defined

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to -1 (negative class) and 1 (positive class)
y_train_binary = 2 * y_train - 1
y_test_binary = 2 * y_test - 1

# Build the SVM model with a surrogate hinge loss
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation=None, kernel_regularizer='l2')  # Regularization for the hinge loss
])

# Define the hinge loss function
def hinge_loss(y_true, y_pred):
    return tf.reduce_mean(tf.maximum(1 - y_true * y_pred, 0))

# Compile the model
model.compile(optimizer='adam', loss=hinge_loss, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_binary, epochs=10, batch_size=32)

# Evaluate the model
y_pred_binary = np.sign(model.predict(X_test).flatten())
y_pred = (y_pred_binary + 1) / 2  # Convert back to 0 and 1

# Print results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

