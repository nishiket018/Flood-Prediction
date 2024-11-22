import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, SimpleRNN, Concatenate
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # For saving and loading scaler

# -------- Data Preprocessing --------

# Load your dataset
df = pd.read_csv(r"C:\Users\ASUS\Downloads\train.csv")

# Drop 'id' column and other non-numeric columns
df = df.drop(columns=['id'])
df = df.select_dtypes(include=[np.number])

# Automatically identify and binarize the target variable
target_col = df.columns[-1]
binarizer = Binarizer(threshold=0.5)
df[target_col] = binarizer.fit_transform(df[[target_col]])

# Split the data into features (X) and target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure the input is reshaped correctly for LSTM/RNN
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# -------- Model Architecture --------

# Input Layer
input_layer = Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))

# Path 1: LSTM Layers with reduced neurons and higher dropout
x1 = LSTM(32, return_sequences=True)(input_layer)
x1 = Dropout(0.6)(x1)
x1 = LSTM(16)(x1)
x1 = Dropout(0.6)(x1)

# Path 2: RNN Layers with reduced neurons and higher dropout
x2 = SimpleRNN(32, return_sequences=True)(input_layer)
x2 = Dropout(0.6)(x2)
x2 = SimpleRNN(16)(x2)
x2 = Dropout(0.6)(x2)

# Concatenate Paths
combined = Concatenate()([x1, x2])

# Output Layer
output_layer = Dense(1, activation='sigmoid')(combined)

# Create the Model
hybrid_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the Model using SGD optimizer and reduced learning rate
hybrid_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = hybrid_model.fit(X_train_scaled, y_train, epochs=1, batch_size=64, validation_data=(X_test_scaled, y_test))

# -------- Evaluate Model on Training Set --------
y_train_pred = (hybrid_model.predict(X_train_scaled) > 0.5).astype("int32")

train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_sensitivity = recall_score(y_train, y_train_pred)
train_specificity = recall_score(y_train, y_train_pred, pos_label=0)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Training F1 Score: {train_f1:.2f}")
print(f"Training Sensitivity: {train_sensitivity:.2f}")
print(f"Training Specificity: {train_specificity:.2f}")

# -------- Evaluate Model on Testing Set --------
y_test_pred = (hybrid_model.predict(X_test_scaled) > 0.5).astype("int32")

test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_sensitivity = recall_score(y_test, y_test_pred)
test_specificity = recall_score(y_test, y_test_pred, pos_label=0)
print(f"Testing Accuracy: {test_accuracy:.2f}")
print(f"Testing F1 Score: {test_f1:.2f}")
print(f"Testing Sensitivity: {test_sensitivity:.2f}")
print(f"Testing Specificity: {test_specificity:.2f}")

# Confusion Matrix for Test Set
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Annotate Confusion Matrix
group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
group_counts = ["{0:0.0f}".format(value) for value in conf_matrix_test.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in conf_matrix_test.flatten()/np.sum(conf_matrix_test)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

# Plot the Confusion Matrix with Annotations
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_test, annot=labels, fmt='', cmap='Blues', cbar=False, square=True, linewidths=2)
plt.title('Confusion Matrix with TP, FP, TN, FN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# -------- Saving the Scaler and Model --------
# Save the scaler using pickle
scaler_filename = r'C:\Users\ASUS\Downloads\scaler.pkl'
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)

# Save the trained model in .h5 format
model_filename = r'C:\Users\ASUS\Downloads\hybrid_model.h5'
hybrid_model.save(model_filename)
