import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv(r'C:\Users\jatin\Desktop\dlProject\new\cs-training.csv')

# Handle missing values
data = data.fillna(0)  # Replace missing values with 0 for simplicity

# Split the dataset into features and labels
X_train = data.drop(columns=['SeriousDlqin2yrs'])
y_train = data['SeriousDlqin2yrs']

datat = pd.read_csv(r'C:\Users\jatin\Desktop\dlProject\new\cs-test.csv')

# Handle missing values
datat = datat.fillna(0)  # Replace missing values with 0 for simplicity

# Split the dataset into features and labels
X_test = data.drop(columns=['SeriousDlqin2yrs'])
y_test = data['SeriousDlqin2yrs']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now X_train_scaled and X_test_scaled are the preprocessed feature sets
# y_train and y_test are the corresponding labels




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the neural network architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification output

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()


# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)