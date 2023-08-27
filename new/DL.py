import tensorflow.kearas
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('credit_scoring_model.h5')

import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('credit_scoring_model.h5')

# Ask the user for input attributes
revolving_utilization = float(input("Enter the revolving utilization of unsecured lines (0 to 1): "))
age = int(input("Enter the age: "))
num_30_59_days_past_due = int(input("Enter the number of times 30-59 days past due: "))
debt_ratio = float(input("Enter the debt ratio: "))
monthly_income = float(input("Enter the monthly income: "))
num_open_credit_lines = int(input("Enter the number of open credit lines and loans: "))
num_90_days_late = int(input("Enter the number of times 90 days late: "))
num_real_estate_loans = int(input("Enter the number of real estate loans or lines: "))
num_60_89_days_past_due = int(input("Enter the number of times 60-89 days past due: "))
num_dependents = int(input("Enter the number of dependents: "))

# Create user input array
user_input = np.array([
    revolving_utilization, age, num_30_59_days_past_due, debt_ratio,
    monthly_income, num_open_credit_lines, num_90_days_late,
    num_real_estate_loans, num_60_89_days_past_due, num_dependents
]).reshape(1, -1)  # Reshape to match input shape

# Preprocess user input
user_input_scaled = scaler.transform(user_input)

# Make predictions
prediction_prob = loaded_model.predict(user_input_scaled)
prediction_class = (prediction_prob > 0.5).astype(int)[0]

print("Probability of Default:", prediction_prob[0][0])
print("Predicted Class:", prediction_class)