import numpy as np
import matplotlib.pyplot as plt

# Given data points
x = [0, 1, 2, 3, 4]
y = [2, 3, 5, 4, 6]

# Step 1: Calculate the means of x and y
x_mean = sum(x) / len(x)
y_mean = sum(y) / len(y)

# Step 2: Calculate the coefficients a and b
# Calculate the numerator and denominator for a
numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
a = numerator / denominator
b = y_mean - a * x_mean

print(f"The linear regression line is y = {a:.2f}x + {b:.2f}")

# Step 3: Estimate the value of y when x = 10
x_new = 10
y_new = a * x_new + b
print(f"The estimated value of y when x = {x_new} is {y_new:.2f}")

# Step 4: Calculate the Mean Squared Error (MSE)
y_pred = [a * xi + b for xi in x]
mse = sum((y[i] - y_pred[i]) ** 2 for i in range(len(y))) / len(y)
print(f"The Mean Squared Error of the model is {mse:.2f}")

# Plot the data points and the regression line
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

