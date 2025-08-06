# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# 2. Load dataset
df = pd.read_csv('d:/data analyst/MACHINE LEARNING/ML PROGRAMS USING PYTHON/USA_Housing.csv')

# 3. Features and target
full_features = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                 'Avg. Area Number of Bedrooms', 'Area Population']
X = df[full_features]
y = df['Price']

# 4. Normalize features
X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / X_std
X_norm.insert(0, 'Intercept', 1)  # Add intercept

# 5. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# 6. Convert to NumPy arrays
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)

# 7. Gradient Descent setup
alpha = 0.001
iterations = 10000
m = len(y_train_np)
theta = np.zeros((X_train_np.shape[1], 1))
cost_history = []
theta_history = []

# Cost function
def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    error = predictions - y
    return (1 / (2 * m)) * np.sum(error ** 2)

# 8. Gradient Descent Loop
for i in range(iterations):
    predictions = X_train_np.dot(theta)
    error = predictions - y_train_np
    gradient = (1 / m) * X_train_np.T.dot(error)
    theta -= alpha * gradient
    cost_history.append(compute_cost(X_train_np, y_train_np, theta))
    theta_history.append(theta.copy())

print("\n✅ Gradient Descent Training Complete.")
print("Final cost:", cost_history[-1])
print("Learned Parameters (theta):", theta.ravel())

# 9. Predictions & Evaluation
y_pred = X_test_np.dot(theta)

print("\n📊 Model Evaluation (Gradient Descent):")
print("Mean Absolute Error:", mean_absolute_error(y_test_np, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test_np, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test_np, y_pred)))
print("R^2 Score:", r2_score(y_test_np, y_pred))

# 10. Actual vs Predicted Prices
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test_np.ravel(), y=y_pred.ravel())
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Gradient Descent)')
plt.show()

# 11. Cost Convergence Plot
plt.figure(figsize=(8,6))
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()

# 12. Random Test Prediction
rand_index = random.randint(0, len(X_test_np)-1)
sample = X_test_np[rand_index].reshape(1, -1)
actual = y_test_np[rand_index][0]
predicted = sample.dot(theta)[0][0]
print("\n🎯 Prediction on Random Test Sample:")
print("Actual Price   :", actual)
print("Predicted Price:", predicted)

# 13. 3D Surface Plot (2 features only)
# Use intercept + 2 features: 'Avg. Area Income' and 'Avg. Area House Age'
X_contour = X_train_np[:, [0,1,2]]  # intercept, income, house age
theta0_vals = np.linspace(-1e6, 1e6, 100)
theta1_vals = np.linspace(-1e6, 1e6, 100)

theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
cost_vals = np.zeros_like(theta0_mesh)

# Fix third parameter (House Age weight) to 0 for visualization
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_mesh[i, j]], [theta1_mesh[i, j]], [0]])  # intercept, income, house age = 0
        cost_vals[i, j] = compute_cost(X_contour, y_train_np, t)

# Convert theta path for plotting
theta_path = np.array(theta_history)
theta0_path = theta_path[:, 0].flatten()
theta1_path = theta_path[:, 1].flatten()
theta2_path = theta_path[:, 2].flatten()

# 3D Plot
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_mesh, theta1_mesh, cost_vals, cmap='viridis', alpha=0.7, edgecolor='none')
ax.set_xlabel('Theta 0 (Intercept)')
ax.set_ylabel('Theta 1 (Avg. Area Income)')
ax.set_zlabel('Cost')
ax.set_title('3D Surface Plot of Cost Function')

# Gradient descent path
ax.plot(theta0_path, theta1_path, cost_history, color='r', marker='o', markersize=2, label='Gradient Descent Path')
ax.legend()

plt.show()
