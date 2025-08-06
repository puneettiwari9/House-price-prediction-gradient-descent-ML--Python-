# 🏡 House Price Prediction using Gradient Descent

This project demonstrates a **Linear Regression model** trained using **Gradient Descent** to predict housing prices using the USA_Housing dataset.

---

## 🔍 Overview

The model uses normalized features from a housing dataset to predict property prices. It includes:

- Feature normalization
- Cost function and manual gradient descent implementation
- Evaluation metrics (MAE, MSE, RMSE, R²)
- Visualization of results, including 3D cost surface and convergence plots

---

## 📁 Dataset Used

- File: `USA_Housing.csv`
- Features used:
  - `Avg. Area Income`
  - `Avg. Area House Age`
  - `Avg. Area Number of Rooms`
  - `Avg. Area Number of Bedrooms`
  - `Area Population`
- Target: `Price`

---

## ⚙️ Model Training

Training is performed using gradient descent for **10,000 iterations** with a learning rate (`α`) of **0.001**.

### ✅ Final Output:

```text
✅ Gradient Descent Training Complete.
Final cost: 5128239149.836445
Learned Parameters (theta):
[1231939.27565796  230732.56333204  163233.22012786
 119944.11581949    3396.54363226  151563.13687833]

##📊 **Model Evaluation**
Mean Absolute Error     : 80888.60
Mean Squared Error      : 10093245137.69
Root Mean Squared Error : 100465.14
R² Score                : 0.91796

🎯 Sample Prediction
🎯 Prediction on Random Test Sample:
Actual Price   : 1599963.8071501006
Predicted Price: 1803815.4780034893

## 📈 Visualizations

The script generates:

📉 **Cost Function Convergence Plot**  
![Cost Convergence Plot]("cost_fn_convergence.png")

📊 **Actual vs Predicted Price Scatter Plot**  
![Actual vs Predicted]("linear_plotting.png")

🌀 **3D Cost Surface with Gradient Descent Path (for two selected features)**  
![3D Cost Surface]("3d_surface_plot_gradient_descent.png")






