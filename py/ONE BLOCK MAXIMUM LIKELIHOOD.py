#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression as LinearRegressionSKL

# linear regression model
class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape

        X = np.hstack((np.ones((n_samples, 1)), X))

        self.weights = np.zeros(n_features + 1)

        # Gradient descent
        for i in range(self.max_iter):
            y_pred = X.dot(self.weights)
            error = y_pred - y
            gradient = X.T.dot(error) / n_samples
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        n_samples = X.shape[0]

        # Add bias term to X
        X = np.hstack((np.ones((n_samples, 1)), X))

        y_pred = X.dot(self.weights)
        return y_pred
    
    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "max_iter": self.max_iter}

# Load data
data = pd.read_csv('C:\\Users\\vigne\\Downloads\\Salary_dataset.csv')
data = data.drop(data.columns[0], axis=1)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(data['YearsExperience'], data['Salary'])
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Salary')
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Fit linear regression model on training data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Calculate residuals for training and testing sets
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

# Calculate residuals for training and testing sets
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

# Create Q-Q plots for residuals
sm.qqplot(residuals_train, line='s')
plt.title('Q-Q Plot for Training Set Residuals')
plt.show()


sm.qqplot(residuals_test, line='s')
plt.title('Q-Q Plot for Testing Set Residuals')
plt.show()


# Predict on training and testing data
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

# Compute performance metrics on training and testing data
mse_train = np.mean((y_pred_train - y_train)**2)
rmse_train = np.sqrt(mse_train)
ssr_train = np.sum((y_pred_train - np.mean(y_train))**2)
sst_train = np.sum((y_train - np.mean(y_train))**2)
r2_train = ssr_train / sst_train

mse_test = np.mean((y_pred_test - y_test)**2)
rmse_test = np.sqrt(mse_test)
ssr_test = np.sum((y_pred_test - np.mean(y_test))**2)
sst_test = np.sum((y_test - np.mean(y_test))**2)
r2_test = ssr_test / sst_test


# Evaluate model using cross-validation
lr_cv = LinearRegression()
scores = cross_val_score(lr_cv, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-scores)

print("Training set:")
print("MSE: ", mse_train)
print("RMSE: ", rmse_train)
print("R^2: ", r2_train)
print("Testing set:")
print("MSE: ", mse_test)
print("RMSE: ", rmse_test)
print("R^2: ", r2_test)

print("Cross-validation:")
print("RMSE scores:", cv_rmse_scores)
print("Mean RMSE:", np.mean(cv_rmse_scores))
print("Std RMSE:", np.std(cv_rmse_scores))


#Check for independence of residuals
plt.scatter(X_train, residuals_train, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Independence of Residuals Check on Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Residuals')
plt.show()


#Check for homoscedasticity
plt.scatter(y_pred_train, residuals_train, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Homoscedasticity Check on Training Set')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

