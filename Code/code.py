# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
dataset = pd.read_csv('data.csv')

# Data Cleaning
# Dropping certain less important attributes and check for the missing values(null values)
dataset.drop(columns = ['Year_of_Release', 'Developer', 'Publisher', 'Platform'], inplace = True)
print(dataset.isna().sum())
dataset = dataset.dropna()

game = dataset.groupby("Genre")["Global_Sales"].count()
print(game)
plt.style.use('dark_background')

# Visualization based on correlation of the data
print(dataset)
non_numeric_columns = dataset.select_dtypes(exclude=['float64', 'int64']).columns
data_numeric = dataset.drop(columns=non_numeric_columns)
corr_matrix = data_numeric.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="rainbow", linewidths=0.5) # type: ignore
plt.title("Correlation Heatmap", fontsize=14)
plt.show()

# Training the model
print(dataset)
x = dataset.iloc[:, :].values
x = np.delete(x, 6, 1)
y = dataset.iloc[:, 6:7].values
# Splitting the dataset into Train and Test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# Saving name of the games in training and test set
games_in_training_set = x_train[:, 0]
games_in_test_set = x_test[:, 0]

# Dropping the column that contains the name of the games
x_train = x_train[:, 1:]
x_test = x_test[:, 1:]

# Imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean')
x_train[:, [5 ,6, 7, 8]] = imputer.fit_transform(x_train[:, [5, 6, 7, 8]])
x_test[:, [5 ,6, 7, 8]] = imputer.transform(x_test[:, [5, 6, 7, 8]])

from sklearn.impute import SimpleImputer
categorical_imputer = SimpleImputer(strategy = 'constant', fill_value = 'NA')
x_train[:, [0, 9]] = categorical_imputer.fit_transform(x_train[:, [0, 9]])
x_test[:, [0, 9]] = categorical_imputer.transform(x_test[:, [0, 9]])

#One-Hot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 9])], remainder = 'passthrough')
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)

#Regression
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators = 200, learning_rate= 0.08)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
# Visualising actual and predicted sales
games_in_test_set = games_in_test_set.reshape(-1, 1)
y_pred = y_pred.reshape(-1, 1)
predictions = np.concatenate([games_in_test_set, y_pred, y_test], axis = 1)
predictions = pd.DataFrame(predictions, columns = ['Name', 'Predicted_Global_Sales', 'Actual_Global_Sales'])
print(predictions[["Predicted_Global_Sales", "Actual_Global_Sales"]])
predictions.plot()

# Visualizing the predicted model
# Residual Plot
# Calculate residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y_pred, residuals, alpha=0.5, color='cyan')
plt.xlabel("Predicted Sales", fontsize=12, color='white')
plt.ylabel("Residuals", fontsize=12, color='white')
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Plot", fontsize=14, color='white')

plt.legend(handles=[scatter], labels=["Data Points"], loc='upper right', fontsize=12)
plt.show()

# Learning Curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error') # type: ignore
train_rmse = np.sqrt(-train_scores)
test_rmse = np.sqrt(-test_scores)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_rmse.mean(axis=1), label='Training RMSE')
plt.plot(train_sizes, test_rmse.mean(axis=1), label='Validation RMSE')
plt.xlabel('Training Examples')
plt.ylabel('RMSE')
plt.legend()
plt.title('Learning Curves')
plt.show()

# Histogram
# Create separate subplots for actual and predicted sales histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.hist(y_test, bins=30, alpha=0.6, color='blue', label='Actual Sales')
ax1.set_xlabel('Sales', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.legend(fontsize=12)
ax1.set_title('Distribution of Actual Sales', fontsize=14)

ax2.hist(y_pred, bins=30, alpha=0.6, color='green', label='Predicted Sales')
ax2.set_xlabel('Sales', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend(fontsize=12)
ax2.set_title('Distribution of Predicted Sales', fontsize=14)
plt.tight_layout()
plt.show()

#Evaluation of the model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
r2_score = r2_score(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
MAE = mean_absolute_error(y_test, y_pred)
print(f"R Squared Score(R2_Score) of the model : {r2_score}")
print(f"Root Mean Squared Error(RMSE) of the model : {rmse}")
print(f"The Mean Absolute Error (MAE) of the model : {MAE}")