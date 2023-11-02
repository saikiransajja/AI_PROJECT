#importing libraries
from re import X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn_pandas
import sklearn
import xgboost


#extract data
data = pd.read_csv("data.csv")
print(data.head())

#check for null values
print(data.isnull().sum())

#cleaning the data by updating the null values
data = data.dropna()

#Top 10 best selling game categories
game = data.groupby("Genre")["Global_Sales"].count().head(10)
print(game)

#Data Visualization
#Visualization based on genres
import matplotlib as mpl
custom_colors = mpl.colors.Normalize(vmin=min(game), vmax=max(game))
colours = [mpl.cm.PuBu(custom_colors(i)) for i in game]
plt.figure(figsize=(7,7))
plt.pie(game, labels=game.index, colors=colours)
central_circle = plt.Circle((0, 0), 0.5, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Top 10 Categories of Games Sold", fontsize=20)
plt.show()

#Visualization based on correlation of the data
print(data)
non_numeric_columns = data.select_dtypes(exclude=['float64', 'int64']).columns
data_numeric = data.drop(columns=non_numeric_columns)
print(data_numeric)
corr_matrix = data_numeric.corr()
sns.heatmap(corr_matrix, cmap="winter_r")
plt.show()


#Train the Data
x = data[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]  #training variables
y = data["Global_Sales"]    #target column
print(x)
print(y)
type(x)

#split the data
from sklearn.model_selection import train_test_split
type(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
type(xtrain)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

#Use the linear regression algorithm to train this model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
print(predictions)

#Alternative .....
