import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


melbourne_file_path = 'melb_data.csv'
new_data_file = 'testdata.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
new_data = pd.read_csv(new_data_file)
print(melbourne_data.columns)
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price
print(y.head())
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.head())

print(X.describe())



# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model
melbourne_model.fit(X_train, y_train)

print("Making predictions for the following 5 houses:")
print(X_test.head())
print("The predictions are")
predictions = melbourne_model.predict(X_test)
print(predictions)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)
r2 = r2_score(y_test, predictions)
print("R² Score:", r2)
print("Accuracy (%):", r2 * 100)

feature_importances = pd.Series(
    melbourne_model.feature_importances_,
    index=melbourne_features
).sort_values(ascending=False)

feature_importances.plot(kind='bar')
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()

