# Energy_consumption_prediction
# Building Energy Consumption Prediction

A small machine learning project where I used linear regression to predict the energy consumption of different buildings based on features such as square footage, number of occupants, appliances used, temperature, building type, and whether the observation was on a weekday or weekend.

I built this mainly to practice working through a regression problem from data exploration and preprocessing to training and evaluating the model.

## Dataset

The dataset contains 100 building records with the following columns:

* `Building Type`
* `Square Footage`
* `Number of Occupants`
* `Appliances Used`
* `Average Temperature`
* `Day of Week`
* `Energy Consumption`

`Energy Consumption` is the value I'm trying to predict.

```python
X = df.drop("Energy Consumption", axis=1)
y = df["Energy Consumption"]
```

## Data exploration

Before building the model, I checked the dataset structure, data types, missing values, and some of the distributions and relationships between the features.

```python
print(df.head())
print(df.info())
```

The dataset did not contain any missing values, but two columns contained categorical data:

* `Building Type`
* `Day of Week`

These needed to be converted into a numerical representation before using them with linear regression.

## Encoding categorical features

`Day of Week` contains two categories:

```text
Weekday
Weekend
```

I encoded them as:

```python
df["Day of Week"] = df["Day of Week"].map({
    "Weekday": 0,
    "Weekend": 1
})
```

For `Building Type`, I initially experimented with category codes but realized that assigning values such as `0`, `1`, and `2` creates an artificial numerical relationship between building types.

Instead, I used one-hot encoding:

```python
df = pd.get_dummies(
    df,
    columns=["Building Type"],
    drop_first=True,
    dtype=int
)
```

Using `drop_first=True` leaves one building type as the reference category and avoids keeping redundant dummy columns.

## Train/test split

I split the dataset into training and testing sets using scikit-learn.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=33
)
```

The model is trained only on the training set, while the test set is kept aside to evaluate how well the model performs on data it did not see during training.

## Linear regression

The model was trained using scikit-learn's `LinearRegression`.

```python
model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)
```

I also looked at the coefficients learned by the model:

```python
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

print(f"Intercept: {model.intercept_}")
```

The learned coefficients were very close to simple values such as:

```text
Square Footage        ≈ 0.05
Number of Occupants   ≈ 10
Appliances Used       ≈ 20
Average Temperature   ≈ -5
Day of Week           ≈ -50
```

The building-type dummy variables also showed clear differences in predicted energy consumption between building categories.

The coefficients being this clean suggests that the dataset follows a very strong linear relationship.

## Model evaluation

I evaluated the predictions using Mean Squared Error, Mean Absolute Error, and R².

```python
mse = mean_squared_error(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
```

Results:

| Metric |   Result |
| ------ | -------: |
| MSE    | ~0.00029 |
| MAE    |  ~0.0147 |
| R²     |    ~1.00 |

The model fits this dataset extremely closely.

An R² value this close to 1 means that almost all of the variation in energy consumption in the test set is explained by the linear model.

This shouldn't be taken as an indication that linear regression will perform this well on normal real-world energy data. The relationships in this particular dataset appear to be almost perfectly linear.

## Baseline comparison

I also compared the model against a simple baseline that predicts the average training-set energy consumption for every test example.

```python
baseline_value = y_train.mean()

baseline_predictions = [baseline_value] * len(y_test)

baseline_mse = mean_squared_error(
    y_test,
    baseline_predictions
)
```

The baseline produced an MSE of roughly:

```text
762800
```

while the trained linear regression model produced an MSE of approximately:

```text
0.00029
```

This confirms that the features contain a very strong signal for predicting energy consumption in this dataset.

## What I learned

This project helped me get more comfortable with:

* exploring a dataset with pandas
* separating features and targets
* working with categorical variables
* binary encoding
* one-hot encoding
* why arbitrary category codes can be problematic
* train/test splitting
* multiple linear regression
* interpreting model coefficients
* MAE, MSE and R²
* comparing a trained model against a simple baseline

It also helped me understand that a good evaluation score needs context. A model should be compared against a baseline, and an unusually perfect result is worth investigating rather than automatically assuming the model is amazing.

## Tools used

* Python
* pandas
* NumPy
* Matplotlib
* Seaborn
* scikit-learn
* Jupyter Notebook
