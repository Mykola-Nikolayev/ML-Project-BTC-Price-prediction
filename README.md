# ML-Project-BTC-Price-prediction

# Documentation: Bitcoin Price Prediction Using Machine Learning and Prophet

This project is designed to predict the price of Bitcoin (BTC) until the year 2030. It combines machine learning techniques, including Linear Regression, and time-series analysis using Facebook Prophet. Additionally, the code includes simulations based on historical market cycles. This documentation covers each step, the rationale behind the operations, and the mathematical formulations involved.

## Table of Contents

1. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
2. [Feature Engineering](#feature-engineering)
3. [Model Training and Evaluation](#model-training-and-evaluation)
4. [Time Series Forecasting with Prophet](#time-series-forecasting-with-prophet)
5. [Historical Cycle Simulation](#historical-cycle-simulation)
6. [Visualization](#visualization)

---

## Data Loading and Preprocessing

### Step 1: Loading and Cleaning Data

```python
file_path = 'btc.csv'
data = pd.read_csv(file_path, delimiter=',')
```
- **Objective**: Load Bitcoin price data from a CSV file.
- **Operation**: Load the dataset using `pandas.read_csv()`.

### Step 2: Column Name Cleaning

```python
data.columns = data.columns.str.strip()
```
- **Objective**: Remove any leading or trailing spaces from column names to prevent errors in referencing columns.
- **Operation**: Use `str.strip()` to clean column names.

### Step 3: Renaming Columns

```python
data.rename(columns={
    'Date': 'date',
    'Dernier': 'priceClose',
    'Ouv.': 'priceOpen',
    'Plus Haut': 'priceHigh',
    'Plus Bas': 'priceLow',
    'Vol.': 'volume'
}, inplace=True)
```
- **Objective**: Simplify column names for easier handling.

### Step 4: Date Conversion and Cleaning

```python
data['date'] = pd.to_datetime(data['date'], dayfirst=True, errors='coerce')
data = data.dropna(subset=['date'])
```
- **Objective**: Convert the `date` column to datetime format.
- **Operation**: Use `pd.to_datetime()`. Any invalid dates are removed using `dropna()`.

### Step 5: Sorting Data

```python
data = data.sort_values('date').reset_index(drop=True)
```
- **Objective**: Ensure that the data is in chronological order for time series analysis.

### Step 6: Data Conversion

Convert the price and volume columns from strings to floats to facilitate further numerical analysis.

```python
data['priceClose'] = data['priceClose'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
# Repeat similar conversion for 'priceOpen', 'priceHigh', 'priceLow', and 'volume'
```
- **Objective**: Convert financial data represented in strings to numeric types to perform mathematical operations.
- **Volume Conversion**: Convert units (e.g., 'M', 'B') to numeric equivalents using `str.replace()`.

---

## Feature Engineering

### Calculating Daily Returns

```python
data['return'] = data['priceClose'].pct_change()
```
- **Objective**: Compute the daily returns of Bitcoin.
- **Formula**:
  
  \[
  r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
  \]
  
  where:
  - \( r_t \) is the daily return.
  - \( P_t \) is the closing price at day \( t \).

### Calculating 7-Day Rolling Volatility

```python
data['volatility_7d'] = data['return'].rolling(window=7).std()
```
- **Objective**: Calculate the 7-day rolling volatility.
- **Formula**:

  \[
  \sigma_{7d} = \sqrt{\frac{1}{7-1} \sum_{i=t-6}^{t} (r_i - \bar{r})^2}
  \]
  
  where:
  - \( \sigma_{7d} \) is the standard deviation of returns over the last 7 days.

### Additional Features

- **Previous Price**: The previous day's price (`Previous_Price`).
- **7-Day Simple Moving Average** (SMA):

  ```python
  data['SMA_7'] = data['priceClose'].rolling(window=7).mean()
  ```
  
  **Formula**:
  
  \[
  SMA_{7} = \frac{1}{7} \sum_{i=t-6}^{t} P_i
  \]
- Extract **Month** and **Year** from the `date` column.

### Handling Missing Values

```python
data = data.dropna()
```
- **Objective**: Remove rows with missing values after feature engineering.

### Calculating Growth Rate

```python
growth_rate = (data['priceClose'].iloc[-1] / data['priceClose'].iloc[0]) ** (1 / num_years) - 1
```
- **Formula**:

  \[
  g = \left( \frac{P_{\text{end}}}{P_{\text{start}}} \right)^{\frac{1}{\text{num\_years}}} - 1
  \]

- **Objective**: Calculate the compounded annual growth rate (CAGR).

---

## Model Training and Evaluation

### Preparing Data for Modeling

```python
X = data[['Previous_Price', 'priceOpen', 'priceHigh', 'priceLow', 'volume', 'SMA_7', 'volatility_7d', 'month', 'year']]
y = data['priceClose']
```
- **Objective**: Define features (`X`) and target (`y`).

### Splitting Data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **Objective**: Split the data into training and test sets (80-20 split).

### Normalizing Features

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- **Objective**: Standardize features using Z-score normalization.
- **Formula**:

  \[
  X_{\text{scaled}} = \frac{X - \mu}{\sigma}
  \]

### Training Linear Regression Model

```python
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```
- **Objective**: Fit a Linear Regression model.
- **Formula**:

  \[
  y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_n X_n + \epsilon
  \]

### Evaluating the Model

- **Mean Squared Error (MSE)**:

  \[
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]
- **Root Mean Squared Error (RMSE)**:

  \[
  RMSE = \sqrt{MSE}
  \]
- **Mean Absolute Error (MAE)**:

  \[
  MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]
- **R-Squared (R²)**:

  \[
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  \]

---

## Time Series Forecasting with Prophet

### Preparing Data for Prophet

```python
data_prophet = data[['date', 'priceClose']].rename(columns={'date': 'ds', 'priceClose': 'y'})
data_prophet['floor'] = 0
```
- **Objective**: Prepare data for Facebook Prophet. Prophet requires specific column names: `ds` for dates and `y` for target values.

### Fitting the Prophet Model

```python
model_prophet = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.5)
model_prophet.fit(data_prophet)
```
- **Objective**: Fit the Prophet model.
- **Hyperparameters**:
  - **Yearly Seasonality**: Account for yearly patterns.
  - **Changepoint Prior Scale**: Control flexibility in trend changes.

### Forecasting Future Values

```python
future = model_prophet.make_future_dataframe(periods=365 * 6, freq='D')
forecast = model_prophet.predict(future)
```
- **Objective**: Forecast BTC prices for an additional 6 years.

---

## Historical Cycle Simulation

### Analyzing Market Phases

- **Objective**: Analyze historical market phases (bull, bear, stable).

#### Log Price Calculation

```python
data['log_price'] = np.log(data['priceClose'])
```
- **Objective**: Convert prices to a logarithmic scale for better volatility analysis.

#### Determining Market Phases

```python
data['price_change'] = data['log_price'].diff()
threshold = data['price_change'].std() * 2
```
- **Objective**: Classify market phases based on a threshold derived from price change volatility.

### Simulating Prices Using Historical Cycles

#### Simulation Logic

- **Daily Return Calculation**: Depending on whether the market is in a bull or bear phase, calculate daily returns:

  \[
  r_{\text{daily}} = (1 + r_{\text{phase}})^{1 / T_{\text{phase}}} - 1
  \]
  
  where \( r_{\text{phase}} \) and \( T_{\text{phase}} \) represent the average return and duration for the current phase.

- **Shock Component**: Add random daily volatility:

  \[
  \text{shock} \sim \mathcal{N}(0, \sigma_{\text{volatility}})
  \]

- **Update Price**:

  \[
  P_{\text{new}} = P_{\text{current}} \times (1 + r_{\text{daily}} + \text{shock})
  \]

---

## Visualization

### Linear Regression Prediction Visualization

```python
plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
```
- **Objective**: Visualize how well the model's predictions match actual values.

### Time Series Plot of Actual vs Predicted

```python
plt.plot(y_test.values, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
```
- **Objective**: Display time series of actual and predicted prices.

### Residuals Distribution

```python
plt.hist(residuals, bins=50, alpha=0.7, color='purple')
```
- **Objective**: Analyze the distribution of residuals to check for biases in the model.

### Interactive Plot Using Plotly

```python
fig = go.Figure()
fig.add_trace(go.Scatter(...))
fig.show()
```
- **Objective**: Create an interactive plot showing historical prices, Prophet predictions, and simulated prices.

---

## Conclusion

This project provides a comprehensive framework for predicting Bitcoin prices. It uses machine learning (Linear Regression), time series analysis (Prophet), and historical cycle-based simulations to make long-term forecasts. Each component contributes to a robust understanding of both the underlying trends and expected market behaviors.

