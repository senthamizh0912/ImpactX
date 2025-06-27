# ImapctX-GDP-
forecasting of GDP per capita across different states in India
# GDP Per Capita Analysis Tool

## Overview
This application provides comprehensive analysis and forecasting of GDP per capita across different states. It leverages advanced machine learning techniques to predict future economic trends, analyze inequality patterns, and generate evidence-based policy recommendations. The tool integrates time series forecasting, feature engineering, and socioeconomic analysis into a user-friendly interface for researchers, policymakers, and economists.

## Features
- **Economic Trend Analysis**: Visualize GDP per capita trends for top-performing states
- **Inequality Assessment**: Track economic disparity using Gini coefficients and other metrics
- **Machine Learning Forecasting**: Predict future GDP values using multiple regression models
- **Growth Rate Comparison**: Identify states with highest and lowest projected growth
- **Policy Recommendations**: Generate data-driven intervention strategies for low-performing regions

## Technologies Used
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Machine learning model implementation
- **Matplotlib & Seaborn**: Data visualization
- **Gradio**: Web interface deployment

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip package manager

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gdp-analysis-tool.git
   cd gdp-analysis-tool
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn gradio
   ```

### Running the Application
Execute the main script to launch the Gradio web interface:
```
python app.py
```

The application will be available at http://127.0.0.1:7860 in your web browser.

## Dataset Requirements
The tool expects CSV data with the following structure:
- A column named `State Name` containing state identifiers
- Multiple columns with pattern `NSDP Per Capita (Nominal)YYYY-YY` containing GDP values
- Each row represents a different state/region

Example format:
```
State Name,NSDP Per Capita (Nominal)2011-12,NSDP Per Capita (Nominal)2012-13,...
State1,₹50000,₹52000,...
State2,₹48000,₹51000,...
```

## Key Technical Components

### 1. Data Preprocessing & Transformation
The application employs sophisticated data preprocessing techniques:

- **Robust Data Extraction**: The `extract_gdp_columns` function dynamically identifies relevant GDP columns from the dataset regardless of specific naming patterns.
  
- **Intelligent Data Cleaning**: The `clean_gdp_value` function handles diverse formats of GDP values, automatically removing currency symbols and non-numeric characters while preserving the decimal structure.
  
- **Time Series Transformation**: The `prepare_time_series_data` function restructures wide-format data into a longitudinal time series format critical for temporal analysis.

```python
# Sample of the data cleaning logic
def clean_gdp_value(value):
    if isinstance(value, str):
        # Remove ₹ symbol and any commas, spaces, etc.
        cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    return value
```

### 2. Advanced Feature Engineering
The feature engineering approach incorporates domain-specific knowledge about economic time series:

- **Temporal Lag Features**: Creates lag variables (1-year and 2-year lags) to capture momentum effects in economic growth.
  
- **Growth Rate Calculation**: Implements percentage change metrics to identify acceleration or deceleration patterns.
  
- **Rolling Statistics**: Computes moving averages to smooth out short-term fluctuations and highlight longer-term trends.
  
- **State-Specific Encoding**: Uses one-hot encoding to allow the model to learn state-specific growth patterns and baselines.

```python
# Key feature engineering implementation
def create_features(df):
    df_features = df.copy()
    
    # Group by state
    for state in df_features['State'].unique():
        state_mask = df_features['State'] == state
        df_state = df_features[state_mask].sort_values('Year')
        
        # Create lag features
        if len(df_state) > 2:
            df_features.loc[state_mask, 'GDP_Lag1'] = df_state['GDP_Per_Capita'].shift(1)
            df_features.loc[state_mask, 'GDP_Lag2'] = df_state['GDP_Per_Capita'].shift(2)
            
            # Growth rate features
            df_features.loc[state_mask, 'Growth_Rate'] = df_state['GDP_Per_Capita'].pct_change()
            df_features.loc[state_mask, 'Growth_Rate_Lag1'] = df_features.loc[state_mask, 'Growth_Rate'].shift(1)
            
            # Rolling average
            df_features.loc[state_mask, 'Rolling_Avg_3Y'] = df_state['GDP_Per_Capita'].rolling(window=min(3, len(df_state)), min_periods=1).mean()
```

### 3. Time Series-Aware Model Evaluation
The application implements time series-specific cross-validation:

- **TimeSeriesSplit**: Uses scikit-learn's TimeSeriesSplit instead of standard k-fold cross-validation to respect the temporal structure of the data and prevent data leakage.
  
- **Multiple Model Comparison**: Evaluates several regression models (Linear, Ridge, Random Forest, Gradient Boosting) to find the optimal algorithm for the specific dataset characteristics.
  
- **Robust Metrics**: Calculates RMSE, MAE, and R² for comprehensive model evaluation, prioritizing error metrics that are relevant for economic forecasting.

```python
# Time series cross-validation implementation
tscv = TimeSeriesSplit(n_splits=min(5, len(X_train) // 2))

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv,
                              scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    results[name] = {
        'cv_rmse_mean': rmse_scores.mean(),
        'cv_rmse_std': rmse_scores.std()
    }
```

### 4. Recursive Multi-Step Forecasting
The forecasting approach uses an advanced recursive technique:

- **Forward Iteration**: The `forecast_future_gdp` function implements a recursive forecasting method where each prediction becomes an input for the next time step.
  
- **Dynamic Feature Updates**: Instead of static forecasting, the model recalculates all features (growth rates, lags, rolling averages) at each step, creating a more realistic simulation of future states.
  
- **State-Specific Projections**: Maintains separate forecasting chains for each state, allowing for distinct growth trajectories based on historical patterns.

```python
# Core of the recursive forecasting approach
for i in range(1, forecast_years + 1):
    future_year = latest_year + i
    
    # Create a feature row with previous predictions
    pred_features = pd.DataFrame({
        'GDP_Lag1': [state_predictions[-1]],
        'GDP_Lag2': [state_data.iloc[-2]['GDP_Per_Capita'] if i == 1 else state_predictions[-2]],
        'Growth_Rate': [(state_predictions[-1] /
                        (state_data.iloc[-2]['GDP_Per_Capita'] if i == 1 else state_predictions[-2]) - 1)],
        # Additional features calculated recursively...
    })
    
    # Make prediction and add to chain
    pred_gdp = best_model.predict(pred_features_scaled)[0]
    state_predictions.append(pred_gdp)
```

### 5. Economic Inequality Analysis
The application incorporates specialized economic inequality metrics:

- **Gini Coefficient**: Implements the mathematical calculation of the Gini coefficient to measure wealth distribution inequality across states.
  
- **Trend Analysis**: Tracks how inequality metrics change over time to identify if economic disparities are increasing or decreasing.
  
- **Coefficient of Variation**: Calculates normalized dispersion metrics that allow for comparison of inequality across different time periods with different baseline GDP levels.

```python
# Implementation of Gini coefficient calculation
def calculate_gini(array):
    array = np.array(array)
    if len(array) < 2:
        return 0
        
    # Sort array
    array = np.sort(array)
    # Calculate cumulative sum of array
    cumulative_sum = np.cumsum(array)
    # Calculate cumulative share of population and income
    n = len(array)
    cumulative_people = np.arange(1, n + 1) / n
    cumulative_income = cumulative_sum / cumulative_sum[-1]
    # Calculate Gini coefficient using the area under the Lorenz curve
    gini = 1 - 2 * np.sum((cumulative_income[:-1] + cumulative_income[1:]) / 2 * np.diff(cumulative_people))
    return gini
```

### 6. Policy Recommendation Engine
The application includes an evidence-based policy recommendation system:

- **Priority Identification**: Automatically identifies states requiring the most urgent intervention based on both current GDP and projected growth.
  
- **Tiered Recommendations**: Provides different levels of recommendations based on the severity of economic challenges (negative growth, low growth, moderate growth).
  
- **Inequality-Focused Strategies**: Suggests specific policies targeted at reducing economic disparities between states.

```python
# Example of the recommendation logic
if rate < 0:
    result += "- **URGENT ACTION NEEDED**: Negative growth projection\n"
    result += "- Implement economic stimulus package\n"
    result += "- Develop infrastructure investment plan\n"
    result += "- Create special economic zones to attract investment\n\n"
elif rate < 5:
    result += "- **HIGH PRIORITY**: Low growth projection\n"
    result += "- Focus on skill development programs\n"
    result += "- Provide tax incentives for new businesses\n"
    result += "- Improve transportation and logistics infrastructure\n\n"
```

### 7. Interactive Visualization Framework
The visualization system implements several advanced techniques:

- **Dynamic Image Generation**: Creates visualizations on-demand and converts them to base64 encoding for seamless integration with the Gradio interface.
  
- **Multi-faceted Analysis**: Generates complementary visualizations that show different aspects of the same data (trends, forecasts, inequality metrics).
  
- **Color-Coded Indicators**: Uses appropriate color schemes to highlight patterns in the data, such as divergent color schemes for heatmaps.

```python
# Visualization generation with proper encoding for web display
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
images['forecast'] = base64.b64encode(buf.read()).decode('utf-8')
```

### 8. Error Handling & Debugging Framework
The application incorporates a robust error handling system:

- **Graceful Degradation**: Implements try-except blocks throughout the code to handle various failure modes without crashing.
  
- **Diagnostic Visualizations**: Includes special debug_visualizations function that can isolate and identify issues with the plotting system.
  
- **Global State Management**: Uses global variables carefully to maintain state between function calls while providing appropriate encapsulation.

```python
# Error handling pattern used throughout the code
try:
    # Attempt operation
    feature_df = create_features(ts_df)
    if len(feature_df) == 0:
        return "Not enough time series data for feature creation", None, None, None, None, None, None
except Exception as e:
    return f"Error in feature creation: {str(e)}", None, None, None, None, None, None
```

## Using the Application

### 1. Analysis Tab
- Upload your CSV file using the file uploader
- Click "Run Analysis" to process the data
- Review model results, visualizations, and social impact analysis

### 2. State Predictions Tab
- Enter a specific state name to view its forecasted GDP values
- Results show year-by-year projections

### 3. Growth Analysis Tab
- Click "Analyze Growth Rates" to compare projected growth across states
- View states with highest and lowest expected economic growth

### 4. Policy Recommendations Tab
- Generate intervention strategies for states with below-average performance
- Review suggested policies for reducing inequality



