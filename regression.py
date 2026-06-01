# -*- coding: utf-8 -*-
# data_regression_final.ipynb

#data preprocessing

"""# 1. library import"""

import pandas as pd
import numpy as np

print("Library loaded successfully!")

"""# 2. Load CSV File"""

# Load two CSV files
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print("movies shape  :", movies.shape)
print("credits shape :", credits.shape)

"""# 3. Merge Data"""

# Merge two datasets based on movie_id
credits = credits.rename(columns={'movie_id': 'id'})
df = movies.merge(credits, on='id')

print("Shape after merge :", df.shape)
df.head(3)

"""# 4. Data Inspection"""

# Check basic information
print(" Basic Dataset Information ")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print()

# Check data types for each column
print(" == Data Types == ")
print(df.dtypes)

"""# 5. Missing Value Handling"""

# Check NULL missing values
print(" NULL Missing Value Status : ")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing)

# Check 0 values
print("== Zero Value Status ==")
num_cols = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']

for col in num_cols:
    zero_count = (df[col] == 0).sum()
    print(f"{col:20s} : {zero_count} items")

# Check invalid inputs
print(" Checking negative values: ")

for col in num_cols:
    neg_count = (df[col] < 0).sum()
    print(f"{col:20s} : {neg_count} items")

print()
print(" Checking status value types: ")
print(df['status'].value_counts())

print()
print(" Checking original_language value types: ")
print(df['original_language'].value_counts().head(10))

# Handle NULL values

# Remove homepage and tagline columns because they are unnecessary for analysis
df = df.drop(columns=['homepage', 'tagline'])

# Replace missing values in overview with empty strings
df['overview'] = df['overview'].fillna('')

# Remove rows with missing release_date values
df = df.dropna(subset=['release_date'])

# Replace missing runtime values with the median
runtime_median = df['runtime'].median()
df['runtime'] = df['runtime'].fillna(runtime_median)

print(" Checking missing values after NULL handling : ")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("No missing values!" if df.isnull().sum().sum() == 0 else "")

# Handle 0 values

# Replace 0 values in budget and revenue with NaN, then fill with median values
df['budget'] = df['budget'].replace(0, np.nan)
df['revenue'] = df['revenue'].replace(0, np.nan)

budget_median  = df['budget'].median()
revenue_median = df['revenue'].median()

df['budget']  = df['budget'].fillna(budget_median)
df['revenue'] = df['revenue'].fillna(revenue_median)

# Replace 0 values in runtime with the median value
df['runtime'] = df['runtime'].replace(0, df['runtime'].median())

# Remove rows where vote_average or vote_count is 0
df = df[df['vote_count'] > 0]
df = df[df['vote_average'] > 0]

print(" Shape after handling 0 values : ")
print(df.shape)

print()
print(" Check after handling 0 values : ")

for col in num_cols:
    zero_count = (df[col] == 0).sum()
    print(f"{col:20s} : {zero_count} items")

# Handle invalid inputs

# Remove movies that are not in Released status
df = df[df['status'] == 'Released']

# Clean duplicated title columns (title_x, title_y → title)
df = df.drop(columns=['title_y'])
df = df.rename(columns={'title_x': 'title'})

print(" Shape after handling invalid inputs : ")
print(df.shape)

print()
print(" Check status values : ")
print(df['status'].value_counts())

"""# 6. Check Duplicate Data"""

# Check duplicate data

print(" Checking duplicate rows : ")
dup_count = df.duplicated().sum()
print(f"Number of duplicate rows : {dup_count} items")

print()
print(" Checking duplicates based on movie titles : ")
dup_title = df.duplicated(subset=['title']).sum()
print(f"Number of duplicated titles : {dup_title} items")

# Remove duplicate data
df = df.drop_duplicates(subset=['title'])

print()
print(" Shape after removing duplicates :")
print(df.shape)

"""# 7. Process JSON Columns"""

# Display only the newly created columns after JSON processing
import ast

# Process JSON format columns

json_columns = [
    'genres',
    'keywords',
    'production_companies',
    'production_countries',
    'spoken_languages',
    'cast',
    'crew'
]

# Function to convert JSON string to list
def json_to_list(text):
    try:
        return ast.literal_eval(text)
    except:
        return []

# Extract only the 'name' values
def extract_names(text):
    try:
        data = ast.literal_eval(text)
        return [item['name'] for item in data]
    except:
        return []

# Process JSON columns
for col in json_columns:

    # Convert string → list(dict)
    df[col] = df[col].apply(json_to_list)

    # Extract only 'name' values
    df[col + '_names'] = df[col].apply(
        lambda x: [i['name'] for i in x if 'name' in i]
    )

    # Convert to string format
    df[col + '_str'] = df[col + '_names'].apply(
        lambda x: ', '.join(x)
    )

print("JSON column processing completed!")

# Display only the newly created columns after JSON processing

json_created_cols = [
    'genres_names', 'genres_str',
    'keywords_names', 'keywords_str',
    'production_companies_names', 'production_companies_str',
    'production_countries_names', 'production_countries_str',
    'spoken_languages_names', 'spoken_languages_str',
    'cast_names', 'cast_str',
    'crew_names', 'crew_str',
]

for col in json_created_cols:
    print(col)

df[json_created_cols].head()

# Create additional features

# Number of crew members
df['crew_count'] = df['crew_names'].apply(len)

# Number of cast members
df['cast_count'] = df['cast_names'].apply(len)

# Main genre
df['main_genre'] = df['genres_names'].apply(
    lambda x: x[0] if len(x) > 0 else np.nan
)

# Extract director
def get_director(crew_list):

    for person in crew_list:

        if person.get('job') == 'Director':
            return person.get('name')

    return np.nan

df['director'] = df['crew'].apply(get_director)

print("Additional columns created successfully!")

# Convert to datetime type
df['release_date'] = pd.to_datetime(df['release_date'])

# Release year
df['release_year'] = df['release_date'].dt.year

# Release month
df['release_month'] = df['release_date'].dt.month

"""# 8. Merge Columns"""

# Feature Engineering

# Profit
df['profit'] = df['revenue'] - df['budget']

# success label 생성 (로그 변환 전에!)
df['success'] = (df['profit'] > 0).astype(int)

print(df['success'].value_counts())

print("Derived variable creation completed!")

"""# 9. Remove Unnecessary Columns"""

# =========================================
# Remove unnecessary columns
# =========================================

drop_cols = [

    # Original JSON columns
    'genres',
    'keywords',
    'production_companies',
    'production_countries',
    'spoken_languages',
    'cast',
    'crew',

    # String / list columns
    'genres_str',
    'keywords_str',
    'keywords_names',
    'cast_str',
    'crew_str',
    'production_companies_names',

    'spoken_languages_names',
    'spoken_languages_str',

    'production_countries_str',
    'production_companies_str',

    # Additional removals
    'crew_names',
    'main_genre',

    # Text columns
    'overview',
    'homepage',
    'tagline',

    # Title columns
    'title',
    'original_title',

    # Original date column
    'release_date',

    # Others
    'id',
    'poster_path',
    'status'
]

# Remove only existing columns
drop_cols = [col for col in drop_cols if col in df.columns]

df = df.drop(columns=drop_cols)

print("Unnecessary columns removed successfully!")
print(df.shape)

# Print all column names

for i, col in enumerate(df.columns):
    print(f"{i+1:2d}. {col}")

"""# 10. Process Categorical Columns"""

# Process original_language column

# Keep only the top 8 languages
top_lang = df['original_language'].value_counts().head(8).index

# Replace the remaining languages with 'Other'
df['original_language'] = df['original_language'].apply(
    lambda x: x if x in top_lang else 'Other'
)

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=['original_language'],
    dtype=int
)

print(df.shape)

# MultiLabel One-Hot Encoding for genres_names

from sklearn.preprocessing import MultiLabelBinarizer

# MultiLabel One-Hot Encoding
mlb = MultiLabelBinarizer()

genre_ohe = pd.DataFrame(
    mlb.fit_transform(df['genres_names']),
    columns=mlb.classes_,
    index=df.index
)

# Merge with the original dataframe
df = pd.concat([df, genre_ohe], axis=1)

# Remove the original column
df = df.drop(columns=['genres_names'])

print(df.shape)

# Check generated genre columns
print(genre_ohe.columns)

df

from collections import Counter

# Calculate country frequencies

country_counter = Counter()

for country_list in df['production_countries_names']:
    country_counter.update(country_list)

# Extract top 10 countries
top_countries = [
    country for country, count
    in country_counter.most_common(10)
]

print(top_countries)

# Process production_countries_names column

# Keep only the top countries
df['production_countries_names'] = df['production_countries_names'].apply(
    lambda x: [c if c in top_countries else 'Other' for c in x]
)

# MultiLabel One-Hot Encoding
mlb_country = MultiLabelBinarizer()

country_ohe = pd.DataFrame(
    mlb_country.fit_transform(df['production_countries_names']),
    columns=mlb_country.classes_,
    index=df.index
)

# Merge with the original dataframe
df = pd.concat([df, country_ohe], axis=1)

# Remove the original column
df = df.drop(columns=['production_countries_names'])

print(df.shape)

df

from collections import Counter

# Calculate actor appearance frequencies

cast_counter = Counter()

for cast_list in df['cast_names']:
    cast_counter.update(cast_list)

print("Number of actors :", len(cast_counter))

# Create cast popularity features

# Sum of actor appearance frequencies
df['cast_freq_sum'] = df['cast_names'].apply(
    lambda x: sum(cast_counter[name] for name in x)
)

# Average actor appearance frequency
df['cast_freq_mean'] = df['cast_names'].apply(
    lambda x: np.mean([cast_counter[name] for name in x]) if len(x) > 0 else 0
)

print(df[['cast_freq_sum', 'cast_freq_mean']].head())

df = df.drop(columns=['cast_names'])

df

# Director Frequency Encoding

# Calculate director appearance frequencies
director_freq = df['director'].value_counts()

# Frequency encoding
df['director_freq'] = df['director'].map(director_freq)

# Check results
print(df[['director', 'director_freq']].head())

df = df.drop(columns=['director'])

for i, col in enumerate(df.columns):
    print(f"{i+1:2d}. {col}")

"""# 11. Graph Visualization"""

import matplotlib.pyplot as plt

# Numerical columns
numeric_cols = [
    'budget',
    'popularity',
    'revenue',
    'runtime',
    'vote_average',
    'vote_count',
    'crew_count',
    'cast_count',
    'release_year',
    'release_month',
    'profit',
    'cast_freq_sum',
    'cast_freq_mean',
    'director_freq'
]

# Display graphs
for col in numeric_cols:

    plt.figure(figsize=(6,4))

    plt.hist(df[col], bins=30)

    plt.title(f'{col} Distribution')

    plt.xlabel(col)
    plt.ylabel('Count')

    plt.show()

# Check the number of movies by genre

genre_cols = [
    'Action', 'Adventure', 'Animation', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Family',
    'Fantasy', 'Foreign', 'History', 'Horror',
    'Music', 'Mystery', 'Romance',
    'Science Fiction', 'TV Movie',
    'Thriller', 'War', 'Western'
]

genre_count = df[genre_cols].sum().sort_values(ascending=False)

genre_count.plot(
    kind='bar',
    figsize=(12,6),
    title='Genre Counts'
)

plt.ylabel('Movie Count')
plt.show()

country_cols = [
    'Australia',
    'Canada',
    'China',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'Other',
    'Spain',
    'United Kingdom',
    'United States of America'
]

country_count = df[country_cols].sum().sort_values(ascending=False)

country_count.plot(
    kind='bar',
    figsize=(10,5),
    title='Production Country Counts'
)

plt.ylabel('Movie Count')
plt.show()

language_cols = [
    'original_language_Other',
    'original_language_de',
    'original_language_en',
    'original_language_es',
    'original_language_fr',
    'original_language_hi',
    'original_language_it',
    'original_language_ja',
    'original_language_zh'
]

language_count = df[language_cols].sum().sort_values(ascending=False)

language_count.plot(
    kind='bar',
    figsize=(10,5),
    title='Original Language Counts'
)

plt.ylabel('Movie Count')
plt.show()

"""# 12. Handle Skewed Data"""

# Check skewness of numerical columns
# (excluding language / genre / country columns)

# Columns to exclude
exclude_cols = [

    # Language OHE
    'original_language_Other',
    'original_language_de',
    'original_language_en',
    'original_language_es',
    'original_language_fr',
    'original_language_hi',
    'original_language_it',
    'original_language_ja',
    'original_language_zh',

    # Genre OHE
    'Action',
    'Adventure',
    'Animation',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Family',
    'Fantasy',
    'Foreign',
    'History',
    'Horror',
    'Music',
    'Mystery',
    'Romance',
    'Science Fiction',
    'TV Movie',
    'Thriller',
    'War',
    'Western',

    # Country OHE
    'Australia',
    'Canada',
    'China',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'Other',
    'Spain',
    'United Kingdom',
    'United States of America'
]

# Select numerical columns
numeric_cols = df.select_dtypes(
    include=['int64', 'float64']
).columns

# Remove excluded columns
numeric_cols = [
    col for col in numeric_cols
    if col not in exclude_cols
]

# Calculate skewness
skew_df = pd.DataFrame({
    'column': numeric_cols,
    'skew': df[numeric_cols].skew()
})

# Sort by absolute skewness
skew_df['abs_skew'] = skew_df['skew'].abs()

skew_df = skew_df.sort_values(
    by='abs_skew',
    ascending=False
)

# Classify skewness level
def skew_level(x):

    if abs(x) < 0.5:
        return 'Normal'

    elif abs(x) < 1:
        return 'Moderate'

    elif abs(x) < 2:
        return 'High'

    else:
        return 'Very High'

skew_df['level'] = skew_df['skew'].apply(skew_level)

print(skew_df[['column', 'skew', 'level']])

# Log Transformation


log_cols = [
    'budget',
    'vote_count',
    'popularity',
    'revenue'
]


for col in log_cols:

    df[col] = np.log1p(df[col])

print("Log Transformation complete")

import matplotlib.pyplot as plt

for col in log_cols:

    plt.figure(figsize=(10,4))

    # After Transformation
    plt.hist(df[col], bins=30)

    plt.title(f'{col} After Log Transform')

    plt.show()

# Check skewness of numerical columns
# (excluding language / genre / country columns)

# Columns to exclude
exclude_cols = [

    # Language OHE
    'original_language_Other',
    'original_language_de',
    'original_language_en',
    'original_language_es',
    'original_language_fr',
    'original_language_hi',
    'original_language_it',
    'original_language_ja',
    'original_language_zh',

    # Genre OHE
    'Action',
    'Adventure',
    'Animation',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Family',
    'Fantasy',
    'Foreign',
    'History',
    'Horror',
    'Music',
    'Mystery',
    'Romance',
    'Science Fiction',
    'TV Movie',
    'Thriller',
    'War',
    'Western',

    # Country OHE
    'Australia',
    'Canada',
    'China',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'Other',
    'Spain',
    'United Kingdom',
    'United States of America'
]

# Select numerical columns
numeric_cols = df.select_dtypes(
    include=['int64', 'float64']
).columns

# Remove excluded columns
numeric_cols = [
    col for col in numeric_cols
    if col not in exclude_cols
]

# Calculate skewness
skew_df = pd.DataFrame({
    'column': numeric_cols,
    'skew': df[numeric_cols].skew()
})

# Sort by absolute skewness
skew_df['abs_skew'] = skew_df['skew'].abs()

skew_df = skew_df.sort_values(
    by='abs_skew',
    ascending=False
)

skew_df['level'] = skew_df['skew'].apply(skew_level)

print(skew_df[['column', 'skew', 'level']])

"""# 13. Determine Box Office Success"""

# Case 1: Considered successful if the movie makes a profit
# profit > 0

# Case 2: Considered successful if the movie is in the top 30% of revenue
# threshold = df['revenue'].quantile(0.7)
# df['success'] = (df['revenue'] >= threshold).astype(int)

# Case 3: Considered successful if the rating is high
# vote_average >= 7

# Case 4: Considered successful if the movie is highly popular (public popularity)
# threshold = df['popularity'].quantile(0.7)
# df['success'] = (df['popularity'] >= threshold).astype(int)

# Case 5: Composite criteria using a combination of Cases 1–4
# df['success'] = ((df['profit'] > 0) & (df['vote_average'] >= 6.5)).astype(int)

# Final Decision in Dataset Splitting

"""# 14. Scaling"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Columns to scale
scale_cols = ['budget', 'runtime', 'popularity', 'vote_average', 'vote_count']

# Apply three different scalers
standard = StandardScaler().fit_transform(df[scale_cols])
minmax   = MinMaxScaler().fit_transform(df[scale_cols])
robust   = RobustScaler().fit_transform(df[scale_cols])

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 5, figsize=(18, 10))
fig.suptitle('Scaler Comparison', fontsize=16, fontweight='bold')

scalers      = [standard, minmax, robust]
scaler_names = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']

for row, (data, name) in enumerate(zip(scalers, scaler_names)):
    for col, col_name in enumerate(scale_cols):

        axes[row][col].hist(
            data[:, col],
            bins=50,
            color='#4C72B0',
            edgecolor='white',
            alpha=0.85
        )

        axes[row][col].set_title(
            f'{name}\n{col_name}',
            fontsize=9
        )

plt.tight_layout()
plt.show()

# Apply an appropriate scaler to each column

robust_cols   = ['budget', 'popularity', 'vote_count']
standard_cols = ['runtime', 'vote_average']

df_scaled = df.copy()

# Apply RobustScaler
robust_scaler = RobustScaler()
df_scaled[robust_cols] = robust_scaler.fit_transform(df[robust_cols])

# Apply StandardScaler
standard_scaler = StandardScaler()
df_scaled[standard_cols] = standard_scaler.fit_transform(df[standard_cols])

print("=== Scaling Completed ===")
print(df_scaled[robust_cols + standard_cols].describe().round(3))

# - budget, popularity, vote_count : Many outliers and right-skewed distribution → RobustScaler
# - runtime, vote_average          : Close to normal distribution → StandardScaler

# Print final columns
for i, col in enumerate(df.columns):
    print(f"{i+1:2d}. {col}")

# director_freq fill nan
df_scaled['director_freq'] = df_scaled['director_freq'].fillna(0)

# 컬럼별 Null 개수 확인
print(df_scaled.isnull().sum())

# 전체 Null 개수 확인
print("\n전체 Null 개수:", df_scaled.isnull().sum().sum())

"""# 15. Split Dataset

# For Classification
"""

# Dataset for Classification


X_cls = df_scaled.drop(columns=[
    'success',
    'profit',
    'revenue'
])

y_cls = df_scaled['success']

print(X_cls.shape)
print(y_cls.shape)

from sklearn.model_selection import train_test_split

# train / temp


X_train_cls, X_temp_cls, y_train_cls, y_temp_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.3,
    random_state=42,
    stratify=y_cls
)

# =========================================
# validation / test
# =========================================

X_val_cls, X_test_cls, y_val_cls, y_test_cls = train_test_split(
    X_temp_cls,
    y_temp_cls,
    test_size=0.5,
    random_state=42,
    stratify=y_temp_cls
)

print("Train :", X_train_cls.shape)
print("Val   :", X_val_cls.shape)
print("Test  :", X_test_cls.shape)

"""# For Regression (1)"""

# target ->  revenue

# Dataset for Regression
# Remove 'revenue' and 'profit' to prevent data leakage

X_reg = df_scaled.drop(columns=[
    'revenue',
    'profit',
    'success'
])

y_reg = df_scaled['revenue']

print(X_reg.shape)
print(y_reg.shape)

# train / temp


X_train_reg, X_temp_reg, y_train_reg, y_temp_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.3,
    random_state=42
)

# validation / test


X_val_reg, X_test_reg, y_val_reg, y_test_reg = train_test_split(
    X_temp_reg,
    y_temp_reg,
    test_size=0.5,
    random_state=42
)
# 7 : 1.5 : 1.5
print("Train :", X_train_reg.shape)
print("Val   :", X_val_reg.shape)
print("Test  :", X_test_reg.shape)

y_test_reg_orig = np.expm1(y_test_reg)

"""# Regression 1. Basic Setup"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Evaluation and Validation Modules
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

# Model Classes
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Visualization Modules (Partial Dependence Plot)
from sklearn.inspection import PartialDependenceDisplay

sns.set_theme(style="whitegrid")

"""# Regression 2. Define Common Used Functions"""

# Evaluation and Visualization Functions

def plot_evaluation_charts(y_true, y_pred, model_name):
    """
    Function to plot Actual vs Predicted scatter and Residual plots.

    [Parameters]
    - y_true (array-like): Actual target values (log-transformed revenue)
    - y_pred (array-like): Predicted target values by the model
    - model_name (str): Model name to use in the visualization titles
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.5, color='royalblue')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_title(f'{model_name}: Actual vs Predicted')
    axes[0].set_xlabel('Actual Revenue (log)')
    axes[0].set_ylabel('Predicted Revenue (log)')

    # Residual Plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, color='darkorange')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_title(f'{model_name}: Residual Plot')
    axes[1].set_xlabel('Predicted Revenue (log)')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')

    plt.tight_layout()
    plt.show()

def evaluate_model(y_true, y_pred, model_name):
    """
    Function to calculate and print regression evaluation metrics (R2, RMSE, MAE).
    Also inversely transforms the log-transformed values to their original scale
    to show intuitive errors.

    [Parameters]
    - y_true (array-like): Actual target values
    - y_pred (array-like): Predicted target values
    - model_name (str): Model name for output
    """
    # Inverse transform: exp(x) - 1
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f'[{model_name} Evaluation Metrics]')
    print(f'R2   : {r2:.4f}')
    print(f'RMSE : {rmse:.4f}')
    print(f'MAE  : {mae:.4f}')

    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    print(f'Original scale RMSE: {rmse_orig:.4f}')
    print(f'Original scale MAE : {mae_orig:.4f}\n')

"""# Regression 3. Top-level Pipeline Function"""

def run_regression_pipeline(model, model_name, data_dict, tuning_type=None, param_space=None):
    """
    Master function that performs model training, hyperparameter tuning,
    prediction, evaluation, and specific visualizations all at once.

    [Parameters]
    - model: Estimator instance to train (e.g., LinearRegression, RandomForestRegressor, XGBRegressor)
    - model_name (str): 'Linear Regression', 'Random Forest', 'XGBoost'
    - data_dict (dict): {'X_train': ..., 'y_train': ..., 'X_val': ..., 'y_val': ..., 'X_test': ..., 'y_test': ...}
    - tuning_type (str):
        - None: Perform basic fit without tuning (for Linear Regression)
        - 'rf_manual': Tune by iterating through a predefined list using the Validation set (for Random Forest)
        - 'xgb_random': Tune using RandomizedSearchCV with CV and extract the Top 3 (for XGBoost)
    - param_space (list or dict): Parameter space for tuning (list or dictionary)

    [Returns]
    - history_df (DataFrame): DataFrame containing parameter combinations and Test RMSE results (for final comparison)
    """
    print(f"- Starting Pipeline for: {model_name}!")

    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']

    best_model = model
    history_df = pd.DataFrame(columns=['params', 'RMSE', 'Model'])


    # Step 1. Training and Hyperparameter Tuning
    if tuning_type == 'rf_manual':
        print("Hyperparameter Tuning In Progress...")
        best_rmse = float('inf')
        best_param = {}
        for i, p in enumerate(param_space, start=1):
            temp_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, **p)
            temp_model.fit(X_train, y_train)

            # Find the Best model using Validation set
            pred_val = temp_model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, pred_val))

            # Save Test RMSE (for record-keeping)
            pred_test = temp_model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))

            history_df.loc[len(history_df)] = [p, test_rmse, model_name]

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model = temp_model
                best_param = p
            print(f'Param #{i}: {p}, Val RMSE: {val_rmse:.4f}')

        print(f"\n Best RF Params: {best_param}")

    elif tuning_type == 'xgb_random':
        print("Hyperparameter Tuning In Progress...")
        random_search = RandomizedSearchCV(
            estimator=model, param_distributions=param_space,
            n_iter=40, scoring='neg_root_mean_squared_error', cv=5,
            verbose=0, random_state=1461, n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        print(f"Best XGB Params: {best_params}")

        # Extract Top 3 from CV results and evaluate on Test set
        cv_df = pd.DataFrame(random_search.cv_results_)
        cv_df['CV_RMSE'] = -cv_df['mean_test_score']
        top3_params = cv_df.nsmallest(3, 'CV_RMSE')['params'].tolist()

        for p in top3_params:
            temp_model = XGBRegressor(n_estimators=500, random_state=1461, n_jobs=-1, early_stopping_rounds=20, **p)
            temp_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            pred_test = temp_model.predict(X_test)
            history_df.loc[len(history_df)] = [p, np.sqrt(mean_squared_error(y_test, pred_test)), model_name]

        # Train the final best model (to record the learning curve)
        best_model = XGBRegressor(n_estimators=500, random_state=1461, n_jobs=-1, early_stopping_rounds=20, **best_params)
        best_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        print(f"Best Iteration: {best_model.best_iteration}\n")

    else:
        # Basic training (for Linear Regression)
        best_model.fit(X_train, y_train)


    # Step 2. Prediction and Common Evaluation/Visualization
    y_pred = best_model.predict(X_test)
    evaluate_model(y_test, y_pred, model_name)
    plot_evaluation_charts(y_test, y_pred, model_name)


    # Step 3. Model-specific Visualization (Feature Importance / Coefficient & PDP)
    if model_name == 'Linear Regression':
        coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': best_model.coef_}).sort_values(by='Coefficient', ascending=False)
        norm = mcolors.TwoSlopeNorm(vmin=coef_df['Coefficient'].min(), vcenter=0, vmax=coef_df['Coefficient'].max())
        colors = [plt.get_cmap('RdYlGn')(norm(val)) for val in coef_df['Coefficient']]

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df, hue='Feature', palette=colors, legend=False)
        plt.title('Linear Regression Feature Coefficients')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
        plt.tight_layout()
        plt.show()

    else:
        # Feature Importance Plot
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = X_train.columns[indices]
        top1_feature = top_features[0]

        plt.figure(figsize=(10, 5))
        sns.barplot(x=importances[indices][:10], y=top_features[:10], hue=top_features[:10], palette='viridis', legend=False)
        plt.title(f'{model_name}: Top 10 Feature Importances')
        plt.tight_layout()
        plt.show()

        # Partial Dependence Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        PartialDependenceDisplay.from_estimator(best_model, X_train, features=[top1_feature], ax=ax)
        ax.set_title(f'{model_name}: Partial Dependence on {top1_feature}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # Learning Curve (XGBoost only)
        if model_name == 'XGBoost':
            results = best_model.evals_result()
            epochs = len(results['validation_0']['rmse'])

            plt.figure(figsize=(8, 5))
            plt.plot(range(epochs), results['validation_0']['rmse'], label='Train RMSE', color='blue')
            plt.plot(range(epochs), results['validation_1']['rmse'], label='Validation RMSE', color='red')
            plt.axvline(x=best_model.best_iteration, color='gray', linestyle='--', label='Best Iteration (Early Stopping)')
            plt.legend()
            plt.title('XGBoost Learning Curve')
            plt.xlabel('Number of Trees (Epochs)')
            plt.ylabel('RMSE (Error)')
            plt.tight_layout()
            plt.show()

    return history_df

"""# Regression 4. Data Setup"""

data_dict = {
    'X_train': X_train_reg, 'y_train': y_train_reg,
    'X_val': X_val_reg, 'y_val': y_val_reg,
    'X_test': X_test_reg, 'y_test': y_test_reg
}

"""# Regression 5. Multiple Linear Regression"""

_ = run_regression_pipeline(LinearRegression(), 'Linear Regression', data_dict, tuning_type=None)

"""# Regression 6. Random Forest Regressor"""

# Predefined list tuning
rf_params = [
    {'max_depth': 5, 'min_samples_leaf': 5},
    {'max_depth': 10, 'min_samples_leaf': 5},
    {'max_depth': 10, 'min_samples_leaf': 3},
    {'max_depth': 15, 'min_samples_leaf': 3},
    {'max_depth': None, 'min_samples_leaf': 1}
]
rf_history = run_regression_pipeline(RandomForestRegressor(), 'Random Forest', data_dict, tuning_type='rf_manual', param_space=rf_params)

"""# Regression 7. XGBoost Regressor"""

# RandomizedSearchCV tuning
xgb_param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}
xgb_base_model = XGBRegressor(n_estimators=100, random_state=1461, n_jobs=-1)
xgb_history = run_regression_pipeline(xgb_base_model, 'XGBoost', data_dict, tuning_type='xgb_random', param_space=xgb_param_dist)

"""# Comparison. Find the Top 8 Combination"""

# Extract Top 8 Combinations
final_results = pd.concat([rf_history, xgb_history], ignore_index=True)
top8 = final_results.sort_values(by='RMSE', ascending=True).head(8)
top8.index = range(1, len(top8) + 1)

# Prevent truncation of long hyperparameter
pd.set_option("display.max_colwidth", None)
print("\n[Final Comparison: Top 8 Combination]")
print(top8)