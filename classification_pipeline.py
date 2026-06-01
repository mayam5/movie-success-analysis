#data preprocessing

import pandas as pd
import numpy as np

print("Library loaded successfully!")

# Load two CSV files
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print("movies shape  :", movies.shape)
print("credits shape :", credits.shape)

# Merge two datasets based on movie_id
credits = credits.rename(columns={'movie_id': 'id'})
df = movies.merge(credits, on='id')

print("Shape after merge :", df.shape)
df.head(3)

# Check basic information
print(" Basic Dataset Information ")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print()

# Check data types for each column
print(" == Data Types == ")
print(df.dtypes)

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

# Feature Engineering

# Profit
df['profit'] = df['revenue'] - df['budget']

# Remove unnecessary columns

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

# target ->  revenue

# Dataset for Regression
# Remove 'revenue' and 'profit' to prevent data leakage

X_reg = df_scaled.drop(columns=[
    'revenue',
    'profit'
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

print("Train :", X_train_reg.shape)
print("Val   :", X_val_reg.shape)
print("Test  :", X_test_reg.shape)

# Classification Libraries
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Main classification experiment pipeline
# This function performs:
# - success label generation
# - hold-out split
# - stratified k-fold validation
# - hyperparameter search
# - top 5 model selection
# - final test evaluation

def classification_pipeline(df_scaled):

    # Save Results
    all_results = []

    # SUCCESS VERSION LOOP
    success_versions = [
        'profit_success',
        'revenue_top30'
    ]

    for success_version in success_versions:
        print("SUCCESS VERSION :", success_version)

        # Version A
        # success = profit > 0
        # imbalance handled with
        # SMOTE / class_weight='balanced'
        if success_version == 'profit_success':

            temp_df = df_scaled.copy()

            temp_df['success'] = (
                temp_df['profit'] > 0
            ).astype(int)

            imbalance_mode = 'balanced'

        # Version B
        # success = revenue top 30%
        # NO SMOTE / NO class_weight
        else:
            temp_df = df_scaled.copy()
            threshold = temp_df['revenue'].quantile(0.7)
            temp_df['success'] = (
                temp_df['revenue'] >= threshold
            ).astype(int)
            imbalance_mode = 'top30_only'

        # Feature / Target
        X = temp_df.drop(columns=[
            'success',
            'profit',
            'revenue'
        ])

        y = temp_df['success']

        print("\nClass Distribution")

        print(y.value_counts())

        # Hold-out Split
        # Test set never used during training
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        print("\nTrain Shape :", X_train.shape)
        print("Test Shape  :", X_test.shape)

        # KFold Candidates
        kfold_list = [2,3,5,7]

        # KNN Parameter Grid
        knn_param_grid = {
            'n_neighbors': [3,5,7,9],
            'weights': [
                'uniform',
                'distance'
            ],
            'metric': [
                'euclidean',
                'manhattan'
            ]
        }
        
        # Decision Tree Parameter Grid
        dt_param_grid = {
            'max_depth': [3,5,7,10],
            'min_samples_split': [2,5,10],
            'min_samples_leaf': [1,2,5],
            'criterion': [
                'gini',
                'entropy'
            ]
        }

        # KFold Loop
        for kfold_n in kfold_list:

            print(f"\nKFold = {kfold_n}")

            skf = StratifiedKFold(
                n_splits=kfold_n,
                shuffle=True,
                random_state=42
            )

            # KNN LOOP
            for n in knn_param_grid['n_neighbors']:
                for w in knn_param_grid['weights']:
                    for m in knn_param_grid['metric']:

                        # Version A (SMOTE)
                        if imbalance_mode == 'balanced':
                            pipeline = Pipeline([
                                ('scaler', StandardScaler()),
                                ('smote', SMOTE(
                                    random_state=42
                                )),
                                ('model', KNeighborsClassifier(
                                    n_neighbors=n,
                                    weights=w,
                                    metric=m
                                ))
                            ])

                        # Version B (NO SMOTE)
                        else:
                            pipeline = Pipeline([
                                ('scaler', StandardScaler()),
                                ('model', KNeighborsClassifier(
                                    n_neighbors=n,
                                    weights=w,
                                    metric=m
                                ))
                            ])

                        # Cross Validation
                        scores = cross_val_score(
                            pipeline,
                            X_train,
                            y_train,
                            cv=skf,
                            scoring='f1_macro'
                        )

                        mean_f1 = scores.mean()

                        all_results.append({
                            'success_version':
                                success_version,
                            'imbalance_mode':
                                imbalance_mode,
                            'model':
                                'KNN',
                            'kfold':
                                kfold_n,
                            'params': {
                                'n_neighbors': n,
                                'weights': w,
                                'metric': m
                            },
                            'cv_f1':
                                mean_f1,
                            'pipeline':
                                pipeline,
                            'X_train':
                                X_train,
                            'y_train':
                                y_train,
                            'X_test':
                                X_test,
                            'y_test':
                                y_test
                        })

            # Decision Tree LOOP
            for depth in dt_param_grid['max_depth']:
                for split in dt_param_grid['min_samples_split']:
                    for leaf in dt_param_grid['min_samples_leaf']:
                        for criterion in dt_param_grid['criterion']:

                            # Version A (class_weight='balanced')
                            if imbalance_mode == 'balanced':
                                model = DecisionTreeClassifier(
                                    max_depth=depth,
                                    min_samples_split=split,
                                    min_samples_leaf=leaf,
                                    criterion=criterion,
                                    class_weight='balanced',
                                    random_state=42
                                )

                            # Version B (NO class_weight)
                            else:
                                model = DecisionTreeClassifier(
                                    max_depth=depth,
                                    min_samples_split=split,
                                    min_samples_leaf=leaf,
                                    criterion=criterion,
                                    random_state=42
                                )

                            # Cross Validation
                            scores = cross_val_score(
                                model,
                                X_train,
                                y_train,
                                cv=skf,
                                scoring='f1'
                            )

                            mean_f1 = scores.mean()
                            all_results.append({
                                'success_version' : success_version,
                                'imbalance_mode' : imbalance_mode,
                                'model' : 'DecisionTree',
                                'kfold' : kfold_n,
                                'params': {
                                    'max_depth': depth,
                                    'min_samples_split': split,
                                    'min_samples_leaf': leaf,
                                    'criterion': criterion
                                },
                                'cv_f1' : mean_f1,
                                'pipeline' : model,
                                'X_train' : X_train,
                                'y_train' : y_train,
                                'X_test' : X_test,
                                'y_test' : y_test
                            })

    # Result DataFrame
    results_df = pd.DataFrame(all_results)

    # TOP 5 Selection
    top5 = results_df.sort_values(

        by='cv_f1',

        ascending=False

    ).head(5)

    print("\n===================================")
    print(" TOP 5 MODELS ")
    print("===================================")

    print(

        top5[[
            'success_version',
            'imbalance_mode',
            'model',
            'kfold',
            'params',
            'cv_f1'
        ]]

    )

    # FINAL HOLD-OUT TEST EVALUATION
    print("\n===================================")
    print(" FINAL TEST EVALUATION ")
    print("===================================")

    final_results = []

    for idx, row in top5.iterrows():
        model = row['pipeline']
        X_train = row['X_train']
        y_train = row['y_train']
        X_test = row['X_test']
        y_test = row['y_test']

        # Train on FULL TRAIN SET
        model.fit(
            X_train,
            y_train
        )

        # Final Prediction
        pred = model.predict(X_test)

        test_f1 = f1_score(
            y_test,
            pred
        )

        test_acc = accuracy_score(
            y_test,
            pred
        )

        print("\n-----------------------------------")

        print("Success Version :",
              row['success_version'])

        print("Imbalance Mode :",
              row['imbalance_mode'])

        print("Model :",
              row['model'])

        print("KFold :",
              row['kfold'])

        print("Params :",
              row['params'])

        print("Test Accuracy :",
              round(test_acc,4))

        print("Test F1-score :",
              round(test_f1,4))

        print("\nClassification Report")

        print(classification_report(
            y_test,
            pred
        ))

        print("\nConfusion Matrix")

        print(confusion_matrix(
            y_test,
            pred
        ))

        final_results.append({

            'success_version' : row['success_version'],

            'imbalance_mode' : row['imbalance_mode'],

            'model' : row['model'],

            'kfold' : row['kfold'],

            'params' : row['params'],

            'cv_f1' : row['cv_f1'],

            'test_f1' : test_f1,

            'test_accuracy' : test_acc
        })

    # Final Ranking
    final_results_df = pd.DataFrame(final_results)

    final_results_df = final_results_df.sort_values(

        by='test_f1',

        ascending=False
    )

    print("\n===================================")
    print(" FINAL TOP 5 ")
    print("===================================")

    print(final_results_df)

    return results_df, final_results_df

# RUN PIPELINE
all_results, top5_results = classification_pipeline(
    df_scaled
)
