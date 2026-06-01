# TMDB Movie Dataset - Data Preprocessing

import ast
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


# 1. Load CSV Files
# If you change the file location, please update the file path.

movies_path = '/kaggle/input/datasets/simyoung321/dataset/tmdb_5000_movies.csv'
credits_path = '/kaggle/input/datasets/simyoung321/dataset/tmdb_5000_credits.csv'

movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

print("movies shape  :", movies.shape)
print("credits shape :", credits.shape)


# 2. Merge Data

credits = credits.rename(columns={'movie_id': 'id'})
df = movies.merge(credits, on='id')

print("Shape after merge:", df.shape)


# 3. Basic Inspection

print("\n== Basic Dataset Information ==")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\n== Data Types ==")
print(df.dtypes)

print("\n== NULL Missing Value Status ==")
missing = df.isnull().sum()
print(missing[missing > 0])

num_cols = [
    'budget',
    'revenue',
    'runtime',
    'popularity',
    'vote_average',
    'vote_count'
]

print("\n== Zero Value Status ==")
for col in num_cols:
    print(f"{col:20s} : {(df[col] == 0).sum()} items")

print("\n== Negative Value Status ==")
for col in num_cols:
    print(f"{col:20s} : {(df[col] < 0).sum()} items")

print("\n== Status Value Counts ==")
print(df['status'].value_counts())

print("\n== Original Language Value Counts ==")
print(df['original_language'].value_counts().head(10))


# 4. Missing Value Handling

# Remove unnecessary columns if they exist
for col in ['homepage', 'tagline']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Replace missing overview with empty string
if 'overview' in df.columns:
    df['overview'] = df['overview'].fillna('')

# Remove rows with missing release_date
if 'release_date' in df.columns:
    df = df.dropna(subset=['release_date'])

# Fill missing runtime with median
runtime_median = df['runtime'].median()
df['runtime'] = df['runtime'].fillna(runtime_median)

print("\n== Missing values after NULL handling ==")
print(df.isnull().sum()[df.isnull().sum() > 0])


# 5. Zero Value Handling

# Replace 0 values in budget and revenue with NaN, then fill with median
for col in ['budget', 'revenue']:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Replace 0 runtime with median
df['runtime'] = df['runtime'].replace(0, df['runtime'].median())

# Remove rows where vote_count or vote_average is 0
df = df[df['vote_count'] > 0]
df = df[df['vote_average'] > 0]

print("\nShape after handling 0 values:", df.shape)


# 6. Invalid Data Handling

# Keep only released movies
df = df[df['status'] == 'Released']

# Clean duplicated title columns after merge
if 'title_y' in df.columns:
    df = df.drop(columns=['title_y'])

if 'title_x' in df.columns:
    df = df.rename(columns={'title_x': 'title'})

print("\nShape after handling invalid inputs:", df.shape)
print(df['status'].value_counts())


# 7. Duplicate Data Handling

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicated titles:", df.duplicated(subset=['title']).sum())

df = df.drop_duplicates(subset=['title'])

print("Shape after removing duplicates:", df.shape)


# 8. JSON Column Processing

json_columns = [
    'genres',
    'keywords',
    'production_companies',
    'production_countries',
    'spoken_languages',
    'cast',
    'crew'
]


def json_to_list(text):
    try:
        return ast.literal_eval(text)
    except Exception:
        return []


for col in json_columns:
    df[col] = df[col].apply(json_to_list)

    df[col + '_names'] = df[col].apply(
        lambda x: [
            item['name']
            for item in x
            if isinstance(item, dict) and 'name' in item
        ]
    )

    df[col + '_str'] = df[col + '_names'].apply(
        lambda x: ', '.join(x)
    )

print("\nJSON column processing completed!")


# 9. Feature Engineering

# Number of crew and cast members
df['crew_count'] = df['crew_names'].apply(len)
df['cast_count'] = df['cast_names'].apply(len)

# Main genre
df['main_genre'] = df['genres_names'].apply(
    lambda x: x[0] if len(x) > 0 else np.nan
)


# Extract director
def get_director(crew_list):
    for person in crew_list:
        if isinstance(person, dict) and person.get('job') == 'Director':
            return person.get('name')
    return np.nan


df['director'] = df['crew'].apply(get_director)

# Release year and month
df['release_date'] = pd.to_datetime(df['release_date'])
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month

# Profit
df['profit'] = df['revenue'] - df['budget']

print("\nFeature engineering completed!")


# 10. Remove Unnecessary Columns

# overview is removed because it needs NLP processing.
# genre information can partially represent movie themes and characteristics.

drop_cols = [
    # Original JSON columns
    'genres',
    'keywords',
    'production_companies',
    'production_countries',
    'spoken_languages',
    'cast',
    'crew',

    # String/list columns not used directly
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

drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=drop_cols)

print("\nUnnecessary columns removed successfully!")
print(df.shape)


# 11. Encoding

# 11-1. original_language One-Hot Encoding

top_lang = df['original_language'].value_counts().head(8).index

df['original_language'] = df['original_language'].apply(
    lambda x: x if x in top_lang else 'Other'
)

df = pd.get_dummies(
    df,
    columns=['original_language'],
    dtype=int
)

print("\nAfter original_language encoding:", df.shape)


# 11-2. Genre MultiLabel One-Hot Encoding

mlb_genre = MultiLabelBinarizer()

genre_ohe = pd.DataFrame(
    mlb_genre.fit_transform(df['genres_names']),
    columns=mlb_genre.classes_,
    index=df.index
)

df = pd.concat([df, genre_ohe], axis=1)
df = df.drop(columns=['genres_names'])

print("After genre encoding:", df.shape)


# 11-3. Production Country MultiLabel One-Hot Encoding

country_counter = Counter()

for country_list in df['production_countries_names']:
    country_counter.update(country_list)

top_countries = [
    country
    for country, count in country_counter.most_common(10)
]

print("Top countries:", top_countries)

df['production_countries_names'] = df['production_countries_names'].apply(
    lambda x: [
        c if c in top_countries else 'Other'
        for c in x
    ]
)

mlb_country = MultiLabelBinarizer()

country_ohe = pd.DataFrame(
    mlb_country.fit_transform(df['production_countries_names']),
    columns=mlb_country.classes_,
    index=df.index
)

df = pd.concat([df, country_ohe], axis=1)
df = df.drop(columns=['production_countries_names'])

print("After country encoding:", df.shape)


# 12. Cast and Director Frequency Encoding

# Cast frequency encoding
cast_counter = Counter()

for cast_list in df['cast_names']:
    cast_counter.update(cast_list)

print("Number of actors:", len(cast_counter))

df['cast_freq_sum'] = df['cast_names'].apply(
    lambda x: sum(cast_counter[name] for name in x)
)

df['cast_freq_mean'] = df['cast_names'].apply(
    lambda x: np.mean([cast_counter[name] for name in x]) if len(x) > 0 else 0
)

df = df.drop(columns=['cast_names'])

# Director frequency encoding
director_freq = df['director'].value_counts()

df['director_freq'] = df['director'].map(director_freq)
df['director_freq'] = df['director_freq'].fillna(0)

df = df.drop(columns=['director'])

print("\nCast and director frequency encoding completed!")


# 13. Graph Visualization

numeric_graph_cols = [
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

for col in numeric_graph_cols:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col], bins=30)
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()


# Genre count graph
genre_cols = list(mlb_genre.classes_)

genre_count = df[genre_cols].sum().sort_values(ascending=False)

genre_count.plot(
    kind='bar',
    figsize=(12, 6),
    title='Genre Counts'
)

plt.ylabel('Movie Count')
plt.show()


# Country count graph
country_cols = list(mlb_country.classes_)

country_count = df[country_cols].sum().sort_values(ascending=False)

country_count.plot(
    kind='bar',
    figsize=(10, 5),
    title='Production Country Counts'
)

plt.ylabel('Movie Count')
plt.show()


# Language count graph
language_cols = [
    col
    for col in df.columns
    if col.startswith('original_language_')
]

language_count = df[language_cols].sum().sort_values(ascending=False)

language_count.plot(
    kind='bar',
    figsize=(10, 5),
    title='Original Language Counts'
)

plt.ylabel('Movie Count')
plt.show()


# 14. Skewness Check Before Log Transformation

exclude_cols = language_cols + genre_cols + country_cols

numeric_cols_for_skew = df.select_dtypes(
    include=['int64', 'float64']
).columns

numeric_cols_for_skew = [
    col
    for col in numeric_cols_for_skew
    if col not in exclude_cols
]


def skew_level(x):
    if abs(x) < 0.5:
        return 'Normal'
    elif abs(x) < 1:
        return 'Moderate'
    elif abs(x) < 2:
        return 'High'
    else:
        return 'Very High'


skew_df = pd.DataFrame({
    'column': numeric_cols_for_skew,
    'skew': df[numeric_cols_for_skew].skew()
})

skew_df['abs_skew'] = skew_df['skew'].abs()
skew_df = skew_df.sort_values(by='abs_skew', ascending=False)
skew_df['level'] = skew_df['skew'].apply(skew_level)

print("\n== Skewness before log transformation ==")
print(skew_df[['column', 'skew', 'level']])


# 15. Success Label Creation

df['success'] = (df['profit'] > 0).astype(int)

print("\nSuccess label counts:")
print(df['success'].value_counts())


# 16. Log Transformation

log_cols = [
    'budget',
    'vote_count',
    'popularity',
    'revenue',
    'profit'
]

for col in log_cols:
    df[col] = np.log1p(df[col])

print("\nLog Transformation complete")

for col in log_cols:
    plt.figure(figsize=(10, 4))
    plt.hist(df[col], bins=30)
    plt.title(f'{col} After Log Transform')
    plt.show()


# 17. Skewness Check After Log Transformation

numeric_cols_for_skew = df.select_dtypes(
    include=['int64', 'float64']
).columns

numeric_cols_for_skew = [
    col
    for col in numeric_cols_for_skew
    if col not in exclude_cols
]

skew_df = pd.DataFrame({
    'column': numeric_cols_for_skew,
    'skew': df[numeric_cols_for_skew].skew()
})

skew_df['abs_skew'] = skew_df['skew'].abs()
skew_df = skew_df.sort_values(by='abs_skew', ascending=False)
skew_df['level'] = skew_df['skew'].apply(skew_level)

print("\n== Skewness after log transformation ==")
print(skew_df[['column', 'skew', 'level']])


# 18. Scaler Comparison

scale_cols = [
    'budget',
    'runtime',
    'popularity',
    'vote_average',
    'vote_count'
]

standard = StandardScaler().fit_transform(df[scale_cols])
minmax = MinMaxScaler().fit_transform(df[scale_cols])
robust = RobustScaler().fit_transform(df[scale_cols])

fig, axes = plt.subplots(3, 5, figsize=(18, 10))
fig.suptitle('Scaler Comparison', fontsize=16, fontweight='bold')

scalers = [standard, minmax, robust]
scaler_names = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']

for row, (data, name) in enumerate(zip(scalers, scaler_names)):
    for col_idx, col_name in enumerate(scale_cols):
        axes[row][col_idx].hist(
            data[:, col_idx],
            bins=50,
            edgecolor='white',
            alpha=0.85
        )
        axes[row][col_idx].set_title(
            f'{name}\n{col_name}',
            fontsize=9
        )

plt.tight_layout()
plt.show()


# 19. Final Scaling

# - budget, popularity, vote_count : many outliers and right-skewed distribution → RobustScaler
# - runtime, vote_average          : close to normal distribution → StandardScaler

robust_cols = [
    'budget',
    'popularity',
    'vote_count'
]

standard_cols = [
    'runtime',
    'vote_average'
]

df_scaled = df.copy()

robust_scaler = RobustScaler()
df_scaled[robust_cols] = robust_scaler.fit_transform(df[robust_cols])

standard_scaler = StandardScaler()
df_scaled[standard_cols] = standard_scaler.fit_transform(df[standard_cols])

print("\n=== Scaling Completed ===")
print(df_scaled[robust_cols + standard_cols].describe().round(3))


# 20. Final Null Check

print("\n== Final Null Check ==")
print(df_scaled.isnull().sum()[df_scaled.isnull().sum() > 0])
print("Total NULL count:", df_scaled.isnull().sum().sum())


# 21. Final Column Check

print("\n== Final Columns ==")

for i, col in enumerate(df_scaled.columns):
    print(f"{i + 1:2d}. {col}")


# 22. Classification Dataset Split

# Classification target: success
# Remove success, profit, revenue to prevent leakage.

X_cls = df_scaled.drop(columns=[
    'success',
    'profit',
    'revenue'
])

y_cls = df_scaled['success']

X_train_cls, X_temp_cls, y_train_cls, y_temp_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.3,
    random_state=42,
    stratify=y_cls
)

X_val_cls, X_test_cls, y_val_cls, y_test_cls = train_test_split(
    X_temp_cls,
    y_temp_cls,
    test_size=0.5,
    random_state=42,
    stratify=y_temp_cls
)

print("\n== Classification Dataset ==")
print("X_cls:", X_cls.shape)
print("y_cls:", y_cls.shape)
print("Train:", X_train_cls.shape)
print("Val  :", X_val_cls.shape)
print("Test :", X_test_cls.shape)


# 23. Regression Dataset Split

# Regression target: revenue
# Remove revenue, profit, success to prevent leakage.

X_reg = df_scaled.drop(columns=[
    'revenue',
    'profit',
    'success'
])

y_reg = df_scaled['revenue']

X_train_reg, X_temp_reg, y_train_reg, y_temp_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.3,
    random_state=42
)

X_val_reg, X_test_reg, y_val_reg, y_test_reg = train_test_split(
    X_temp_reg,
    y_temp_reg,
    test_size=0.5,
    random_state=42
)

print("\n== Regression Dataset ==")
print("X_reg:", X_reg.shape)
print("y_reg:", y_reg.shape)
print("Train:", X_train_reg.shape)
print("Val  :", X_val_reg.shape)
print("Test :", X_test_reg.shape)


# 24. Optional: Save Preprocessed Data

# If you need preprocessed data(.CSV FILE), uncomment the following lines.

# df_scaled.to_csv('tmdb_preprocessed.csv', index=False)

# X_train_cls.to_csv('X_train_cls.csv', index=False)
# X_val_cls.to_csv('X_val_cls.csv', index=False)
# X_test_cls.to_csv('X_test_cls.csv', index=False)

# y_train_cls.to_csv('y_train_cls.csv', index=False)
# y_val_cls.to_csv('y_val_cls.csv', index=False)
# y_test_cls.to_csv('y_test_cls.csv', index=False)

# X_train_reg.to_csv('X_train_reg.csv', index=False)
# X_val_reg.to_csv('X_val_reg.csv', index=False)
# X_test_reg.to_csv('X_test_reg.csv', index=False)

# y_train_reg.to_csv('y_train_reg.csv', index=False)
# y_val_reg.to_csv('y_val_reg.csv', index=False)
# y_test_reg.to_csv('y_test_reg.csv', index=False)
