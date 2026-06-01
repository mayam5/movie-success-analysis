# TMDB Movie Dataset - Data Preprocessing
# remove print, grape output // only print final result( Columns , dataset) 

import ast
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")


# 1. Load CSV Files
# If you change the file location, please update the file path.

movies_path = '/kaggle/input/datasets/simyoung321/dataset/tmdb_5000_movies.csv'
credits_path = '/kaggle/input/datasets/simyoung321/dataset/tmdb_5000_credits.csv'

movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)


# 2. Merge Data

credits = credits.rename(columns={'movie_id': 'id'})
df = movies.merge(credits, on='id')


# 3. Basic Inspection

num_cols = [
    'budget',
    'revenue',
    'runtime',
    'popularity',
    'vote_average',
    'vote_count'
]


# 4. Missing Value Handling

for col in ['homepage', 'tagline']:
    if col in df.columns:
        df = df.drop(columns=[col])

if 'overview' in df.columns:
    df['overview'] = df['overview'].fillna('')

if 'release_date' in df.columns:
    df = df.dropna(subset=['release_date'])

runtime_median = df['runtime'].median()
df['runtime'] = df['runtime'].fillna(runtime_median)


# 5. Zero Value Handling

for col in ['budget', 'revenue']:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

df['runtime'] = df['runtime'].replace(0, df['runtime'].median())

df = df[df['vote_count'] > 0]
df = df[df['vote_average'] > 0]


# 6. Invalid Data Handling

df = df[df['status'] == 'Released']

if 'title_y' in df.columns:
    df = df.drop(columns=['title_y'])

if 'title_x' in df.columns:
    df = df.rename(columns={'title_x': 'title'})


# 7. Duplicate Data Handling

df = df.drop_duplicates(subset=['title'])


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


# 9. Feature Engineering

df['crew_count'] = df['crew_names'].apply(len)
df['cast_count'] = df['cast_names'].apply(len)

df['main_genre'] = df['genres_names'].apply(
    lambda x: x[0] if len(x) > 0 else np.nan
)


def get_director(crew_list):
    for person in crew_list:
        if isinstance(person, dict) and person.get('job') == 'Director':
            return person.get('name')
    return np.nan


df['director'] = df['crew'].apply(get_director)

df['release_date'] = pd.to_datetime(df['release_date'])
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month

df['profit'] = df['revenue'] - df['budget']


# 10. Remove Unnecessary Columns

drop_cols = [
    'genres',
    'keywords',
    'production_companies',
    'production_countries',
    'spoken_languages',
    'cast',
    'crew',

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

    'overview',
    'homepage',
    'tagline',

    'title',
    'original_title',

    'release_date',

    'id',
    'poster_path',
    'status'
]

drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=drop_cols)


# 11. Encoding

top_lang = df['original_language'].value_counts().head(8).index

df['original_language'] = df['original_language'].apply(
    lambda x: x if x in top_lang else 'Other'
)

df = pd.get_dummies(
    df,
    columns=['original_language'],
    dtype=int
)


mlb_genre = MultiLabelBinarizer()

genre_ohe = pd.DataFrame(
    mlb_genre.fit_transform(df['genres_names']),
    columns=mlb_genre.classes_,
    index=df.index
)

df = pd.concat([df, genre_ohe], axis=1)
df = df.drop(columns=['genres_names'])


country_counter = Counter()

for country_list in df['production_countries_names']:
    country_counter.update(country_list)

top_countries = [
    country
    for country, count in country_counter.most_common(10)
]

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


# 12. Cast and Director Frequency Encoding

cast_counter = Counter()

for cast_list in df['cast_names']:
    cast_counter.update(cast_list)

df['cast_freq_sum'] = df['cast_names'].apply(
    lambda x: sum(cast_counter[name] for name in x)
)

df['cast_freq_mean'] = df['cast_names'].apply(
    lambda x: np.mean([cast_counter[name] for name in x]) if len(x) > 0 else 0
)

df = df.drop(columns=['cast_names'])

director_freq = df['director'].value_counts()

df['director_freq'] = df['director'].map(director_freq)
df['director_freq'] = df['director_freq'].fillna(0)

df = df.drop(columns=['director'])


# 13. Success Label Creation

df['success'] = (df['profit'] > 0).astype(int)


# 14. Log Transformation

log_cols = [
    'budget',
    'vote_count',
    'popularity',
    'revenue',
    'profit'
]

for col in log_cols:
    df[col] = np.log1p(df[col])


# 15. Final Scaling

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


# 16. Final Column Check

print("\n== Final Columns ==")

for i, col in enumerate(df_scaled.columns):
    print(f"{i + 1:2d}. {col}")


# 17. Classification Dataset Split

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


# 18. Regression Dataset Split

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


# 19. Optional: Save Preprocessed Data

# If you need preprocessed data, uncomment the following lines.

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
