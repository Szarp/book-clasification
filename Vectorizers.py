from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def count_vectorizer(max_fatures: int = 1500, min_df: float = 5, max_df: float = 0.7):
    return CountVectorizer(max_features=max_fatures, min_df=min_df, max_df=max_df)


def tfid_vectorizer(max_fatures: int = 1500, min_df: float = 5, max_df: float = 0.7):
    return TfidfVectorizer(max_features=max_fatures, min_df=min_df, max_df=max_df)
