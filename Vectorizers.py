from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def count_vectorizer(max_fatures: int = 1500, min_df: float = 1, max_df: float = 0.3,ngram_range=(1,1)):
    return CountVectorizer(max_features=max_fatures, min_df=min_df, max_df=max_df,ngram_range=ngram_range)


def tfid_vectorizer(max_fatures: int = 1500, min_df=0.02,  max_df=0.45,ngram_range=(1,1)):
    return TfidfVectorizer(max_features=max_fatures,ngram_range=ngram_range,sublinear_tf=True,use_idf=False)
# , min_df=min_df, max_df=max_df