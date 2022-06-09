import numpy as np
from sklearn.model_selection import train_test_split

# from sklearn import datasets
from sklearn import svm
from dataset import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
from Vectorizers import tfid_vectorizer, count_vectorizer
from Models import gausian_NB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score


def split_data(
    path: str = "data/booksummaries1.txt",
    test_size: float = 0.2,
    random_state: int = 0,
    categories=["Science Fiction", "Fantasy", "Mystery", "Historical novel"],
):
    b: Dataset = Dataset(path=path, categories=categories)
    b.load_csv(have_header=True)
    # Replace category names with indexes
    le = LabelEncoder()
    X = b.df["Summary"].values
    # pd.Summary conversion to np.array
    y = b.df["Genres"].apply(lambda genre_str: genre_str.split(";")).values
    y = np.concatenate(y).ravel()
    # Creating dictionary for categories
    le.fit(y)
    # Convert y to indexes
    X_train, X_test, y_train, y_test = train_test_split(
        X, le.transform(y), test_size=test_size, random_state=random_state
    )
    # Return data and dictionary
    return (X_train, X_test, y_train, y_test, le.classes_)


def train_vectorizer(vectorizer, bag_of_words=[]):
    vec = vectorizer.fit_transform(bag_of_words)
    return (vectorizer, vec)


def prepare_testing_vector(vectorizer, bag_of_words=[]):
    return vectorizer.transform(bag_of_words)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, classes = split_data(path="data/books_four_genres.txt")
    model = gausian_NB()
    vectorizer, X_train_vector = train_vectorizer(count_vectorizer(max_fatures=500), X_train)
    X_test_vector = prepare_testing_vector(vectorizer, X_test)
    model.fit(X_train_vector.toarray(), y_train)
    predictions = model.predict(X_test_vector.toarray())
    prediction_matrix = confusion_matrix(predictions, y_test)
    result = pd.DataFrame(prediction_matrix, columns=classes, index=classes)

    print(result)
    print(accuracy_score(predictions, y_test, normalize=True))

    pass
