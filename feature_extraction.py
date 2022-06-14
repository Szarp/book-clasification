import numpy as np
from sklearn.model_selection import train_test_split
from dataset import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
from Vectorizers import tfid_vectorizer, count_vectorizer
from Models import gausian_NB,svm,svm_grid
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, SelectFwe,SelectFpr,SelectPercentile, chi2
import matplotlib.pyplot as plt

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
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

if __name__ == "__main__":
    useChi:bool = True
    k_best:int = 200
    X_train, X_test, y_train, y_test, classes = split_data(path="data/books_four_genres.txt")
    # model = gausian_NB()
    model = svm_grid()
    for k_best in range(100,1500,200):
    # for alpha in np.arange(10,60,5):
        vectorizer, X_train_vector = train_vectorizer(tfid_vectorizer(max_fatures=1500,ngram_range=(1,2)), X_train)
        X_test_vector = prepare_testing_vector(vectorizer, X_test)
        
        kbest = SelectKBest(score_func = chi2, k = k_best).fit(X_train_vector, y_train)
        # kbest = SelectFwe(chi2, alpha=alpha).fit(X_train_vector, y_train)
        # kbest = SelectPercentile(chi2, percentile=alpha).fit(X_train_vector, y_train)
        # kbest = SelectFpr(chi2, alpha=alpha).fit(X_train_vector, y_train)
        if useChi:
            X_train_vector = kbest.transform(X_train_vector)
            X_test_vector = kbest.transform(X_test_vector)
        model.fit(X_train_vector.toarray(), y_train)
        # X_new = SelectKBest(chi2, k=10).fit_transform(X, y)
        # predictions = model.predict(X_test_vector.toarray())
        plot_grid_search(model.cv_results_,[1,4],['linear', 'rbf'],"C","Kernel")
        # prediction_matrix = confusion_matrix(predictions, y_test)
        # result = pd.DataFrame(prediction_matrix, columns=classes, index=classes)

        # print(result)
        # print("params " + model.best_params_)

## {'C': 1, 'kernel': 'linear'}

        # print("estimator " +model.best_estimator_)
        # print(f'Accurency: {accuracy_score(predictions, y_test, normalize=True)} Kbest:{k_best}, vector shape: {X_train_vector.shape}')
        # print(f'Accurency: {accuracy_score(predictions, y_test, normalize=True)} Kbest:{alpha}, vector shape: {X_train_vector.shape}')

    pass
