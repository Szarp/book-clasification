from sklearn.linear_model import LogisticRegression

# model=LogisticRegression()
# model.fit(X_train_vectorized, y_train)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB


def gausian_NB():
    return GaussianNB()


def tree():
    return tree.DecisionTreeClassifier()


def svm(C: float = 1.0, kernel: str = "linear", degree: int = 3, gamma: str = "auto"):
    return SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
