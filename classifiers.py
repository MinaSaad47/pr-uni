from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score

class PolyRegModel:
    def __init__(self, train_data):
        self.__lr = LinearRegression()
        self.__x_train = train_data.iloc[:, :-1]
        self.__y_train = train_data.iloc[:, -1]
        pf = PolynomialFeatures(degree=5)
        self.__x_train = pf.fit_transform(self.__x_train)

    def train(self):
            self.__lr.fit(self.__x_train, self.__y_train)


    def predict(self, test_data):
        self.__x_test = test_data.iloc[:, :-1]
        self.__y_test = test_data.iloc[:, -1]
        pf = PolynomialFeatures(degree=5)
        self.__x_test = pf.fit_transform(self.__x_test)
        self.__y_pred = self.__lr.predict(self.__x_test)

    def test_accuracy(self):
        return self.__lr.score(self.__x_test, self.__y_pred)

    def r2_score(self):
        return r2_score(self.__y_test, self.__y_pred)


from sklearn.tree import  DecisionTreeClassifier
class DecisionTreeModel:
    def __init__(self, train_data):
        self.__model = DecisionTreeClassifier()
        self.__x_train = train_data.iloc[:, :-1]
        self.__y_train = train_data.iloc[:, -1]

    def train(self):
            self.__model.fit(self.__x_train, self.__y_train)


    def predict(self, test_data):
        self.__x_test = test_data.iloc[:, :-1]
        self.__y_test = test_data.iloc[:, -1]
        self.__y_pred = self.__model.predict(self.__x_test)

    def test_accuracy(self):
        return self.__model.score(self.__x_test, self.__y_test)

    def conf_matrix(self):
        return confusion_matrix(self.__y_test, self.__y_pred)

    def r2_score(self):
        return r2_score(self.__y_test, self.__y_pred)

from sklearn.linear_model import LogisticRegression
class LogisticRegressionModel:
    def __init__(self, train_data):
        self.__model = LogisticRegression()
        self.__x_train = train_data.iloc[:, :-1]
        self.__y_train = train_data.iloc[:, -1]

    def train(self):
        self.__model.fit(self.__x_train, self.__y_train)


    def predict(self, test_data):
        self.__x_test = test_data.iloc[:, :-1]
        self.__y_test = test_data.iloc[:, -1]
        self.__y_pred = self.__model.predict(self.__x_test)

    def test_accuracy(self):
        return self.__model.score(self.__x_test, self.__y_test)

    def conf_matrix(self):
        return confusion_matrix(self.__y_test, self.__y_pred)

    def r2_score(self):
        return r2_score(self.__y_test, self.__y_pred)

from sklearn.svm import SVC
class SVMModel:
    def __init__(self, train_data):
        self.__model = SVC(probability=True, kernel='rbf', gamma='auto', C=1)
        self.__x_train = train_data.iloc[:, :-1]
        self.__y_train = train_data.iloc[:, -1]

    def train(self):
            self.__model.fit(self.__x_train, self.__y_train)


    def predict(self, test_data):
        self.__x_test = test_data.iloc[:, :-1]
        self.__y_test = test_data.iloc[:, -1]
        self.__y_pred = self.__model.predict(self.__x_test)

    def test_accuracy(self):
        return self.__model.score(self.__x_test, self.__y_test)

    def conf_matrix(self):
        return confusion_matrix(self.__y_test, self.__y_pred)

    def r2_score(self):
        return r2_score(self.__y_test, self.__y_pred)
