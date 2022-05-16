from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import math
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

class Preprocessor:
    def __init__(self, dataset_type='train'):
        self.__dataset_type = dataset_type

    def __preprocess_genre(self, total_data):
        # Handling `Genre` with Mode and OneHotEncoder
        if total_data['Genre'].isnull().all():
            total_data['Genre'] = 0
            return total_data
        elif not total_data['Genre'].notnull().all():
            total_data['Genre'].fillna(total_data['Genre'].mode()[0], inplace=True)
        ohe = OneHotEncoder()
        genre = ohe.fit_transform(np.expand_dims(total_data['Genre'], axis=1)).toarray()
        total_data.drop(columns='Genre', inplace=True)
        total_data[ohe.categories_[0]] = genre
        target_column = total_data['MovieSuccessLevel']
        total_data.drop(columns='MovieSuccessLevel', inplace=True)
        total_data = pd.concat([total_data, target_column], axis=1)
        return total_data


    def __preprocess_mpaa(self, total_data):
        # Handling `MPAA` with mode
        if total_data['MPAA'].isnull().all():
            total_data['MPAA'] = 0
            return total_data
        elif not total_data['MPAA'].notnull().all():
            total_data['MPAA'].fillna(total_data['MPAA'].mode()[0], inplace=True)
        le = LabelEncoder()
        total_data['MPAA'] = le.fit_transform(total_data['MPAA'])
        return total_data

    def __preprocess_date(self, total_data):
        # Handling `Date` with Linear Regression
        if total_data['Date'].isnull().all():
            total_data['Date'] = 0
            return total_data

        total_data['Date'] = pd.to_datetime(total_data['Date'], errors='coerce').apply(lambda dt: dt.timestamp() if not pd.isnull(dt) else dt)
        if not total_data['Date'].notnull().all():
            le = LabelEncoder()
            lr = LinearRegression()
            total_data['Actor'] = le.fit_transform(total_data['Actor'])
            prep_data = total_data[['Actor', 'Date']]
            test_data = prep_data[prep_data['Date'].isnull()]
            train_data = prep_data.dropna()
            y_train = train_data['Date']
            x_train = train_data.drop('Date', axis=1)
            lr.fit(x_train, y_train)
            x_test = test_data.drop('Date', axis=1)
            y_pred = lr.predict(x_test)
            prep_data.loc[prep_data['Date'].isnull(), 'Date'] = y_pred
            total_data[['Date', 'Actor']] = prep_data[['Date', 'Actor']]

        return total_data

    def __preprocess_title_and_character_and_director_actor(self, total_data):
        # Handling `Title` & `Character` with label encoding
        le = LabelEncoder()
        if total_data['Actor'].notnull().all():
            total_data['Actor'] = le.fit_transform(total_data['Actor'])
        else:
            total_data['Actor'] = 0

        if total_data['Title'].notnull().all():
            total_data['Title'] = le.fit_transform(total_data['Title'])
        else:
            total_data['Title'] = 0

        if total_data['Character'].notnull().all():
            total_data['Character'] = le.fit_transform(total_data['Character'])
        else:
            total_data['Character'] = 0

        if total_data['Director'].notnull().all():
            total_data['Director'] = le.fit_transform(total_data['Director'])
        else:
            total_data['Director'] = 0

        return total_data

    def __preprocess_movie_success_level(self, total_data):
        le = LabelEncoder()
        total_data['MovieSuccessLevel'][total_data['MovieSuccessLevel'].notnull()] = le.fit_transform(total_data['MovieSuccessLevel'][total_data['MovieSuccessLevel'].notnull()])
        if total_data['MovieSuccessLevel'].notnull().all():
            total_data['MovieSuccessLevel'] = total_data['MovieSuccessLevel'].astype(int)
            return total_data
        prep_data = total_data
        prep_data.head(n=10)
        test_data = prep_data[prep_data['MovieSuccessLevel'].isnull()]
        train_data = prep_data.dropna()
        y_train = train_data['MovieSuccessLevel']
        x_train = train_data.drop('MovieSuccessLevel', axis=1)
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        x_test = test_data.drop('MovieSuccessLevel', axis=1)
        y_pred = lr.predict(x_test)
        y_pred = np.array([math.floor(x) for x in y_pred])
        prep_data.loc[prep_data['MovieSuccessLevel'].isnull(), 'MovieSuccessLevel'] = y_pred
        prep_data['MovieSuccessLevel'] = prep_data['MovieSuccessLevel'].astype(int)
        total_data = prep_data
        return total_data

    def preprocess(self, total_data, train_data=None):
        total_data = self.__preprocess_genre(total_data)
        total_data = self.__preprocess_mpaa(total_data)
        total_data = self.__preprocess_date(total_data)
        total_data = self.__preprocess_title_and_character_and_director_actor(total_data)
        total_data = self.__preprocess_movie_success_level(total_data)
        if self.__dataset_type == 'test':
            for feature in train_data.columns:
                if not feature in total_data.columns:
                    total_data.insert(train_data.columns.get_loc(feature), feature, train_data[feature])
                    total_data[feature] = 0
        return total_data

