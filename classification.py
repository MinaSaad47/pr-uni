#!/usr/bin/python3
import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

from loading_data import load_classification_data
from preprocessing import Preprocessor
from classifiers import *

TRAIN_MOVIE_DIRECTOR_FILE = 'movie-director.csv'
TRAIN_MOVIE_REVENUE_FILE  = 'movies-revenue-classification.csv'
TRAIN_MOVIE_ACTOR_FILE    = 'movie-voice-actors.csv'

TEST_MOVIE_DIRECTOR_FILE = 'movie-director-test-samples.csv'
TEST_MOVIE_REVENUE_FILE  = 'movies-revenue-test-samples.csv'
TEST_MOVIE_ACTOR_FILE    = 'movie-voice-actors-test-samples.csv'

CLS = {
    "svm": ("SVM Model", SVMModel),
    "dt":  ("Decission Tree Model", DecisionTreeModel),
    "pr":  ("Polynominal Regressition Model", PolyRegModel),
    "lr":  ("Logistic Regression Model", LogisticRegressionModel),
}

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Movie Revenue Classifier")
    sub_parser = arg_parser.add_subparsers(title='action', help='Action to perform', required=True, dest='action')

    train_parser = sub_parser.add_parser('train')
    train_parser.add_argument('train_directory', type=str, help='Directory containing train dataset')
    train_parser.add_argument('-c', '--classifier', type=str, choices=['dt', 'lr', 'svm', 'pr'], required=True, help='Classifier type', dest='classifier')
    train_parser.add_argument('-o', '--output-pic', type=str, required=True, help='Output file to picke model to', dest='pic_file')

    train_parser = sub_parser.add_parser('test')
    train_parser.add_argument('test_directory', type=str, help='Directory containing test dataset')
    train_parser.add_argument('-i', '--input-pic', type=str, required=True, help='Input file to pickled model', dest='pic_file')

    train_parser = sub_parser.add_parser('bench')
    train_parser.add_argument('train_directory', type=str, help='Directory containing train dataset')
    train_parser.add_argument('test_directory', type=str, help='Directory containing test dataset')
    train_parser.add_argument('-p', '--png', help='Output PNG file', action='store_true', dest='png_flag')
    train_parser.add_argument('-w', '--win', help='Output Window', action='store_true', dest='win_flag')
    parsed_args = arg_parser.parse_args()

    # Train Action
    if parsed_args.action == 'train':
        TRAIN_MOVIE_DIRECTOR_FILE = os.path.join(parsed_args.train_directory, TRAIN_MOVIE_DIRECTOR_FILE)
        TRAIN_MOVIE_REVENUE_FILE = os.path.join(parsed_args.train_directory, TRAIN_MOVIE_REVENUE_FILE)
        TRAIN_MOVIE_ACTOR_FILE = os.path.join(parsed_args.train_directory, TRAIN_MOVIE_ACTOR_FILE)

        train_data = load_classification_data(director_file=TRAIN_MOVIE_DIRECTOR_FILE,
                               actor_file=TRAIN_MOVIE_ACTOR_FILE,
                               revenue_file=TRAIN_MOVIE_REVENUE_FILE)

        train_pp = Preprocessor()
        train_data = train_pp.preprocess_classification(train_data)

        cls = parsed_args.classifier

        print(CLS[cls][0])
        cls_model = CLS[cls][1](train_data)

        cls_model.train()

        with open(parsed_args.pic_file, 'wb') as pic:
            pickle.dump((train_data, cls_model), pic)


    # Test Action
    elif parsed_args.action == 'test':
        TEST_MOVIE_DIRECTOR_FILE = os.path.join(parsed_args.test_directory, TEST_MOVIE_DIRECTOR_FILE)
        TEST_MOVIE_REVENUE_FILE = os.path.join(parsed_args.test_directory, TEST_MOVIE_REVENUE_FILE)
        TEST_MOVIE_ACTOR_FILE = os.path.join(parsed_args.test_directory, TEST_MOVIE_ACTOR_FILE)

        test_data = load_classification_data(director_file=TEST_MOVIE_DIRECTOR_FILE,
                              actor_file=TEST_MOVIE_ACTOR_FILE,
                              revenue_file=TEST_MOVIE_REVENUE_FILE)

        with open(parsed_args.pic_file, 'rb') as pic:
            (train_data, cls_model) = pickle.load(pic)

        test_pp = Preprocessor(dataset_type='test')
        test_data = test_pp.preprocess_classification(test_data, train_data)

        cls_model.predict(test_data)
        print(f"Accuracy: {cls_model.test_accuracy()}")

    # Bench Action
    elif parsed_args.action == 'bench':
        TRAIN_MOVIE_DIRECTOR_FILE = os.path.join(parsed_args.train_directory, TRAIN_MOVIE_DIRECTOR_FILE)
        TRAIN_MOVIE_REVENUE_FILE = os.path.join(parsed_args.train_directory, TRAIN_MOVIE_REVENUE_FILE)
        TRAIN_MOVIE_ACTOR_FILE = os.path.join(parsed_args.train_directory, TRAIN_MOVIE_ACTOR_FILE)

        train_data = load_classification_data(director_file=TRAIN_MOVIE_DIRECTOR_FILE,
                               actor_file=TRAIN_MOVIE_ACTOR_FILE,
                               revenue_file=TRAIN_MOVIE_REVENUE_FILE)

        train_pp = Preprocessor()
        train_data = train_pp.preprocess_classification(train_data)

        TEST_MOVIE_DIRECTOR_FILE = os.path.join(parsed_args.test_directory, TEST_MOVIE_DIRECTOR_FILE)
        TEST_MOVIE_REVENUE_FILE = os.path.join(parsed_args.test_directory, TEST_MOVIE_REVENUE_FILE)
        TEST_MOVIE_ACTOR_FILE = os.path.join(parsed_args.test_directory, TEST_MOVIE_ACTOR_FILE)

        test_data = load_classification_data(director_file=TEST_MOVIE_DIRECTOR_FILE,
                              actor_file=TEST_MOVIE_ACTOR_FILE,
                              revenue_file=TEST_MOVIE_REVENUE_FILE)

        test_pp = Preprocessor(dataset_type='test')
        test_data = test_pp.preprocess_classification(test_data, train_data)

        train_times = []
        test_times = []
        accuracies = []

        for i, model in enumerate(CLS.keys()):
            print(f"[{i + 1}] {CLS[model][0]}:")
            cls = CLS[model][1](train_data)
            # timing train
            stime = time.time()
            cls.train()
            etime = time.time()
            train_time = etime - stime
            print(f'\tTrain Time: {train_time} Sec')
            train_times.append(train_time)

            # timing test
            stime = time.time()
            cls.predict(test_data)
            etime = time.time()
            test_time = etime - stime
            print(f'\tTest Time: {test_time} Sec')
            test_times.append(test_time)

            # calculating accuracy
            accuracy = cls.test_accuracy()
            print(f'\tAccuracy Time: {accuracy} %')
            accuracies.append(accuracy)

        pos = np.arange(len(CLS.keys()))

        if parsed_args.png_flag or parsed_args.win_flag:
            fig = plt.figure()
            plt.title("Train Times")
            plt.xlabel('Models')
            plt.ylabel('Time in Sec')
            plt.bar(pos, train_times)
            plt.xticks(pos, CLS.keys())
            if parsed_args.png_flag:
                plt.savefig('train_times.png')
            if parsed_args.win_flag:
                plt.show()

            fig = plt.figure()
            plt.title("Test Times")
            plt.xlabel('Models')
            plt.ylabel('Time in Sec')
            plt.bar(pos, test_times)
            plt.xticks(pos, CLS.keys())
            if parsed_args.png_flag:
                plt.savefig('test_times.png')
            if parsed_args.win_flag:
                plt.show()


            fig = plt.figure()
            plt.title("Accuracies")
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.bar(pos, accuracies)
            plt.xticks(pos, CLS.keys())
            if parsed_args.png_flag:
                plt.savefig('accuracies.png')
            if parsed_args.win_flag:
                plt.show()

