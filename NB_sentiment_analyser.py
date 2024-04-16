# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

"""
import argparse
import string
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from classify import *
from data_processing import *
from feature_train import *
from feature_test import *

USER_ID = "acb20rtw"

TEST = False # Set to true to use unlablled data

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args

# Get the counts used for confusion matrix
def get_matrix_counts(results, data, classes):
    counts = np.zeros((classes, classes))
    for review in data:
        review_id = review[0]
        result = results[review_id]
        counts[result][review[2]] += 1
                
    return counts

def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test

    # Load Files
    train_data = pd.read_csv(f'moviereviews/{training}',  sep='\t').values.tolist()
    dev_data = pd.read_csv(f'moviereviews/{dev}', sep='\t').values.tolist()
    test_data = pd.read_table(f'moviereviews/{test}').values.tolist()

    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    # Intialise Classes and Preprocess data
    # IF TEST DATA IS USED
    if TEST:
        process_test = Pre_Process(test_data, number_classes)

    # IF DEV DATA IS USED
    else:
        process_test = Pre_Process(dev_data, number_classes)

    process_test.data = process_test.pre_process()

    process_tr = Pre_Process(train_data, number_classes)
    train = process_tr.pre_process()

    if features == 'features':
        # Initialise classes for feature selection/extraction
        extract_tr = Feature_Train(train, number_classes)
        extract_test = Feature_Test()

        train = extract_tr.extract_relevant_terms()
        # train = extract_tr.extract_relevant_tfidfs_threshold()
        # train = extract_tr.extract_relevant_chi()

        process_test.data = extract_test.extract_dfs_from_test(process_test.data, extract_tr.most_common)
        # process_test.data = extract_test.get_tfidfs_threshold(process_test.data, extract_tr.tfs, extract_tr.idfs)
        # process_test.data = extract_test.extract_relevant_chi_test(process_test.data, extract_tr.chi)


    # Inintalise Naive Bayes Classification Class
    bayes = Naive_Bayes_Classify(train, number_classes)

    # Stores sentence id : predicted sentiment
    labels = dict()

    # Classify Dev Data
    for d in process_test.data:
        bayes.posteriors = bayes.get_posteriors(d[1])
        labels[d[0]] = bayes.classify_review()
    
    # Evaluations
    # Calculates F1 Scores
    f1_score = 0
    f_scores = 0
    if not TEST:
        totals = 0
        for c in range(number_classes):
            tp, fp, tn, fn = bayes.evaluate(labels, process_test.data, c)
            totals += tp
            f_scores += bayes.f_measure(tp, fp, tn, fn)

        # Macro F1-Score
        f1_score = f_scores / number_classes
    
    # Displays Confusion Matrix
    if confusion_matrix and not TEST:   
        counts = get_matrix_counts(labels, process_test.data, number_classes)
        sns.heatmap(counts, annot=True, cmap="crest", fmt='.0f')
        plt.xlabel('Expected Class')
        plt.ylabel('Predicted Class')
        plt.title(f'Confusion Matrix for {number_classes} Class Sentiment Analysis')
        plt.show()

    # Write results to file
    # Uncomment to write to a file

    # results = pd.DataFrame(list(labels.items()), columns=['Sentence ID', 'Sentiment'])
    # results.to_csv('test_predictions_3classes_acb20rtw.tsv', sep='\t', index=False)

    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()
