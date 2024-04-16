
import numpy as np
from collections import Counter
from NB_sentiment_analyser import *

class Naive_Bayes_Classify:
    def __init__(self, data, num_classes):
        '''
        Initalisation

        train: training dictionary also referenced as sent_dict
        classes: number of classes (default = 3)
        priors: array containing the prior probabilities for each class
        posteriors: array containing the prior probabilities for each class
        lengths: array containing the size of reviews for each class
        '''
        
        self.data = data
        self.classes = num_classes
        self.train = self.create_train_dict()
        self.priors = self.get_priors()
        self.posteriors = []
        self.lengths = self.class_lengths()
        self.distinct = self.distinct_features()
    
    # Calculates Priors
    def get_priors(self):
        # print(self.train)
        priors = np.zeros(self.classes)
        # Gets amount of reviews in class and total number of reviews
        for reviews in self.data:
            sent_val = reviews[2]
            review = reviews[1]
            priors[sent_val] += 1

        # Gets the total amount of reviews
        total = len(self.data)

        # calculate priors
        priors = [priors[c] / total for c in range(self.classes)]
        return priors
    
    # Calculates likelihoods for a single review then converts them to posteriors for each class
    def get_posteriors(self, test_review):
        self.posteriors = np.ones(self.classes)
        for t in test_review:
            for c in range(self.classes):
                # Gets counts of terms in class and in total
                class_length = self.lengths[c]
                class_occurences = np.sum([Counter(r)[t] for r in self.train[c] if t in r])

                # Calculates likelihood and posteriors
                # replace 1 in denominator with self.distinct to use distinct features
                self.posteriors[c] *= ((class_occurences + 1) / (class_length + self.distinct)) 
                # self.posteriors[c] *= ((class_occurences + 1) / (class_length + 1)) 


        return self.posteriors

    # Classifys review by getting the max class probability using argmax
    def classify_review(self):
        probs = [self.priors[c] * self.posteriors[c] for c in range(self.classes)]
        return np.argmax(probs)

    # Evaluate
    def evaluate(self, results, data, sentiment):
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for review in data:
            review_id = review[0]
            result = results[review_id]
            if review[2] == sentiment:
                # If belongs to class
                # if results == expected == sentiment
                if result == sentiment:
                    tp += 1
                # if results != sentiment and expected == sentiment    
                else:
                    fn += 1
            else:
                # If it doesn't belong to class
                # expected != sentiment and results == sentiment
                if result == sentiment:
                    fp += 1
                # expected != sentiment and results != sentiment
                else:
                    tn += 1

        return tp, fp, tn, fn

    # Calculates f score for a single class
    def f_measure(self, tp, fp, tn, fn):
        denominator = (2*tp)+fp+fn

        # Catches dividing by 0
        if denominator != 0:
            return (2*tp)/denominator
        return 0

    # Gets the length of all reviews in the class
    def class_lengths(self):
        lengths = np.zeros(self.classes)
        for review in self.data:
            lengths[review[2]] += len(review[1])
        
        return lengths  

    # Creates a dictionary based of training data
    # sent_dict = {sentiment/class : [{review strings}, ]}
    def create_train_dict(self):
        sent_dict = dict()
        for review in self.data:
            # Create Dictionary
            if review[2] in sent_dict:
                sent_dict[review[2]].append(review[1])
            else:
                sent_dict[review[2]] = [review[1]]
        return sent_dict

    # Gets the number of distinct features in reviews
    def distinct_features(self):
        distinct = 0
        sets = set()
        for review in self.data:
            sets.update(review[1])
        return len(sets)

            