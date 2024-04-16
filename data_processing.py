
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import string


# nltk.download("wordnet")
# nltk.download('stopwords')

lemma = WordNetLemmatizer()

# Addtional Punctuation and Strings not included in Strings.punctuation list but occur in the datasets
PUNCT = {'...', '``', "''", '-lrb-', '-rrb-', '--', '//', '/ /', '\\/', '!?', '?!'} | set(string.punctuation)

# Extra terms that are added to the stoplist
EXTRA_TERMS = {"'s", "'re", "'ll", "'ve", "'wo", "e.t.c", "e.t", "i.e", "'le"}

# Words that cause Negations
NEGATIONS = {'not', "n't", 'never', 'hardly', 'barely', 'rarely', 'nowhere', 'without' , 'nor', 'neither', "'nt"}
# Words that increase the magnitude of others
INTENSIFIERS = {'very', 'really', 'extremely', 'amazingly', 'exceptionally', 'incredibly', 'particularly', 'remarkably', 'unusually',
                'absolutely', 'exceptionally', 'completely', 'vividly', 'ultimately', 'passionately', 'profoundly'}

# Adds new terms and removes Negation and Intensifier terms to the stoplist
STOPS = (set(stopwords.words('english')) - NEGATIONS - INTENSIFIERS) | EXTRA_TERMS

class Pre_Process():

    def __init__(self, data, num_classes):

        '''
        Initalisation

        data: data passed in to be processed
        classes: number of classes (default = 3)
        '''
        self.data = data
        self.classes = num_classes

    # Preprocesses the train and dev sets
    # review[0] is the review id
    # review[1] is the review string
    # review[2] is the sentiment value
    # Returns the processed data
    def pre_process(self):
        '''
        Preprocesses the train and dev sets

        review[0] is the review id
        review[1] is the review string
        review[2] is the sentiment value

        Returns the processed data
        '''
        for review in self.data:
            # Preprocess review
            review[1] = review[1].lower().split()
            review[1] = self.remove_stops(review[1])     
            review[1] = self.apply_rules(review[1])
            review[1] = self.exclamatives(review[1])
            review[1] = self.remove_punctuation(review[1])

            # Map Sentiment
            if self.classes == 3 and len(review) > 2:
                review[2] = self.set_sentiment(review[2])
            
        return self.data

    # Applies preprocessing to the test data
    def pre_process_test(self, test):
        for review in test:
            review[1] = remove_unwanted(review[1].lower())
            review[1] = set(review[1].split())
        return test

    # sets sentiment values
    def set_sentiment(self, sentiment_val):
        '''
        Maps the sentiment score for 3 class

        sentiment_val = class/sentiment value (0 <= x <= 5)

        Returns new sentiment value
        '''
        if sentiment_val > 2: 
            return 0
        elif sentiment_val < 2:
            return 2
        else:
            return 1

    # Removes punctuation from the review string
    def remove_punctuation(self, review):
        review_set = set(review)
        # Checks if any punctuation is in the review
        removal = review_set & PUNCT

        # Removes any character from above using symmetric difference operator
        # This also creates binarization since we return the review set
        return removal ^ review_set

    # Removes items in the stop list
    def remove_stops(self, review):
        # Gets the relevant terms
        removal = set(review) & STOPS
        return [term for term in review if term not in removal]

    # Applies lemmaisation, negation rule and intensifier rule on the review
    def apply_rules(self, review):
        '''
        Applies rules to the review

        Rules:
        Negation Rule add 'NOT_' prefix to the next word
        Intensifier Rule add 'INT_' prefix to the next word
        Apllies Lemmaisation

        returns processed review string
        '''
        # Check if review contains either negations or intensifiers
        for t in range(len(review)-1):
            # Apply lemmaisation
            review[t] = lemma.lemmatize(review[t])
            # Apply Negations
            if review[t] in NEGATIONS:
                # Checks double negations
                if (review[t+1] not in NEGATIONS) and (review[t+1] not in PUNCT):
                    review[t+1] = 'NOT_' + review[t+1]

            # Apply Intensifiers
            if (review[t] in INTENSIFIERS) and (review[t+1] not in PUNCT):
                review[t+1] = 'INT_' + review[t+1]
                
        return review

    # Applies the Exclamative Rule
    # Applies the 'EX_' prefix to the preceding word
    def exclamatives(self, review):
        # Checks for exclamation marks 
        for t in range(len(review)):
            if review[t].__contains__('!') & (t-1 > 0):
                review[t-1] = 'EX_' + review[t-1]
        return review


    # N-gram feature selection
    # NOT USED
    def ngram(self, review):
        n = 2
        new_review = set()
        if n > 1:
            for t in range(len(review)):
                splice = t+(n-1)
                if splice < len(review):
                    splice = len(review)
            
                new_review.add(review[t:splice])

        else:
            return review

        return new_review


            


