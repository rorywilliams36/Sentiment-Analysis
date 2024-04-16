import numpy as np
import math

'''
Class used for feature selection on the training data
Had to make two separate files for train and test data as they are both different data structures
'''

class Feature_Train():
    def __init__(self, data, number_classes):
        '''
        Initalisation

        ---Params---

        data: training data
        sent_dict: dictionary storing sentiment and its reviews
        classes: number of classes (default = 3)
        NUM_DOCS: total number of reviews
        tfs: dictionary containing term frequencies
        idfs: dictionary containing idf values for terms
        dfs: dictiionary containing document frequencies for terms
        all_tfidfs: dictionary containing all tfidf values for terms
        most_common: list of 2500 terms in the dataset
        '''

        self.data = data
        self.classes = number_classes
        self.NUM_DOCS = len(data)
        self.tfs = self.term_freqs()
        # self.idfs = self.get_idfs()
        # self.dfs = self.document_freqs()
        # self.all_tfidfs = dict()

        # self.chi = self.chi_square()
        self.most_common = self.get_most_relevant()
        # self.most_common = self.get_most_common()


# ------------- Term/Document Frequency Selection -------------


    # Gets term frequencies for every term
    def term_freqs(self):
        self.tfs = dict()
        for reviews in self.data:
            review = reviews[1]
            for word in review:
                if word not in self.tfs:
                    self.tfs[word] = 1
                else:
                    self.tfs[word] += 1

        return self.tfs

    # Gets the document frequency of a term
    # Function is not used
    def document_freqs(self):
        self.dfs = dict()
        for term in self.tfs:
            for review in self.data:
                if term in review[1]:
                    if term in self.dfs:
                        self.dfs[term] += 1
                    else:
                        self.dfs[term] = 1

        return self.dfs    

    
    # Gets the 2500 most common words from tfs
    def get_most_common(self):
        sorted_tfs = sorted(self.tfs.items(), key=lambda x:x[1], reverse=True)[:2500]
        return [term[0] for term in sorted_tfs]

    # Get relevant terms
    # Ignores terms with too high and too low frequencies
    def get_most_relevant(self):
        sorted_tfs = sorted(self.tfs.items(), key=lambda x:x[1], reverse=True)
        top = 180
        if sorted_tfs[11][1] < top:
            top = sorted_tfs[11][1]
        bottom = 4
        relevant = [term[0] for term in sorted_tfs if term[1] < top and term[1] > bottom]
        return relevant[:2500]
 
    # Extract most common words from test/dev
    # Removes these words from the review
    def extract_relevant_terms(self, temp = None):
        for reviews in self.data:
            rel_terms = set()
            review = reviews[1]
            for term in review:
                if temp == None:
                    if term in self.most_common:
                        rel_terms.add(term)
                else:
                    if term in self.all_tfidfs:
                        rel_terms.add(term)

            if len(rel_terms) > 0:
                reviews[1] = rel_terms

        return self.data

# ------------- Tf.idf -------------

    # calculates idf for each term in the data
    def get_idfs(self):
        self.idfs = dict()
        for term in self.tfs:
            df = 0
            for review in self.data:
                if term in review[1]:
                    df += 1

            if df != 0:
                self.idfs[term] = math.log10(self.NUM_DOCS/df)
            else:
                self.idfs[term] = 0

        return self.idfs

    # calculates the tfidfs for data and removes lowest scores from the review
    def extract_relevant_tfidfs(self):
        for reviews in self.data:
            tfidfs = dict()
            for term in reviews[1]:
                if term in self.tfs and term in self.idfs:
                    tfidfs[term] = self.tfs[term] * self.idfs[term]
                else:
                    tfidfs[term] = 0
            
            # Removes the worst tfidf scores from the review
            sorted_tfidfs = set(sorted(tfidfs, reverse=False)[:int(len(reviews[1])/3)])
            if len(reviews[1]) > 2:
                reviews[1] = reviews[1] - sorted_tfidfs

        return self.data

    # Extracts features from review using a threshold value
    # Removes terms from review if tfidf score is lower than thresholds
    def extract_relevant_tfidfs_threshold(self):
        removal = set()
        for review in self.data:
            for term in review[1]:
                if term in self.tfs and term in self.idfs:
                    tfidfs = self.tfs[term] * self.idfs[term]
                    if tfidfs < 55 and len(review[1]) > 2:
                        removal.add(term)
            
            review[1] = review[1] - removal

        return self.data

    # Calculates tfidf for every term in the dataset at once
    def get_all_tfidfs(self):
        self.all_tfidfs = dict()
        for term in self.tfs:
            if (term in self.tfs) and (term in self.idfs):
                self.all_tfidfs[term] = self.tfs[term] * self.idfs[term]
            else:
                self.all_tfidfs[term] = 0
        
        self.all_tfidfs = set(sorted(self.all_tfidfs, reverse=True)[:2500])

        return self.all_tfidfs

# ------------- Chi Square -------------

    # Extracts features from review using a threshold value
    # Removes terms from review if chi score is lower than thresholds
    def extract_relevant_chi(self):
        removal = set()
        for review in self.data:
            for term in review[1]:
                if self.chi[term] < 1.4 and len(review[1]) > 2:
                    removal.add(term)
        
            review[1] = review[1] - removal

        return self.data

    # calculates chi square values for all terms and stores in dictionary
    def chi_square(self):
        self.chi = dict()
        contingency = self.create_contingency()
        contingency_list = list(contingency.values())

        total = np.sum(contingency_list)
        col = np.sum(contingency_list, axis=0)

        for term in contingency:
            if term not in self.chi:
                self.chi[term] = 0

            row = np.sum(contingency[term])

            for c in range(self.classes):
                expected_freq = (row * col[c]) /total

                self.chi[term] += (contingency[term][c] - expected_freq)**2 / expected_freq

        return self.chi
    
    # Creates contingency table for terms
    def create_contingency(self):
        contingency = dict()
        for review in self.data:
            for term in review[1]:
                if term not in contingency:
                    contingency[term] = np.zeros(self.classes)

                contingency[term][review[2]] += 1 

        return contingency