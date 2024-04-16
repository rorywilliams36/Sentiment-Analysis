
class Feature_Test:

    # Extract most common words from test/dev
    def extract_dfs_from_test(self, data, most_common):
        for reviews in data:
            rel_terms = set()
            review = reviews[1]
            for term in review:
                if term in most_common:
                    rel_terms.add(term)

            if len(rel_terms) > 0:
                reviews[1] = rel_terms

        return data

    # calculates the tfidfs for data and removes lowest scores from the review
    def get_tfidfs_test(self, data, tfs, idfs):
        for reviews in data:
            tfidfs = dict()
            for term in reviews[1]:
                if term in tfs and term in idfs:
                    tfidfs[term] = tfs[term] * idfs[term]
                else:
                    tfidfs[term] = 0
            
            #print(tfidfs)
            # Removes the worst tfidf scores from the review
            sorted_tfidfs = set(sorted(tfidfs, reverse=False)[:int(len(reviews[1])/3)])
            if len(reviews[1]) > 2:
                reviews[1] = reviews[1] - sorted_tfidfs

        return data 

    # Extracts features from review using a threshold value
    # Removes terms from review if tfidf score is lower than thresholds
    def get_tfidfs_threshold(self, data, tfs, idfs):
        removal = set()
        for review in data:
            for term in review[1]:
                if term in tfs and term in idfs:
                    tfidfs = tfs[term] * idfs[term]
                    if tfidfs < 55 and len(review[1]) > 2:
                        removal.add(term)
            
            review[1] = review[1] - removal

        return data

    # Extracts features from review using a threshold value
    # Removes terms from review if chi score is lower than thresholds
    def extract_relevant_chi_test(self, data, chi):
        removal = set()
        for review in data:
            for term in review[1]:
                if term in chi:
                    if chi[term] < 1.4 and len(review[1]) > 2:
                        removal.add(term)
                        
            review[1] = review[1] - removal

        return data