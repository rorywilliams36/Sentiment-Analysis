# Sentiment Analysis Assignment

## Installation

Navigate to the directory that the code is found and run  

```pip install -r requirements.txt```  

On first run you must uncomment both ```nltk.download('xxx')``` lines in the .\data_processing.py file (lines 9 and 10) to install the stoplist and lemmaization packages.
After which you can comment them out again

## Usage

To run program navigate to relevant directory and run  

```python .\NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes {3, 5} -features {features/all_words} {-confusion_matrix}```  


If you wish to evaluate on test data (dataset with no sentiment labels) go into the .\NB_sentiment_analyser.py and set the constant ```TEST = True``` at line 20. False as default

## Experiments
The program is set ot our propesed method using document frequencies.  

To pick features to use this is located in the .\NB_sentiment_analyser.py lines 85-96 where each option is commented out. If you wish to use any options you must uncomment the same line for both the training and testing data.  

If you wish to use chi square go to the .\Feature_Train.py file and uncomment ```self.chi = xxx``` in the initalisation  

For tfidf uncomment self.tfs, self.idfs and self.dfs lines under the same location above.  

To use 2500 most common terms uncomment ```self.most_common = self.get_most_common()``` in the same file  

