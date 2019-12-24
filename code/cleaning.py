# Import libraries
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
import re
import string

# Open dataframe
movie_df = pd.read_csv('csvs/wiki_movie_plots.csv')

# Drop duplicates (same Wiki page)
movie_df = movie_df.drop_duplicates(subset = ['Wiki Page'], keep = 'first').reset_index(drop = True)

# Text preprocessing

# Remove apostrophes
remove_apostrophes = lambda x: x.replace('\'', '')

# Keep only letters
remove_numbers = lambda x: ' '.join(re.sub('\w*\d\w*', ' ', x).split())

# Remove new line characters
no_new_line = lambda x: x.replace('\n',' ')

# Make them lowercase and remove punctuation
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x.lower()).strip()

# Get part of speech for lemmatization
def get_wordnet_pos(word):
    ''' 
    Map POS tag to first character lemmatize() accepts.
    '''
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Lemmatization function
def lemmatizer(text):
    '''
    Lemmatizes a given string.
    '''
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in word_tokenize(text)]
    lemmatized_text = ' '.join(tokens)
    return lemmatized_text

# Named Entity function
def named_entities(text):
    '''
    Replaces all named entities
    before vectorization.
    '''
    for k, v in entities.items():
        text = text.replace(k, v)
    return text

# Get rid of spaces on the ends of titles
movie_df["Title"] = movie_df["Title"].str.strip()

# Clean punctuation
movie_df["Plot"] = movie_df["Plot"].map(remove_apostrophes).map(remove_numbers).map(punc_lower)

# Lemmatization
movie_df["Plot"] = movie_df["Plot"].apply(lambda x: lemmatizer(x))

# Fix the dataset typo
movie_df.iloc[14640,1] = "The Conjuring"

with open('cleaned_data','wb') as file:
	pickle.dump(movie_df, file)