from keras.models import load_model
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.preprocessing.text import Tokenizer

from sklearn.utils import shuffle

from IPython.display import HTML

from itertools import chain
from keras.utils import plot_model
import numpy as np
import pandas as pd
import random
import json
import re

RANDOM_STATE = 50
TRAIN_FRACTION = 0.7

def format_sequence(s):
    """Add spaces around punctuation and remove references to images/citations."""
    
    # Add spaces around punctuation
    s =  re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', s)
    
    # Remove references to figures
    s = re.sub(r'\((\d+)\)', r'', s)
    
    # Remove double spaces
    s = re.sub(r'\s\s', ' ', s)
    return s

def get_data(file, training_length = 50, lower = False):
    filters='!"%;[\\]^_`{|}~\t\n'
    data = pd.read_csv(file,parse_dates=['patent_date']).dropna(subset = ['patent_abstract'])
    abstracts = [format_sequence(a) for a in list(data['patent_abstract'])]
    word_idx, idx_word, num_words, word_counts, texts, sequences, features, labels = make_sequences(abstracts, training_length, lower, filters)

    X_train, X_valid, y_train, y_valid = createTrainingData(features, labels, num_words)

    training_dict = {'X_train': X_train, 'X_valid': X_valid, 
                         'y_train': y_train, 'y_valid': y_valid}

    return training_dict, word_idx, idx_word, sequences


def make_sequences(texts, training_length, lower = True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    tokenizer = Tokenizer(lower=lower, filters=filters)
    tokenizer.fit_on_texts(texts)
    
    word_idx = tokenizer.word_index
    idx_word = tokenizer.index_word
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts
    
    sequences = tokenizer.texts_to_sequences(texts)
    seq_lengths = [len(x) for x in sequences]
    over_idx = [i for i, l in enumerate(seq_lengths) if l > (training_length + 20)]
    
    new_texts = []
    new_sequences = []
    
    for i in over_idx:
        new_texts.append(texts[i])
        new_sequences.append(sequences[i])
        
    features = []
    labels = []
    
    
    for seq in new_sequences:
        
        # Create multiple training examples from each sequence
        for i in range(training_length, len(seq)):
            # Extract the features and label
            extract = seq[i - training_length: i + 1]
            
            # Set the features and label
            features.append(extract[:-1])
            labels.append(extract[-1])
    
    print(f'There are {len(features)} sequences.')
    
    # Return everything needed for setting up the model
    return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, features, labels
    



def createTrainingData(features,labels,num_words,train_fraction=0.7):
    
    ## TO shuffle the data in Random State
    features, labels = shuffle(features, labels, random_state=RANDOM_STATE)
    
    train_end = int(train_fraction*len(labels))  # Dividing the data in 70/30 Ratio.
    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])
    
    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]
    
    X_train, X_valid = np.array(train_features), np.array(valid_features)
    
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)
    
    for example_index, word_index in enumerate(train_labels):      # Creating a one hot encoder vector with the size of number of training feature set*Num_words
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1
        
    import gc
    gc.enable()
    del features, labels, train_features, valid_features, train_labels, valid_labels
    gc.collect()

    return X_train, X_valid, y_train, y_valid
