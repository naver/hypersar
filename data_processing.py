import pandas as pd
import numpy as np
import torch
import pickle
import gzip
import io
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

def load_data(file, delimiter):
    load_data = pd.read_csv(file, delimiter=delimiter)
    return load_data

def collect_token_set(data, options):
    tokens = set() # Set of unique words or characters
    for _, row in data.iterrows():
        query = row['query']
        if query == "-" or query == "": # Recommendation instance: no query
            continue
        query = query.split(" ")
        for token in query:
            print(token)
            tokens.add(token)
    return tokens

def load_w2v_model(vectorizer, options):
    w2v_path = "w2v/" + options.w2v_dir
    data_dir = options.data_dir
    tokens = vectorizer.get_feature_names()

    print("Loading the raw word2vec model...", datetime.now(), flush=True)
    raw_w2v_model = {}
    for line in io.open(w2v_path + os.sep + "raw_w2v", "r", encoding="utf-8", newline="\n", errors="ignore"):
        line_split = line.rstrip().split(" ")
        raw_w2v_model[line_split[0]] = np.asarray([float(e) for e in line_split[1:]])
    w2v_dim = len(list(raw_w2v_model.values())[0]) # Take the embedding size from any element

    print("Restricting the raw word2vec model to identified vocabulary...", datetime.now(), flush=True)
    w2v_model = {}
    for token in tokens:
        if token in raw_w2v_model:
            w2v_model[token] = raw_w2v_model[token]
        else:
            # Randomly initialize the embeddings for out-of-vocabulary tokens
            w2v_model[token] = np.random.randn(w2v_dim)
    w2v_filename = "w2v_" + str(options.num_keyword) + ".p"
    pickle.dump(w2v_model, gzip.open(data_dir + os.sep + w2v_filename, "wb"))

    return w2v_model

def process_interactions(data, vectorizer, options, verbose=False):
    user_interactions = {}
    num_rec_sample = 0
    num_search_sample = 0
    for _, row in data.iterrows():
        user_id = row['user_id']
        if user_id not in user_interactions:
            user_interactions[user_id] = []
        interactions = [int(row['item_id'])]

        # Check whether the interaction is a search (with query) or recommendation (without query) instance
        query = row['query']
        if query == "-" or query == "": # Recommendation instance: no query
            num_rec_sample += 1
            interactions += [0]
        else: # Search instance: a query is present
            num_search_sample += 1
            interactions += [1]

            if options.use_query: # Add the query keywords into the interaction
                query = query.split(" ")

                keywords = []
                for keyword in query:
                    if keyword in vectorizer.vocabulary_:
                        keyword_id = vectorizer.vocabulary_[keyword] # ID of the keyword term in the vectorizer
                        keywords.append((keyword_id))
                interactions += keywords

        user_interactions[user_id].append(interactions)
    if verbose:
        print("Number of recommendation instances:", num_rec_sample, flush=True)
        print("Number of search instances:", num_search_sample, flush=True)
    return user_interactions

def build_vectorizer(data, options):
    print("Building a vectorizer on the query data...", datetime.now(), flush=True)
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), binary=True, max_features=options.num_keyword,
                                 lowercase=False, max_df=0.1, norm=None)
    try:
        vectorizer.fit(data['query'].to_list())
    except ValueError: # Empty vocabulary
        vectorizer = None

    return vectorizer

def build_keyword_embed(vectorizer, w2v_model, options):
    keyword_pre_embeddings = torch.zeros((options.num_keyword, options.w2v_dim), dtype=torch.float)

    # Get the keyword embeddings from w2v_model
    for (keyword, keyword_id) in vectorizer.vocabulary_.items():
        keyword_pre_embeddings[keyword_id, :] = torch.tensor(w2v_model[keyword])

    return keyword_pre_embeddings