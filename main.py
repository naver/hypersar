import pandas as pd
import numpy as np
import torch
import os
import sys
import pickle
import gzip
from option_parser import OptParser
from data_processing import process_interactions, load_data, build_vectorizer, load_w2v_model, build_keyword_embed
from torch.utils.data import RandomSampler, DataLoader
from evaluation import predict_evaluate
from models import MatrixFactorization, LightGCN, FactorizationMachine, DeepFM, JSR, DREM, HyperSaR
from datetime import datetime
import random


#######################################
# Define settings and preprocess data #
#######################################

# Option definition
optparser = OptParser()
options = optparser.parse_args()[0]

# Seeds for reproducibility
seed = int(options.seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

# Settings
## Dataset
if options.dataset is None:
    print("Missing dataset!", flush=True)
    exit()
## Model
model_type = options.model
if model_type == "MatrixFactorization":
    options.use_query = False
    options.use_w2v = False
    from data_loader import InteractionDataset as TrainDataset
    from data_loader import InteractionDataset as EvalDataset
    from data_loader import interaction_collate_fn as collate_fn
elif model_type == "LightGCN":
    options.use_query = False
    options.use_w2v = False
    from data_loader import GraphInteractionDataset as TrainDataset
    from data_loader import InteractionDataset as EvalDataset
    from data_loader import interaction_collate_fn as collate_fn
elif model_type in ["FactorizationMachine", "DREM"]:
    options.use_query = True
    options.use_w2v = False
    from data_loader import SaRInteractionDataset as TrainDataset
    from data_loader import SaRInteractionDataset as EvalDataset
    from data_loader import sar_interaction_collate_fn as collate_fn
elif model_type == "DeepFM":
    options.use_query = True
    options.use_w2v = False
    from data_loader import SaRInteractionDataset as TrainDataset
    from data_loader import SaRInteractionDataset as EvalDataset
    from data_loader import sar_interaction_collate_fn as collate_fn
elif model_type == "JSR":
    options.use_query = True
    options.use_w2v = True
    from data_loader import SaRInteractionDataset as TrainDataset
    from data_loader import SaRInteractionDataset as EvalDataset
    from data_loader import sar_interaction_collate_fn as collate_fn
elif model_type == "HyperSaR":
    options.use_query = True
    options.use_w2v = False
    from data_loader import GraphSaRInteractionDataset as TrainDataset
    from data_loader import SaRInteractionDataset as EvalDataset
    from data_loader import sar_interaction_collate_fn as collate_fn
elif model_type is None:
    print("Missing model!", flush=True)
    exit()
else:
    print("Unknown model!", flush=True)
    exit()
## Training
num_epoch = options.num_epoch
batch_size = options.batch_size
eval_batch_size = options.eval_batch_size if options.eval_batch_size != None else batch_size
num_workers = options.num_workers
## Evaluation
use_valid = options.use_valid # Use a validation set, otherwise the validation set is integrated to the training set
options.metrics = [("recall", 1), ("recall", 10), ("recall", 20), ("ndcg", 20)]
## Pre-trained word embeddings
use_w2v = options.use_w2v # Indicates whether word2vec embeddings should be used for keywords (only for SaR)
## GPU usage
device_embed = torch.device(str(options.device_embed) + ":" + str(options.cuda)) if options.device_embed == "cuda" \
    else torch.device(options.device_embed)
device_ops = torch.device(str(options.device_ops) + ":" + str(options.cuda)) if options.device_ops == "cuda" \
    else torch.device(options.device_ops)
print("Using device_embed: {0}".format(options.device_embed), flush=True)
print("Using device_ops: {0}".format(options.device_ops), flush=True)
options.device_embed = device_embed
options.device_ops = device_ops

# Set up log files
## Log file name
experiment_id = options.dataset
experiment_id += "-valid" if use_valid else "-novalid"
experiment_id += "-" + model_type
if model_type in ["LightGCN", "DeepFM", "JSR", "HyperSaR"]:
    experiment_id += "-l" + str(options.num_layer)
if model_type in ["FactorizationMachine", "DeepFM", "JSR", "DREM", "HyperSaR"]:
    experiment_id += "-z" + str(options.num_keyword)
if model_type in ["LightGCN", "HyperSaR"]:
    experiment_id += "-ed" + str(options.edge_dropout)
if model_type in ["DeepFM", "JSR"]:
    experiment_id += "-wd" + str(options.weight_dropout)
if model_type in ["JSR", "HyperSaR"]:
    experiment_id += "-lw" + str(options.loss_weight)
if model_type in ["JSR"]:
    experiment_id += "-lm" + str(options.lm_weight)
experiment_id += "-e" + str(num_epoch) + "-b" + str(batch_size) + "-r" + str(options.lr) + \
                 "-h" + str(options.embed_dim) + "-n" + str(options.num_neg_sample) + \
                 "-w" + str(options.weight_decay) + "-s" + str(seed)
## File creation
if not os.path.isdir('logs'):
    os.mkdir('logs')
if not os.path.isdir('res'):
    os.mkdir('res')
logfile = open('./logs/%s.log' % experiment_id, 'w')
resfile = open('./res/%s.tsv' % experiment_id, 'w')
outputs = [sys.stdout, logfile]


###################
# Preprocess data #
###################

# Load data
## Paths
options.data_dir = "data" + os.sep + options.dataset
data_size_path = options.data_dir + os.sep + "data_size.txt"
train_path = options.data_dir + os.sep + "train.txt"
valid_path = options.data_dir + os.sep + "valid.txt"
test_path = options.data_dir + os.sep + "test.txt"
## Load the dataset
data_size = [int(e) for e in open(data_size_path, "r").readline().split("\t")] # Gives (num_user, num_item)
options.num_user = data_size[0]
options.num_item = data_size[1]
print("Loading dataset...", datetime.now(), flush=True)
delimiter = '\t'
train_data = load_data(train_path, delimiter)
valid_data = load_data(valid_path, delimiter)
if not use_valid:
    train_data = pd.concat([train_data, valid_data]) # Use training and validation data as training set
test_data = load_data(test_path, delimiter)

# Query processing to extract keywords
if options.use_query:
    # Build a vectorizer on the query data to identify the most frequent tokens
    if not use_valid:
        vectorizer = build_vectorizer(pd.concat([train_data, test_data]), options)
    else:
        vectorizer = build_vectorizer(pd.concat([train_data, valid_data, test_data]), options)
    if vectorizer is not None:
        options.num_keyword = len(vectorizer.vocabulary_)
    else: # Empty vocabulary
        options.num_keyword = 0

    if options.use_w2v:
        # Load dense keyword embeddings based on a pretrained word2vec model
        w2v_filename = "w2v_" + str(options.num_keyword) + ".p"
        if not os.path.exists(options.data_dir + os.sep + w2v_filename):
            print("Preparing w2v model... ", datetime.now(), flush=True)
            w2v_model = load_w2v_model(vectorizer, options)
        else:
            print("Loading the preprocessed w2v model... ", datetime.now(), flush=True)
            w2v_model = pickle.load(gzip.open(options.data_dir + os.sep + w2v_filename, "rb"))
        options.w2v_dim = len(list(w2v_model.values())[0])

        # Build pre-trained keyword embeddings based on word2vec embeddings
        options.keyword_pre_embeddings = build_keyword_embed(vectorizer, w2v_model, options)
else:
    vectorizer = None

# Output dataset stats
print("Number of users:", options.num_user, flush=True)
print("Number of items:", options.num_item, flush=True)
if options.use_query:
    print("Number of keywords:", options.num_keyword, flush=True)

# Preprocess data
print("Preprocessing data...", datetime.now(), flush=True)
## Interactions
print("Training set...", flush=True)
train_user_interactions = process_interactions(train_data, vectorizer, options, verbose=True)
if use_valid:
    print("Validation set...", flush=True)
    valid_user_interactions = process_interactions(valid_data, vectorizer, options, verbose=True)
print("Test set...", flush=True)
test_user_interactions = process_interactions(test_data, vectorizer, options, verbose=True)
if use_valid:
    train_valid_user_interactions = process_interactions(pd.concat([train_data, valid_data]), vectorizer, options)
else: # Validation data already in training data
    train_valid_user_interactions = process_interactions(train_data, vectorizer, options)

# Define the dataset and dataloader variables
## Training set
train_dataset = TrainDataset(train_user_interactions, options)
train_random_sampler = RandomSampler(train_dataset, replacement=False) # No replacement to process all users
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_random_sampler,
                               num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch))
if use_valid:
    ## Validation set
    valid_dataset = EvalDataset(valid_user_interactions, options)
    valid_random_sampler = RandomSampler(valid_dataset, replacement=False) # No replacement to process all users
    valid_data_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, sampler=valid_random_sampler,
                                   num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch))
## Test set
test_dataset = EvalDataset(test_user_interactions, options)
test_random_sampler = RandomSampler(test_dataset, replacement=False) # No replacement to process all samples
test_data_loader = DataLoader(test_dataset, batch_size=eval_batch_size, sampler=test_random_sampler,
                              num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch))


#############################################
# Train a new model or load an existing one #
#############################################

# Model initialization
print("Initializing the model...", datetime.now(), flush=True)
if model_type == "MatrixFactorization":
    model = MatrixFactorization(options)
elif model_type == "LightGCN":
    model = LightGCN(options, train_dataset)
elif model_type == "FactorizationMachine":
    model = FactorizationMachine(options)
elif model_type == "DeepFM":
    model = DeepFM(options)
elif model_type == "JSR":
    model = JSR(options, train_dataset)
elif model_type == "DREM":
    model = DREM(options)
elif model_type == "HyperSaR":
    model = HyperSaR(options, train_dataset)
else:
    print("Model unknown!", flush=True)
    exit()

if options.load:
    # Load a previously trained model
    print("Loading an existing model...", datetime.now(), flush=True)
    checkpoint = torch.load('checkpoint/%s.t7' % experiment_id, map_location=device_embed)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
else:
    # Training
    print("Starting training...", datetime.now(), flush=True)
    best_eval = -1 # For model selection
    for epoch in range(num_epoch):
        # Train for an epoch
        model.train()
        epoch_loss = 0.0
        for (n, batch) in enumerate(train_data_loader):
            batch = {k: v.to(options.device_embed) for (k, v) in batch.items()}
            loss = model(batch)
            epoch_loss += loss
        epoch_loss /= (n + 1) # Divide by the number of batches

        [print('Epoch {}: train loss {} -- {}'.format(epoch, epoch_loss, datetime.now()), file=f, flush=True)
         for f in outputs]

        # Save only for the last epoch if no validation set is used for model selection
        if epoch == num_epoch - 1:
            print('Saving...', flush=True)
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, 'checkpoint/%s.t7' % experiment_id)

        logfile.flush()

    # Model selection
    if use_valid:
        # Load the selected model based on the best performance on the validation set
        print("Selected model evaluation...", datetime.now(), flush=True)
        checkpoint = torch.load('checkpoint/%s.t7' % experiment_id)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']


######################
# Evaluate the model #
######################

if model_type in ["JSR"]:
    # For JSR, compute language models once the model is trained
    print("Computing language models...", datetime.now(), flush=True)
    model.compute_language_models()

print("Starting evaluation...", datetime.now(), flush=True)
model.eval()
# Header of the result file
metrics_str = ["{}@{}".format(metric[0], metric[1]) for metric in options.metrics]
metrics_str = "\t".join(metrics_str)
print("set\ttype\t{}".format(metrics_str), file=resfile, flush=True)
if use_valid:
    # Evaluate on the validation set
    eval_results = predict_evaluate(valid_data_loader, options, model, train_user_interactions)
    for type in ['all', 'rec', 'search']:
        type_eval_results = eval_results[type]
        eval_res_str = ["{}@{} {:.5f}".format(metric[0], metric[1], type_eval_results[metric])
                        for metric in options.metrics]
        eval_res_str = ", ".join(eval_res_str)
        [print('Epoch {}: valid, {}-type -- {} -- {}'.format(epoch, type, eval_res_str, datetime.now()),
               file=f, flush=True) for f in outputs]
        eval_res_str = ["{:.5f}".format(type_eval_results[metric]) for metric in options.metrics]
        eval_res_str = "\t".join(eval_res_str)
        print("valid\t{}-type\t{}".format(type, eval_res_str), file=resfile, flush=True)
else:
    # Evaluate on the test set
    eval_results = predict_evaluate(test_data_loader, options, model, train_valid_user_interactions)
    for type in ['all', 'rec', 'search']:
        type_eval_results = eval_results[type]
        eval_res_str = ["{}@{} {:.5f}".format(metric[0], metric[1], type_eval_results[metric])
                        for metric in options.metrics]
        eval_res_str = ", ".join(eval_res_str)
        [print('Epoch {}: test, {}-type -- {} -- {}'.format(epoch, type, eval_res_str, datetime.now()),
               file=f, flush=True) for f in outputs]
        eval_res_str = ["{:.5f}".format(type_eval_results[metric]) for metric in options.metrics]
        eval_res_str = "\t".join(eval_res_str)
        print("test\t{}-type\t{}".format(type, eval_res_str), file=resfile, flush=True)

logfile.close()
