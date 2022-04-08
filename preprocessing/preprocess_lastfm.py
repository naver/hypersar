import random
import pandas as pd
import numpy as np

# Parameters
## Split
training_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2
## Seed
seed = 2021
np.random.seed(seed)
random.seed(seed)

#################
# Load the data #
#################

print("Loading data...", flush=True)

dir_path = "../data/lastfm/"
original_path = dir_path + "original_data"
final_path = dir_path
tag_path = dir_path + "tags.dat"
listen_interaction_path = dir_path + "user_artists.dat"
tag_interaction_path = dir_path + "user_taggedartists.dat"
tag_df = pd.read_csv(tag_path, sep="\t", encoding="unicode_escape")
listen_interaction_df = pd.read_csv(listen_interaction_path, sep="\t")
listen_interaction_df["tagValue"] = "-"
tag_interaction_df = pd.read_csv(tag_interaction_path, sep="\t")
tag_interaction_df = pd.merge(tag_interaction_df, tag_df, how='left', on="tagID")
df = pd.concat([listen_interaction_df[["userID", "artistID", "tagValue"]],
                tag_interaction_df[["userID", "artistID", "tagValue"]]])

###################################
# Make the train/valid/test split #
###################################

print("Splitting data into train/valid/test set...", flush=True)

training_tuples = []
validation_tuples = []
test_tuples = []
user_list = set(df['userID'])
item_list = set(df['artistID'])
user_dict = {u: user_id for (user_id, u) in enumerate(user_list)}
item_dict = {p: item_id for (item_id, p) in enumerate(item_list)}
for (u, user_id) in user_dict.items():
    user_df = df[df['userID'] == u].sample(frac = 1) # Shuffle the user's interactions for random split
    if user_id % 1000 == 0:
        print("Number of users processed: " + str(user_id), flush=True)
    user_items = user_df['artistID'].unique()
    random.shuffle(user_items)
    n_user_items = len(user_items)
    n_test = int(test_ratio * n_user_items)
    n_validation = int(validation_ratio * n_user_items)
    n_training = n_user_items - n_validation - n_test

    test_items = user_items[:n_test]
    validation_items = user_items[n_test:n_test+n_validation]
    train_items = user_items[n_test+n_validation:]

    for interaction_count, (row, interaction) in enumerate(user_df.iterrows()):
        item_id = item_dict[interaction['artistID']] # Item interacted
        tag_text = '"' + interaction['tagValue'].replace('\t', ' ') + '"'

        # Process the query and add it to training, validation or test set
        if interaction['artistID'] in train_items: # Training set
            training_tuples.append((user_id, item_id, tag_text))
        elif interaction['artistID'] in validation_items: # Validation set
            validation_tuples.append((user_id, item_id, tag_text))
        elif interaction['artistID'] in test_items: # Test set
            test_tuples.append((user_id, item_id, tag_text))

n_user = len(user_list)
n_item = len(item_dict)

print("Training", len(training_tuples), flush=True)
print("Validation", len(validation_tuples), flush=True)
print("Test", len(test_tuples), flush=True)

##########################
# Save preprocessed data #
##########################

print("Saving preprocessed data...", flush=True)

# Save data size file
data_size_path = final_path + "/data_size.txt"
with open(data_size_path, "w+", encoding='utf-8') as f:
    f.write(str(n_user) + "\t" + str(n_item) + "\n")

# Save training file
training_path = final_path + "/train.txt"
with open(training_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\tquery\n")
    for (user_id, item_id, tag_text) in training_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + tag_text + "\n")

# Save validation file
validation_path = final_path + "/valid.txt"
with open(validation_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\tquery\n")
    for (user_id, item_id, tag_text) in validation_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + tag_text + "\n")

# Save test file
test_path = final_path + "/test.txt"
with open(test_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\tquery\n")
    for (user_id, item_id, tag_text) in test_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + tag_text + "\n")

# Save item dict file
item_dict_path = final_path + "/item_dict.txt"
with open(item_dict_path, "w+", encoding='utf-8') as f:
    for (old_id, new_id) in item_dict.items():
        f.write(str(old_id) + "\t" + str(new_id) + "\n")

# Save user dict file
user_dict_path = final_path + "/user_dict.txt"
with open(user_dict_path, "w+", encoding='utf-8') as f:
    for (old_id, new_id) in user_dict.items():
        f.write(str(old_id) + "\t" + str(new_id) + "\n")