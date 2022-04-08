import pandas as pd

# Parameters
## Filter
min_i = 10
min_u = 10
## Split
training_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

#################
# Load the data #
#################

print("Loading data...", flush=True)

dir_path = "../data/movielens/"
original_path = dir_path + "original_data"
final_path = dir_path
rating_df = pd.read_csv(original_path + "/ratings.csv", sep=",")
rating_df["tag"] = -1
tag_df = pd.read_csv(original_path + "/tags.csv", sep=",")
tag_df["rating"] = -1
df = pd.concat([rating_df, tag_df], ignore_index=True)
df = df[~df['tag'].isnull()] # Remove NaN tags
df = df[(df["rating"] >= 4.0) | (df["rating"] == -1)] # Remove interactions corresponding to low (< 4) ratings
df = df.sort_values(["timestamp"], axis=0, ascending=True, ignore_index=True)

####################
# K-core filtering #
####################

print("K-core filtering...", flush=True)

# Filter out users with no tagging interaction
tagging_user_ids = set(df[df['tag'] != -1].userId)
df = df[df['userId'].isin(tagging_user_ids)]

# Filter out items with less than min_i interactions
print("Filtering out items...", flush=True)
dist_df = df['movieId'].value_counts()
dist_df = dist_df[dist_df >= min_i]
filtered_item_ids = dist_df.keys()
df = df[df['movieId'].isin(filtered_item_ids)]

# Filter out users with less than min_u interactions
print("Filtering out users...", flush=True)
dist_df = df['userId'].value_counts()
dist_df = dist_df[dist_df >= min_u]
filtered_user_ids = dist_df.keys()
df = df[df['userId'].isin(filtered_user_ids)]

filtered_user_ids = set(df['userId']) # Update list of users filtered to remove those with no interaction
filtered_item_ids = set(df['movieId']) # Update list of items filtered to remove those with no interaction
print("Number of users:", len(filtered_user_ids))
print("Number of items:", len(filtered_item_ids))

###################################
# Make the train/valid/test split #
###################################

print("Splitting data into train/valid/test set...", flush=True)

training_tuples = []
validation_tuples = []
test_tuples = []
user_list = set(df['userId'])
item_list = set(df['movieId'])
user_dict = {u: user_id for (user_id, u) in enumerate(user_list)}
item_dict = {p: item_id for (item_id, p) in enumerate(item_list)}
for (u, user_id) in user_dict.items():
    user_query_df = df[df['userId'] == u].sort_values(by=['timestamp'])
    if user_id % 1000 == 0:
        print("Number of users processed: " + str(user_id), flush=True)
    n_interaction = user_query_df.shape[0]
    n_test = int(test_ratio * n_interaction)
    n_validation = int(validation_ratio * n_interaction)
    n_training = n_interaction - n_validation - n_test

    for interaction_count, (row, interaction) in enumerate(user_query_df.iterrows()):
        item_id = item_dict[interaction['movieId']] # Item interacted
        time = interaction['timestamp'] # Time of the interaction
        rating = interaction['rating'] # Rating of the interaction or -1 if no rating
        rating = "-" if rating == -1 else str(rating)
        tag = interaction['tag'] # ID of the tag or -1 if no tag
        tag_text = "-" if tag == -1 else str(tag)
        tag_text = '"' + tag_text.replace('\t', ' ') + '"'

        # Process the query and add it to training, validation or test set
        if interaction_count < n_training: # Training set
            training_tuples.append((user_id, item_id, time, rating, tag_text))
        elif interaction_count >= n_training and interaction_count < n_training + n_validation: # Validation set
            validation_tuples.append((user_id, item_id, time, rating, tag_text))
        else: # Test set
            test_tuples.append((user_id, item_id, time, rating, tag_text))

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
    f.write("user_id\titem_id\ttime\trating\tquery\n")
    for (user_id, item_id, time, rating, tag_text) in training_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + str(time) + "\t" + rating + "\t" + tag_text + "\n")

# Save validation file
validation_path = final_path + "/valid.txt"
with open(validation_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\ttime\trating\tquery\n")
    for (user_id, item_id, time, rating, tag_text) in validation_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + str(time) + "\t" + rating + "\t" + tag_text + "\n")

# Save test file
test_path = final_path + "/test.txt"
with open(test_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\ttime\trating\tquery\n")
    for (user_id, item_id, time, rating, tag_text) in test_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + str(time) + "\t" + rating + "\t" + tag_text + "\n")

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