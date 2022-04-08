from torch.utils.data import Dataset
import torch
import numpy as np
import scipy.sparse as sp

def interaction_collate_fn(batch):
    batch_size = len(batch)

    # Collate interactions, sequence sizes, and queries from the current minibatch
    collated_interaction_ids = torch.zeros(batch_size, dtype=torch.long) # IDs of the interactions in the batch
    collated_user_ids = torch.zeros(batch_size, dtype=torch.long) # IDs of the users in the batch
    collated_item_ids = torch.zeros(batch_size, dtype=torch.long) # IDs of the interacted items in the batch
    collated_interaction_types = torch.zeros(batch_size, dtype=torch.long) # Types of the interactions in the batch

    for (i, sample) in enumerate(batch):
        # Save the ID of the interaction in the sample
        collated_interaction_ids[i] = sample['interaction_id']

        # Save the ID of the user in the sample
        collated_user_ids[i] = sample['user_id']

        # Save the ID of the interacted item in the sample
        collated_item_ids[i] = sample['item_id']

        # Save the type of the interaction in the sample
        collated_interaction_types[i] = sample['interaction_type']

    return {'interaction_ids': collated_interaction_ids, 'user_ids': collated_user_ids, 'item_ids': collated_item_ids,
            'interaction_types': collated_interaction_types}

def sar_interaction_collate_fn(batch):
    batch_size = len(batch)
    batch_query_size = max([len(sample['keyword_id']) for sample in batch]) # Max #keywords per interaction in batch

    # Collate interactions, sequence sizes, and queries from the current minibatch
    collated_interaction_ids = torch.zeros(batch_size, dtype=torch.long) # IDs of the interactions in the batch
    collated_user_ids = torch.zeros(batch_size, dtype=torch.long) # IDs of the users in the batch
    collated_item_ids = torch.zeros(batch_size, dtype=torch.long) # IDs of the interacted items in the batch
    collated_interaction_types = torch.zeros(batch_size, dtype=torch.long) # Types of the interactions in the batch
    collated_keyword_ids = torch.zeros((batch_size, batch_query_size), dtype=torch.long) # IDs of keywords per interaction
    collated_keyword_ids.fill_(-1) # The last item (-1) in the keyword embeddings is used for padding
    interaction_query_sizes = torch.zeros(batch_size, dtype=torch.long) # Number of keywords per interaction

    for (i, sample) in enumerate(batch):
        # Save the ID of the interaction in the sample
        collated_interaction_ids[i] = sample['interaction_id']

        # Save the ID of the user in the sample
        collated_user_ids[i] = sample['user_id']

        # Save the ID of the interacted item in the sample
        collated_item_ids[i] = sample['item_id']

        # Save the type of the interaction in the sample
        collated_interaction_types[i] = sample['interaction_type']

        # Save the IDs of keywords, with padding
        keyword_ids = sample['keyword_id']
        query_size = len(keyword_ids)
        keyword_ids = torch.tensor(keyword_ids, dtype=torch.long) # Convert to a tensor
        collated_keyword_ids[i, :query_size] = keyword_ids

        # Save the number of keywords per interaction for unpadding
        interaction_query_sizes[i] = query_size

    return {'interaction_ids': collated_interaction_ids, 'user_ids': collated_user_ids, 'item_ids': collated_item_ids,
            'interaction_types': collated_interaction_types, 'keyword_ids': collated_keyword_ids,
            'query_sizes': interaction_query_sizes}

class InteractionDataset(Dataset):
    """
        Dataset where an interaction (user + clicked item) is a single sample.
    """
    def __init__(self, user_interactions, options):

        # Build the user and clicked item vectors by considering each interaction as a sample
        (self.num_user, self.num_item) = (options.num_user, options.num_item)
        self.user_ids = []
        self.item_ids = []
        self.interaction_types = []
        for user in user_interactions.keys():
            for interaction in user_interactions[user]:
                item = interaction[0]
                self.item_ids.append(item)
                interaction_type = interaction[1] # Interaction type: 0 = recommendation, 1 = search
                self.interaction_types.append(interaction_type)
            self.user_ids.extend([user] * len(user_interactions[user]))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        sample = {'interaction_id': idx, 'user_id': self.user_ids[idx], 'item_id': self.item_ids[idx],
                  'interaction_type': self.interaction_types[idx]}
        return sample

class GraphInteractionDataset(Dataset):
    """
        Dataset where an interaction (user + clicked item) is a single sample and the samples are organized in a
        graph structure.
    """
    def __init__(self, user_interactions, options):
        # Build the user and clicked item vectors by considering each interaction as a sample
        (self.num_user, self.num_item) = (options.num_user, options.num_item)
        self.user_ids = []
        self.item_ids = []
        self.interaction_types = []
        for user in user_interactions.keys():
            for interaction in user_interactions[user]:
                item = interaction[0]
                self.item_ids.append(item)
                interaction_type = interaction[1]  # Interaction type: 0 = recommendation, 1 = search
                self.interaction_types.append(interaction_type)
            self.user_ids.extend([user] * len(user_interactions[user]))

        # Build the sparse normalized adjacency matrix D^(-1/2) * A * D^(-1/2)
        ## User-item interaction matrix
        num_interaction = len(self.user_ids)
        user_item_mat = sp.csr_matrix((np.ones(num_interaction), (self.user_ids, self.item_ids)),
                                      shape=(self.num_user, self.num_item))
        ## Adjacency matrix A
        adj_mat = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        user_item_mat = user_item_mat.tolil()
        adj_mat[:self.num_user, self.num_user:] = user_item_mat
        adj_mat[self.num_user:, :self.num_user] = user_item_mat.T
        adj_mat = adj_mat.todok()
        ## Degree matrix D^(-1/2)
        deg_mat = np.array(adj_mat.sum(axis=1))
        deg_mat = np.power(deg_mat, -0.5).flatten()
        deg_mat[np.isinf(deg_mat)] = 0.0 # Isolated node in the graph
        deg_mat = sp.diags(deg_mat)
        ## Normalized adjacency matrix D^(-1/2) * A * D^(-1/2)
        norm_adj_mat = deg_mat.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(deg_mat)
        ## Graph as a sparse tensor
        coo = norm_adj_mat.tocoo().astype(np.float32)
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        index = torch.stack([row, col])
        data = torch.tensor(coo.data, dtype=torch.float32)
        self.norm_adj_mat = torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
        self.norm_adj_mat = self.norm_adj_mat.coalesce()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        sample = {'interaction_id': idx, 'user_id': self.user_ids[idx], 'item_id': self.item_ids[idx],
                  'interaction_type': self.interaction_types[idx]}
        return sample

class SaRInteractionDataset(Dataset):
    """
        Dataset where an interaction (user + clicked item + possibly query) is a single sample.
    """
    def __init__(self, user_interactions, options):
        # Build the user, clicked item, and keyword vectors by considering each interaction as a sample
        (self.num_user, self.num_item, self.num_keyword) = (options.num_user, options.num_item, options.num_keyword)
        self.user_ids = []
        self.item_ids = []
        self.interaction_types = []
        self.keyword_ids = []
        for user in user_interactions.keys():
            for interaction in user_interactions[user]:
                self.user_ids.append(user)

                item = interaction[0]
                self.item_ids.append(item)

                interaction_type = interaction[1] # Interaction type: 0 = recommendation, 1 = search
                self.interaction_types.append(interaction_type)

                keywords = interaction[2:] # 0 keyword (recommendation interaction) or n>0 keywords (search interaction)
                if keywords == []:
                    self.keyword_ids.append([self.num_keyword]) # num_keyword will be the embedding index of "no keyword"
                else:
                    interaction_keywords = []
                    for keyword_id in keywords:
                        interaction_keywords.append(keyword_id)
                    self.keyword_ids.append(interaction_keywords)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        sample = {'interaction_id': idx, 'user_id': self.user_ids[idx], 'item_id': self.item_ids[idx],
                  'interaction_type': self.interaction_types[idx], 'keyword_id': self.keyword_ids[idx]}
        return sample

class GraphSaRInteractionDataset(Dataset):
    """
        Dataset where an interaction (user + clicked item + possibly query) is a single sample and the samples are
        organized in a graph structure such that users, items, and keywords are nodes and interactions are edges.
    """
    def __init__(self, user_interactions, options):
        # Build the user, clicked item, and keyword vectors by considering each interaction as a sample
        (self.num_user, self.num_item, self.num_keyword) = (options.num_user, options.num_item, options.num_keyword)
        self.user_ids = []
        self.item_ids = []
        self.interaction_types = []
        self.keyword_ids = []
        row_ids = [] # Row IDs for the sparse incidence matrix construction
        column_ids = [] # Column IDs for the sparse incidence matrix construction
        vals = [] # Values for the sparse incidence matrix construction
        interaction_count = 0
        for user in user_interactions.keys():
            for interaction in user_interactions[user]:
                self.user_ids.append(user)
                # Users row from index 0 to (num_user - 1)
                row_ids.append(user)
                column_ids.append(interaction_count)
                vals.append(1.0)

                item = interaction[0]
                self.item_ids.append(item)
                # Item rows from index num_user to (num_user + num_item - 1)
                row_ids.append(item + self.num_user)
                column_ids.append(interaction_count)
                vals.append(1.0)

                interaction_type = interaction[1]  # Interaction type: 0 = recommendation, 1 = search
                self.interaction_types.append(interaction_type)

                keywords = interaction[2:] # 0 keyword (recommendation interaction) or n>0 keywords (search interaction)
                if keywords == []:
                    self.keyword_ids.append([self.num_keyword])  # num_keyword will be the embedding index of "no keyword"
                else:
                    interaction_keywords = []
                    for keyword_id in keywords:
                        interaction_keywords.append(keyword_id)
                        # keyword rows from index (num_user + num_item) to (num_user + num_item + num_keyword - 1)
                        row_ids.append(keyword_id + self.num_user + self.num_item)
                        column_ids.append(interaction_count)
                        vals.append(1.0)
                    self.keyword_ids.append(interaction_keywords)

                interaction_count += 1

        # Build the normalized adjacency matrix of the clique-expanded graph from incidence matrix H of the hypergraph:
        # D_V^(-1/2) * H * D_E^(-1) * H^T * D_V^(-1/2)
        ## Incidence matrix
        num_interaction = interaction_count
        incid_mat = sp.csr_matrix((vals, (row_ids, column_ids)),
                                  shape=(self.num_user + self.num_item + self.num_keyword, num_interaction))
        node_deg_vec = np.array(incid_mat.sum(axis=1)).flatten()
        edge_deg_vec = np.array(incid_mat.sum(axis=0)).flatten()
        ## Degree matrices for normalization
        inv_sqrt_d_v = sp.diags(np.power(node_deg_vec, -0.5)) # D_V^(-1/2)
        inv_sqrt_d_e = sp.diags(np.power(edge_deg_vec, -0.5)) # D_E^(-1/2)
        ## Normalized incidence matrix: D_V^(-1/2) * H * D_E^(-1/2)
        norm_incid_mat = inv_sqrt_d_v @ incid_mat @ inv_sqrt_d_e
        ## Transposed normalized incidence matrix: D_E^(-1/2) * H^T * D_V^(-1/2)
        norm_incid_mat_t = norm_incid_mat.transpose()
        ## Normalized adjacency matrix: D_V^(-1/2) * H * D_E^(-1) * H^T * D_V^(-1/2)
        norm_adj_mat = norm_incid_mat @ norm_incid_mat_t
        ## Normalized adjacency matrix as a sparse tensor
        coo = norm_adj_mat.tocoo().astype(np.float32)
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        index = torch.stack([row, col])
        data = torch.tensor(coo.data, dtype=torch.float32)
        self.norm_adj_mat = torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
        self.norm_adj_mat = self.norm_adj_mat.coalesce()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        sample = {'interaction_id': idx, 'user_id': self.user_ids[idx], 'item_id': self.item_ids[idx],
                  'interaction_type': self.interaction_types[idx], 'keyword_id': self.keyword_ids[idx]}
        return sample