import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math

class DotProdScorer(nn.Module):
    def __init__(self, device):
        super(DotProdScorer, self).__init__()

        # Define the device
        self.device = device

    def forward(self, user_embeddings, item_embeddings):
        # Scores based on the learned user/item embeddings
        if self.training:
            assert user_embeddings.size()[0] == item_embeddings.size()[0] # Equals to batch_size
            # Score user-item pairs aligned in user_embeddings and item_embeddings
            scores = (user_embeddings * item_embeddings).sum(-1).squeeze()
            ## Shape of scores: (batch_size)
        else:
            # Score every pair made of a row from user_embeddings and any row from item_embeddings
            scores = torch.mm(user_embeddings, item_embeddings.t())
            ## Shape of scores: (batch_size, num_item)

        return scores

class DenseScorer(nn.Module):
    def __init__(self, device, embed_dim, num_layer=3, dropout_prob=0.0):
        super(DenseScorer, self).__init__()

        # Define the device
        self.device = device

        # Define the hyperparameters and settings
        self.num_layer = num_layer
        self.dropout_prob = dropout_prob
        list_dims = [embed_dim for _ in range(self.num_layer)] + [1]
        layer_in_size = [dim for dim in list_dims[:-1]]
        layer_out_size = [dim for dim in list_dims[1:]]

        # Layers and components
        self.linear_layers = nn.ModuleList()
        for i in range(self.num_layer):
            self.linear_layers.append(nn.Linear(layer_in_size[i], layer_out_size[i]).to(self.device))
        self.dropout = nn.Dropout(self.dropout_prob) # Drop out some of the neurons
        self.activation = nn.ReLU() # Non-linearity

    def forward(self, user_embeddings, item_embeddings):
        # Scores based on the learned user/item embeddings
        if self.training:
            assert user_embeddings.size()[0] == item_embeddings.size()[0] # Equals to batch_size
            # Score user-item pairs aligned in user_embeddings and item_embeddings
            scores = user_embeddings * item_embeddings
            ## Shape of scores: (batch_size, embed_dim)
        else:
            # Score every pair made of a row from user_embeddings and any row from item_embeddings
            scores = user_embeddings.unsqueeze(1) * item_embeddings.unsqueeze(0) # Element-wise product
            ## Shape of scores: (batch_size, num_item, embed_dim)
        for i in range(self.num_layer):
            scores = self.dropout(self.linear_layers[i](scores))
            if i != self.num_layer - 1: # Non-linearity on all layers except the last one
                scores = self.activation(scores)
        scores = scores.squeeze()

        return scores

class DotProd3DScorer(nn.Module):
    def __init__(self, device, drem_version=False):
        super(DotProd3DScorer, self).__init__()

        self.drem_version = drem_version # The DREM scorer version ignores the item_embed * keyword_embed term

        # Define the device
        self.device = device

    def forward(self, user_embeddings, item_embeddings, keyword_embeddings):
        # Scores based on the learned user/item/keyword embeddings
        if self.training:
            assert (user_embeddings.size()[0] == item_embeddings.size()[0]) and \
                   (user_embeddings.size()[0] == keyword_embeddings.size()[0]) # Equals to batch_size
            # Score user-item-keyword tuples aligned in user_embeddings, item_embeddings and keyword_embeddings
            if not self.drem_version:
                scores = (user_embeddings * item_embeddings
                          + user_embeddings * keyword_embeddings
                          + item_embeddings * keyword_embeddings).sum(-1).squeeze()
            else:
                scores = (user_embeddings * item_embeddings
                          + user_embeddings * keyword_embeddings).sum(-1).squeeze()
            ## Shape of scores: (batch_size)
        else:
            assert user_embeddings.size()[0] == keyword_embeddings.size()[0] # Equals to batch_size
            # Score every tuple made of a row from user_embeddings, the corresponding row from keyword_embeddings and
            # any row from item_embeddings
            if not self.drem_version:
                scores = torch.mm(user_embeddings, item_embeddings.t()) \
                         + (user_embeddings * keyword_embeddings).sum(-1).unsqueeze(-1) \
                         + torch.mm(keyword_embeddings, item_embeddings.t())
            else:
                scores = torch.mm(user_embeddings, item_embeddings.t()) \
                         + (user_embeddings * keyword_embeddings).sum(-1).unsqueeze(-1)
            ## Shape of scores: (batch_size, num_item)

        return scores

class Dense3DScorer(nn.Module):
    def __init__(self, device, embed_dim, num_layer=3, dropout_prob=0.0):
        super(Dense3DScorer, self).__init__()

        # Define the device
        self.device = device

        # Define the hyperparameters and settings
        self.num_layer = num_layer
        self.dropout_prob = dropout_prob
        list_dims = [3 * embed_dim] + [embed_dim for _ in range(self.num_layer - 1)] + [1]
        layer_in_size = [dim for dim in list_dims[:-1]]
        layer_out_size = [dim for dim in list_dims[1:]]

        # Layers and components
        self.linear_layers = nn.ModuleList()
        for i in range(self.num_layer):
            self.linear_layers.append(nn.Linear(layer_in_size[i], layer_out_size[i]).to(self.device))
        self.dropout = nn.Dropout(self.dropout_prob) # Drop out some of the neurons
        self.activation = nn.ReLU() # Non-linearity

    def forward(self, user_embeddings, item_embeddings, keyword_embeddings):
        # Scores based on the learned user/item embeddings
        if self.training:
            assert (user_embeddings.size()[0] == item_embeddings.size()[0]) and \
                   (user_embeddings.size()[0] == keyword_embeddings.size()[0]) # Equals to batch_size
            # Score user-item-keyword tuples aligned in user_embeddings, item_embeddings and keyword_embeddings
            scores = torch.cat([user_embeddings, item_embeddings, keyword_embeddings], dim=1) # Concatenation
            ## Shape of scores: (batch_size, 3 * embed_dim)
        else:
            assert user_embeddings.size()[0] == keyword_embeddings.size()[0] # Equals to batch_size
            # Score every tuple made of a row from user_embeddings, the corresponding row from keyword_embeddings and
            # any row from item_embeddings
            batch_size = user_embeddings.size()[0]
            num_item = item_embeddings.size()[0]
            scores = torch.cat([user_embeddings.unsqueeze(1).expand(-1, num_item, -1), # Concatenation
                                item_embeddings.unsqueeze(0).expand(batch_size, -1, -1),
                                keyword_embeddings.unsqueeze(1).expand(-1, num_item, -1)], dim=2)
            ## Shape of scores: (batch_size, num_item, 3 * embed_dim)
        for i in range(self.num_layer):
            scores = self.dropout(self.linear_layers[i](scores))
            if i != self.num_layer - 1: # Non-linearity on all layers except the last one
                scores = self.activation(scores)
        scores = scores.squeeze()

        return scores

class SumAggregator(nn.Module):
    def __init__(self, device):
        super(SumAggregator, self).__init__()

        # Define the device
        self.device = device

    def forward(self, input, query_sizes):
        ## Shape of input: (batch_size, batch_query_size, embed_dim)
        input = torch.sum(input, dim=1)
        return input

class MeanAggregator(nn.Module):
    def __init__(self, device):
        super(MeanAggregator, self).__init__()

        # Define the device
        self.device = device

    def forward(self, input, query_sizes):
        ## Shape of input: (batch_size, batch_query_size, embed_dim)
        input = torch.sum(input, dim=1)
        input /= query_sizes.unsqueeze(-1)
        return input

class FullyConnectedAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(FullyConnectedAggregator, self).__init__()

        # Define the device_ops
        self.device = device

        # Define the fully-connected layer
        self.fc_layer = ProjectionLayer(input_dim, output_dim, True, device)
        self.activation = nn.Tanh() # Non-linearity

    def forward(self, input, query_sizes):
        ## Shape of input: (batch_size, batch_query_size, embed_dim)
        input = torch.sum(input, dim=1)
        input /= query_sizes.unsqueeze(-1)
        input = self.activation(self.fc_layer(input))
        return input

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias, device):
        super(ProjectionLayer, self).__init__()

        # Define the device_ops
        self.device = device

        # Define the parameters
        self.weight = Parameter(torch.Tensor(output_dim, input_dim))
        nn.init.uniform_(self.weight, -1 / math.sqrt(output_dim), 1 / math.sqrt(output_dim))

        if use_bias:
            self.bias = Parameter(torch.Tensor(output_dim))
            nn.init.uniform_(self.bias, -1 / math.sqrt(output_dim), 1 / math.sqrt(output_dim))
        else:
            self.bias = None

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class SparseDropout(torch.nn.Module):
    def __init__(self, device, dropout_prob = 0.3):
        super(SparseDropout, self).__init__()

        # Define the device
        self.device = device

        # Probability to keep an edge
        self.keep_prob = 1.0 - dropout_prob

    def forward(self, x):
        if self.training and self.keep_prob < 1.0:
            mask = torch.bernoulli(self.keep_prob * torch.ones_like(x.values(), device=self.device)).type(torch.bool)
            sparse_indices = x.indices()[:, mask]
            sparse_values = x.values()[mask] * (1.0 / self.keep_prob)
            return torch.sparse_coo_tensor(sparse_indices, sparse_values, x.size())
        else:
            return x
