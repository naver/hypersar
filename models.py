import torch.nn as nn
import torch
import numpy as np
from layers import DotProdScorer, DenseScorer, DotProd3DScorer, Dense3DScorer, SumAggregator, MeanAggregator,\
    FullyConnectedAggregator, ProjectionLayer, SparseDropout
from utils import bpr_loss, ce_loss, sample_items

class MatrixFactorization(nn.Module):
    """
        Implementation of the Matrix Factorization model with a BPR loss and trained with SGD.
    """
    def __init__(self, options):
        super(MatrixFactorization, self).__init__()

        self.num_user = options.num_user
        self.num_item = options.num_item
        self.embed_dim = options.embed_dim
        self.lr = options.lr
        self.num_neg_sample = options.num_neg_sample
        self.device_embed = options.device_embed
        self.device_ops = options.device_ops

        # Embeddings
        ## Definition
        self.user_embeddings = nn.Embedding(self.num_user, self.embed_dim) # User embeddings to be learned
        self.item_embeddings = nn.Embedding(self.num_item, self.embed_dim) # Item embeddings to be learned
        ## Initialization
        nn.init.normal_(self.user_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.item_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        ## Move embeddings to device
        self.user_embeddings = self.user_embeddings.to(self.device_embed)
        self.item_embeddings = self.item_embeddings.to(self.device_embed)

        # Components of the model
        self.scorer = DotProdScorer(self.device_ops).to(self.device_ops)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=options.weight_decay)

    @torch.no_grad()
    def predict(self, user_ids, item_ids=None):
        """
        Compute the score predictions at test time
        Args:
            user_ids: (array<int>) users for whom to return items
            item_ids: (array<int>) items for which prediction scores are desired; if not provided, predictions for all
            items will be computed
        Returns:
            scores: (tensor<float>) predicted scores for all items in item_ids
        """
        if item_ids is None:
            item_ids = torch.tensor(np.arange(self.num_item), dtype=torch.long, device=self.device_embed)

        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (num_item, embed_dim)

        scores = self.scorer(batch_user_embeddings, batch_item_embeddings)
        ## Shape of scores: (batch_size, num_item)
        return scores

    def forward(self, batch):
        # Unpack the content of the minibatch
        user_ids = batch['user_ids']
        ## Shape of user_ids: (batch_size)
        item_ids = batch['item_ids']
        ## Shape of item_ids: (batch_size)

        # Fetch the user embeddings for the minibatch
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)

        # Fetch the (positive) item embeddings for the minibatch
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (batch_size, embed_dim)

        # Calculate the recommendation loss on the minibatch using BPR
        positive_score = self.scorer(batch_user_embeddings, batch_item_embeddings)
        ## Shape of positive_score: (batch_size)
        loss = torch.tensor(0.0, dtype=torch.float, device=self.device_ops)
        for i in range(self.num_neg_sample):
            # Negative sampling
            negative_item_ids = sample_items(self.num_item, item_ids.size())
            negative_item_ids = torch.tensor(negative_item_ids, dtype=torch.long, device=self.device_embed)
            batch_negative_item_embeddings = self.item_embeddings(negative_item_ids).to(self.device_ops)
            ## Shape of batch_negative_item_embeddings: (batch_size, embed_dim)
            negative_score = self.scorer(batch_user_embeddings, batch_negative_item_embeddings)
            ## Shape of negative_score: (batch_size)

            # Compute the BPR loss on the positive and negative scores while masking padded elements in the sequences
            loss += bpr_loss(positive_score, negative_score)
        loss /= self.num_neg_sample

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class LightGCN(nn.Module):
    """
        Implementation of the LightGCN recommendation model. The LightGCN model was originally proposed in:
        He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and Powering Graph
        Convolution Network for Recommendation. SIGIR '20, 639–648.
    """
    def __init__(self, options, train_dataset):
        super(LightGCN, self).__init__()

        self.num_user = options.num_user
        self.num_item = options.num_item
        self.embed_dim = options.embed_dim
        self.lr = options.lr
        self.num_neg_sample = options.num_neg_sample
        self.edge_dropout = options.edge_dropout
        self.num_layer = options.num_layer
        self.device_embed = options.device_embed
        self.device_ops = options.device_ops
        self.norm_adj_mat = train_dataset.norm_adj_mat.to(self.device_embed)

        # Embeddings
        ## Definition
        self.user_embeddings = nn.Embedding(self.num_user, self.embed_dim) # User embeddings to be learned
        self.item_embeddings = nn.Embedding(self.num_item, self.embed_dim) # Item embeddings to be learned
        ## Initialization
        nn.init.normal_(self.user_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.item_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        ## Move embeddings to device
        self.user_embeddings = self.user_embeddings.to(self.device_embed)
        self.item_embeddings = self.item_embeddings.to(self.device_embed)

        # Components of the model
        self.scorer = DotProdScorer(self.device_ops).to(self.device_ops)
        self.dropout_layer = SparseDropout(self.device_embed, self.edge_dropout).to(self.device_embed)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=options.weight_decay)

    def compute_embeddings(self):
        """
        Compute the higher-order embeddings for all users and items after propagation in the graph
        Returns:
            (all_user_embeddings, all_item_embeddings): (tensor<float>,tensor<float>) embeddings for all user and item
            nodes after propagation in the graph
        """
        layer_all_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight]) # 0th layer
        ## Shape of layer_all_embeddings: (num_user + num_item, embed_dim)
        all_embeddings = [layer_all_embeddings]
        norm_adj_mat = self.dropout_layer(self.norm_adj_mat) # Perform layer-shared edge dropout
        for layer in range(self.num_layer):
            layer_all_embeddings = torch.sparse.mm(norm_adj_mat, layer_all_embeddings)
            all_embeddings.append(layer_all_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        ## Shape of all_embeddings: (num_user + num_item, num_layer, embed_dim)

        aggreg_all_embeddings = torch.mean(all_embeddings, dim=1)
        ## Shape of aggregated_all_embeddings: (num_user + num_item, embed_dim)
        all_user_embeddings, all_item_embeddings = torch.split(aggreg_all_embeddings, [self.num_user, self.num_item])
        ## Shape of all_user_embeddings: (num_user, embed_dim)
        ## Shape of all_item_embeddings: (num_item, embed_dim)

        return (all_user_embeddings, all_item_embeddings)

    @torch.no_grad()
    def predict(self, user_ids, item_ids=None):
        """
        Compute the score predictions at test time
        Args:
            user_ids: (array<int>) users for whom to return items
            item_ids: (array<int>) items for which prediction scores are desired; if not provided, predictions for all
            items will be computed
        Returns:
            scores: (tensor<float>) predicted scores for all items in item_ids
        """
        # If no item_ids is provided, consider all items as potential predictions
        if item_ids is None:
            item_ids = torch.tensor(np.arange(self.num_item), dtype=torch.long, device=self.device_embed)

        # Compute the higher-order item/user embeddings based on the graph
        all_user_embeddings, all_item_embeddings = self.compute_embeddings()
        batch_user_embeddings = all_user_embeddings[user_ids].to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = all_item_embeddings[item_ids].to(self.device_ops)
        ## Shape of batch_item_embeddings: (num_item, embed_dim)

        # Compute the scores
        scores = self.scorer(batch_user_embeddings, batch_item_embeddings)
        ## Shape of scores: (batch_size, num_item)
        return scores

    def forward(self, batch):
        # Unpack the content of the minibatch
        user_ids = batch['user_ids']
        item_ids = batch['item_ids']

        # Compute the higher-order user/item embeddings based on the graph
        all_user_embeddings, all_item_embeddings = self.compute_embeddings()
        batch_user_embeddings = all_user_embeddings[user_ids].to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = all_item_embeddings[item_ids].to(self.device_ops)
        ## Shape of batch_item_embeddings: (batch_size, embed_dim)

        # Calculate the recommendation loss on the minibatch using BPR
        positive_score = self.scorer(batch_user_embeddings, batch_item_embeddings)
        ## Shape of positive_score: (batch_size)
        loss = torch.tensor(0.0, dtype=torch.float, device=self.device_ops)
        for i in range(self.num_neg_sample):
            # Negative sampling
            negative_item_ids = sample_items(self.num_item, item_ids.size())
            negative_item_ids = torch.tensor(negative_item_ids, dtype=torch.long, device=self.device_embed)

            # Compute the BPR loss on the positive and negative scores
            batch_negative_item_embeddings = all_item_embeddings[negative_item_ids].to(self.device_ops)
            ## Shape of batch_negative_item_embeddings: (batch_size, embed_dim)
            negative_score = self.scorer(batch_user_embeddings, batch_negative_item_embeddings)
            ## Shape of negative_score: (batch_size)
            loss += bpr_loss(positive_score, negative_score)
        loss /= self.num_neg_sample

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class FactorizationMachine(nn.Module):
    """
        Implementation of the Factorization Machine model for search and recommendation with a BPR loss. Each
        interaction corresponds to a user, an item, and zero, one or more keywords. The Factorization Machine model was
        originally proposed in:
        Rendle, S. (2010). Factorization Machines. ICDM '10, 995–1000.
    """
    def __init__(self, options):
        super(FactorizationMachine, self).__init__()

        self.num_user = options.num_user
        self.num_item = options.num_item
        self.num_keyword = options.num_keyword
        self.embed_dim = options.embed_dim
        self.lr = options.lr
        self.num_neg_sample = options.num_neg_sample
        self.device_embed = options.device_embed
        self.device_ops = options.device_ops

        # Embeddings
        ## Definition
        self.user_embeddings = nn.Embedding(self.num_user, self.embed_dim) # User embeddings to be learned
        self.item_embeddings = nn.Embedding(self.num_item, self.embed_dim) # Item embeddings to be learned
        self.keyword_embeddings = nn.Embedding(self.num_keyword, self.embed_dim) # Keyword embeddings to be learned
        ## Initialization
        nn.init.normal_(self.user_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.item_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.keyword_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        ## Move embeddings to device
        self.user_embeddings = self.user_embeddings.to(self.device_embed)
        self.item_embeddings = self.item_embeddings.to(self.device_embed)
        self.keyword_embeddings = self.keyword_embeddings.to(self.device_embed)

        # Components of the model
        self.scorer = DotProd3DScorer(self.device_ops).to(self.device_ops)
        self.keyword_aggregator = MeanAggregator(self.device_ops).to(self.device_ops)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=options.weight_decay)

    @torch.no_grad()
    def predict(self, user_ids, keyword_ids, query_sizes, item_ids=None):
        """
        Compute the score predictions at test time
        Args:
            user_ids: (array<int>) users for whom to return items
            keyword_ids: (array<int>) keywords for which to return items
            query_sizes: (tensor<int>) number of keywords for each query
            item_ids: (array<int>) items for which prediction scores are desired; if not provided, predictions for all
            items will be computed
        Returns:
            scores: (tensor<float>) predicted scores for all items in item_ids
        """
        # If no item_ids is provided, consider all items as potential predictions
        if item_ids is None:
            item_ids = torch.tensor(np.arange(self.num_item), dtype=torch.long, device=self.device_embed)

        # Fetch the user/item/keyword embeddings for the minibatch
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (num_item, embed_dim)

        # Extend the keyword embeddings with the missing keyword embedding and the padding keyword embedding (both zero)
        extended_keyword_embeddings = torch.zeros((self.num_keyword + 2, self.embed_dim), device=self.device_embed)
        extended_keyword_embeddings[:self.num_keyword] = self.keyword_embeddings.weight
        batch_keyword_embeddings = extended_keyword_embeddings[keyword_ids].to(self.device_ops)
        ## Shape of batch_keyword_embeddings: (batch_size, batch_query_size, embed_dim)
        # For each interaction, aggregate the keyword embeddings for all the keywords in the query
        batch_keyword_embeddings = self.keyword_aggregator(batch_keyword_embeddings, query_sizes)
        ## Shape of batch_keyword_embeddings: (batch_size, embed_dim)

        # Compute the scores
        scores = self.scorer(batch_user_embeddings, batch_item_embeddings, batch_keyword_embeddings)
        ## Shape of scores: (batch_size, num_item)
        return scores

    def forward(self, batch):
        # Unpack the content of the minibatch
        user_ids = batch['user_ids']
        ## Shape of user_ids: (batch_size)
        item_ids = batch['item_ids']
        ## Shape of item_ids: (batch_size)
        keyword_ids = batch['keyword_ids']
        ## Shape of keyword_ids: (batch_size, batch_query_size)
        query_sizes = batch['query_sizes']
        ## Shape of query_sizes: (batch_size)

        # Fetch the user/item/keyword embeddings for the minibatch
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (batch_size, embed_dim)

        # Extend the keyword embeddings with the missing keyword embedding and the padding keyword embedding (both zero)
        extended_keyword_embeddings = torch.zeros((self.num_keyword + 2, self.embed_dim), device=self.device_embed)
        extended_keyword_embeddings[:self.num_keyword] = self.keyword_embeddings.weight
        batch_keyword_embeddings = extended_keyword_embeddings[keyword_ids].to(self.device_ops)
        ## Shape of batch_keyword_embeddings: (batch_size, batch_query_size, embed_dim)
        # For each interaction, aggregate the keyword embeddings for all the keywords in the query
        batch_keyword_embeddings = self.keyword_aggregator(batch_keyword_embeddings, query_sizes)
        ## Shape of batch_keyword_embeddings: (batch_size, embed_dim)

        # Calculate the recommendation loss on the minibatch using BPR
        positive_score = self.scorer(batch_user_embeddings, batch_item_embeddings, batch_keyword_embeddings)
        ## Shape of positive_score: (batch_size)
        loss = torch.tensor(0.0, dtype=torch.float, device=self.device_ops)
        for i in range(self.num_neg_sample):
            # Negative sampling
            negative_item_ids = sample_items(self.num_item, item_ids.size())
            negative_item_ids = torch.tensor(negative_item_ids, dtype=torch.long, device=self.device_embed)

            # Compute the BPR loss on the positive and negative scores
            batch_negative_item_embeddings = self.item_embeddings(negative_item_ids).to(self.device_ops)
            ## Shape of batch_negative_item_embeddings: (batch_size, embed_dim)
            negative_score = self.scorer(batch_user_embeddings, batch_negative_item_embeddings, batch_keyword_embeddings)
            ## Shape of negative_score: (batch_size)
            loss += bpr_loss(positive_score, negative_score)
        loss /= self.num_neg_sample

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class DeepFM(nn.Module):
    """
        Implementation of the Deep Factorization Machine (DeepFM) model for search and recommendation with a BPR loss.
        Each interaction corresponds to a user, an item, and zero, one or more keywords. The DeepFM model was originally
        proposed in:
        Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: A Factorization-Machine based Neural Network for CTR
        Prediction. IJCAI '17, 1725–1731.
    """
    def __init__(self, options):
        super(DeepFM, self).__init__()

        self.num_user = options.num_user
        self.num_item = options.num_item
        self.num_keyword = options.num_keyword
        self.embed_dim = options.embed_dim
        self.lr = options.lr
        self.num_layer = options.num_layer
        self.weight_dropout = options.weight_dropout
        self.num_neg_sample = options.num_neg_sample
        self.device_embed = options.device_embed
        self.device_ops = options.device_ops

        # Embeddings
        ## Definition
        self.user_embeddings = nn.Embedding(self.num_user, self.embed_dim) # User embeddings to be learned
        self.item_embeddings = nn.Embedding(self.num_item, self.embed_dim) # Item embeddings to be learned
        self.keyword_embeddings = nn.Embedding(self.num_keyword, self.embed_dim) # Keyword embeddings to be learned
        ## Initialization
        nn.init.normal_(self.user_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.item_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.keyword_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        ## Move embeddings to device
        self.user_embeddings = self.user_embeddings.to(self.device_embed)
        self.item_embeddings = self.item_embeddings.to(self.device_embed)
        self.keyword_embeddings = self.keyword_embeddings.to(self.device_embed)

        # Components of the model
        self.fm_scorer = DotProd3DScorer(self.device_ops).to(self.device_ops)
        if self.num_layer == 0:
            self.dnn_scorer = DotProd3DScorer(self.device_ops).to(self.device_ops)
        else:
            self.dnn_scorer = Dense3DScorer(self.device_ops, self.embed_dim, self.num_layer,
                                            self.weight_dropout).to(self.device_ops)
        self.keyword_aggregator = MeanAggregator(self.device_ops).to(self.device_ops)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=options.weight_decay)

    @torch.no_grad()
    def predict(self, user_ids, keyword_ids, query_sizes, item_ids=None):
        """
        Compute the score predictions at test time
        Args:
            user_ids: (array<int>) users for whom to return items
            keyword_ids: (array<int>) keywords for which to return items
            query_sizes: (tensor<int>) number of keywords for each query
            item_ids: (array<int>) items for which prediction scores are desired; if not provided, predictions for all
            items will be computed
        Returns:
            scores: (tensor<float>) predicted scores for all items in item_ids
        """
        # If no item_ids is provided, consider all items as potential predictions
        if item_ids is None:
            item_ids = torch.tensor(np.arange(self.num_item), dtype=torch.long, device=self.device_embed)

        # Fetch the user/item/keyword embeddings for the minibatch
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (num_item, embed_dim)

        # Extend the keyword embeddings with the missing keyword embedding and the padding keyword embedding (both zero)
        extended_keyword_embeddings = torch.zeros((self.num_keyword + 2, self.embed_dim), device=self.device_embed)
        extended_keyword_embeddings[:self.num_keyword] = self.keyword_embeddings.weight
        batch_keyword_embeddings = extended_keyword_embeddings[keyword_ids].to(self.device_ops)
        ## Shape of batch_keyword_embeddings: (batch_size, batch_query_size, embed_dim)
        # For each interaction, aggregate the keyword embeddings for all the keywords in the query
        batch_keyword_embeddings = self.keyword_aggregator(batch_keyword_embeddings, query_sizes)
        ## Shape of batch_keyword_embeddings: (batch_size, embed_dim)

        # Compute the scores
        scores = self.fm_scorer(batch_user_embeddings, batch_item_embeddings, batch_keyword_embeddings) + \
                 self.dnn_scorer(batch_user_embeddings, batch_item_embeddings, batch_keyword_embeddings)
        ## Shape of scores: (batch_size, num_item)
        return scores

    def forward(self, batch):
        # Unpack the content of the minibatch
        user_ids = batch['user_ids']
        ## Shape of user_ids: (batch_size)
        item_ids = batch['item_ids']
        ## Shape of item_ids: (batch_size)
        keyword_ids = batch['keyword_ids']
        ## Shape of keyword_ids: (batch_size, batch_query_size)
        query_sizes = batch['query_sizes']
        ## Shape of query_sizes: (batch_size)

        # Fetch the user/item/keyword embeddings for the minibatch
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (batch_size, embed_dim)

        # Extend the keyword embeddings with the missing keyword embedding and the padding keyword embedding (both zero)
        extended_keyword_embeddings = torch.zeros((self.num_keyword + 2, self.embed_dim), device=self.device_embed)
        extended_keyword_embeddings[:self.num_keyword] = self.keyword_embeddings.weight
        batch_keyword_embeddings = extended_keyword_embeddings[keyword_ids].to(self.device_ops)
        ## Shape of batch_keyword_embeddings: (batch_size, batch_query_size, embed_dim)
        # For each interaction, aggregate the keyword embeddings for all the keywords in the query
        batch_keyword_embeddings = self.keyword_aggregator(batch_keyword_embeddings, query_sizes)
        ## Shape of batch_keyword_embeddings: (batch_size, embed_dim)

        # Calculate the recommendation loss on the minibatch using BPR
        positive_score = self.fm_scorer(batch_user_embeddings, batch_item_embeddings, batch_keyword_embeddings) + \
                         self.dnn_scorer(batch_user_embeddings, batch_item_embeddings, batch_keyword_embeddings)
        ## Shape of positive_score: (batch_size)
        loss = torch.tensor(0.0, dtype=torch.float, device=self.device_ops)
        for i in range(self.num_neg_sample):
            # Negative sampling
            negative_item_ids = sample_items(self.num_item, item_ids.size())
            negative_item_ids = torch.tensor(negative_item_ids, dtype=torch.long, device=self.device_embed)

            # Compute the BPR loss on the positive and negative scores
            batch_negative_item_embeddings = self.item_embeddings(negative_item_ids).to(self.device_ops)
            ## Shape of batch_negative_item_embeddings: (batch_size, embed_dim)
            negative_score = self.fm_scorer(batch_user_embeddings, batch_negative_item_embeddings,
                                            batch_keyword_embeddings) + \
                             self.dnn_scorer(batch_user_embeddings, batch_negative_item_embeddings,
                                             batch_keyword_embeddings)
            ## Shape of negative_score: (batch_size)
            loss += bpr_loss(positive_score, negative_score)
        loss /= self.num_neg_sample

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class JSR(nn.Module):
    """
        Implementation of the Joint Search and Recommendation (JSR) model for search and recommendation. The JSR model
        was originally proposed in:
        Zamani, H., & Croft, W. B. (2020). Learning a Joint Search and Recommendation Model from User-Item Interactions.
        WSDM '20, 717–725.
    """
    def __init__(self, options, train_dataset):
        super(JSR, self).__init__()

        self.num_user = options.num_user
        self.num_item = options.num_item
        self.num_keyword = options.num_keyword
        self.embed_dim = options.embed_dim
        self.lr = options.lr
        self.num_layer = options.num_layer
        self.weight_dropout = options.weight_dropout
        self.num_neg_sample = options.num_neg_sample
        self.loss_weight = options.loss_weight
        self.lm_weight = options.lm_weight
        self.device_embed = options.device_embed
        self.device_ops = options.device_ops

        # Embeddings
        ## Definition
        self.user_embeddings = nn.Embedding(self.num_user, self.embed_dim) # User embeddings to be learned
        self.item_embeddings = nn.Embedding(self.num_item, self.embed_dim) # Item embeddings to be learned
        self.keyword_embeddings = nn.Embedding(self.num_keyword, options.w2v_dim) # keyword embeddings (pre-trained)
        ## Initialization
        nn.init.normal_(self.user_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.item_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        ### Initialize the keyword embeddings based on pre-trained word embeddings and freeze them
        keyword_pre_embeddings = options.keyword_pre_embeddings
        self.keyword_embeddings.weight.data.copy_(keyword_pre_embeddings)
        self.keyword_embeddings.weight.requires_grad = False # Embeddings are frozen
        ## Move embeddings to device
        self.user_embeddings = self.user_embeddings.to(self.device_embed)
        self.item_embeddings = self.item_embeddings.to(self.device_embed)
        self.keyword_embeddings = self.keyword_embeddings.to(self.device_embed)

        # Components of the model
        if self.num_layer == 0:
            self.scorer = DotProdScorer(self.device_ops).to(self.device_ops)
        else:
            self.scorer = DenseScorer(self.device_ops, self.embed_dim, self.num_layer,
                                      self.weight_dropout).to(self.device_ops)
        self.item_projector = ProjectionLayer(self.embed_dim, options.w2v_dim, False, self.device_ops).to(self.device_ops)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=options.weight_decay)

    @torch.no_grad()
    def compute_language_models(self):
        """
        Compute the item-specific and corpus language models used during inference
        """
        # Fetch item and keyword embeddings
        item_embeddings = self.item_embeddings.weight.to(self.device_ops)
        ## Shape of item_embeddings: (num_item, embed_dim)
        keyword_embeddings = self.keyword_embeddings.weight.to(self.device_ops)
        ## Shape of keyword_embeddings: (num_keyword, embed_dim)

        # Compute item-specific language models: p(w | theta_i)
        item_language_models = self.item_projector(item_embeddings) @ keyword_embeddings.t()
        item_language_models = torch.softmax(item_language_models, dim=-1)
        ## Shape of item_language_models: (num_item, num_keyword)

        # Compute the corpus-wide language model: p(w | C) = 1/|I| sum_i p(w | theta_i)
        corpus_language_model = torch.mean(item_language_models, dim=0)
        ## Shape of corpus_language_model: (num_keyword)

        # Extend the language models to integrate special keywords with id num_keyword (used when the query is missing)
        # and num_keyword+1 (used as padding token)
        self.item_language_models = torch.zeros((self.num_item, self.num_keyword + 2),
                                                dtype=torch.float, device=self.device_ops)
        self.item_language_models[:, :self.num_keyword] = item_language_models
        self.corpus_language_model = torch.zeros((self.num_keyword + 2), dtype=torch.float, device=self.device_ops)
        self.corpus_language_model[:self.num_keyword] = corpus_language_model

    @torch.no_grad()
    def predict(self, user_ids, keyword_ids, query_sizes, item_ids=None):
        """
        Compute the score predictions at test time
        Args:
            user_ids: (array<int>) users for whom to return items
            keyword_ids: (array<int>) keywords for which to return items
            query_sizes: (tensor<int>) number of keywords for each query
            item_ids: (array<int>) items for which prediction scores are desired; if not provided, predictions for all
            items will be computed
        Returns:
            scores: (tensor<float>) predicted scores for all items in item_ids
        """
        # If no item_ids is provided, consider all items as potential predictions
        if item_ids is None:
            item_ids = torch.tensor(np.arange(self.num_item), dtype=torch.long, device=self.device_embed)

        # Fetch the user/item embeddings
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (num_item, embed_dim)

        # Compute the recommendation scores
        recommendation_scores = self.scorer(batch_user_embeddings, batch_item_embeddings)
        ## Shape of recommendation_scores: (batch_size, num_item)

        # Compute the retrieval scores
        batch_item_language_models = self.item_language_models[item_ids].to(self.device_ops)
        ## Shape of batch_item_language_models: (num_item, num_keyword)
        retrieval_scores = self.lm_weight * batch_item_language_models[:, keyword_ids] \
                           + (1 - self.lm_weight) * self.corpus_language_model[keyword_ids]
        ## Shape of retrieval_scores: (num_item, batch_size, batch_query_size)
        eps = 1e-7 # Smooth the argument of the log to prevent potential numerical underflows
        retrieval_scores = torch.log(retrieval_scores + eps)
        retrieval_scores = torch.sum(retrieval_scores, dim=-1)
        retrieval_scores = retrieval_scores / query_sizes.unsqueeze(0)
        ## Shape of retrieval_scores: (num_item, batch_size)
        retrieval_scores = retrieval_scores.t()
        ## Shape of retrieval_scores: (batch_size, num_item)

        # Compute the final scores depending on the nature of the interaction: search or recommendation
        scores = recommendation_scores # Default scores are recommendation scores
        search_selector = keyword_ids[:, 0] != self.num_keyword # Only non-empty queries don't start with num_keyword
        scores[search_selector, :] = retrieval_scores[search_selector, :] # If non-empty query, use retrieval score instead

        return scores

    def forward(self, batch):
        # Unpack the content of the minibatch
        user_ids = batch['user_ids']
        ## Shape of user_ids: (batch_size)
        item_ids = batch['item_ids']
        ## Shape of item_ids: (batch_size)
        keyword_ids = batch['keyword_ids']
        ## Shape of keyword_ids: (batch_size, batch_query_size)
        query_sizes = batch['query_sizes']
        ## Shape of query_sizes: (batch_size)

        # Fetch the user/item embeddings
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (batch_size, embed_dim)

        # Calculate the recommendation (cross-entropy) loss on the minibatch
        recommendation_loss = torch.tensor(0.0, dtype=torch.float, device=self.device_ops)
        positive_score = self.scorer(batch_user_embeddings, batch_item_embeddings)
        recommendation_loss += ce_loss(positive_score, 1.0)
        ## Shape of positive_score: (batch_size)
        for i in range(self.num_neg_sample):
            # Negative sampling
            negative_item_ids = sample_items(self.num_item, item_ids.size())
            negative_item_ids = torch.tensor(negative_item_ids, dtype=torch.long, device=self.device_embed)

            # Compute the CE loss on the negative scores
            batch_negative_item_embeddings = self.item_embeddings(negative_item_ids).to(self.device_ops)
            ## Shape of batch_negative_item_embeddings: (batch_size, embed_dim)
            negative_score = self.scorer(batch_user_embeddings, batch_negative_item_embeddings)
            ## Shape of negative_score: (batch_size)
            recommendation_loss += ce_loss(negative_score, 0.0)
        recommendation_loss /= (self.num_neg_sample + 1)

        # Calculate the reconstruction loss on the items in the minibatch and their text description (based on keywords)
        ## Consider only the search interactions
        search_selector = keyword_ids[:, 0] != self.num_keyword # Only non-empty queries don't start with num_keyword
        batch_item_embeddings = batch_item_embeddings[search_selector]
        query_sizes = query_sizes[search_selector]
        ### Shape of batch_item_embeddings: (search_batch_size, embed_dim)
        keyword_ids = keyword_ids[search_selector]
        ### Shape of keyword_ids: (search_batch_size, batch_query_size)
        ## Calculate the softmax over all keywords for each interaction
        keyword_embeddings = self.keyword_embeddings.weight.to(self.device_ops)
        batch_keyword_logit = self.item_projector(batch_item_embeddings) @ keyword_embeddings.t()
        ### Shape of batch_keyword_logit: (search_batch_size, num_keyword)
        eps = 1e-7 # Smooth the argument of the log to prevent potential numerical underflows
        batch_keyword_log_prob = -torch.log(torch.softmax(batch_keyword_logit, dim=-1) + eps)
        ### Shape of batch_keyword_log_prob: (search_batch_size, num_keyword)
        ## Build a mask to ignore padding keywords in the log probabilities later
        mask = (keyword_ids == -1)
        ## Gather the log probabilities for the actual keywords in each interaction
        keyword_ids[keyword_ids == -1] = 0 # Negative indices are not supported in gather function, replace with any item
        batch_keyword_log_prob = torch.gather(batch_keyword_log_prob, 1, keyword_ids)
        ### Shape of batch_keyword_log_prob: (search_batch_size, batch_query_size)
        ## Mask the log probabilities to ignore padding keywords
        batch_keyword_log_prob.masked_fill_(mask, 0.0)
        ### Shape of batch_keyword_log_prob: (search_batch_size, batch_query_size)
        ## Calculate the reconstruction loss
        reconstruction_loss = torch.sum(batch_keyword_log_prob, 1) / query_sizes.unsqueeze(-1) # Average on query keywords
        reconstruction_loss = torch.mean(reconstruction_loss) # Average on batch search instances

        loss = recommendation_loss + self.loss_weight * reconstruction_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class DREM(nn.Module):
    """
        Implementation of the Dynamic Relation Embedding Model (DREM) model for search and recommendation. Each
        interaction corresponds to a user, an item, and zero, one or more keywords. The DREM model was originally
        proposed in:
        Ai, Q., Zhang, Y., Bi, K., & Bruce Croft, W. (2020). Explainable Product Search with a Dynamic Relation
        Embedding Model. ACM Transactions on Information Systems, 38(1).
    """
    def __init__(self, options):
        super(DREM, self).__init__()

        self.num_user = options.num_user
        self.num_item = options.num_item
        self.num_keyword = options.num_keyword
        self.embed_dim = options.embed_dim
        self.lr = options.lr
        self.num_neg_sample = options.num_neg_sample
        self.device_embed = options.device_embed
        self.device_ops = options.device_ops

        # Embeddings
        ## Definition
        self.user_embeddings = nn.Embedding(self.num_user, self.embed_dim) # User embeddings to be learned
        self.item_embeddings = nn.Embedding(self.num_item, self.embed_dim) # Item embeddings to be learned
        self.keyword_embeddings = nn.Embedding(self.num_keyword, self.embed_dim) # Keyword embeddings to be learned
        ## Initialization
        nn.init.normal_(self.user_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.item_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.keyword_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        ## Move embeddings to device
        self.user_embeddings = self.user_embeddings.to(self.device_embed)
        self.item_embeddings = self.item_embeddings.to(self.device_embed)
        self.keyword_embeddings = self.keyword_embeddings.to(self.device_embed)

        # Components of the model
        self.scorer = DotProd3DScorer(self.device_ops, drem_version=True).to(self.device_ops)
        self.keyword_aggregator = FullyConnectedAggregator(self.embed_dim, self.embed_dim,
                                                           self.device_ops).to(self.device_ops)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=options.weight_decay)

    @torch.no_grad()
    def predict(self, user_ids, keyword_ids, query_sizes, item_ids=None):
        """
        Compute the score predictions at test time
        Args:
            user_ids: (array<int>) users for whom to return items
            keyword_ids: (array<int>) keywords for which to return items
            query_sizes: (tensor<int>) number of keywords for each query
            item_ids: (array<int>) items for which prediction scores are desired; if not provided, predictions for all
            items will be computed
        Returns:
            scores: (tensor<float>) predicted scores for all items in item_ids
        """
        # If no item_ids is provided, consider all items as potential predictions
        if item_ids is None:
            item_ids = torch.tensor(np.arange(self.num_item), dtype=torch.long, device=self.device_embed)

        # Fetch the user/item/keyword embeddings for the minibatch
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (num_item, embed_dim)

        # Extend the keyword embeddings with the missing keyword embedding (mean) and the padding keyword embedding (zero)
        extended_keyword_embeddings = torch.zeros((self.num_keyword + 2, self.embed_dim))
        extended_keyword_embeddings[:self.num_keyword] = self.keyword_embeddings.weight
        #extended_keyword_embeddings[self.num_keyword] = torch.mean(self.keyword_embeddings.weight, dim=0).detach()
        batch_keyword_embeddings = extended_keyword_embeddings[keyword_ids].to(self.device_ops)
        ## Shape of batch_keyword_embeddings: (batch_size, batch_query_size, embed_dim)
        # For each interaction, aggregate the keyword embeddings for all the keywords in the query
        batch_keyword_embeddings = self.keyword_aggregator(batch_keyword_embeddings, query_sizes)
        ## Shape of batch_keyword_embeddings: (batch_size, embed_dim)

        # Compute the scores
        scores = self.scorer(batch_user_embeddings, batch_item_embeddings, batch_keyword_embeddings)
        ## Shape of scores: (batch_size, num_item)
        return scores

    def forward(self, batch):
        # Unpack the content of the minibatch
        user_ids = batch['user_ids']
        ## Shape of user_ids: (batch_size)
        item_ids = batch['item_ids']
        ## Shape of item_ids: (batch_size)
        keyword_ids = batch['keyword_ids']
        ## Shape of keyword_ids: (batch_size, batch_query_size)
        query_sizes = batch['query_sizes']
        ## Shape of query_sizes: (batch_size)

        # Fetch the user/item/keyword embeddings for the minibatch
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (batch_size, embed_dim)

        # Extend the keyword embeddings with the missing keyword embedding (mean) and the padding keyword embedding (zero)
        extended_keyword_embeddings = torch.zeros((self.num_keyword + 2, self.embed_dim))
        extended_keyword_embeddings[:self.num_keyword] = self.keyword_embeddings.weight
        #extended_keyword_embeddings[self.num_keyword] = torch.mean(self.keyword_embeddings.weight, dim=0).detach()
        batch_keyword_embeddings = extended_keyword_embeddings[keyword_ids].to(self.device_ops)
        ## Shape of batch_keyword_embeddings: (batch_size, batch_query_size, embed_dim)
        # For each interaction, aggregate the keyword embeddings for all the keywords in the query
        batch_keyword_embeddings = self.keyword_aggregator(batch_keyword_embeddings, query_sizes)
        ## Shape of batch_keyword_embeddings: (batch_size, embed_dim)

        # Calculate the recommendation loss on the minibatch using cross-entropy
        positive_score = self.scorer(batch_user_embeddings, batch_item_embeddings, batch_keyword_embeddings)
        loss = ce_loss(positive_score, 1.0)
        ## Shape of positive_score: (batch_size)
        for i in range(self.num_neg_sample):
            # Negative sampling
            negative_item_ids = sample_items(self.num_item, item_ids.size())
            negative_item_ids = torch.tensor(negative_item_ids, dtype=torch.long, device=self.device_embed)

            # Compute the CE loss on the negative scores
            batch_negative_item_embeddings = self.item_embeddings(negative_item_ids).to(self.device_ops)
            ## Shape of batch_negative_item_embeddings: (batch_size, embed_dim)
            negative_score = self.scorer(batch_user_embeddings, batch_negative_item_embeddings, batch_keyword_embeddings)
            ## Shape of negative_score: (batch_size)
            loss += ce_loss(negative_score, 0.0)
        loss /= (self.num_neg_sample + 1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class HyperSaR(nn.Module):
    """
        Implementation of the HyperSaR model for search and recommendation. The nodes of the hypergraph correspond to
        users, items and keywords. The (hyper)edges are associated with a user and an item for the recommendation
        interactions, and a user, an item and any number of keywords for the search interactions.
    """
    def __init__(self, options, train_dataset):
        super(HyperSaR, self).__init__()

        self.num_user = options.num_user
        self.num_item = options.num_item
        self.num_keyword = options.num_keyword
        self.embed_dim = options.embed_dim
        self.lr = options.lr
        self.num_neg_sample = options.num_neg_sample
        self.edge_dropout = options.edge_dropout
        self.num_layer = options.num_layer
        self.loss_weight = options.loss_weight
        self.device_embed = options.device_embed
        self.device_ops = options.device_ops
        self.norm_adj_mat = train_dataset.norm_adj_mat.to(self.device_embed)

        # Embeddings
        ## Definition
        self.user_embeddings = nn.Embedding(self.num_user, self.embed_dim) # User embeddings to be learned
        self.item_embeddings = nn.Embedding(self.num_item, self.embed_dim) # Item embeddings to be learned
        self.keyword_embeddings = nn.Embedding(self.num_keyword, self.embed_dim) # Keyword embeddings to be learned
        ## Initialization
        nn.init.normal_(self.user_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.item_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        nn.init.normal_(self.keyword_embeddings.weight, 0.0, 1.0 / self.embed_dim)
        ## Move embeddings to device
        self.user_embeddings = self.user_embeddings.to(self.device_embed)
        self.item_embeddings = self.item_embeddings.to(self.device_embed)
        self.keyword_embeddings = self.keyword_embeddings.to(self.device_embed)

        # Components of the model
        self.scorer = DotProd3DScorer(self.device_ops).to(self.device_ops)
        self.dropout_layer = SparseDropout(self.device_embed, self.edge_dropout).to(self.device_embed)
        self.keyword_aggregator = SumAggregator(self.device_ops).to(self.device_ops)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=options.weight_decay)

    def compute_embeddings(self):
        """
        Compute the higher-order embeddings for all users, items and keywords after propagation in the hypergraph
        Returns:
            (all_user_embeddings, all_item_embeddings, all_keyword_embeddings): (tensor<float>,tensor<float>,
            tensor<float>) embeddings for all user, item and keyword nodes after propagation in the hypergraph
        """

        # Propagate embeddings in the hypergraph
        user_embeddings = self.user_embeddings.weight
        item_embeddings = self.item_embeddings.weight
        keyword_embeddings = self.keyword_embeddings.weight
        layer_all_embeddings = torch.cat([user_embeddings, item_embeddings, keyword_embeddings]) # 0th layer
        ## Shape of layer_all_embeddings: (num_user + num_item + num_keyword, embed_dim)
        all_embeddings = [layer_all_embeddings]
        norm_adj_mat = self.dropout_layer(self.norm_adj_mat) # Perform layer-shared edge dropout
        for layer in range(self.num_layer):
            # Propagate node embeddings to edges and edge embeddings to node in a single operation
            layer_all_embeddings = torch.sparse.mm(norm_adj_mat, layer_all_embeddings)
            # Store the the resulting node embeddings for this layer
            all_embeddings.append(layer_all_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        ## Shape of all_embeddings: (num_user + num_item + num_keyword, num_layer, embed_dim)

        # Aggregate embeddings at different layers
        aggreg_all_embeddings = torch.mean(all_embeddings, dim=1)
        ## Shape of aggregated_all_embeddings: (num_user + num_item + num_keyword, embed_dim)
        all_user_embeddings, all_item_embeddings, all_keyword_embeddings = torch.split(
            aggreg_all_embeddings, [self.num_user, self.num_item, self.num_keyword]
        )
        ## Shape of all_user_embeddings: (num_user, embed_dim)
        ## Shape of all_item_embeddings: (num_item, embed_dim)
        ## Shape of all_keyword_embeddings: (num_keyword, embed_dim)

        # Extend the keyword embeddings with the missing keyword embedding and the padding keyword embedding (both zero)
        extended_all_keyword_embeddings = torch.zeros((self.num_keyword + 2, self.embed_dim), device=self.device_embed)
        extended_all_keyword_embeddings[:self.num_keyword] = keyword_embeddings # Layer 0 only
        all_keyword_embeddings = extended_all_keyword_embeddings
        ## Shape of all_keyword_embeddings: (num_keyword + 2, embed_dim)

        return (all_user_embeddings, all_item_embeddings, all_keyword_embeddings)

    @torch.no_grad()
    def predict(self, user_ids, keyword_ids, query_sizes, item_ids=None):
        """
        Compute the score predictions at test time
        Args:
            user_ids: (array<int>) users for whom to return items
            keyword_ids: (array<int>) keywords for which to return items
            query_sizes: (tensor<int>) number of keywords for each query
            item_ids: (array<int>) items for which prediction scores are desired; if not provided, predictions for all
            items will be computed
        Returns:
            scores: (tensor<float>) predicted scores for all items in item_ids
        """
        # If no item_ids is provided, consider all items as potential predictions
        if item_ids is None:
            item_ids = torch.tensor(np.arange(self.num_item), dtype=torch.long, device=self.device_embed)

        # Compute the higher-order user/item/keyword embeddings based on the graph
        all_user_embeddings, all_item_embeddings, all_keyword_embeddings = self.compute_embeddings()
        batch_user_embeddings = all_user_embeddings[user_ids].to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = all_item_embeddings[item_ids].to(self.device_ops)
        ## Shape of batch_item_embeddings: (num_item, embed_dim)
        batch_keyword_embeddings = all_keyword_embeddings[keyword_ids].to(self.device_ops)
        ## Shape of batch_keyword_embeddings: (batch_size, batch_query_size, embed_dim)
        # For each interaction, aggregate the keyword embeddings for all the keywords in the query
        batch_keyword_embeddings = self.keyword_aggregator(batch_keyword_embeddings, query_sizes)
        ## Shape of batch_keyword_embeddings: (batch_size, embed_dim)

        # Compute the scores
        scores = self.scorer(batch_user_embeddings, batch_item_embeddings, batch_keyword_embeddings)
        ## Shape of scores: (batch_size, num_item)

        return scores

    def forward(self, batch):
        # Unpack the content of the minibatch
        user_ids = batch['user_ids']
        ## Shape of user_ids: (batch_size)
        item_ids = batch['item_ids']
        ## Shape of item_ids: (batch_size)
        keyword_ids = batch['keyword_ids']
        ## Shape of keyword_ids: (batch_size, batch_query_size)
        query_sizes = batch['query_sizes']
        ## Shape of query_sizes: (batch_size)

        # Compute the higher-order user/item/keyword embeddings based on the graph
        all_user_embeddings, all_item_embeddings, all_keyword_embeddings = self.compute_embeddings()
        batch_user_embeddings = all_user_embeddings[user_ids].to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)
        batch_item_embeddings = all_item_embeddings[item_ids].to(self.device_ops)
        ## Shape of batch_item_embeddings: (batch_size, embed_dim)
        batch_keyword_embeddings = all_keyword_embeddings[keyword_ids].to(self.device_ops)
        ## Shape of batch_keyword_embeddings: (batch_size, batch_query_size, embed_dim)
        # For each interaction, aggregate the keyword embeddings for all the keywords in the query
        batch_query_embeddings = self.keyword_aggregator(batch_keyword_embeddings, query_sizes)
        ## Shape of batch_query_embeddings: (batch_size, embed_dim)

        # Calculate the context-item matching loss on the minibatch using BPR
        cim_loss = torch.tensor(0.0, dtype=torch.float, device=self.device_ops)
        positive_score = self.scorer(batch_user_embeddings, batch_item_embeddings, batch_query_embeddings)
        ## Shape of positive_score: (batch_size)
        for i in range(self.num_neg_sample):
            # Negative sampling
            negative_item_ids = sample_items(self.num_item, item_ids.size())
            negative_item_ids = torch.tensor(negative_item_ids, dtype=torch.long, device=self.device_embed)

            # Compute the BPR loss on the positive and negative scores
            batch_negative_item_embeddings = all_item_embeddings[negative_item_ids].to(self.device_ops)
            ## Shape of batch_negative_item_embeddings: (batch_size, embed_dim)
            negative_score = self.scorer(batch_user_embeddings, batch_negative_item_embeddings, batch_query_embeddings)
            ## Shape of negative_score: (batch_size)
            cim_loss += bpr_loss(positive_score, negative_score)
        cim_loss /= self.num_neg_sample

        # Calculate the query likelihood loss on the minibatch using user- and item-specific probabilities over keywords
        ql_loss = torch.tensor(0.0, dtype=torch.float, device=self.device_ops)
        if self.loss_weight > 0:
            ## Consider only the search interactions with a non-empty query
            search_selector = keyword_ids[:, 0] != self.num_keyword # Only non-empty queries don't start with num_keyword
            batch_user_embeddings = batch_user_embeddings[search_selector]
            batch_item_embeddings = batch_item_embeddings[search_selector]
            query_sizes = query_sizes[search_selector]
            ### Shape of batch_item_embeddings: (search_batch_size, embed_dim)
            keyword_ids = keyword_ids[search_selector]
            ### Shape of keyword_ids: (search_batch_size, batch_query_size)
            ## Calculate the user-specific and item-specific probability over all keywords for each search interaction
            keyword_embeddings = all_keyword_embeddings[:self.num_keyword].to(self.device_ops)
            batch_user_logit = batch_user_embeddings @ keyword_embeddings.t()
            batch_item_logit = batch_item_embeddings @ keyword_embeddings.t()
            eps = 1e-7 # Smooth the argument of the log to prevent potential numerical underflows
            batch_keyword_log_prob = -torch.log(torch.softmax(batch_user_logit, dim=-1)
                                               * torch.softmax(batch_item_logit, dim=-1) + eps)
            ### Shape of batch_keyword_log_prob: (search_batch_size, num_keyword)
            ## Build a mask to ignore padding keywords in the log probabilities later
            mask = (keyword_ids == -1)
            ## Gather the log probabilities for the actual keywords in each interaction
            keyword_ids[keyword_ids == -1] = 0 # Negative indices aren't supported in gather fn, replace with any item
            batch_keyword_log_prob = torch.gather(batch_keyword_log_prob, 1, keyword_ids)
            ### Shape of batch_keyword_log_prob: (search_batch_size, batch_query_size)
            ## Mask the log probabilities to ignore padding keywords
            batch_keyword_log_prob.masked_fill_(mask, 0.0)
            ### Shape of batch_keyword_log_prob: (search_batch_size, batch_query_size)
            ## Calculate the reconstruction loss
            ql_loss = torch.sum(batch_keyword_log_prob, 1) / query_sizes.unsqueeze(-1) # Avg on query keywords
            ql_loss = torch.mean(ql_loss) # Avg on batch search instances

        loss = cim_loss + self.loss_weight * ql_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
