import numpy as np
import torch
from datetime import datetime

def precision(correct_predictions, k):
    num_hit = torch.sum(correct_predictions, dim=-1)
    return num_hit / k

def recall(correct_predictions, num_relevant):
    num_hit = torch.sum(correct_predictions, dim=-1)
    return num_hit / num_relevant

def ndcg(correct_predictions, num_relevant, k):
    ideal_correct_predictions = torch.zeros_like(correct_predictions)
    batch_size = ideal_correct_predictions.shape[0]
    for sample in range(batch_size):
        ideal_correct_predictions[sample, :num_relevant[sample]] = 1
    return dcg(correct_predictions, k) / dcg(ideal_correct_predictions, k)

def dcg(correct_predictions, k):
    result = 0.0
    for rank in range(k):
        result += correct_predictions[:, rank] / np.log2(rank + 2)
    return result

def map(correct_predictions, num_relevant, k):
    result = 0.0
    for rank in range(k):
        result += precision(correct_predictions[:, :rank + 1], rank + 1) * correct_predictions[:, rank]
    result /= num_relevant
    return result

def evaluate(correct_predicted_interactions, num_true_interactions, metrics):
    """
    Evaluates a ranking model in terms of precision and recall for the given cutoff values
    Args:
        correct_predicted_interactions: (array<bool>: n_rows * max(cutoffs)) 1 iff prediction matches a true interaction
        num_true_interactions: (array<bool>: n_rows) number of true interactions associated to each row
        metrics: (list<tuple<string,int>>) list of metrics to consider, with tuples made of the metric type and cutoff

    Returns:
        eval_results: dictionary with evaluation results for each metric cumulated over all rows; keys are the metrics
    """
    eval_results = {}
    for metric in metrics:
        (metric_type, k) = metric # Get the metric type and cutoff e.g. ("precision", 5)
        correct_predictions = correct_predicted_interactions[:, :k]
        k = min(k, correct_predictions.shape[1])
        if metric_type == "precision":
            eval_results[metric] = precision(correct_predictions, k)
        elif metric_type == "recall":
            eval_results[metric] = recall(correct_predictions, num_true_interactions)
        elif metric_type == "ndcg":
            eval_results[metric] = ndcg(correct_predictions, num_true_interactions, k)
        elif metric_type == "map":
            eval_results[metric] = map(correct_predictions, num_true_interactions, k)

    return eval_results

def predict_evaluate(data_loader, options, model, known_interactions):
    max_k = max([metric[1] for metric in options.metrics])
    max_k = min(max_k, options.num_item)
    types = ['all', 'rec', 'search']
    eval_results = {type: {metric: torch.tensor([], dtype=torch.float, device=options.device_ops)
                           for metric in options.metrics} for type in types}

    for (batch_id, batch) in enumerate(data_loader):
        if batch_id % 1 == 0:
            print("Number of batches processed: " + str(batch_id) + "...", datetime.now(), flush=True)

        device_embed = options.device_embed
        device_ops = options.device_ops
        user_ids = batch['user_ids'].to(device_embed)
        item_ids = batch['item_ids'].to(device_ops)
        interaction_types = batch['interaction_types'].to(device_ops)
        batch_size = len(user_ids)

        # Predict the items interacted for each user and mask the items which appeared in known interactions
        if options.model in ["FactorizationMachine", "DeepFM", "JSR", "DREM", "HyperSaR"]:
            keyword_ids = batch['keyword_ids'].to(device_embed)
            query_sizes = batch['query_sizes'].to(device_ops)
            predicted_scores = model.predict(user_ids, keyword_ids, query_sizes)
        else:
            predicted_scores = model.predict(user_ids)
        ## Shape of predicted_scores: (batch_size, num_item)

        # Mask for each user the items from their training set
        mask_value = -np.inf
        for i, user in enumerate(user_ids):
            if int(user) in known_interactions:
                for interaction in known_interactions[int(user)]:
                    item = interaction[0]
                    predicted_scores[i, item] = mask_value
        _, predicted_interactions = torch.topk(predicted_scores, k=max_k, dim=1, largest=True, sorted=True)
        ## Shape of predicted_interactions: (batch_size, num_item)

        # Identify the correct interactions in the top-k predicted items
        correct_predicted_interactions = (predicted_interactions == item_ids.unsqueeze(-1)).float()
        ## Shape of correct_predicted_interactions: (batch_size, max_k)
        num_true_interactions = torch.ones([batch_size], dtype=torch.long, device=options.device_ops) # 1 relevant item
        ## Shape of num_true_interactions: (batch_size)

        # Perform the evaluation
        batch_results = {}
        batch_results['all'] = evaluate(correct_predicted_interactions, num_true_interactions, options.metrics)
        ## Separate results for recommendation and search instances
        recommendation_ids = torch.where(interaction_types == 0)[0]
        batch_results['rec'] = {metric: batch_results['all'][metric][recommendation_ids] for metric in options.metrics}
        search_ids = torch.where(interaction_types == 1)[0]
        batch_results['search'] = {metric: batch_results['all'][metric][search_ids] for metric in options.metrics}

        eval_results = {type: {metric: torch.cat((eval_results[type][metric], batch_results[type][metric]), dim=0)
                                for metric in options.metrics} for type in types}

    eval_results = {type: {metric: torch.mean(eval_results[type][metric], dim=0) for metric in options.metrics}
                    for type in types}

    return eval_results
