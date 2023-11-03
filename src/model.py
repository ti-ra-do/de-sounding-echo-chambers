import sys
sys.path.append('src')

import torch
import random
import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from collections import defaultdict
from torch.utils.data import DataLoader

from data import TrainDataset, TestDataset, OneShotIterator


class TrainPipeline(object):
    def __init__(self, model, optimizer, evaluator, warm_up_steps, max_steps, negative_adversarial_sampling, adversarial_temperature,  
        regularization, uni_weight):

        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator

        # TODO: Investigate adjusting learning rate
        self.current_learning_rate = optimizer.param_groups[0]['lr']
        self.max_steps = max_steps

        # TODO: Investigate
        if warm_up_steps:
            self.warm_up_steps = warm_up_steps
        else:
            self.warm_up_steps = self.max_steps // 2

        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.adversarial_temperature = adversarial_temperature
        self.regularization = regularization
        self.uni_weight = uni_weight

        self.cuda = next(model.parameters()).is_cuda
        
    def reset_community_embeddings(self, new_node_ranges):
        for node_type in [n for n in new_node_ranges if n != 'community']:
            assert new_node_ranges[node_type][0] < new_node_ranges['community'][0]

        keep_ids, nentity = new_node_ranges['community']
    
        self.model.reset_embeddings(nentity, keep_ids=keep_ids)

        if self.cuda:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.current_learning_rate
        )

    def train(self, data):
        # TODO: Think about on which level to loop: For now in every step we train each dataset once
        # Would it be better to loop and then randomly select a sample from one dataset?
        logs = []
        for set_name in data:
            log = self.train_step(data[set_name]['train_iterator'])
            logs.append(log)
        return logs

    def train_step(self, iterator):
        if self.cuda:
            torch.cuda.empty_cache()

        self.model.train()
        self.optimizer.zero_grad()
        qtype = iterator.qtype

        positive_sample, negative_sample, subsampling_weight, mode = next(iterator)

        if self.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = self.model((positive_sample, negative_sample), mode, qtype=qtype)
        
        if self.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = self.model(positive_sample, qtype=qtype)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if self.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if self.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = self.regularization * (
                    self.model.entity_embedding.norm(p=3) ** 3 +
                    self.model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        self.optimizer.step()

        log = {
            'metrics': {
                **regularization_log,
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            },
        }

        return log

    def test(self, data, edge_chains, test_set='test'):
        metrics = dict()

        for set_name in data:
            test_triples = data[set_name]['{}_triples'.format(test_set)]
            chains = edge_chains[set_name]
            for chain in chains:
                chain_tuple = edge_chains[set_name][chain]
                '''
                chain_tuple = [edge_chains[set_name][chain]] if set_name == '1-chain' \
                    else list(edge_chains[set_name][chain])
                '''
                target_chain_indices = []
                for index, v in enumerate(test_triples['chain']):
                    if v == chain_tuple:
                        target_chain_indices.append(index)

                chain_test_triples = dict()
                for key, values in test_triples.items():
                    values = np.asarray(values, dtype=object)
                    reduced_values = values[target_chain_indices]
                    chain_test_triples[key] = reduced_values.tolist()
                metrics[chain] = self.test_step(chain_test_triples, self.test_batch_size)

        return metrics

    def test_step(self, test_triples, batch_size, random_sampling=True):
        if self.cuda:
            torch.cuda.empty_cache()

        self.model.eval()

        test_dataset = TestDataset(
            test_triples,
            'tail-batch',
            random_sampling,
            self.negative_eval_size, 
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            #num_workers=max(1, self.cpu_num // 2),
            num_workers=0,
            collate_fn=test_dataset.collate_fn
        )

        test_logs = defaultdict(list)

        with torch.no_grad():
            for batch in test_dataloader: 
                positive_sample, negative_sample, mode = batch

                if self.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()

                score = self.model((positive_sample, negative_sample), mode, qtype=test_triples['qtype'][0])

                batch_results = self.evaluator.eval({
                    'y_pred_pos': score[:, 0],
                    'y_pred_neg': score[:, 1:]})

                for metric in batch_results:
                    test_logs[metric].append(batch_results[metric])

            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()
        return metrics

    def adjust_optimizer(self, model):
        logging.info("info | Change learning_rate to {}".format(self.current_learning_rate))
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=self.current_learning_rate
        )
        self.warm_up_steps = self.warm_up_steps * 3


class MeanSet(nn.Module):
    def __init__(self):
        super(MeanSet, self).__init__()

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if len(embeds3) > 0:
            return torch.mean(torch.stack([embeds1, embeds2, embeds3], dim=0), dim=0)
        else:
            return torch.mean(torch.stack([embeds1, embeds2], dim=0), dim=0)


class CenterSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, agg_func=torch.min, bn='no', nat=1, name='Real_center'):
        super(CenterSet, self).__init__()
        assert nat == 1, 'vanilla method only support 1 nat now'
        self.center_use_offset = center_use_offset
        self.agg_func = agg_func
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims*2, mode_dims))
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))

        nn.init.xavier_uniform(self.pre_mats)
        self.register_parameter("premat_%s"%name, self.pre_mats)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn2 = nn.BatchNorm1d(mode_dims)
            self.bn3 = nn.BatchNorm1d(mode_dims)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat_%s"%name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if self.center_use_offset:
            temp1 = torch.cat([embeds1, embeds1_o], dim=1)
            temp2 = torch.cat([embeds2, embeds2_o], dim=1)
            if len(embeds3) > 0:
                temp3 = torch.cat([embeds3, embeds3_o], dim=1)
        else:
            temp1 = embeds1
            temp2 = embeds2
            if len(embeds3) > 0:
                temp3 = embeds3

        if self.bn == 'no':
            temp1 = F.relu(temp1.mm(self.pre_mats))
            temp2 = F.relu(temp2.mm(self.pre_mats))
        elif self.bn == 'before':
            temp1 = F.relu(self.bn1(temp1.mm(self.pre_mats)))
            temp2 = F.relu(self.bn2(temp2.mm(self.pre_mats)))
        elif self.bn == 'after':
            temp1 = self.bn1(F.relu(temp1.mm(self.pre_mats)))
            temp2 = self.bn2(F.relu(temp2.mm(self.pre_mats)))

        if len(embeds3) > 0:
            if self.bn == 'no':
                temp3 = F.relu(temp3.mm(self.pre_mats))
            elif self.bn == 'before':
                temp3 = F.relu(self.bn3(temp3.mm(self.pre_mats)))
            elif self.bn == 'after':
                temp3 = self.bn3(F.relu(temp3.mm(self.pre_mats)))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        if model_name in ['RotatE', 'ComplEx']:
            self.entity_dim = hidden_dim * 2 
            self.relation_dim = hidden_dim * 2 
            double_entity_embedding = True
            double_relation_embedding = True
        else:
            self.entity_dim = hidden_dim
            self.relation_dim = hidden_dim
            

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        #self.deepsets = CenterSet(self.relation_dim, self.relation_dim, False, agg_func = torch.mean, bn=bn, nat=nat)
        self.deepsets = MeanSet()

        if model_name not in ['MF', 'TransE', 'DistMult', 'ComplEx', 'RotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        self.models = {
            'MF': MF,
            'TransE': TransE,
            'DistMult': DistMult,
            'ComplEx': ComplEx,
            'RotatE': RotatE,
        }
        if self.model_name == 'TransE':
            self.model_func = self.models[model_name](self.gamma)
        else:
            self.model_func = self.models[model_name]()

    def reset_embeddings(self, nentity, keep_ids):
        old_emb = self.entity_embedding.data[:keep_ids, :]
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.entity_embedding.data[:keep_ids, :] = old_emb 
        
    def forward(self, sample, mode='single', qtype='1-chain'):
        if qtype == 'chain-inter':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1

                head_1 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).view(batch_size, negative_sample_size, -1)

                relation_11 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                relation_12 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 4]).unsqueeze(1).unsqueeze(1)
                relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                # todo: community & topic embeddings as sentence embeddings
                # for simplicity: include a mapper community_idx -> {entity_idx}_c
                head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

                relation_11 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1).unsqueeze(1)
                relation_12 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1).unsqueeze(1)
                relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

        elif qtype == 'inter-chain' or qtype == 'union-chain':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
            head = torch.cat([head_1, head_2], dim=0)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1) 

            relation_11 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1)
            relation_12 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
            relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

        elif qtype == '2-inter' or qtype == '2-union':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1

                head_1 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
                tail = torch.cat([tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                
            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
                tail = torch.cat([tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)

        elif qtype == '1-chain' or qtype == '2-chain':
            if mode == 'single':
                head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)

                if qtype == '1-chain' or qtype == '2-chain':
                    relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                if qtype == '2-chain':
                    relation2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)
                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)

                if qtype == '1-chain' or qtype == '2-chain':
                    relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1).unsqueeze(1)
                if qtype == '2-chain':
                    relation2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            elif mode == 'head-batch':
                pass
        else:
            raise ValueError('qtype %s not supported' % qtype)

        score = self.score(head, relation, tail, qtype)

        return score

    @staticmethod
    def normalized_euclidean_distance(x1, x2, dim=1, eps=1e-8):
        """
        Normalized eucledian distance in pytorch.

        Cases:
            1. For comparison of two vecs directly make sure vecs are of size [B] e.g. when using nes as a loss function.
                in this case each number is not considered a representation but a number and B is the entire vector to
                compare x1 and x2.
            2. For comparison of two batch of representation of size 1D (e.g. scores) make sure it's of shape [B, 1].
                In this case each number *is* the representation of the example. Thus a collection of reps
                [B, 1] is mapped to a rep of the same size [B, 1]. Note usually D does decrease since reps are not of size 1
                (see case 3)
            3. For the rest specify the dimension. Common use case [B, D] -> [B, 1] for comparing two set of
                activations of size D. In the case when D=1 then we have [B, 1] -> [B, 1]. If you meant x1, x2 [D, 1] to be
                two vectors of size D to be compare feed them with shape [D].

        https://discuss.pytorch.org/t/how-does-one-compute-the-normalized-euclidean-distance-similarity-in-a-numerically-stable-way-in-a-vectorized-way-in-pytorch/110829
        https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance/498753?noredirect=1#comment937825_498753
        """
        # to compute ned for two individual vectors e.g to compute a loss (NOT BATCHES/COLLECTIONS of vectorsc)
        if len(x1.size()) == 1:
            # [K] -> [1]
            ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps))
        # if the input is a (row) vector e.g. when comparing two batches of acts of D=1 like with scores right before sf
        elif x1.size() == torch.Size([x1.size(0), 1]):  # note this special case is needed since var over dim=1 is nan (1 value has no variance).
            # [B, 1] -> [B]
            ned_2 = 0.5 * ((x1 - x2)**2 / (x1**2 + x2**2 + eps)).squeeze()  # Squeeze important to be consistent with .var, otherwise tensors of different sizes come out without the user expecting it
        # common case is if input is a batch
        else:
            # e.g. [B, D] -> [B]
            ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
        return ned_2 ** 0.5

    @staticmethod
    def prospect_weighting(x, gamma=0.5, k=10): # gamma = 0.4, k = 10
        return 1.0 / (1.0 + torch.exp(-k * (x - gamma)))
        #return (x ** gamma) / (x ** gamma + (1.0 - x) ** gamma) ** (1.0 / gamma)

    def distance_fn(self, relation, query_type):
        relation_emb = self.get_relation_embedding(relation).unsqueeze(1)
        def distance(head, target, weight=True, gamma=0.5, k=10):
            head_emb = self.get_entity_embedding(head)
            target_emb = self.get_entity_embedding(target)
            
            head_proj = self.project(head_emb, relation_emb, query_type)

            dist = KGEModel.normalized_euclidean_distance(head_proj, target_emb)
            if weight:
                return KGEModel.prospect_weighting(dist, gamma, k)
            else:
                return dist

        return distance   

    def score(self, head, relation, tail, qtype='1-chain'):
        return self.model_func.score(head, relation, tail, qtype)

    def project(self, head, relation, qtype='1-chain'):
        return self.model_func.project(head, relation, qtype)

    def get_embedding(self, embedding, idx):
        idx = torch.tensor(idx, dtype=torch.long)

        if self.cuda():
            idx = idx.cuda()

        return torch.index_select(embedding, dim=0, index=idx)

    def get_relation_embedding(self, idx):
        return self.get_embedding(self.relation_embedding, idx)
        
    def get_entity_embedding(self, idx):
        return self.get_embedding(self.entity_embedding, idx)

    def rank_fn(self, rank_chain, query_type, max_num_tails=None):
        rank_chain_tensor = torch.tensor(rank_chain, dtype=torch.long)
        def rank(head, tail):
            head_tensor = torch.tensor(head, dtype=torch.long)

            if max_num_tails is not None and len(tail) > max_num_tails:
                tail = random.sample(tail, k=max_num_tails)
                
            tail_tensor = torch.tensor(tail, dtype=torch.long)
            ranked_tails = self.rank(head_tensor, rank_chain_tensor, tail_tensor, query_type)
            return [tail[c] for c in ranked_tails][::-1]

        return rank

    def rank(self, head, edge_tuple, tail, qtype, head_2=None):
        if head_2 is None:
            head_2 = head 

        sample = [
            head.repeat(tail.shape[0]).unsqueeze(1),
            head_2.repeat(tail.shape[0]).unsqueeze(1),
            edge_tuple.repeat(tail.shape[0], 1),
            tail.unsqueeze(1)
        ]

        sample = torch.cat(sample, dim=1)

        if self.cuda:
            sample = sample.cuda()

        scores = self.forward(sample, mode='single', qtype=qtype)
        scores = F.logsigmoid(scores).squeeze(dim=1)
        scores = scores.argsort(dim=0, descending=False).view(-1).tolist()
        return scores  


class Projection(ABC):
    @abstractmethod
    def project(self, head, relation, qtype):
        pass

    @abstractmethod
    def score(self, head, relation, tail, qtype):
        pass


class MF(Projection):
    def project(self, head, relation, qtype):
        return head

    def score(self, head, relation, tail, qtype):
        if qtype == '2-inter':
            head = head.view([-1, 2, head.shape[-1]])
            head = torch.mean(head, dim=1, keepdim=True)

            tail = tail.view([-1, 2, head.shape[-1]])
            tail = torch.mean(head, dim=1, keepdim=True) 
        return torch.sum(head * tail, dim=2)


class TransE(Projection):
    def __init__(self, gamma):
        self.gamma = gamma

    def project(self, head, relation, qtype):
        if qtype == 'chain-inter':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)

            score_1 = (heads[0] + relations[0][:, 0, :, :] + relations[1][:, 0, :, :]).squeeze(1)
            score_2 = (heads[1] + relations[2][:, 0, :, :]).squeeze(1)
            conj_score = self.deepsets(score_1, None, score_2, None).unsqueeze(1)
            return conj_score

        elif qtype == 'inter-chain':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)

            score_1 = (heads[0] + relations[0][:, 0, :, :]).squeeze(1)
            score_2 = (heads[1] + relations[1][:, 0, :, :]).squeeze(1)
            conj_score = self.deepsets(score_1, None, score_2, None).unsqueeze(1)
            score = conj_score + relations[2][:, 0, :, :]
            return score
        elif qtype == 'union-chain':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)

            score_1 = heads[0] + relations[0][:, 0, :, :] + relations[2][:, 0, :, :]
            score_2 = heads[1] + relations[1][:, 0, :, :] + relations[2][:, 0, :, :]
            conj_score = torch.stack([score_1, score_2], dim=0)
            return conj_score
        else:
            rel_len = relation.shape[1]
            score = head
            for rel in range(rel_len):
                score = score + relation[:, rel, :, :]
            if 'inter' not in qtype and 'union' not in qtype:
                return score
                #score = score - tail
            else:
                num_heads = int(qtype.split('-')[0])
                score = score.squeeze(1)
                scores = torch.chunk(score, num_heads, dim=0)
                if 'inter' in qtype:
                    conj_score = self.deepsets(scores[0], None, scores[1], None)
                    conj_score = conj_score.unsqueeze(1)
                    return conj_score
                else:
                    assert False, 'qtype does not exist: {}'.format(qtype)

    def score(self, head, relation, tail, qtype):
        score = self.project(head, relation, qtype)
            
        if 'inter' in qtype or 'union' in qtype and 'chain' not in qtype:
            if 'inter' in qtype:
                num_heads = int(qtype.split('-')[0])
                tail = torch.chunk(tail, num_heads, dim=0)[0]  
            else:
                assert False, 'qtype does not exist: {}'.format(qtype)
        
        score = score - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        return score


class DistMult(Projection):
    def project(self, head, relation, qtype):
        rel_len = relation.shape[1]

        score = head
        for rel in range(rel_len):
            score = score * relation[:, rel, :]

        return score

    def score(self, head, relation, tail, qtype):
        score = self.project(head, relation, qtype)
        score = score * tail
        score = score.sum(dim=-1)
        return score


class ComplEx(Projection):
    def project(self, head, relation, qtype):
        rel_len = relation.shape[1]

        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)

        re_score = re_head * re_relation[:, 0, :] - im_head * im_relation[:, 0, :]
        im_score = re_head * im_relation[:, 0, :] + im_head * re_relation[:, 0, :]

        for rel in range(1, rel_len):
            re_score = re_score * re_relation[:, rel, :] - im_score * im_relation[:, rel, :]
            im_score = re_score * im_relation[:, rel, :] + im_score * re_relation[:, rel, :]

        return re_score, im_score

    def score(self, head, relation, tail, qtype):
        re_score, im_score = self.project(head, relation, qtype)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
        score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=-1)
        return score


class RotatE(Projection):
    def project(self, head, relation, qtype):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)


        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        return re_score, im_score

    def score(self, head, relation, tail, qtype):
        re_score, im_score = self.project(head, relation, qtype)

        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score


