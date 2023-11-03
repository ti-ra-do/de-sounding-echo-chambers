import os
import csv
import ast
import json
import torch
import random
import logging
import datetime

import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from community import community_louvain
from sklearn.model_selection import train_test_split

from utils import bidict, merge, jaccard_similarity

import leidenalg as la
import igraph as ig


class DataPipeline(object):
    def __init__(self, num_epochs, node_info_file, edge_info_file, community_info_file):

        self.epoch = 0
        self.num_epochs = num_epochs

        self.epoch_edges = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        self.node_info = pd.read_csv(node_info_file)
        self.edge_info = pd.read_csv(edge_info_file, index_col='edge_type')
        self.edge_info['dependency'] = self.edge_info['dependency'].apply(ast.literal_eval)
        self.edge_info = DataPipeline.add_edge_tuples(self.edge_info)
        self.community_info = pd.read_csv(community_info_file, index_col='id')

    @staticmethod
    def get_community_mapper(user_graph):
        return bidict(nx.get_node_attributes(user_graph, 'community'))


    @staticmethod
    def add_edge_tuples(edge_info):
        ''' Split between single and longer chains '''
        single_chains = edge_info[edge_info['dependency'] == ()]
        multi_chains = pd.concat([edge_info,single_chains]).drop_duplicates(keep=False)

        ''' Get edge tuples for single chains '''
        single_chains.loc[:,'edge_tuple'] = [(x,) for x in single_chains.reset_index().index]

        ''' Get edge tuples for longer chains '''
        multi_edge_tuples = []
        for _, chain in multi_chains.iterrows():
            edge_tuple = [single_chains.loc[c]['edge_tuple'][0] for c in chain['dependency']]
            multi_edge_tuples.append(tuple(edge_tuple))
        multi_chains.loc[:,'edge_tuple'] = multi_edge_tuples

        return pd.concat([single_chains, multi_chains])

    def get_num_nodes(self):
        return self.node_info['count'].sum()

    def get_node_ranges(self):
        node_ranges = dict()
        cur_idx = 0
        for _, row in self.node_info.iterrows():
            count = row['count']
            node_ranges[row['node_type']] = (cur_idx, cur_idx + count)
            cur_idx += count

        return node_ranges

    def get_num_edge_tuples(self, single_only=False):
        if single_only:
            return self.edge_info[self.edge_info['query_type'] == '1-chain'].shape[0]
        else:
            return self.edge_info.shape[0]

    def add_edges(self, edges, epoch=None):
        if epoch is None:
            self.epoch_edges[self.epoch+1] = edges
        else:
            self.epoch_edges[epoch] = edges

    def remove_edges(self, edges):
        if len(edges) == 0:
            return 

        def dict_difference(d1, d2):
            return {k:d1.get(k,d2.get(k)) for k in set(d1) | set(d2)}

        for epoch in self.epoch_edges:
            for query_type in self.epoch_edges[epoch]:
                for edge_type in self.epoch_edges[epoch][query_type]:
                    to_remove = set(frozenset(d.items()) for d in self.epoch_edges[epoch][query_type][edge_type]) - set(frozenset(d.items()) for d in edges[query_type][edge_type])

                    self.epoch_edges[epoch][query_type][edge_type] = [dict(s) for s in to_remove]

    def get_edges(self, epoch=None):
        if epoch is None:
            edges = defaultdict(lambda: defaultdict(list))

            for epoch in self.epoch_edges:
                epoch_edges = self.epoch_edges[epoch]
                edges = merge(edges, epoch_edges)

            return edges
        
        return self.epoch_edges[epoch]

    def init_all_edges(self):
        return self.init_edges(self.df_edges)

    def init_edges(self, df_edges, epoch=None):
        epoch_edges = defaultdict(lambda: defaultdict(list))
        for _, edge in df_edges.iterrows():
            query_type = edge['query_type']
            edge_type = edge['edge_type']

            nodes = {'head_1_id': edge['head'], 'head_2_id': None, 'tail_id': edge['tail']}

            epoch_edges[query_type][edge_type].append(nodes)

        if epoch is None:
            epoch = self.epoch
        self.epoch_edges[epoch] = epoch_edges

    def init_epoch_nodes(self):
        nodes = set()
        for edge_type in self.epoch_edges['1-chain']:
            for edge in self.epoch_edges['1-chain'][edge_type]:
                head_1_id = edge['head_1_id']
                nodes.add(head_1_id)

                tail_id = edge['tail_id']
                nodes.add(tail_id)

        return nodes 

    def get_user_graph_test(self):
        initial_edges = self.init_edges(self.df_edges)
        user_graph = self.init_user_graph(initial_edges)
        return user_graph

    def get_epoch_user_graph(self):
        sim_edges = self.get_sim_edges()

        if self.free_simulation:
            initial_edge_df = self.df_edges[self.df_edges['bin']==0]
        else:
            initial_edge_df = self.df_edges[self.df_edges['bin']<=self.epoch]
        
        initial_edges = self.init_edges(initial_edge_df)
        user_graph = self.init_user_graph(merge(sim_edges, initial_edges))
        nx.set_node_attributes(user_graph, self.community_mapper, 'community')       

        return user_graph

    def init_user_graph(self):
        user_graph = nx.DiGraph()

        edges = self.get_edges()
        for edge in edges['2-chain']['follow']:
            user_1_node = edge['head_1_id']
            user_2_node = edge['tail_id']
            user_graph.add_edge(user_1_node, user_2_node)
        
        user2community = dict()
        for edge in edges['1-chain']['member']:
            user_node = edge['head_1_id']
            community_node = edge['tail_id']
            user2community[user_node] = community_node

        nx.set_node_attributes(user_graph, user2community, 'community') 
        return user_graph

    def init_message_graph(self):
        twitter_graph = nx.DiGraph()
        
        edges = self.get_edges()

        for retweet in edges['1-chain']['repost']:
            user_node = retweet['head_1_id']
            tweet_node = retweet['tail_id']
            twitter_graph.add_edge(user_node, tweet_node)

        for tweet in edges['1-chain']['post']:
            user_node = tweet['tail_id']
            tweet_node = tweet['head_1_id']
            twitter_graph.add_edge(tweet_node, user_node)

        node_ranges = self.get_node_ranges()
        node_attributes = dict()
        for i in range(node_ranges['user'][0], node_ranges['user'][1]):
            node_attributes[i] = 'user'

        for i in range(node_ranges['tweet'][0], node_ranges['tweet'][1]):
            node_attributes[i] = 'tweet'
        
        nx.set_node_attributes(twitter_graph, node_attributes, 'type')
       
        return twitter_graph


    def update_communities(self, nx_user_graph):
        reference_community_mapper = bidict(nx.get_node_attributes(nx_user_graph, 'community'))

        # TODO: Delete?
        comm2comm = dict()

        user_graph = ig.Graph(len(nx_user_graph), list(zip(*list(zip(*nx.to_edgelist(nx_user_graph)))[:2])), directed=True) 
        #partition = la.ModularityVertexPartition(user_graph, initial_membership=self.initial_communities)
        partition = la.ModularityVertexPartition(user_graph)
        optimiser = la.Optimiser()
        optimiser.optimise_partition(partition, n_iterations=10)

        user2comm = dict()
        for community in partition:
            sims = dict()
            for reference_community_idx, reference_community in reference_community_mapper.inverse.items():
                sim = jaccard_similarity(community, reference_community)
                sims[reference_community_idx] = sim

            sims = sorted(sims.items(), key=lambda item: item[1], reverse=True)
            matched_reference_community_idx = sims[0][0]

            community_idx = matched_reference_community_idx
            for user in community:
                user2comm[user] = community_idx 

            comm2comm[community_idx] = matched_reference_community_idx
        
        nx.set_node_attributes(nx_user_graph, user2comm, 'community')

        # TODO: Replace Member Posts as well
        for epoch in self.epoch_edges:
            self.epoch_edges[epoch]['1-chain']['member'] = []

        for user, community in user2comm.items():
            self.epoch_edges[self.epoch]['1-chain']['member'].append({
                'head_1_id': user,
                'head_2_id': None,
                'tail_id': community
            })

    def init_community_edges(self, community_mapper):   
        community_edges = defaultdict(lambda: defaultdict(list))        
        for user, community in community_mapper.items():
            edge = {
                'head_1_id': user,
                'head_2_id': None,
                'tail_id': community
            }
            community_edges['1-chain'][('member', 'user', None, 'community')].append(edge)

        return community_edges

    def init_community_tweet_edges(self, retweet_edges, community_mapper):
        community_tweet_edges = defaultdict(lambda: defaultdict(list))

        for retweet in retweet_edges:
            user_node = retweet['head_1_id']
            tweet_node = retweet['tail_id']

            user_community = community_mapper[user_node]
            edge = {
                'head_1_id': user_community,
                'head_2_id': None,
                'tail_id': tweet_node
            }
            community_tweet_edges['1-chain'][('tweet_link'), 'community', None, 'tweet'].append(edge)

        return community_tweet_edges

    @staticmethod
    def get_num_edges(edge_dict):
        all_edges = []
        for _, edge_types in edge_dict.items():
            for _, edges in edge_types.items():
                all_edges += edges
        return len(all_edges)

    def append_community_intersection(self):
        print(self.relations)

        chain = str((self.relations['retweet'], self.relations['tweet'], self.relations['link']))

        if 'inter-chain' not in self.epoch_data['edges']:
            self.epoch_data['edges']['chain_inter'] = {}

        self.epoch_data['edges']['chain_inter'][('community_intersection', 'user', 'community', 'user')] = []
        community_edges = self.epoch_data['edges']['1-chain'][('link', 'community', None, 'user')]
        user_edges = self.epoch_data['edges']['2-chain'][('user_connect', 'user', None, 'user')]
        for c_edge in community_edges:
            community_id = c_edge['head_1_id']
            c_user_id = c_edge['tail_id']

            for u_edge in user_edges[:5000]:
                u_user_head_id = u_edge['head_1_id']
                u_user_tail_id = u_edge['tail_id']

                if c_user_id == u_user_tail_id:
                    edge = {
                        'edge_type': 'community_intersection',
                        'chain': chain,
                        'query_type': 'chain-inter',
                        'head_1_id': u_user_head_id,
                        'head_1_text': "",
                        'head_1_type': 'user',
                        'head_2_id': community_id,
                        'head_2_text': "",
                        'head_2_type': 'community',
                        'tail_id': c_user_id,
                        'tail_text': "",
                    'tail_type': 'user'
                    }
                    self.epoch_data['edges']['chain_inter'][('community_intersection', 'user', 'community', 'user')].append(edge)
        
    def append_boundary_projection(self, data, G, u2c, c2u):
        user_sample = random.choices(list(G.nodes), k=5000)

        data['1-chain'][('user_boundary', 'user', None, 'user')] = []
        data['2-chain'][('boundary_retweet', 'user', None, 'tweet')] = []

        for user in user_sample:
            user_community = u2c[user]
            community_users = c2u[user_community]
            boundary_users = list(nx.algorithms.boundary.node_boundary(G, community_users))

            retweets = self.db_handler.filter_edges('retweet', head_1_list=[user])

            for boundary_user in boundary_users[:20]:
                edge = {
                    'edge_type': 'user_boundary',
                    'chain': '(4)',
                    'query_type': '1-chain',
                    'head_1_id': user,
                    'head_1_text': "",
                    'head_1_type': 'user',
                    'head_2_id': -1,
                    'head_2_text': None,
                    'head_2_type': None,
                    'tail_id': boundary_user,
                    'tail_text': "",
                    'tail_type': 'user'
                }
                if edge not in data['1-chain'][('user_boundary', 'user', None, 'user')]:
                    data['1-chain'][('user_boundary', 'user', None, 'user')].append(edge)

                for retweet in retweets[:20]:
                    edge = {
                        'edge_type': 'boundary_retweet',
                        'chain': '(4, 0)',
                        'query_type': '2-chain',
                        'head_1_id': user,
                        'head_1_text': "",
                        'head_1_type': 'user',
                        'head_2_id': -1,
                        'head_2_text': None,
                        'head_2_type': None,
                        'tail_id': retweet['tail_id'],
                        'tail_text': "",
                        'tail_type': 'tweet'
                    }
                    if edge not in data['2-chain'][('boundary_retweet', 'user', None, 'tweet')]:
                        data['2-chain'][('boundary_retweet', 'user', None, 'tweet')].append(edge)

        return data
 
    def get_epoch_train_data(self, batch_size, negative_sample_size, num_workers, do_test, full):
        if full:
            epoch_edges = self.get_edges()
        else:
            epoch_edges = self.get_edges(epoch=self.epoch)

        positives_mapper = self.init_positives_mapper()

        data_out = defaultdict(lambda: defaultdict(lambda: list))
        for query_type in tqdm(epoch_edges):
            if do_test:
                train_edges, test_edges = dict(), dict()
                q_data = epoch_edges[query_type]
                for edge_type, edges in q_data.items():
                    train_edges[edge_type], test_edges[edge_type] = train_test_split(
                        edges, train_size=0.8, shuffle=True)

                data_out[query_type]['train_triples'] = self.produce_data(train_edges, query_type, positives_mapper, negatives_size=50)
                data_out[query_type]['test_triples'] = self.produce_data(test_edges, query_type, positives_mapper, negatives_size=500)
            else:
                train_edges = dict()
                q_data = epoch_edges[query_type]
                for edge_type, edges in q_data.items():
                    random.shuffle(edges)
                    train_edges[edge_type] = edges

                data_out[query_type]['train_triples'] = self.produce_data(train_edges, query_type, positives_mapper, negatives_size=50)
                
        return self.init_train_iterator(data_out, batch_size, negative_sample_size, num_workers)

    def init_train_iterator(self, data, batch_size, negative_sample_size, num_workers):
        for set_name in data:
            train_triples = data[set_name]['train_triples']   
            train_count = defaultdict(lambda: 4)
            for i in range(len(train_triples['head_1'])):
                chain = tuple(train_triples['chain'][i])

                head_1 = train_triples['head_1'][i]
                head_1_type = train_triples['head_1_type'][i]

                head_2 = train_triples['head_2'][i]
                head_2_type = train_triples['head_2_type'][i]

                train_count[(head_1, head_2, chain, head_1_type, head_2_type)] += 1
            dataset = TrainDataset(train_triples, negative_sample_size, 'tail-batch',
                train_count)

            train_dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                #num_workers=max(1, self.cpu_num//2),  # TODO: Fix
                num_workers=num_workers,
                collate_fn=dataset.collate_fn,
                #prefetch_factor=100,
                #persistent_workers=True,
                #pin_memory=True
            )
            data[set_name]['train_iterator'] = OneShotIterator(train_dataloader, train_triples['qtype'][0])
        
        return data

    def init_positives_mapper(self, sort=True):
        positives_mapper = defaultdict(lambda: defaultdict(list))

        epoch_edges = self.get_edges()

        for query_type in epoch_edges:
            for edge_type, edges in epoch_edges[query_type].items():
                for edge in edges:
                    head_id = edge['head_1_id']
                    tail_id = edge['tail_id'] 
                    positives_mapper[head_id][edge_type].append(tail_id)
        
        if sort:
            for head_id in positives_mapper:
                for edge_type in positives_mapper[head_id]:
                    positives_mapper[head_id][edge_type] = sorted(positives_mapper[head_id][edge_type])
        return positives_mapper

    def produce_data(self, edges, qtype, positives_mapper, negatives_size):
        # TODO: Verify
        def sample_negatives(pos_inds, a, b, n_samp=32):
            """ Pre-verified with binary search
            `pos_inds` is assumed to be ordered
            """
            raw_samp = np.random.randint(a, b - len(pos_inds), size=n_samp)
            pos_inds_adj = pos_inds - np.arange(len(pos_inds))
            ss = np.searchsorted(pos_inds_adj, raw_samp, side='right')
            neg_inds = raw_samp + ss
            return neg_inds

        node_ranges = self.get_node_ranges()

        data = defaultdict(list)

        for edge_type in edges:
            edge_info = self.edge_info.loc[edge_type]
            edge_tuple = edge_info['edge_tuple']
            head_1_type = edge_info['head_1_type']
            head_2_type = edge_info['head_2_type']
            tail_type = edge_info['tail_type']
            
            for edge in edges[edge_type]:
                data['chain'].append(edge_tuple)
                data['qtype'].append(qtype)
                data['head_1'].append(edge['head_1_id'])
                data['head_1_type'].append(head_1_type)
                data['head_2'].append(edge['head_2_id'])
                data['head_2_type'].append(head_2_type)
                data['tail'].append(edge['tail_id'])
                data['tail_type'].append(tail_type)
    
                positives = positives_mapper[edge['head_1_id']][edge_type]
                negatives = sample_negatives(positives, node_ranges[tail_type][0], node_ranges[tail_type][1], n_samp=negatives_size)
                data['negatives'].append(tuple(negatives))

        return data

    def get_epoch_timestamps(self, num_epochs):
        timestamps = []
        for t in self.db_handler.get_edge_timestamps():
            if t['timestamp']:
                timestamp = datetime.datetime.strptime(t['timestamp'], '%a %b %d %X %z %Y')
                timestamps.append(timestamp)

        timestamps.sort()

        timestamps = np.array_split(np.array(timestamps), num_epochs)

        epoch_timestamps = []
        for epoch in range(num_epochs):
            batch_timestamp = timestamps[epoch][-1]
            epoch_timestamps.append(batch_timestamp)
            
        return epoch_timestamps
    

class TrainDataset(Dataset):
    def __init__(self, triples, negative_sample_size, mode, count):
        self.len = len(triples['head_1'])
        self.triples = triples
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = count  # TODO: adjust dynamically

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        chain = self.triples['chain'][idx]
        head_1 = self.triples['head_1'][idx]
        head_1_type = self.triples['head_1_type'][idx]
        head_2 = self.triples['head_2'][idx]
        head_2_type = self.triples['head_2_type'][idx]
        tail = self.triples['tail'][idx]
        tail_type = self.triples['tail_type'][idx]

        if head_2:
            positive_sample = np.array([head_1] + [head_2] + list(chain) + [tail])
        else:
            positive_sample = np.array([head_1] + [head_1] + list(chain) + [tail])
        positive_sample = torch.LongTensor(positive_sample)
      
        subsampling_weight = self.count[(head_1, head_2, chain, head_1_type, head_2_type)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        if 'negatives' in self.triples.keys():
            negative_sample = np.random.choice(self.triples['negatives'][idx], self.negative_sample_size)
            #negative_sample = np.array([t for t in negative_sample])
            negative_sample = torch.LongTensor(negative_sample)
        else:
            # TODO: Buggy
            raise Exception
            negative_sample = torch.randint(self.entity_dict[tail_type][0], self.entity_dict[tail_type][1],
                                            (self.negative_sample_size,))
        
        
        return positive_sample, negative_sample, subsampling_weight, self.mode


    def collate_fn(self, data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]

        return positive_sample, negative_sample, subsample_weight, mode


class TestDataset(Dataset):
    def __init__(self, triples, mode, random_sampling, neg_size):
        self.len = len(triples['head_1'])
        self.triples = triples
        self.mode = mode
        self.random_sampling = random_sampling
        if random_sampling:
            self.neg_size = neg_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        chain = self.triples['chain'][idx]
        head_1 = self.triples['head_1'][idx]
        head_1_type = self.triples['head_1_type'][idx]
        head_2 = self.triples['head_2'][idx]
        head_2_type = self.triples['head_2_type'][idx]
        tail = self.triples['tail'][idx]
        tail_type = self.triples['tail_type'][idx]

        if head_2:
            positive_sample = [head_1] + [head_2] + list(chain) + [tail]
        else:
            positive_sample = [head_1] + [head_1] + list(chain) + [tail]
        positive_sample = torch.LongTensor(positive_sample)

        if 'negatives' in self.triples.keys():
            negative_sample = np.random.choice(self.triples['negatives'][idx], self.neg_size)
            #negative_sample = np.array([t for t in negative_sample])
            negative_sample = torch.LongTensor(negative_sample)
        else:
            # TODO: Buggy
            raise Exception
            negative_sample = torch.randint(self.entity_dict[tail_type][0], self.entity_dict[tail_type][1],
                                            (self.neg_size,))

        negative_sample = torch.cat([torch.LongTensor([tail]), torch.LongTensor(negative_sample)])

        return positive_sample, negative_sample, self.mode
       
    def collate_fn(self, data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, negative_sample, mode


class OneShotIterator(object):
    def __init__(self, dataloader_tail, qtype):
        # TODO: Check whether this influences performance in linux
        torch.set_num_threads(1)
        self.iterator = self.one_shot_iterator(dataloader_tail)
        self.qtype = qtype
        self.step = 0

    def __next__(self):
        self.step += 1
        return next(self.iterator)

    @staticmethod
    def one_shot_iterator(dataloader):
        ''' Transform a PyTorch Dataloader into python iterator '''
        while True:
            for data in dataloader:
                yield data


class Serializer(object):
    def save(self, pipeline, step, args):
        model = pipeline.model
        optimizer = pipeline.optimizer

        save_variable_list = {
            'step': step, 
            'current_learning_rate': pipeline.current_learning_rate,
            'warm_up_steps': pipeline.warm_up_steps,
            'entity_dict': pipeline.entity_dict
        }

        argparse_dict = vars(args)
        with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(args.save_path, 'checkpoint')
        )
        
        entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'entity_embedding'), 
            entity_embedding
        )
        
        relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'relation_embedding'), 
            relation_embedding
        )

    def init_checkpoint(self, init_checkpoint):
        logging.info('Loading checkpoint %s...' % init_checkpoint)
        checkpoint = torch.load(os.path.join(init_checkpoint, 'checkpoint'))
        logging.info('Checkpoint loaded')
        return checkpoint

    def load_checkpoint(self, checkpoint, pipeline, do_train=True):
        init_step = checkpoint['step']
        pipeline.model.load_state_dict(checkpoint['model_state_dict'])
        
        if do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            pipeline.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            pipeline.current_learning_rate = current_learning_rate
            pipeline.warm_up_steps = warm_up_steps
            
        return init_step
