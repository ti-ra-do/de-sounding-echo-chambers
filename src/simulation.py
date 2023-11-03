import random
from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import bidict, merge


class SimulationPipeline(object):
    def __init__(self, condition, latitude_of_acceptance, sharpness):
        self.condition = condition
        self.latitude_of_acc = latitude_of_acceptance
        self.sharpness = sharpness

        #self.avg_acceptance_prob = []

    def sample_users(self, user_graph, sample_size, sampling_strategy='random'):
        if sampling_strategy == 'random':
            return random.choices(list(user_graph.nodes), k=sample_size)
        elif sampling_strategy == 'boundary':
            community_mapper = bidict(nx.get_node_attributes(user_graph, 'community'))
            
            user_sample = []
            for _, users in community_mapper.inverse.items():
                boundary_users = nx.algorithms.boundary.node_boundary(user_graph, users)
                user_sample += list(boundary_users)
        elif self.user_sampling_strategy == 'center':
            community_mapper = bidict(nx.get_node_attributes(user_graph, 'community'))

            user_sample = []
            negative_user_sample = []
            for _, users in community_mapper.inverse.items():
                boundary_users = nx.algorithms.boundary.node_boundary(user_graph, users)
                for user in boundary_users:
                    ego_graph = nx.generators.ego.ego_graph(user_graph, user, 1)
                    negative_user_sample += list(ego_graph.nodes)
            user_sample = [n for n in user_graph.nodes if n not in negative_user_sample]
        else:
            raise Exception("Sampling strategy not supported: {}".format(sampling_strategy))

        random.shuffle(user_sample)
        user_sample = user_sample[:sample_size]

        return user_sample 

    def recommend_users(self, user, user_graph, strategy, num_recs, rank_fn=None):
        # TODO: Look into old project for further implementations
        if strategy == 'all':
            user_recs = list(user_graph.nodes)
        elif strategy == 'ego':
            ego_graph = user_graph[user]
            user_recs = list(ego_graph.keys())
        elif strategy == 'boundary':
            community_mapper = bidict(nx.get_node_attributes(user_graph, 'community'))

            community_nodes = community_mapper.inverse[community_mapper[user]]
            outside_nodes = [n for n in user_graph.nodes if n not in community_nodes]

            user_recs = list(nx.algorithms.boundary.node_boundary(user_graph, outside_nodes, community_nodes))
        else:
            raise Exception('strategy not supported: {}'.format(strategy))

        if rank_fn is not None:
            return rank_fn(user, user_recs)[:num_recs]
        else:
            return random.sample(user_recs, num_recs)

    def recommend_messages(self, user, user_recs, message_graph, strategy, num_recs, rank_fn=None):
        message_recs = dict()

        if strategy in ['all', 'post']:
            posts = message_graph.in_edges(user_recs) 
            
            for candidate_post, candidate_user in posts:
                if not message_graph.has_edge(user, candidate_post):
                    message_recs[candidate_post] = (candidate_user, candidate_user)
                    #message_recs.append((candidate_post, candidate_user, candidate_user))
        
        if strategy in ['all', 'repost']:
            reposts = message_graph.out_edges(user_recs) 

            for candidate_user, candidate_post in reposts:
                try:
                    author_user = list(message_graph.out_edges(candidate_post))[0][1]
                    if not message_graph.has_edge(user, candidate_post) and author_user != candidate_user:
                        #message_recs.append((candidate_post, candidate_user, author_user))
                        message_recs[candidate_post] = (candidate_user, author_user)

                except:
                    print('Something went wrong for post {}'.format(candidate_post))
                    print(message_graph.out_edges(candidate_post))
                    raise 
        
        message_recs = pd.DataFrame(message_recs.values(), columns=['rec_user_id', 'author_user_id'], index=message_recs.keys())
        if rank_fn is not None:
            ranked_messages = rank_fn(user, message_recs.index)[:num_recs]
            return message_recs.loc[ranked_messages].reset_index().to_numpy()
        else:
            return message_recs.sample(n=num_recs).reset_index().to_numpy()

    def make_ideological_decision(self, repost_dist, target_community_dist, source_community_dist):
        epsilon = torch.minimum(
            torch.tensor(1.0, dtype=torch.float32),
            2 * (target_community_dist ** self.sharpness) / (target_community_dist ** self.sharpness + source_community_dist ** self.sharpness)
        )
        return (epsilon * self.latitude_of_acc) ** self.sharpness / ((epsilon * self.latitude_of_acc) ** self.sharpness + repost_dist ** self.sharpness)

    def make_epistemic_decision(self, repost_dist):
        return self.latitude_of_acc ** self.sharpness / (self.latitude_of_acc ** self.sharpness + repost_dist ** self.sharpness)

    def model_disconnect(self, target_user, user_graph, message_graph, disconnect_ratio, user_rank_fn=None):
        disconnect_edges = defaultdict(lambda: defaultdict(list))

        followed_users = [u for _, u in user_graph.out_edges(target_user) ]
        reposts = [t for _, t in message_graph.out_edges(target_user)]

        num_disconnects = int(len(followed_users) * disconnect_ratio)
        if num_disconnects == 0:
            return disconnect_edges

        if user_rank_fn is not None:
            followed_users = user_rank_fn(target_user, followed_users)
        else:
            random.shuffle(followed_users)

        disconnect_users = followed_users[-num_disconnects:]

        for disconnect_user in disconnect_users:
            disconnect_edges['2-chain']['follow'].append({
                'head_1_id': target_user,
                'head_2_id': None,
                'tail_id': disconnect_user,               
            })
            
            disconnect_user_messages = [t for t, _ in message_graph.in_edges(disconnect_user)]
            disconnect_reposts = list(set(reposts).intersection(disconnect_user_messages))
            for repost in disconnect_reposts:
                disconnect_edges['1-chain']['repost'].append({
                    'head_1_id': target_user,
                    'head_2_id': None,
                    'tail_id': repost,               
                })            

        return disconnect_edges

    def agent_based_modeling(self, target_user, recommendations, user_graph, community_info, repost_dist_fn, community_dist_fn,
        community_distances, I):
        sim_edges = defaultdict(lambda: defaultdict(list))

        community_mapper = bidict(nx.get_node_attributes(user_graph, 'community'))
        
        log = defaultdict(int)
        log['community'] = community_mapper[target_user]
        log['potential_edges'] = len(recommendations)         

        for message_rec, user_rec, message_author in recommendations:
            ''' Identify Communities '''
            target_user_community = community_mapper[target_user]
            user_rec_community = community_mapper[user_rec]
            message_author_community = community_mapper[message_author]
            
            ''' Calculate Acceptance Probability '''
            repost_dist = repost_dist_fn(target_user, message_rec)
            if self.condition == 'ideological':
                ideological_strength = community_info.loc[target_user_community]['ideological']

                if ideological_strength > 0.0:
                    '''
                    target_community_dist = community_dist_fn(
                        head=target_user, 
                        target=target_user_community,
                        gamma=ideological_strength)
                    rec_community_dist = community_dist_fn(
                        head=target_user, 
                        target=user_rec_community,
                        gamma=ideological_strength)
                    '''
                    target_community_dist = community_distances[target_user_community][I]
                    rec_community_dist = community_distances[user_rec_community][I]

                    acceptance_probability = self.make_ideological_decision(
                        repost_dist, target_community_dist, rec_community_dist)
                else:
                    acceptance_probability = self.make_epistemic_decision(repost_dist)
            elif self.condition == 'epistemic':
                acceptance_probability = self.make_epistemic_decision(repost_dist)
            else:
                raise Exception("Decision style not supported!")

            ''' Add Edges '''
            if random.random() <= acceptance_probability:
                sim_edges['1-chain']['repost'].append({
                    'head_1_id': target_user,
                    'head_2_id': None,
                    'tail_id': message_rec,
                })

                log['retweet_edges'] += 1
                #if target_user_community_idx == source_user_community_idx:
                if target_user_community == message_author_community:
                    log['intra_retweet_edges'] += 1
                else:
                    log['inter_retweet_edges'] += 1

                if not user_graph.has_edge(target_user, message_author):
                    # TODO: Potentially duplicate
                    sim_edges['2-chain']['follow'].append({
                        'head_1_id': target_user,
                        'head_2_id': None,
                        'tail_id': message_author,
                    })

                    log['follow_edges'] += 1
                    #if target_user_community_idx == source_user_community_idx:
                    if target_user_community == message_author_community:
                        log['intra_follow_edges'] += 1
                    else:
                        log['inter_follow_edges'] += 1

            if target_user_community == message_author_community:
                log['potential_intra_edges'] += 1
            else:
                log['potential_inter_edges'] += 1

        return sim_edges, log



    