import os
import json
import torch
import random 
import logging

import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats

from tqdm import tqdm
from collections import Counter, defaultdict
from tensorboardX import SummaryWriter

from graph import performance as perf
from graph import connectivity as conn
from graph import homophily as hom

''' If still runs, may delete'''
def merge(d1, d2):
    for qtype, q_data in d2.items():
        for rel, r_data in q_data.items():
            for edge in r_data:
                d1[qtype][rel].append(edge)
    
    return d1


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / float(len(s1.union(s2))))


def int_object_hook(obj):
    """If a value in obj is a string, try to convert it to an int"""
    rv = {}
    for k, v in obj.items():
        if isinstance(k, str):
            try:
                rv[int(k)] = v
            except ValueError:
                rv[k] = v
        else:
            rv[k] = v
    return rv

'''
def merge(d1, d2):
    return {**d1, **d2}
    '''
    
def truncated_power_law(a, m):
    x = np.arange(1, m+1, dtype='float')
    pmf = 1/x**a
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(range(1, m+1), pmf))


class TruncatedNormalDistribution(object):
    def __init__(self, center, range, scale):
        self.center = center

        self.x = np.arange(-range, range+1)

        xU, xL = self.x + 0.5, self.x - 0.5 
        prob = stats.norm.cdf(xU, scale = scale) - stats.norm.cdf(xL, scale = scale)
        self.prob = prob / prob.sum() # normalize the probabilities so their sum is 1

    def draw(self):
        return self.center + np.random.choice(self.x, size = 1, p = self.prob)[0]


class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


class SimLogger(object):
    def __init__(self, save_path, verbose=False):
        self.save_path = save_path
        self.verbose = verbose
        
        ''' Define Tensorboard Writer '''
        self.board_writer = SummaryWriter(self.save_path)

        ''' Define Logging '''
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=os.path.join(self.save_path, 'simulation.log'),
            filemode='w'
        )
        self.sim_metrics = defaultdict(list)

        self.pbar_desc = "[epoch {} | loss: {:.5f}]"

    def init_pbar(self, max_steps):
        return tqdm(range(0, max_steps), desc=self.pbar_desc.format(0, 0))

    def update_pbar(self, pbar, epoch, step, logs):
        loss = sum([log['metrics']['loss'] for log in logs])/len(logs)

        pbar.set_description(self.pbar_desc.format(
            epoch, loss))

    def log_info(self, args):
        logging.info(args)

    def log_epoch_info(self, epoch, data, nentity, total_nentity, nrelation):
        logging.info("info | Epoch: {} | Training {:.2f}% of entities ({}/{}) on {} relations".format(
            epoch, nentity / total_nentity * 100, nentity, total_nentity, nrelation))
            
        for _, query_type_data in data.items():
            for edge_tuple, edges in query_type_data.items():
                logging.info("info | Epoch: {} | #{}: {}".format(epoch, edge_tuple[0], len(edges)))

    def log_training(self, epoch, step, logs):
        metrics = {}
        for metric in logs[0]['metrics'].keys():
            metrics[metric] = sum([log['metrics'][metric] for log in logs])/len(logs)
    
        for metric in metrics:
            ''' Readable Logfile'''
            logging.info('{} | Epoch: {} | Step: {} | {}: {:.3f}'.format(
                'train', epoch, step, metric, metrics[metric]))

            ''' Tensorboard Logging '''
            self.board_writer.add_scalar(
                'Training/epoch_{}/{}'.format(epoch, metric),
                metrics[metric],
                step
            )

    def log_test(self, epoch, step, logs, set_name):
        for edge_type in logs:
            for metric in logs[edge_type]:
                logging.info('{} | Epoch: {} | Step: {} | {} | {}: {:.3f}'.format(
                    'test_at_{}'.format(set_name), epoch, step, edge_type, metric, logs[edge_type][metric]))

                self.board_writer.add_scalar(
                    'Test_at_{}/epoch_{}/{}/{}'.format(set_name, epoch, edge_type, metric),
                    logs[edge_type][metric],
                    step
                )

    def parallel_logging(self, user_graph, community_mapper, node_ranges, epoch, sim_logs):
        self.calculate_metrics(user_graph, community_mapper, node_ranges)
        self.log_simulation(epoch, sim_logs)

    def log_simulation(self, epoch, sim_logs=None):
        if sim_logs is not None:
            combined_sim_logs = defaultdict(list)

            for user in sim_logs:
                community = user['community']
                potential_edges = user['potential_edges']
                potential_intra_edges = user['potential_intra_edges']
                potential_inter_edges = user['potential_inter_edges']

                if potential_edges > 0:
                    combined_sim_logs['{}_intra_edge_rate'.format(community)].append(potential_intra_edges / potential_edges)
                    combined_sim_logs['{}_inter_edge_rate'.format(community)].append(potential_inter_edges / potential_edges)
                    
                    combined_sim_logs['{}_retweet_rate'.format(community)].append(user['retweet_edges'] / potential_edges)
                    combined_sim_logs['{}_follow_rate'.format(community)].append(user['follow_edges'] / potential_edges)

                if potential_inter_edges > 0:
                    combined_sim_logs['{}_inter_retweet_rate'.format(community)].append(user['inter_retweet_edges'] / potential_inter_edges)
                    combined_sim_logs['{}_inter_follow_rate'.format(community)].append(user['inter_follow_edges'] / potential_inter_edges)

                if potential_intra_edges > 0:
                    combined_sim_logs['{}_intra_retweet_rate'.format(community)].append(user['intra_retweet_edges'] / potential_intra_edges)
                    combined_sim_logs['{}_intra_follow_rate'.format(community)].append(user['intra_follow_edges'] / potential_intra_edges)

            for key, values in combined_sim_logs.items():
                mean = np.mean(values)
                logging.info("sim | Epoch: {} | {}: {:.4f}".format(
                    epoch, key, mean
                ))

                self.board_writer.add_scalar(
                    'ABM/{}'.format(key),
                    mean,
                    epoch
                )

        for metric in self.sim_metrics:
            ''' Readable Logfile '''
            logging.info("sim | Epoch: {} | {}: {}".format(
                epoch, metric, self.sim_metrics[metric][-1]))

            ''' Tensorboard Logging '''
            self.board_writer.add_scalar(
                'ABM/{}'.format(metric),
                self.sim_metrics[metric][-1],
                epoch
            )

        ''' CSV-Logfile'''
        pd.DataFrame(self.sim_metrics).to_csv(os.path.join(self.save_path, 'simulation.csv'))
    
        '''
        TODO
        if len(global_ap) > 0:
            mean = np.mean(global_ap)
            var = np.var(global_ap)
            min = np.min(global_ap)
            max = np.max(global_ap)

            print('Acceptance Prob - Mean: {:.4f} | Var: {:.4f} | Min: {:.4f} | Max: {:.4f}'.format(
                mean, var, min, max
            ))
        '''

    def calculate_graph_statistics(self, G):
        degree = list(G.degree())
        degree = [x[1] for x in degree]
        global_degree = np.mean(degree)

        self.sim_metrics['degree'].append(global_degree)
        self.sim_metrics['max_degree'].append(np.max(degree))
        self.sim_metrics['density'].append(nx.density(G))

        try:
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G, tol=1e-03)
            eigenvector_centrality = np.mean(list(eigenvector_centrality.values()))
        except Exception:
            eigenvector_centrality = 0.0
        self.sim_metrics['eigenvector_centrality'].append(eigenvector_centrality)

        degree_centrality = nx.degree_centrality(G)
        degree_centrality = np.mean(list(degree_centrality.values()))
        self.sim_metrics['degree_centrality'].append(degree_centrality)

    def calculate_community_performance(self, G, community2users, node_ranges):
        community_mapper = bidict(nx.get_node_attributes(G, 'community'))
        community2users = community_mapper.inverse

        community_range = node_ranges['community']
        for k, p in community2users.items():
            self.sim_metrics['size_{}'.format(k)].append(len(p))

        for k in [c for c in range(community_range[0], community_range[1]) if c not in community2users]:
            self.sim_metrics['size_{}'.format(k)].append(0)

        homophily = hom(G, community2users)
        if isinstance(homophily, dict):
            for k, v in homophily.items():
                self.sim_metrics['homophily_{}'.format(k)].append(v)
            for k in [c for c in range(community_range[0], community_range[1]) if c not in community2users]:
                self.sim_metrics['homophily_{}'.format(k)].append(0.0)
        else:
            self.sim_metrics['homophily'].append(homophily)

        
        coverage = nx.algorithms.community.quality.coverage(G, list(community2users.values()))
        self.sim_metrics['coverage'].append(coverage)
        
        #performance = nx.algorithms.community.quality.performance(G, list(community2users.values()))
        performance = perf(G, list(community2users.values()))
        self.sim_metrics['performance'].append(performance)

        modularity = nx.algorithms.community.quality.modularity(G, list(community2users.values()))
        self.sim_metrics['modularity'].append(modularity)
        #connectivity = conn(G, list(community2users.values()))

        #self.sim_metrics['performance'].append(performance)
        #self.sim_metrics['connectivity'].append(connectivity)

    def calculate_garimella_controversy(self, G, u2c, c2u):
        # TODO: Improve    
        high_degree_nodes = []
        for _, community_users in c2u.items():
            degrees = G.degree(community_users)
            sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
            high_degree_nodes.append(sorted_degrees[0][0])

        def perform_random_walk(c_0=None, c_1=None):
            assert (c_0 is not None and c_1 is not None) or (c_0 is None and c_1 is None)

            if c_0 and c_1:
                S = G.subgraph(c2u[c_0]+c2u[c_1]).copy()
            else:
                S = G.copy()

            stayed = 1.0
            left = 1.0
            random_walk_length = 100
            for user in S.nodes:
                user_community = u2c[user]
                current_target = user
                for i in range(random_walk_length):
                    neighbors = list(S.neighbors(current_target))
                    if len(neighbors) == 0:
                        current_target = user
                    else:
                        current_target = random.choice(neighbors)
                    if current_target in high_degree_nodes:
                        target_community = u2c[current_target]
                        if target_community == user_community:
                            stayed += 1
                        else:
                            left += 1
                        break
            controversy = stayed / (stayed + left)
            if c_0 is not None and c_1 is not None:
                if (c_0, c_1) not in self.metrics['garimella']:
                    self.sim_metrics['garimella'][(c_0, c_1)] = []
                self.sim_metrics['garimella'][(c_0, c_1)].append(controversy)
            else:
                self.sim_metrics['garimella'].append(controversy)
        
        if self.verbose:
            for c_0 in c2u.keys():
                for c_1 in c2u.keys():
                    if c_0 != c_1:  
                        perform_random_walk(c_0, c_1)
        perform_random_walk()

    def calculate_garimella_user_controversy(self, G, u2c, c2u):
        # TODO: Improve
        max_len = 100 # todo: adjust to 100

        high_degree_nodes = {}
        for community, community_users in c2u.items():
            degrees = G.degree(community_users)
            sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
            high_degree_nodes[community] = sorted_degrees[0][0]

        def perform_random_walk(user, c_0=None, c_1=None):
            if c_1:
                S = G.subgraph(c2u[c_0]+c2u[c_1])
            else:
                S = G

            community_representative = high_degree_nodes[c_0]
            if c_1:
                other_representatives = [high_degree_nodes[c_1]]
            else:
                other_representatives = [n for n in high_degree_nodes.values() if n != community_representative]
            num_steps_own = 0
            current_target = user
            while True:
                num_steps_own += 1

                neighbors = list(S.neighbors(current_target))
                if len(neighbors) == 0:
                    current_target = user
                else:
                    current_target = random.choice(neighbors)
                if current_target == community_representative or num_steps_own == max_len:
                    break
            #stayed_time.append((user, num_steps_own))

            num_steps_other = 0
            current_target = user
            while True:
                num_steps_other += 1
                neighbors = list(S.neighbors(current_target))
                if len(neighbors) == 0:
                    current_target = user
                else:
                    current_target = random.choice(neighbors)
                if current_target in other_representatives or num_steps_other == max_len:
                    break

            user_controversy = num_steps_other / (num_steps_other + num_steps_own)
            #left_time.append((user, num_steps_other))
            return user_controversy

        global_controversy = 0.0
        controversy = {}

        user_sample = random.choices(list(G.nodes), k=10000)

        for user in user_sample:
            user_community = u2c[user]
            global_controversy += perform_random_walk(user, user_community)
            if self.verbose:
                for other_community in c2u.keys():
                    if user_community != other_community:
                        local_controversy = perform_random_walk(user, user_community, other_community)
                        if (user_community, other_community) not in controversy:
                            controversy[(user_community, other_community)] = []
                        controversy[(user_community, other_community)].append(local_controversy)

        if self.verbose:
            for (c_0, c_1), cont in controversy.items():
                    cont = np.mean(cont)
                    if (c_0, c_1) not in self.sim_metrics['garimella_user']:
                        self.sim_metrics['garimella_user'][(c_0, c_1)] = []
                    self.sim_metrics['garimella_user'][(c_0, c_1)].append(cont)

        global_controversy /= len(user_sample)
        self.sim_metrics['garimella_user'].append(global_controversy)

        return controversy

    def calculate_metrics(self, user_graph, model, node_ranges):
        community_mapper = bidict(nx.get_node_attributes(user_graph, 'community'))
        c2u = community_mapper.inverse

        self.sim_metrics['num_nodes'].append(len(user_graph.nodes))
        self.sim_metrics['num_edges'].append(len(user_graph.edges))
        self.sim_metrics['num_communities'].append(len(c2u))

        ''' General Graph Statistics '''
        self.calculate_graph_statistics(user_graph)

        ''' Latent Space Statistics '''
        self.calculate_latent_statistics(user_graph, model, node_ranges)

        ''' Community Performance '''
        self.calculate_community_performance(user_graph, c2u, node_ranges)

        ''' Controversy '''
        #self.calculate_garimella_controversy(user_graph, u2c, c2u)
        #self.calculate_garimella_user_controversy(user_graph, u2c, c2u)


    def calculate_latent_statistics(self, user_graph, model, node_ranges):
        cuda = next(model.parameters()).is_cuda
        community_mapper = bidict(nx.get_node_attributes(user_graph, 'community'))

        with torch.no_grad(): 
            model.eval()
            entity_embeddings = model.entity_embedding

            internal_distances = []
            external_distances = []
            pol_alt = torch.tensor(0.0, dtype=torch.float32)
            if cuda:
                pol_alt = pol_alt.cuda()
            for c0 in community_mapper.inverse:
                c0_users = community_mapper.inverse[c0]
                c0_user_tensor = torch.tensor(c0_users)
                
                if cuda:
                    c0_user_tensor = c0_user_tensor.cuda()                
                
                c0_user_embeddings = torch.index_select(entity_embeddings, 0, c0_user_tensor)

                internal_dist = torch.cdist(c0_user_embeddings, c0_user_embeddings)
                internal_triu = torch.triu(internal_dist)
                internal_triu = internal_triu.view(-1)
                internal_distances.append(internal_triu)
                
                c1 = [c for c in community_mapper.inverse if c != c0]
                c1_users = []
                for c in c1:
                    c_users = community_mapper.inverse[c]
                    c1_users += c_users

                c1_user_tensor = torch.tensor(c1_users)
                if cuda:
                    c1_user_tensor = c1_user_tensor.cuda()                
                
                c1_user_embeddings = torch.index_select(entity_embeddings, 0, c1_user_tensor)

                external_dist = torch.cdist(c0_user_embeddings, c1_user_embeddings)
                external_triu = torch.triu(external_dist)
                external_triu = external_triu.view(-1)
                external_distances.append(external_triu)
                
                mean_internal_dist = internal_dist.mean()
                mean_external_dist = external_dist.mean()
                pol = mean_external_dist - mean_internal_dist
                pol_alt += pol / c0_user_embeddings.shape[0]
                self.sim_metrics['pol_{}'.format(c0)].append(pol.detach().cpu().numpy())
                print('{} | Dist: {:.3f} | Pol: {:.3f}'.format(c0, mean_external_dist, mean_external_dist-mean_internal_dist))

        internal_distances = torch.cat(internal_distances, dim=0)
        print(internal_distances.shape)
        external_distances = torch.cat(external_distances, dim=0)
        
        avg_internal_distance = torch.mean(internal_distances)
        avg_external_distance = torch.mean(external_distances)
        pol = avg_external_distance - avg_internal_distance
        self.sim_metrics['pol_global'].append(pol.detach().cpu().numpy())    

        num_nodes = torch.tensor(user_graph.number_of_nodes())
        num_communities = torch.tensor(len(community_mapper.inverse))
        if cuda:
            num_nodes = num_nodes.cuda()
            num_communities = num_communities.cuda()

        pol_alt = pol_alt / num_communities * num_nodes
        self.sim_metrics['pol_alt'].append(pol_alt.detach().cpu().numpy())    

        community_range = node_ranges['community']
        for k in [c for c in range(community_range[0], community_range[1]) if c not in community_mapper.inverse]:
            self.sim_metrics['pol_{}'.format(k)].append(0)

def log_dataset_statistics(db_handler, entities, args):
    log_file = os.path.join(args.dataset_dir, args.log_file)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    dataset_name = args.input_file.split("/")[-1]
    logging.info("Statistics for {}".format(dataset_name))

    entities_count = db_handler.count_entities(entities)
    for entity, count in entities_count.items():
        logging.info("Num {}: {}".format(entity, count))

    relations = db_handler.get_relations()
    edge_types = [r['edge_type'] for r in relations]
    for edge_type in edge_types:
        edges = db_handler.get_edges(edge_type)
        logging.info("Num {}: {}".format(edge_type, len(edges)))

    if 'tweet' in edge_types:
        user_tweet_tweet = db_handler.get_edges('tweet')
        users = [t['tail_id'] for t in user_tweet_tweet]
        tweets_per_user = Counter(users)
        avg_tweets_per_user = np.mean(list(tweets_per_user.values()))
        logging.info('Avg. Tweets per User: {:.3f}'.format(avg_tweets_per_user))

    if 'retweet' in edge_types:
        user_tweet_retweet = db_handler.get_edges('retweet')
        users = [t['head_1_id'] for t in user_tweet_retweet]
        retweets_per_user = Counter(users)
        avg_retweets_per_user = np.mean(list(retweets_per_user.values()))
        logging.info('Avg. Retweets per User: {:.3f}'.format(avg_retweets_per_user))

        tweets = [t['tail_id'] for t in user_tweet_retweet]
        retweets_per_tweet = Counter(tweets)
        avg_retweets_per_tweet = np.mean(list(retweets_per_tweet.values()))
        logging.info('Avg. Retweets per Tweet: {:.3f}'.format(avg_retweets_per_tweet)) 


@DeprecationWarning
def log_dataset_statistics2(data, args):
    logging.info('Model: %s' % args.model)
    logging.info('Dataset: %s' % args.dataset)
    logging.info('#entity: %d' % args.nentity)
    logging.info('#relation: %d' % args.nrelation)
    
    for set_name in data:
        logging.info("--- {} ----".format(set_name))
        logging.info('#train: %d' % len(data[set_name]['train_triples']['head_1']))
        logging.info('#valid: %d' % len(data[set_name]['valid_triples']['head_1']))
        logging.info('#test: %d' % len(data[set_name]['test_triples']['head_1']))


@DeprecationWarning
def log_train_statistics(args, init_step):
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)


@DeprecationWarning
def log_model_statistics(kge_model):
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))


@DeprecationWarning
def calculate_path_lengths(G, c2u):
    avg_path_len = nx.average_shortest_path_length(G)
    print("Graph Statistics - Avg. Path Length (total): {:.2f}".format(avg_path_len))
    for community, users in c2u.items():
        S = G.subgraph(users).copy()
        avg_path_len = nx.average_shortest_path_length(S)
        print("Graph Statistics - Avg. Path Length ({}): {:.2f}".format(community, avg_path_len))


@DeprecationWarning
def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.dataset = argparse_dict['dataset']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
