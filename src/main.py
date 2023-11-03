import sys
sys.path.append('../src')

import os
import ast
import pprint
import time
import torch
import random
import argparse

import pandas as pd
import networkx as nx

from tqdm import tqdm
from collections import defaultdict

from utils import SimLogger, bidict, merge
from data import DataPipeline
from evaluate import Evaluator
from simulation import SimulationPipeline
from model import KGEModel, TrainPipeline


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    ''' Dataset '''
    parser.add_argument('--dataset', type=str, default='twitter', help='dataset name, default to twitter')
    parser.add_argument('--data_dir', type=str, default='dataset/processed')

    ''' Model Parameters '''
    parser.add_argument('--model', default='DistMult', type=str)
    parser.add_argument('-d', '--hidden_dim', default=20, type=int)
    parser.add_argument('-g', '--gamma', default=20.0, type=float)

    ''' Training Parameters '''
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('--max_steps', default=15000, type=int) # 15000
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('-n', '--negative_sample_size', default=10, type=int)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-r', '--regularization', default=0.000015, type=float)  
    parser.add_argument('--uni_weight', action='store_true', help='Otherwise use subsampling weighting like in word2vec')

    ''' Simulation Parameters '''    
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--condition', default='ideological', type=str, help='Options: epistemic, ideological')
    parser.add_argument('--user_sampling_strategy', default='random', type=str)
    parser.add_argument('--user_sample_size', default=2000, type=int)
    parser.add_argument('--num_user_recs', default=20, type=int)
    parser.add_argument('--num_message_recs', default=30, type=int)
    parser.add_argument('--user_rec_pool_size', default=1000, type=int, help='Maximum number of users considered for recommendation. Reduce to improve performance.')
    parser.add_argument('--message_rec_pool_size', default=1000, type=int, help='Maximum number of messages considered for recommendation. Reduce to improve performance.')
    parser.add_argument('--user_rec_strategy', default='all', type=str)
    parser.add_argument('--message_rec_strategy', default='all', type=str)
    parser.add_argument('--rank_users', action='store_true')
    parser.add_argument('--rank_messages', action='store_true')
    parser.add_argument('--user_rank_edge', default='follow', type=str, help='Set the edge according to which user recommendations are ranked, default follow.')    
    parser.add_argument('--message_rank_edge', default='repost', type=str, help='Set the edge according to which item recommendations are ranked, default repost.')    
    parser.add_argument('--latitude_of_acceptance', default=0.5, type=float)
    parser.add_argument('--sharpness', default=5, type=int)
    parser.add_argument('--do_disconnect', action='store_true', help='Apply disconnect behavior')
    parser.add_argument('--disconnect_strategy', default='ranked', help='Options: ranked, random')
    parser.add_argument('--disconnect_ratio', default=0.1, type=float)
    parser.add_argument('--keep_communities', action='store_true', help='Keep initial communities over time')

    ''' Evaluation Parameters '''
    parser.add_argument('--test_batch_size', default=128, type=int, help='valid/test batch size')
    parser.add_argument('--test_negative_sample_size', default=500, type=int)
    #negative_eval_size (?)

    ''' Run Setup '''
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('-cpu', '--cpu_num', default=2, type=int) # 16
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    ''' Logging '''
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--save_checkpoint_steps', default=1000, type=int)
    parser.add_argument('--valid_steps', default=500, type=int)
    parser.add_argument('--log_steps', default=500, type=int, help='train log every xx steps')

    ''' Set during Runtime '''
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)


def main(args):
    args.full = True
    args.do_disconnect = True

    ''' Set up paths & directories '''
    args.data_dir = os.path.join(args.data_dir, args.dataset)

    ''' Logging Setup '''
    if not args.save_path:
        args.save_path = os.path.join(
            'log', args.dataset, args.condition, 'ranked' if args.rank_users else 'random', args.model, str(time.time()))
    sim_logger = SimLogger(args.save_path)
    sim_logger.log_info(args)

    ''' Set up Data Pipeline '''
    node_info_file = os.path.join(args.data_dir, 'node_info.csv')
    edge_info_file = os.path.join(args.data_dir, 'edge_info.csv')
    community_info_file = os.path.join(args.data_dir, 'community_info.csv')
    data_pipeline = DataPipeline(args.num_epochs, node_info_file, edge_info_file, community_info_file)

    ''' Initialize Edges '''
    df_edges = pd.read_csv(os.path.join(args.data_dir, 'edges.csv'), index_col=0)
    data_pipeline.init_edges(df_edges)

    args.nentity = data_pipeline.get_num_nodes()
    args.nrelation = data_pipeline.get_num_edge_tuples(single_only=True)

    ''' Training Pipeline Setup '''
    model = KGEModel(
        model_name=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
    )

    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate
    )

    evaluator = Evaluator(name=args.dataset, eval_metric='mrr')

    train_pipeline = TrainPipeline(
        model, optimizer, evaluator, 
        args.warm_up_steps, args.max_steps, args.negative_adversarial_sampling, 
        args.adversarial_temperature, args.regularization, args.uni_weight)
    
    ''' Simulation Pipeline Setup '''
    simulation_pipeline = SimulationPipeline(args.condition, args.latitude_of_acceptance, args.sharpness)

    ''' Run Training & Agent-based Modeling '''
    epoch_message_graph = data_pipeline.init_message_graph()
    epoch_user_graph = data_pipeline.init_user_graph()
    for _ in range(0, args.num_epochs):
        ''' Prepare Training Data '''
        if data_pipeline.epoch > 0:
            args.batch_size = 128
            #train_pipeline.current_learning_rate = 0.0001
        data = data_pipeline.get_epoch_train_data(args.batch_size, args.negative_sample_size, args.cpu_num, args.do_test, args.full)
 
        ''' Training Loop '''
        training_logs = []
        #if step >= train_pipeline.warm_up_steps:
        train_pipeline.adjust_optimizer(model)

        pbar = sim_logger.init_pbar(args.max_steps if data_pipeline.epoch == 0 else int(args.max_steps * 0.2))
        for step in pbar:
            training_logs += train_pipeline.train(data)


            if step % args.log_steps == 0:
                sim_logger.update_pbar(pbar, data_pipeline.epoch, step, training_logs)
                sim_logger.log_training(data_pipeline.epoch, step, training_logs)
                training_logs = []

                if args.do_test:
                    test_logs = train_pipeline.test(data, data_pipeline.edge_chains)
                    sim_logger.log_test(data_pipeline.epoch, step, test_logs, 'valid')

        ''' Agent-based Modeling '''
        with torch.no_grad(): 
            model.eval()

            simulation_logs = []
            simulation_edges = defaultdict(lambda: defaultdict(list))
            disconnect_edges = defaultdict(lambda: defaultdict(list))

            ''' Initialize Ranking Functions for Users & Messages '''
            edge_info = data_pipeline.edge_info
            user_rank_fn = model.rank_fn(
                rank_chain=edge_info.loc[args.user_rank_edge]['edge_tuple'],
                query_type=edge_info.loc[args.user_rank_edge]['query_type'],
                max_num_tails=args.user_rec_pool_size)

            message_rank_fn = model.rank_fn(
                rank_chain=edge_info.loc[args.message_rank_edge]['edge_tuple'],
                query_type=edge_info.loc[args.message_rank_edge]['query_type'],
                max_num_tails=args.message_rec_pool_size)

            ''' Initialize Distance Functions for Repost & Community Propotypicality '''
            repost_dist_fn = model.distance_fn(
                relation=edge_info.loc['repost']['edge_tuple'], 
                query_type=edge_info.loc['repost']['edge_tuple'])

            community_dist_fn = model.distance_fn(
                relation=edge_info.loc['member']['edge_tuple'], 
                query_type=edge_info.loc['member']['edge_tuple'])
       
            ''' Simulation Loop '''
            if args.user_sampling_strategy is 'all':
                user_sample = epoch_user_graph.nodes
            else:
                user_sample = simulation_pipeline.sample_users(
                    epoch_user_graph, args.user_sample_size, args.user_sampling_strategy)

            # TODO: Testing
            if args.condition == 'ideological':
                node_ranges = data_pipeline.get_node_ranges()
                community_mapper =  bidict(nx.get_node_attributes(epoch_user_graph, 'community'))
                community_distances = {}
                for community in community_mapper.inverse:
                    target_community_dist = community_dist_fn(
                        head=user_sample, 
                        target=community,
                        gamma=1.0)
                    community_distances[community] = target_community_dist
            else:
                community_distances = None

            all_message_recs = []
            for i, user in enumerate(tqdm(user_sample, desc='[epoch {} | ab-modeling]'.format(data_pipeline.epoch))):
                user_recs = simulation_pipeline.recommend_users(user, epoch_user_graph, 
                    args.user_rec_strategy, args.num_user_recs, user_rank_fn if args.rank_users else None)

                message_recs = simulation_pipeline.recommend_messages(user, user_recs, epoch_message_graph,
                    args.message_rec_strategy, args.num_message_recs, message_rank_fn if args.rank_messages else None)

                all_message_recs.append(message_recs)
                
                user_simulation_edges, log = simulation_pipeline.agent_based_modeling(user, message_recs,
                    epoch_user_graph, data_pipeline.community_info, repost_dist_fn, community_dist_fn, community_distances, i)
                simulation_edges = merge(simulation_edges, user_simulation_edges)

                if args.do_disconnect:
                    user_disconnect_edges = simulation_pipeline.model_disconnect(user, epoch_user_graph, epoch_message_graph,
                        args.disconnect_ratio, user_rank_fn if args.disconnect_strategy == 'ranked' else None)
                    disconnect_edges = merge(disconnect_edges, user_disconnect_edges)
                simulation_logs.append(log)
                
        data_pipeline.add_edges(simulation_edges)
        data_pipeline.remove_edges(disconnect_edges)
        data_pipeline.epoch += 1

        ''' Init Graphs '''
        epoch_message_graph = data_pipeline.init_message_graph()
        epoch_user_graph = data_pipeline.init_user_graph()

        ''' Update Communities '''
        if not args.keep_communities:
            data_pipeline.update_communities(epoch_user_graph)

        ''' Log Simulation Statistics '''
        node_ranges = data_pipeline.get_node_ranges()
        sim_logger.calculate_metrics(epoch_user_graph, model, node_ranges)
        sim_logger.log_simulation(data_pipeline.epoch, simulation_logs) 
        nx.write_gexf(epoch_user_graph, os.path.join(args.save_path, "user_graph_{}.gexf".format(data_pipeline.epoch)))
        user_embeddings = model.entity_embedding.data[node_ranges['user'][0]:node_ranges['user'][1], :]
        torch.save(user_embeddings, os.path.join(args.save_path, 'user_emb_{}.pt'.format(data_pipeline.epoch)))

        train_pipeline.reset_community_embeddings(node_ranges)


if __name__ == '__main__':
    main(parse_args())