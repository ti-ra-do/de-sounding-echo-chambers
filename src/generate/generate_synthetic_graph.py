import sys
sys.path.append('src')

import os
import csv
import json
import random
import argparse 

import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from collections import defaultdict

from utils import bidict, int_object_hook, truncated_power_law, TruncatedNormalDistribution


def create_user_graph(community_config, min_in_degree=0):
    def get_community_node_ranges():
        node_ranges = dict()
        cur_idx = 0

        for community in community_config:
            count = np.sum(list(community_config[community]['num_members'].values()))
            node_ranges[community] = (cur_idx, cur_idx + count)
            cur_idx += count

        return node_ranges

    def connect_node(node, community, member_type):
        user_graph.add_node(node, community=community)
        tn = TruncatedNormalDistribution(center=community_config[community]['num_followers'][member_type], range=6, scale=3)

        num_edges = tn.draw()

        connection_prob = community_config[community]['connection_probs'][member_type]
        for _ in range(num_edges):
            tail_community = np.random.choice(list(connection_prob.keys()), p=list(connection_prob.values()))
            tail = community_user_dist[tail_community].rvs(size=1)[0] - 1
            #tail = int(np.sum([np.sum(list(s.values())) for idx, s in num_members.items() if idx < tail_community])) + tail
            tail = community_node_ranges[tail_community][0] + tail
            
            if node != tail:
                user_graph.add_edge(node, tail)         
    
    def densify(user_graph, min_connections):
        community_mapper = bidict(nx.get_node_attributes(user_graph, 'community'))

        # in_degree == 0 is allowed as respective users retweet only
        low_degree_users = [user for (user, in_degree) in user_graph.in_degree()
            if in_degree < min_connections and in_degree != 0]  

        for user in tqdm(low_degree_users, desc="densify user graph"):
            in_edges = list(user_graph.in_edges(user))
            # TODO: Sometimes assign edges across communities
            user_community = [x for x in community_mapper.inverse[community_mapper[user]]
                if x is not user and x not in [y[0] for y in in_edges]]

            num_missing_edges = min_connections - user_graph.in_degree(user)
            additional_edges = [(u, user) for u in random.sample(user_community, num_missing_edges)]
            user_graph.add_edges_from(additional_edges)
        
        return user_graph
                    
    community_node_ranges = get_community_node_ranges()

    user_graph = nx.DiGraph()

    community_user_dist = dict([
        (i, truncated_power_law(a=community_config[i]['centrality'], m=np.sum(list(community_config[i]['num_members'].values()))))
            for i in community_config
    ])

    head = 0
    for community in tqdm(community_config, desc='create user graph'):
        for _ in range(community_config[community]['num_members']['gatekeepers']):
            connect_node(head, community, 'gatekeepers')
            head += 1

        for _ in range(community_config[community]['num_members']['members']):
            connect_node(head, community, 'members')
            head += 1

    if min_in_degree > 0:
        return densify(user_graph, min_in_degree)

    return user_graph


def create_twitter_graph(user_graph, expected_tweets_per_user, expected_retweets_per_user, min_retweets_per_tweet):
    tweets = []
    tweet_edges = defaultdict(list)
    retweet_edges = defaultdict(list)

    node_type = {}

    for user in tqdm(user_graph.nodes, desc='create twitter graph'):
        user_tweets = []
        node_type[user] = 'user'
        in_edges = list(user_graph.in_edges(user))
        if len(in_edges) > 0:
            num_tweets = np.random.poisson(expected_tweets_per_user - 1) + 1
        
            for _ in range(num_tweets):
                tweet = len(tweets) + len(user_graph.nodes)
                tweets.append(tweet)
                user_tweets.append(tweet)
                node_type[tweet] = 'tweet'
                tweet_edges[user].append(tweet)

                retweeting_users = [u for u, _ in random.sample(in_edges, min_retweets_per_tweet)]
                for retweeting_user in retweeting_users:
                    retweet_edges[retweeting_user].append(tweet)
            
            for retweeting_user, _ in in_edges:
                if len(set(retweet_edges[retweeting_user]).intersection(user_tweets)) == 0:
                    tweet = random.choice(user_tweets)
                    retweet_edges[retweeting_user].append(tweet)

    
    for user in tqdm(user_graph.nodes, desc='densify twitter graph'):
        retweets = retweet_edges[user]
        num_retweets = len(retweets)
        connected_users = list(user_graph[user].keys())

        tweet_candidates = [t for u in connected_users for t in tweet_edges[u]]
        tweet_candidates = set(tweet_candidates) - set(retweets)

        missing_retweets = np.minimum(np.maximum(0, expected_retweets_per_user - num_retweets), len(tweet_candidates))
        
        if missing_retweets > 0:
            tweet_candidates = random.sample(tweet_candidates, missing_retweets)
            retweet_edges[user] += tweet_candidates
    
    
    twitter_graph = nx.DiGraph()
    for user, tweets in tweet_edges.items():
        for tweet in tweets:
            twitter_graph.add_edge(tweet, user, t='tweet')
    for user, tweets in retweet_edges.items():
        for tweet in tweets:
            twitter_graph.add_edge(user, tweet, t='retweet')

    nx.set_node_attributes(twitter_graph, node_type, 'node_type')

    return twitter_graph


def transform_edges(edges, edge_type, chain_type, head_type, tail_type):
    result_edges = []
    for head, tail in edges: 
        result_edges.append((head, tail, edge_type, chain_type, head_type, tail_type))
    return result_edges

 
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Create Random Twitter Graph',
        usage='generate_synthetic_graph.py [<args>] [-h | --help]'
    )

    ''' Dataset Parameters '''
    parser.add_argument('--input_file', type=str, default='dataset/raw/synth_2_comms.json')
    parser.add_argument('--output_dir', type=str, default='dataset/processed/synth_2_comms',
        help='output directory relative to current working directory')
    parser.add_argument('--num_bins', default=1, type=int)
    parser.add_argument('--expected_tweets_per_user', default=10, type=int) # 10
    parser.add_argument('--expected_retweets_per_user', default=5, type=int) # 20
    parser.add_argument('--min_retweets_per_tweet', default=3, type=int) # 10
    
    return parser.parse_args(args)


def main(args):
    with open(args.input_file) as f:
        community_config = json.load(f, object_hook=int_object_hook)

    ''' Create User Graph'''
    user_graph = create_user_graph(
        community_config,
        min_in_degree=args.min_retweets_per_tweet)

    tweeting_users = [n for n in user_graph.nodes if user_graph.in_degree(n) > 0]
    
    community_mapper = bidict(nx.get_node_attributes(user_graph, 'community'))

    ''' Create Twitter Graph '''
    twitter_graph = create_twitter_graph(
        user_graph, 
        args.expected_tweets_per_user, 
        args.expected_retweets_per_user,
        args.min_retweets_per_tweet)

    ''' Create Dataset '''
    users = user_graph.nodes
    tweets = [x for x,y in twitter_graph.nodes(data=True) if y['node_type'] == 'tweet']
    random.shuffle(tweets)

    posts = transform_edges(
        [(h,t) for h,t,d in twitter_graph.out_edges(tweets, data=True) if d['t'] == 'tweet'],
        'post', '1-chain', 'tweet', 'user')

    reposts = transform_edges(
        [(h,t) for h,t,d in twitter_graph.in_edges(tweets, data=True) if d['t'] == 'retweet'],
        'repost', '1-chain', 'user', 'tweet')

    follows = transform_edges(
        user_graph.edges,
        'follow', '2-chain', 'user', 'user')

    community_start_id = len(users) + len(tweets)
    community2id = dict([(c,i+community_start_id) for i,c in enumerate(community_mapper.inverse.keys())])

    member = transform_edges(
        [(u, community2id[d['community']]) for u, d in user_graph.nodes(data=True)],
        'member', '1-chain', 'user', 'community')

    post_member = transform_edges(
        [(p,community2id[community_mapper[u]]) for p,u,d in twitter_graph.out_edges(tweets, data=True) if d['t'] == 'tweet'],
        'post_member', '2-chain', 'tweet', 'community')

    ''' Save to Disk '''
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    edge_df = pd.DataFrame(
        posts + reposts + follows + member + post_member, 
        columns=['head', 'tail', 'edge_type', 'query_type', 'head_type', 'tail_type'])
    edge_df.to_csv(os.path.join(args.output_dir, 'edges.csv'))

    with open(os.path.join(args.output_dir, 'node_info.csv'), 'w', newline='') as f:
        csv_writer = csv.DictWriter(f, delimiter=',', fieldnames=['node_type', 'count'])
        csv_writer.writeheader()
        csv_writer.writerows([
            {'node_type': 'user', 'count': len(users)},
            {'node_type': 'tweet', 'count': len(tweets)},
            {'node_type': 'community', 'count': len(community_mapper.inverse.keys())}])

    with open(os.path.join(args.output_dir, 'edge_info.csv'), 'w', newline='') as f:
        csv_writer = csv.DictWriter(f, delimiter=',', fieldnames=['edge_type', 'count', 'query_type', 'head_1_type', 'head_2_type', 'tail_type', 'dependency'])
        csv_writer.writeheader()
        csv_writer.writerows([
            {'edge_type': 'post', 'count': len(posts), 'query_type': '1-chain',
             'head_1_type': 'tweet', 'head_2_type': '', 'tail_type': 'user', 'dependency': ()},
            {'edge_type': 'repost', 'count': len(reposts), 'query_type': '1-chain',
             'head_1_type': 'user', 'head_2_type': '', 'tail_type': 'tweet', 'dependency': ()},
            {'edge_type': 'member', 'count': len(member), 'query_type': '1-chain', 
             'head_1_type': 'user', 'head_2_type': '', 'tail_type': 'community', 'dependency': ()},
            {'edge_type': 'follow', 'count': len(follows), 'query_type': '2-chain', 
             'head_1_type': 'user', 'head_2_type': '', 'tail_type': 'user', 'dependency': ('repost', 'post')},
            {'edge_type': 'post_member', 'count': len(post_member), 'query_type': '2-chain', 
             'head_1_type': 'tweet', 'head_2_type': '', 'tail_type': 'community', 'dependency': ('post', 'member')}])

    with open(os.path.join(args.output_dir, 'community_info.csv'), 'w', newline='') as f:
        csv_writer = csv.DictWriter(f, delimiter=',', fieldnames=['label', 'id', 'ideological', 'count'])
        csv_writer.writeheader()
        csv_writer.writerows([{'label': k, 'id': v, 'ideological': community_config[k]['ideological'], 'count': len(community_mapper.inverse[k])} for k,v in community2id.items()])

    nx.write_gexf(user_graph, os.path.join(args.output_dir, 'user_graph.gexf'))
    nx.write_gexf(twitter_graph, os.path.join(args.output_dir, 'twitter_graph.gexf'))
  

if __name__ == "__main__":
    main(parse_args())