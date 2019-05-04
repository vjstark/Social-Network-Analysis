from collections import defaultdict
from tinydb import TinyDB
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import time
from itertools import combinations
from random import sample
from math import ceil,floor
import pickle

def read_topics_file(filename):
    with open(filename) as f:
        screen_names = f.read().splitlines()
    return screen_names

def dump_data(summary, pick = 'data_from_clusterpy'):
    with open(pick,'wb') as f:
        pickle.dump(summary,f)
        
def create_graph(sampled_data):
    graph = nx.Graph()
    for key,value in sampled_data.items():
        edge = key.split(':')
        for key in edge:
            for v in value:
                graph.add_edge(key,v)
    return graph

def draw_network(graph, topics, filename):
    label_dict = {}
    position_dict = {}
    position = nx.spring_layout(graph)
    
    for t in topics:
        if t in position.keys():
            label_dict[t] = t
            position_dict[t] = position[t]
    
    
    fig = plt.figure(figsize=(30,25))
    plt.axis('off')
    nx.draw_networkx(graph, pos=position, with_labels=False,node_size=100,\
        linewidths=0.5, linecolor='g',width=0.5, edge_color='g')
    nx.draw_networkx_labels(graph,position_dict,label_dict,font_color='b')
    plt.savefig(filename)
    plt.show
    
def optimize_GN(G):
    GN_generator = nx.algorithms.community.girvan_newman(G)
    best_k = 0
    curr_k = 1
    node_count = 0
    comm_dict = dict()
    while node_count < 2:
        current_gn_instance = next(GN_generator)
        curr_k += 1
        current_cluster_set = list(map(lambda x:x, current_gn_instance))
        comm_dict[curr_k] = current_cluster_set
        for cluster in current_cluster_set:
            if len(cluster) == 1:
                node_count += 1
                if node_count == 1:
                    best_k = curr_k - 1            
    return len(comm_dict[best_k]) ,[i & set(topics) for i in comm_dict[best_k]]

groups_by_ht = defaultdict(set)
uid_set = set() #id_list
user_dict = defaultdict(set)
user_count = 0

topics = read_topics_file('topics')
path = 'twitter_data/'
user_db={}
tweet_db= {}
msg_id = set()

for i in topics:
    user_db[i] = TinyDB(f'{path}/user_{i}.json')
    tweet_db[i] = TinyDB(path+i+'.json')
    for j in tweet_db[i]:
        msg_id.add(j['tweet_id'])
        
for i in topics:
    for j in user_db[i]:
        groups_by_ht[i].add(j['user_id'])
        if j['user_id'] not in uid_set:
            user_dict[j['user_id']] = j['screen_name']
        uid_set.add(j['user_id'])
        user_count += 1

overlaps = defaultdict(set)
for index in range(1, len(topics) + 1):
    combination_list = list(combinations(topics,index))
    for value in combination_list:
        group = groups_by_ht[value[0]]    
        for v in value[1:]:
            group = group.intersection(groups_by_ht[v])
        if len(group) > 0:
            overlaps[':'.join(value)] = group
            
sampled_data = defaultdict(set)
sample_size = 500
for k,v in overlaps.items():
    sample_num = ceil((len(v)/user_count)*sample_size)
    sample_topic = sample(v,sample_num)
    if len(sample_topic) != 0:
        sampled_data[k] = sample_topic


G = create_graph(sampled_data)
num_communities, cluster_topics = optimize_GN(G)

summary = []
users = f'Number of users collected: {len(uid_set)}'
messages = f'Number of messages collected: {len(msg_id)}'
communities = f'Number of communities discovered: {num_communities}'
avg_users = f'Average number of users per community: {len(uid_set)/num_communities}'
summary.append(users)
summary.append(messages)
summary.append(communities)
summary.append(avg_users)

dump_data(summary)

print(f'Number of users collected: {len(uid_set)}')
print(f'Number of messages collected: {len(msg_id)}')
print(f'Number of communities discovered: {num_communities}')
print(f'Average number of users per community: {len(uid_set)/num_communities}')

draw_network(G, topics, 'cluster.jpg')
