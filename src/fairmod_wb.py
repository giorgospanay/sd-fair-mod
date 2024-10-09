##### ------------------------------------
#
#     Fair-mod (weighted balance)
# 	  Author: Georgios Panayiotou 
#     
#     Based on NetworkX implementation of the Louvain community detection algorithm.
#	  See here: https://networkx.org/documentation/stable/_modules/networkx/algorithms/community/louvain.html
# 
##### ------------------------------------

import itertools
from collections import defaultdict, deque
import random

import networkx as nx
from networkx.algorithms.community import modularity
from networkx.utils import py_random_state

import pickle
import sys
import time
import matplotlib.pyplot as plt


### --------------- GLOBALS ------------------------------------
obj_path="../data/obj"

### --------------- FUNCTIONS FOR FAIR-MOD ---------------------

@py_random_state("seed")
def fair_louvain_communities(
	G, weight="weight", resolution=1, threshold=0.000000001, max_level=None, seed=None, sensitive_attribute="color", alpha=0.9
):	
	partitions = fair_louvain_partitions(G, weight, resolution, threshold, seed, sensitive_attribute="color", alpha=alpha)
	if max_level is not None:
		if max_level <= 0:
			raise ValueError("max_level argument must be a positive integer or None")
		partitions = itertools.islice(partitions, max_level)
	final_partition = deque(partitions, maxlen=1)
	return final_partition.pop()


@py_random_state("seed")
def fair_louvain_partitions(G, weight="weight", resolution=1, threshold=0.000000001, seed=None, sensitive_attribute="color", alpha=0.9):
	partition = [{u} for u in G.nodes()]

	colors=nx.get_node_attributes(G, "color")

	# Fill initial partition colors table. Calculate phi
	partition_colors = []
	blue_ct=0
	red_ct=0
	for i,u in enumerate(G.nodes()):
		partition_t=dict()
		if colors[u]=="blue":
			partition_t["red"]=0
			partition_t["blue"]=1
			partition_t["score"]=0
			blue_ct+=1
		elif colors[u]=="red":
			partition_t["red"]=1
			partition_t["blue"]=0
			partition_t["score"]=0
			red_ct+=1
		partition_colors.append(partition_t)

	if nx.is_empty(G):
		yield partition
		return

	phi=min(blue_ct/len(G.nodes()),red_ct/len(G.nodes()))

	mod = modularity(G, partition, resolution=resolution, weight=weight)
	fair, _f_dist = fairness(G, partition)
	opt = alpha * mod + (1-alpha) * fair

	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		graph.add_nodes_from(G)
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	n = graph.number_of_nodes()
	m = graph.size(weight="weight")
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition(
		graph, 
		n,
		m,
		partition, 
		colors,
		partition_colors,
		sensitive_attribute=sensitive_attribute, 
		alpha=alpha,
		phi=phi,
		resolution=resolution, 
		threshold=threshold,
		is_directed=is_directed, 
		seed=seed
	)

	improvement = True
	while improvement:
		yield [s.copy() for s in partition]
		new_mod = modularity(
			graph, inner_partition, resolution=resolution, weight="weight"
		)
		new_fair, _new_f_dist = fairness(
			G, partition
		)

		new_opt = alpha * new_mod + (1-alpha) * new_fair

		if new_opt - opt <= threshold:
			return

		mod = new_mod
		fair = new_fair
		opt = new_opt

		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)
		partition_colors = partition_colors_new

		partition, inner_partition, improvement, partition_colors_new = _calculate_partition(
			graph, 
			n,
			m,
			partition,
			colors, 
			partition_colors_new,
			sensitive_attribute=sensitive_attribute, 
			alpha=alpha,
			phi=phi,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)

# Calculates average balance fairness of G for a given partition into communities
def fairness(G, partition):
	colors=nx.get_node_attributes(G, "color")
	n=G.number_of_nodes()

	sum_scores=0.0
	F_dist=[]

	# For all communities discovered
	for i, ci in enumerate(partition):
		if len(ci)>0:
			# For all nodes u in ci, check if R/B
			sum_r=0
			sum_b=0
			for u in ci:
				if colors[u]=="red":
					sum_r+=1
				elif colors[u]=="blue":
					sum_b+=1
				# else goes here if extended to multiple colors
			balance_ci=0.0
			if sum_r>0 and sum_b>0:
				balance_ci=1.0*min(sum_r/sum_b,sum_b/sum_r)

			# Add to total sum
			ci_score=((sum_r+sum_b)*balance_ci)
			sum_scores+=ci_score
			F_dist.append(ci_score/n)

	return sum_scores/n, F_dist

# Helper function to remove all empty dicts
def _full_partition_colors(part_dict):
	# If partition is empty, return False to discard
	if part_dict["blue"]==0 and part_dict["red"]==0:
		return False
	return True

def _calculate_partition(G, n, m, partition, colors, partition_cols, sensitive_attribute="color", alpha=0.9, phi=0, resolution=1, threshold=0.0000001, is_directed=False, seed=None):
	# At start, assign each node to a community 
	comms = {u: i for i,u in enumerate(G.nodes())}
	inner_partition = [{u} for u in G.nodes()]

	partition_colors=partition_cols.copy()
	original_partition=partition.copy()

	# @TODO: fix when supporting DiGraphs
	if is_directed==True:
		print("Directed networks not supported.")
		return None

	# Get all neighbours, including R/B
	nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
	
	# Get sum of edge weights for all B/R nodes
	sum_all = {u: len(nbrs[u]) for u in G}

	degrees=dict(G.degree(weight="weight"))
	Stot=list(degrees.values())


	# Do random shuffle on seed
	rand_nodes=list(G.nodes)
	seed.shuffle(rand_nodes)

	# Initial weighted fairness score calculation for all partitions (unnormalized)
	sum_scores=sum([
		(partition_colors[comms[ci]]["blue"] + partition_colors[comms[ci]]["red"]) 
			* partition_colors[comms[ci]]["score"]
		for ci in comms])


	# Start calculating movements to other communities
	n_moves=1
	improvement=False

	while n_moves>0:
		n_moves=0

		avg_fair = sum_scores/n 
			
		# For each node:
		for u in rand_nodes:
			# Initialize with current community
			opt_best=0
			sum_scores_new=sum_scores
			comm_best=comms[u]
			new_fair_update=0.0
			post_fair_update=0.0

			# Get degrees
			degree=degrees[u]
			Stot[comm_best]-=degree

			# For each node, calculate the weights of its neighbours in the same comm
			w2c=neighbor_weights(nbrs[u],comms)

			
			# Calculate modularity remove cost
			mod_rem = -w2c[comm_best]/m + resolution*(Stot[comm_best]*degree)/(2* m**2)

			# Calculate fairness (balance) current cost
			## @TODO: slowest part, probably. Can be optimized?
			com = G.nodes[u].get("nodes", {u})

			# Curr red, blue --> only for nodes under u
			curr_red=0
			curr_blue=0
			for i in com:
				if colors[i]=="blue":
					curr_blue+=1
				elif colors[i]=="red":
					curr_red+=1

			# Curr fair: all nodes in comms(u)
			all_red=partition_colors[comms[u]]["red"]
			all_blue=partition_colors[comms[u]]["blue"]

			curr_fair=0.0
			if all_red>0 and all_blue>0:
				curr_fair=(all_red+all_blue) * min(all_red/all_blue,all_blue/all_red)
			
			# Calculate score of current community after move of u
			post_red=all_red-curr_red
			post_blue=all_blue-curr_blue

			# Important: add failsafe here. If post drops <0, do not allow move. cont
			if post_red<0 or post_blue<0: continue

			post_fair=0.0
			if post_red>0 and post_blue>0:
				post_fair=(post_red+post_blue) * min(post_red/post_blue,post_blue/post_red)

			# For each neighbour of u: 
			for nbr_comm, wt in w2c.items():

				# Calculate new modularity
				mod_gain = mod_rem + wt/m - resolution*(Stot[nbr_comm]*degree)/(2* m**2)

				# Calculate gain (on sum_scores) for move to nbr_comm
				new_score = partition_colors[nbr_comm]["score"]
				new_red = partition_colors[nbr_comm]["red"] + curr_red
				new_blue = partition_colors[nbr_comm]["blue"] + curr_blue
				# Fairness score for newly created community
				new_fair=0.0
				if new_red>0 and new_blue>0:
					new_fair = (new_red + new_blue) * min(new_red/new_blue, new_blue/new_red)
				## Sum scores update:
				# new_fair (nbr) + post_fair (comms(u)) - new_score (nbr) - curr_fair (comms(u))
				sum_scores_update = new_fair + post_fair - new_score - curr_fair

				# Overall fairness score gain
				fair_gain = sum_scores_update / n

				# Calculate new opt score
				opt_gain = alpha * mod_gain + (1-alpha) * fair_gain

				# If opt gain tops previous high score, set as new best
				if opt_gain - opt_best > threshold:
					opt_best=opt_gain
					comm_best=nbr_comm

					new_fair_update=new_fair
					post_fair_update=post_fair
					sum_scores_new=sum_scores_update


			# Then finalize move if necessary
			Stot[comm_best]+=degree
			if comm_best != comms[u]:
				# Finalize move. Get nodes in com
				com = G.nodes[u].get("nodes", {u})

				# Update new community colors
				partition_colors[comm_best]["blue"]+=curr_blue
				partition_colors[comm_best]["red"]+=curr_red
				partition_colors[comm_best]["score"]=new_fair_update
				
				# Update old community colors
				partition_colors[comms[u]]["blue"]-=curr_blue
				partition_colors[comms[u]]["red"]-=curr_red
				partition_colors[comms[u]]["score"]=post_fair_update


				# Update sum_scores
				sum_scores+=sum_scores_new

				# Update partition, remove com nodes from comms(u)
				partition[comms[u]].difference_update(com)
				# Update inner partition, remove u from comms(u)
				inner_partition[comms[u]].remove(u)
				# Update partition, add com nodes to comm_best
				partition[comm_best].update(com)
				# Update inner partition, add u to comm_best
				inner_partition[comm_best].add(u)
				# Signal improvement
				improvement = True
					
				# Comment for infinite loop on a=0
				n_moves += 1

				# Change new best community for u
				comms[u] = comm_best

	partition = list(filter(len, partition))
	inner_partition = list(filter(len, inner_partition))
	partition_colors_new = list(filter(_full_partition_colors, partition_colors))

	return partition, inner_partition, improvement, partition_colors_new



# Get weights between node and its neighbor communities. Also for blue and red nodes
def neighbor_weights(nbrs, comms):
	weights=defaultdict(float)
	for nbr, w in nbrs.items():
		weights[comms[nbr]]+=w
	return weights

# Generate a new graph based on the partitions of a given graph.
# Also update partition colors
def _gen_graph(G, partition, colors):
	H = G.__class__()
	node2com = {}
	partition_colors = {}
	for i, part in enumerate(partition):
		nodes = set()
		for node in part:
			partition_colors[i]={}
			node2com[node] = i
			nodes.update(G.nodes[node].get("nodes", {node}))
		H.add_node(i, nodes=nodes)

	for node1, node2, wt in G.edges(data=True):
		wt = wt["weight"]
		com1 = node2com[node1]
		com2 = node2com[node2]
		temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
		H.add_edge(com1, com2, weight=wt + temp)
	return H, partition_colors


## -------------------------------------
##         Experiments & tests
## -------------------------------------


def test_network():
	import matplotlib.pyplot as plt

	# Test network, Figure 1
	G = nx.Graph()
	G.add_nodes_from([0,1,2,3,4,7,8,9],color="red")
	G.add_nodes_from([5,6,10,11,12,13],color="blue")
	elist = [
		(0,1),(0,2),
		(1,2),(1,4),(1,5),
		(2,3),(2,4),
		(3,4),
		(4,6),
		(5,7),(5,8),
		(6,9),
		(7,8),(7,9),
		(8,9),(8,10),
		(9,12),
		(10,11),(10,12),(10,13),
		(11,12),(11,13),
		(12,13)
	]
	for e in elist:
		G.add_edge(*e,weight=1.0)

	# Run here!
	n=10
	for _ in range(n):
		res=fair_louvain_communities(G,alpha=0.65)
		print(res)

	return

if __name__ == '__main__':
	test_network()
