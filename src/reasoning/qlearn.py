import math
import numpy as np 
import random

###
# Quality-based planning
###
def create_qtable(actions):
	qtable = {}
	for a in actions:
		qtable[str(a)] = {'qvalue':0.0,'sumvalue':0.0,'trials':0}
	return qtable

def ucb_select_action(node,c=0.5,mode='max'):
	# 1. Initialising the support values
	if mode == 'max':
		targetUCB, targetA = -np.inf, None
	elif mode == 'min':
		targetUCB, targetA = np.inf, None
	else:
		print('Invalid mode for UCB:', mode)
		raise NotImplemented

	# 2. Checking the best action via UCT algorithm
	for a in node.actions:
		qvalue = node.qtable[str(a)]['qvalue']
		trials = node.qtable[str(a)]['trials']
		if trials > 0:
			if mode == 'max':
				current_ucb = qvalue + c * \
				np.sqrt(np.log(float(node.visits)) / float(trials))
				if current_ucb > targetUCB:
					targetUCB = current_ucb
					targetA = a
			elif mode == 'min':
				current_ucb = qvalue - c * \
					np.sqrt(np.log(float(node.visits)) / float(trials))
				if current_ucb < targetUCB:
					targetUCB = current_ucb
					targetA = a
		else:
			return a

	# 3. Checking if the best action was found
	if targetA is None:
		targetA = random.sample(node.actions,1)[0]

	# 4. Returning the best action
	return targetA

###
# Information-based planning
###
def create_etable(actions):
	etable = {}
	for a in actions:
		etable[str(a)] = {'entropy':0.0, 'cumentropy':0.0,\
							 'trials':0, 'max_entropy': 1}
	return etable

def iucb_select_action(node,alpha,mode='max'):
	# 1. Initialising the support values
	if mode == 'max':
		targetUCB, targetA = -np.inf, None
	elif mode == 'min':
		targetUCB, targetA = np.inf, None
	else:
		print('Invalid mode for I-UCB:', mode)
		raise NotImplemented

	# 2. Checking the best action via UCT algorithm
	actions = [a for a in node.actions]
	np.random.shuffle(actions)
	for a in actions:
		qvalue = node.qtable[str(a)]['qvalue']
		trials = node.qtable[str(a)]['trials']
		if trials > 0:
			exploration_value = np.sqrt(np.log(float(node.visits)) / float(trials))

			
			information_value = (node.etable[str(a)]['entropy'] / node.etable[str(a)]['max_entropy'])

			current_ucb =  qvalue + \
				((1-alpha) * exploration_value) + (alpha * information_value)

			if mode == 'max' and current_ucb > targetUCB:
				targetUCB = current_ucb
				targetA = a
			elif mode == 'min' and current_ucb < targetUCB:
				targetUCB = current_ucb
				targetA = a
		else:
			return a

	# 3. Checking if the best action was found
	if targetA is None:
		targetA = random.sample(node.actions,1)[0]

	# 4. Returning the best action
	return targetA

def entropy(set):
	H = 0
	norm = sum([set[y] for y in set])
	for x in set:
		Px = set[x]/norm
		H += Px*math.log(Px)
	return -H

def kl_divergence_uniform(set):
	n = len(set)
	if n <= 1:
		return 0.0
	norm = sum([set[y] for y in set])
	kl = 0.0
	for x in set:
		Px = set[x]/norm
		if Px > 0:
			kl += Px*math.log(Px*n)  # log(Px / (1/n)) = log(Px*n)
	return kl

def kl_divergence(p, q, eps=1e-9):
	# KL divergence D_KL(P || Q) between two distributions 
	if len(p) == 0 or len(q) == 0:
		return 0.0
	pnorm = sum([p[y] for y in p])
	qnorm = sum([q[y] for y in q])
	kl = 0.0
	for x in p:
		Px = p[x]/pnorm
		if Px > 0:
			Qx = q[x]/qnorm if x in q else eps  
			kl += Px*math.log(Px/Qx)
	return kl