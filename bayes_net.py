import sys

def learn(graph, training_set):
	#Count the number of occurrences of each configuration
	count = {} #use defaultdict
	for row in training_set:
		for i in range(len(graph)):
			parents = graph[i]
			parent_vals = []
			for j in range(len(parents)):
				parent_vals.append(row[parents[j]])
			parent_vals = tuple(parent_vals)
			node_val = row[i]
			if i in count and parent_vals in count[i] and node_val in count[i][parent_vals]:
				count[i][parent_vals][node_val] += 1
			elif i in count and parent_vals in count[i]:
				count[i][parent_vals][node_val] = 1
			elif i in count:
				count[i][parent_vals] = { node_val : 1 }
			else:
				count[i] = { parent_vals : { node_val : 1 } }

	#Apply MLE to obtain theta
	for node in count:
		for parent_vals in count[node]:
			sum = 0
			for node_val in count[node][parent_vals]: #turn into list comprehension
				sum += count[node][parent_vals][node_val]
			for node_val in count[node][parent_vals]:
				count[node][parent_vals][node_val] = 1.0*count[node][parent_vals][node_val]/sum

	return count

def classify(graph, theta, training_set):
	result = []

	#Predict HD value for each row in the training set
	for row in training_set:
		p = conditional_query(graph, theta, row, -1, [1, 2], 0)
		
		if p > 1-p :
			result.append(1)
		else:
			result.append(2)

	return result

def conditional_query(graph, theta, row, query_idx, query_vals, query_val_idx):
	"""Assumes that all other variables are observed.
	query_idx is the column index of the query variable in the training set,
	query_vals is a list of values that the query variable can take,
	query_val_idx is the index of the value of query variable in question
	in the query_vals list."""

	#Compute the joint probability of observed variables and each possible
	#value of the query variable
	p_joint = [1]*len(query_vals)
	j = 0
	for val in query_vals:
		row[query_idx] = val
		for i in range(len(row)):
			p_joint[j] = joint_query(graph, theta, row)
		j += 1

	#Obtain the sum of all probabilities in p_joint
	sum = 0
	for p in p_joint:
		sum += p

	#Obtain the conditional probability
	p_cond = p_joint[query_val_idx]/sum
	return p_cond

def joint_query(graph, theta, row):
	p = 1
	for i in range(len(row)):
		parents = graph[i]
		parent_vals = []
		for j in range(len(parents)):
			parent_vals.append(row[parents[j]])
		parent_vals = tuple(parent_vals)
		node_val = row[i]
		if parent_vals in theta[i] and node_val in theta[i][parent_vals]:
			p *= theta[i][parent_vals][node_val]
		else: #If unseen during training
			p *= 0.000000001

	return p

def evaluate(result, gold):
	#Count the number of correct predictions
	correct_count = 0
	for i in range(len(result)):
		if result[i] == gold[i]:
			correct_count += 1

	#Obtain average
	accuracy = 1.0*correct_count/len(result)
	print "Accuracy =", accuracy

def main():
	#Open a graph file containing a list of parents of each node.
	#Each row in the file corresponds to a random variable (a column) in the
	#training data (i.e., first row lists the parents of random variable A, second row of
	#random variable G, and so on).
	f = open(sys.argv[1], 'r')
	lines = f.readlines()
	graph = []
	for line in lines:
		graph.append(tuple([int(x) for x in line.split()]))

	#Open training file
	f = open(sys.argv[2], 'r')
	lines = f.read().splitlines()
	training_set = []
	for line in lines:
		training_set.append([int(x) for x in line.split(',')])

	#Compute theta
	theta = learn(graph, training_set)

	#Open test file
	f = open(sys.argv[3], 'r')
	lines = f.read().splitlines()
	test_set = []
	gold = []
	for line in lines:
		line = line.split(',')
		test_set.append([int(x) for x in line])
		gold.append(int(line[-1]))

	#Predict HD value and evaluate the predictions
	result = classify(graph, theta, test_set)
	evaluate(result, gold)

	#P(CH|A=2, G=M, CP=N, BP=L, ECG=N, HR=L, EIA=N, HD=N)
	#The variable row lists the observed values in the order random variables appear in
	#the training set (i.e., row[0] contains the value of the random variable A,
	#row[1] of random variable G, and so on).
	row = [2, 2, 4, 1, 1, 1, 1, 1, 1]
	p = conditional_query(graph, theta, row, 4, [1, 2], 0) #P(CH=L|...)
	print "P(CH=L|A=2, ..., HD=N) =", p
	print "P(CH=H|A=2, ..., HD=N) =", 1-p

	#P(BP|A=2, CP=T, CH=H, ECG=N, HR=H, EIA=Y, HD=N)
	row = [2, 1, 1, 1, 2, 1, 2, 2, 1]
	pg1bp1 = joint_query(graph, theta, row) #P(BP=L, G=F, A=2, ...)

	row = [2, 1, 1, 2, 2, 1, 2, 2, 1]
	pg1bp2 = joint_query(graph, theta, row) #P(BP=H, G=F, A=2, ...)

	row = [2, 2, 1, 1, 2, 1, 2, 2, 1]
	pg2bp1 = joint_query(graph, theta, row) #P(BP=L, G=M, A=2, ...)

	row = [2, 2, 1, 2, 2, 1, 2, 2, 1]
	pg2bp2 = joint_query(graph, theta, row) #P(BP=H, G=M, A=2, ...)
	
	sum = pg1bp1+pg1bp2+pg2bp1+pg2bp2
	print "P(BP=L, G=F|A=2, ..., HD=N) =", pg1bp1/sum
	print "P(BP=H, G=F|A=2, ..., HD=N) =", pg1bp2/sum
	print "P(BP=L, G=M|A=2, ..., HD=N) =", pg2bp1/sum
	print "P(BP=H, G=M|A=2, ..., HD=N) =", pg2bp2/sum

	#Print the probability tables of A, BP, HD and HR in the order
	#specified in the template
	print "CPTs in the order specified in the template:"
	a_cpt = theta[0][()]
	for p in a_cpt:
		print a_cpt[p]

	bp_cpt = theta[3]
	for i in range(1, 3):
		for j in range (1, 3):
			print bp_cpt[(i,)][j]

	hd_cpt = theta[8]
	for i in range(1, 3):
		for j in range(1, 3):
			for k in range(1, 3):
				print hd_cpt[(j, i)][k]

	hr_cpt = theta[6]
	for i in range (1, 3):
		for j in range(1, 3):
			for k in range(1, 4):
				for l in range(1, 3):
					if (k, j, i) in hr_cpt and l in hr_cpt[(k, j, i)]:
						print hr_cpt[(k, j, i)][l]
					else:
						print 0

if __name__ == '__main__':
	main()
