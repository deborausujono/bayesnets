Run the code by typing:
	python bayes_net.py (graph_file) (train_file) (test_file)

The train_file and test_file have to be in the format provided for the assignment. Each row in the graph_file should contain a space-separated list of the parents of each node. If a node does not have any parents, then the line is left blank. The input graph file used in the assignment is included in the zip file. The first row in the graph file corresponds to the node in the first column in the data set, second row corresponds to the second column, and so on.

The answers to questions 4 and 5 are printed out to standard output when running the code. It also prints out the prediction accuracy with the given training and test set. I obtain the mean and standard deviation of these values by hand.

The code was written in Python 2.7.3.