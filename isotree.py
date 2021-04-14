import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv('creditcard.csv')
# data.drop('V20','V21','V22','V23','V24','V25','V26','V27','V28','V29')

# select a column (feature) from the data:
def select_feature(data):
    return random.choice(data.columns)

# select a random value within the range for that column:
def select_value(data,feat):
    mini = data[feat].min()
    maxi = data[feat].max()
    return (maxi-mini)*np.random.random()+mini

# Split data
def split_data(data, split_column, split_value):
    data_below = data[data[split_column] <= split_value]
    data_above = data[data[split_column] > split_value]
    return data_below, data_above 

# all together for the tree
def isolation_tree(data, counter = 0, max_depth = 50, random_subspace = False):

    # end loop if max depth or isolated:
    if (counter == max_depth) or data.shape[0] <= 1:
        classification = classify_data(data)
        return classification
    
    else:
        # counter
        counter += 1
        # select feature
        split_column = select_feature(data)
        # select value
        split_value = select_value(data, split_column)
        # split the data
        data_below, data_above = split_data(data, split_column, split_value)
        # instantiate sub-tree
        question = '{} <= {}'.format(split_column, split_value)
        sub_tree = {question: []}

        # recursive part
        below_answer = isolation_tree(data_below, counter, max_depth = max_depth)
        above_answer = isolation_tree(data_above, counter, max_depth = max_depth)

        if below_answer == above_answer:
            sub_tree = below_answer
        else:
            sub_tree[question].append(below_answer)
            sub_tree[question].append(above_answer)
        return sub_tree


def isolation_forest(df, n_trees = 5, max_depth = 5, subspace = 256):
    forest = []

    for i in range(n_trees):
        # sample the subspace
        if subspace <= 1:
            df = df.sample(frac = subspace)
        else:
            df = df.sample(subspace)
    
        # fit tree
        tree = isolation_tree(df, max_depth = max_depth)
        # add tree to forest
        forest.append(tree)
    return forest 

# this is just counting how many nodes an instance goes through given how data has been stored before
def pathLength(example, iTree, path = 0, trace = False):
    # init question and counter
    path += 1
    question = list(iTree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    # ask question
    if example[feature_name].values <= float(value):
        answer = iTree[question][0]
    else:
        answer = iTree[question][1]
    
    # base case
    if not isinstance(answer, dict):
        return path
    
    # recursive part
    else:
        residual_tree = answer
        return pathLength(example, residual_tree,path=path)
        
    return path


# evaluation
mean = [0, 0]
cov = [[1, 0], [0, 1]] # diagonal covariance
Nobjs = 2000
x, y = np.random.multivariate_normal(mean, cov, Nobjs).T 
# add manual outlier
x[0] = 3.3
y[0] = 3.3
X = np.array([x,y]).T 
X = pd.DataFrame(X, columns = ['feat1', 'feat2'])
plt.figure(figsize = (7,7))
plt.plot(x, y, 'bo')
# plt.show()

iForest = isolation_forest(X, n_trees = 20, max_depth = 100, subspace = 256)

