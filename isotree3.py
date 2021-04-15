import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pprint import pprint
import seaborn as sns
import pdb
sns.set_style(style="whitegrid")
from matplotlib import rcParams


plt.style.use('fivethirtyeight')
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['text.color'] = 'k'
rcParams['figure.figsize'] = 16,8


X = pd.read_csv('creditcard.csv').sample(frac = 0.1)
# other options: X = creditcard.sample(frac=0.1,random_state=1).reset_index(drop=True)
fraudClass = X.loc[:,'Class']
x = X.loc[:, 'V1']
y = X.loc[:, 'V2']
toDrop = ['Time','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']
X.drop(toDrop, inplace = True, axis = 1)

print(X[:5])
print(X.columns)
print(X.describe)
plt.figure(figsize=(7,7))
plt.plot(x,y,'bo');
# plt.savefig('sample.png')
plt.show()

def select_feature(data): 
    '''
    Randomly select a feature of a dataframe
    '''
    return random.choice(data.columns)

def select_value(data,feat):
    '''
    Select values of 
    '''
    mini = data[feat].min()
    maxi = data[feat].max()
    return (maxi-mini)*np.random.random()+mini

# this is just a test...leave commented
# var = select_feature(X) 
# value = select_value(X,var)
# print(var, value)


def split_data(data, split_column, split_value):
    '''
    Split data based on the value of a column
    '''
    data_below = data[data[split_column] <= split_value]
    data_above = data[data[split_column] >  split_value]
    
    return data_below, data_above


var = select_feature(X) 
value = select_value(X,select_feature(X))
print(var,value)
a,b =split_data(X,
           var,
           value)

print(a.shape)
print(b.shape)




def classify_data(data):
    
    label_column = data.values[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification
classify_data(X)

def isolation_tree(data,counter=0, max_depth=50):
    
    # End Loop
    if (counter == max_depth) or data.shape[0]<=1:
        classification = classify_data(data)
        return classification
    
    else:
        # Counter
        counter +=1
        
        # Select feature
        split_column = select_feature(data)
        
        # Select value
        split_value = select_value(data,split_column)

        # Split data
        data_below, data_above = split_data(data,split_column,split_value)
        
        # instantiate sub-tree      
        question = "{} <= {}".format(split_column, split_value)
        sub_tree = {question: []}
        
        # Recursive part
        below_answer = isolation_tree(data_below, counter,max_depth=max_depth)
        above_answer = isolation_tree(data_above, counter,max_depth=max_depth)
        
        if below_answer == above_answer:
            sub_tree = below_answer
        else:
            sub_tree[question].append(below_answer)
            sub_tree[question].append(above_answer)
        
        return sub_tree


tree = isolation_tree(X, max_depth=1)
print(tree)


def isolation_forest(df,n_trees=5, max_depth=5, subspace=256):
    forest = []

    for i in range(n_trees):
        # Sample the subspace
        if subspace<=1:
            df = df.sample(frac=subspace)
        else:
            df = df.sample(subspace)
        

        # Fit tree
        tree = isolation_tree(df,max_depth=max_depth)
        
        # Save tree to forest
        forest.append(tree)
    
    return forest

forestmodel = isolation_forest(X,n_trees=5,max_depth=1)
print(forestmodel)



def pathLength(example,iTree,path=0,trace=False):
    
    path=path+1
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

tree = isolation_tree(X.head(50),max_depth=3)

ins = X.sample(1)
path1 = pathLength(ins,tree)
print(path1)



def makeline(data,example,iTree,path=0,line_width=1):
    
    #line_width = line_width +2
    path=path+1
    question = list(iTree.keys())[0]
    feature_name, comparison_operator, value = question.split()
    print(question)
    
    # ask question
    if example[feature_name].values <= float(value):
        answer = iTree[question][0]
        data = data[data[feature_name] <= float(value)]
    else:
        answer = iTree[question][1]
        data = data[data[feature_name] > float(value)]
        

    if feature_name == 'V1':
        plt.hlines(float(value),xmin=data.V1.min(),xmax=data.V1.max(),linewidths=line_width)
    else:
        plt.vlines(float(value),ymin=data.V2.min(),ymax=data.V2.max(),linewidths=line_width)
             
        
    # base case
    if not isinstance(answer, dict):
        return path
    
    # recursive part
    else:
        if feature_name == 'V1':
            plt.hlines(float(value),xmin=data.V1.min(),xmax=data.V1.max(),linewidths=line_width)
        else:
            plt.vlines(float(value),ymin=data.V2.min(),ymax=data.V2.max(),linewidths=line_width)
        residual_tree = answer
        return makeline(data,example, residual_tree,path=path,line_width=line_width)
    
    return path

def make_plot(data,example,iTree):
    plt.figure()
    plt.plot(data['V1'],data['V2'],'bo',alpha=0.2)
    plt.xlabel('V1')
    plt.ylabel('V2')
    plt.xlim(data.V1.min(),data.V2.max())
    plt.ylim(-3,3)
    plt.xlim(-3,3)
      
    
    # Plot H,v line
    makeline(data,example,tree)
    

    # Plot the point we are looking for
    plt.scatter(x=example.V1,y=ins.V2,c='r',marker='o')
    
    plt.show()

data_plot = X.sample(10)
ins = data_plot.sample(1)
print(ins)

tree = isolation_tree(data_plot,max_depth=50)
make_plot(data_plot,ins,tree)

'''
iForest = isolation_forest(X,n_trees=20, max_depth=100, subspace=256)

def evaluate_instance(instance,forest):
    paths = []
    for tree in forest:
        paths.append(pathLength(instance,tree))
    return paths

def c_factor(n) :
    """
    Average path length of unsuccesful search in a binary search tree given n points
    
    Parameters
    ----------
    n : int
        Number of data points for the BST.
    Returns
    -------
    float
        Average path length of unsuccesful search in a BST
        
    """
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


def anomaly_score(data_point,forest,n):
    '''
    # Anomaly Score
    
    # Returns
    # -------
    # 0.5 -- sample does not have any distinct anomaly
    # 0 -- Normal Instance
    # 1 -- An anomaly
    '''
    # Mean depth for an instance
    E = np.mean(evaluate_instance(data_point,forest))
    
    c = c_factor(n)
    
    return 2**-(E/c)


an= []
for i in range(X.shape[0]):
    an.append(anomaly_score(X.iloc[[i]],iForest,256))

def instance_depth_plot(instance,outlier,forest):
    bars1 = evaluate_instance(outlier,forest)

    bars2 = evaluate_instance(instance,forest)

    # width of the bars
    barWidth = 0.3

    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    
    # Create cyan bars
    plt.bar(r2, bars2, width = barWidth, capsize=7, label='Normal Sample')

    # Create blue bars
    plt.bar(r1, bars1, width = barWidth,  capsize=7, label='Outlier')
    #sns.barplot(x=r1, y=bars1,capsize=7, label='Outlier')

    
    #sns.barplot(x=r2, y=bars2, label='Normal')
    
    # general layout

    plt.ylabel('Tree Depth')
    plt.xlabel('Trees')
    plt.legend()

    # Show graphic
    plt.savefig('images/normal_vs_outlier.png')

    plt.show()

instance_depth_plot(X.sample(1),X.head(1),iForest)

outlier  = evaluate_instance(X.head(1),iForest)
normal  = evaluate_instance(X.sample(1),iForest)

np.mean(outlier)

np.mean(normal)

X.shape

iForest = isolation_forest(X,n_trees=20, max_depth=100, subspace=1_560)

print('Anomaly score for outlier:',anomaly_score(X.head(1),iForest,1560))
print('Anomaly score for normal:',anomaly_score(X.sample(1),iForest,1560))


















