# Perceptron Algorithm on the Sonar Dataset
from random import seed
from random import randrange
import pandas as pd
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
 
# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0
 
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights
 
# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)
 
# Test the Perceptron algorithm on the sonar dataset
seed(1)
dataset = pd.read_csv('SPECT.csv')
dataset.to_csv('test.csv')
filename="test.csv"
attributes = []
rows = []
with open(filename,'r') as csvfile:
	csvreader=csv.reader(csvfile)
	attributes=next(csvreader)
	for row in csvreader:
		rows.append(row)
        
for i in range(len(rows)):
    rows[i].pop(0)
    
for i in range(len(rows)):
    if rows[i][0]=='Yes':
        rows[i][0]=1
    elif rows[i][0]=='No':
        rows[i][0]=0

for i in range(len(rows)):
    for j in range(1,len(rows[i])):
        if rows[i][j]=='1':
            rows[i][j]=1
        elif rows[i][j]=='0':
            rows[i][j]=0
for i in range(len(rows)):
    val=int(rows[i][0])
    rows[i].pop(0);
    rows[i].append(val)
n_folds = 10
l_rate = 0.01
n_epoch = 500
scores = evaluate_algorithm(rows, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))