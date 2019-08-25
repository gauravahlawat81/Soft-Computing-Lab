#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:23:40 2019

@author: gaurav
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:37:41 2019

@author: gaurav
"""
import pandas as pd
import csv
import random
from random import seed
def predict(weight,row,size):
    #print("Size of weight is {}".format(len(weight)))
    activation=weight[size-1]
    for i in range(len(row)-1):
        activation=activation + weight[i]*row[i]
    #print("activation is {}".format(activation))
    #print("weight[-1] is {}".format(weight[-1]))
    return 1.0 if activation>=1 else 0.0


def train_weights(train,rate,count,size,test):
    #weight=[0.0 for i in range(len(train[0]))]
    weight=[]
    for i in range(len(train[0])):
        weight.append(random.uniform(0,1))
    #print(weight)
    for i in range(count):
        sum_error=0.0
        for row in train:
            prediction=predict(weight,row,size)
            error=row[size-1]-prediction
            sum_error=sum_error+error**2
            weight[size-1]=weight[size-1]+rate*error
            for j in range(len(row)-1):
                weight[j]=weight[j]+rate*error*row[j]
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (i,rate, sum_error))
        #print(weight)
    #return weight
    count=0
    ll=0
    for row in test:
        ll=ll+1
        prediction=predict(weight,row,size)
        if(prediction==row[size-1]):
            count=count+1
    
    score=count/ll;
    score=score*100
    #print("The score is {}".format(score))
    return score
        
    


def fold(dataset,i,k):
	l=len(dataset)
	start_index_test=l*(i-1)//k
	end_index_test=l*i//k
	#print(end_index_test)
	if start_index_test==0:
		start_index_train=end_index_test
		end_index_train=l
		return [dataset[start_index_train:end_index_train],dataset[start_index_test:end_index_test]]
	elif end_index_test==l:
		start_index_train=0
		end_index_train=start_index_test
		return [dataset[start_index_train:end_index_train],dataset[start_index_test:end_index_test]]
	else:
		new_dataset=[]
		for i in range(start_index_test):
			new_dataset.append(dataset[i])
		for j in range(end_index_test,l):
			new_dataset.append(dataset[j])

		return [new_dataset,dataset[start_index_test:end_index_test]]

def main():
    dataset = pd.read_csv('SPECT.csv')
    dataset = dataset.sample(frac=1)
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
            rows[i][j]=int(rows[i][j])
    
    for i in range(len(rows)):
        val=int(rows[i][0])
        rows[i].pop(0);
        rows[i].append(val)
    
    rate=0.01
    count=2000
    size=len(rows[0])
    #print(len(rows[0]))
    #weight=train_weights(rows,rate,count,size)
    
    #print(weight)
    
    #count=0
    #for row in rows:
     #   val=predict(weight,row,size)
      #  if(val==row[-1]):
       #     count=count+1;
        #print("Expected {} and predicted {}".format(row[-1],val))
        
    #x=count/len(rows)
    #x=x*100
    #print(x)
    k=int(input("Enter the value of k: "))
    accuracy=[]
    
    for i in range(1,k+1):
        after_fold=fold(rows,i,k)
        train_set=after_fold[0]
        #print("Train set is ")
        #print(train_set)
        test_set=after_fold[1]
        #print("Test set is")
        #print(test_set)
        acc=train_weights(train_set,rate,count,size,test_set)
        
        accuracy.append(acc)
    #print("The Accuracy for each fold is as follows : ")
    sum = 0
    for i in accuracy:
        print(i)
        sum=sum+i;
    print("\nAccuracy of {}- fold  Single Layer Perceptron is:{}".format(k,sum/k))
    
if __name__=='__main__':
    main()

    


    



        
