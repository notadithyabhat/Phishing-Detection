import numpy as np
import pandas as pd
from collections import Counter
import random
import time
from scipy.io import arff


df = pd.read_csv('data.csv')
df=df.drop(columns=['id'])


start=time.time()

def k_nearest_neighbours(data,predict,k):
    distances=[]
    for group in data:
        for features in data[group]:
            euclid_distance=np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclid_distance,group])
    #print(sorted(distances))
    votes=[]
    for i in sorted(distances)[:k]:
         votes.append(i[1])
    #print(votes)

    cnt=Counter(votes)
    result=cnt.most_common(1)
    #print(result)
    return result[0][0]


df.fillna(-99999,inplace=True)
full_data=df.values.tolist()
#print(full_data)
#random.shuffle(full_data)

test_size=0.01
train_set={-1:[],1:[]}
test_set={-1:[],1:[]}

train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]
#print(test_data)
#train_data=train_data1[:6000]
#test_data=test_data1[:2000]


for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

#print(len(full_data),len(train_data),len(test_data))

data=[-1,0,-1,1,-1,0,1,1,-1,1,1,-1,1,0,0,-1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1]
print("The website to be predicted has the following features")
print(data)
vote=k_nearest_neighbours(train_set,data,7)
print(vote)
if vote==1:
    print("The website is Legitimate")
if vote==-1:
    print("The webisite is a Phishing website")

correct=0
total=0

'''
#finding the accuracy

for group in test_set:
    for data in test_set[group]:
        #print("reached")
        vote=k_nearest_neighbours(train_set,data,7)
        print(vote,group)
        if group==vote:
            correct=correct+1
        total=total+1

accuracy=correct/total*100
print("Accuracy=",accuracy)
end=time.time()
print("Time elapsed=",round((end-start),2),"seconds")

'''