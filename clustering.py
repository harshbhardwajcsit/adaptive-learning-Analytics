import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from kmodes import kmodes
import json
data=pd.read_csv('sample access record data (2).csv')
data = data.drop(["identifier", "topic_id","course_id"], axis = 1)
#data.head(10)   
for i in range(0,5):
    print(data['parent_topic'].unique()[i])
    trim_data=data[data['parent_topic']==data['parent_topic'].unique()[i]]

    km = kmodes.KModes(n_clusters=100, init='Huang', n_init=100, verbose=1)
    
    clusters = km.fit_predict(trim_data)
    dict={}
    dict[data['parent_topic'].unique()[i]]=0
    for i in range(0,km.cluster_centroids_.shape[0]):
        if km.cluster_centroids_[i][1] in dict.keys():
            dict[km.cluster_centroids_[i][1]]=dict.get(km.cluster_centroids_[i][1])+1
        else:
            dict[km.cluster_centroids_[i][1]]=1
            
 
    with open(r'''D:\Users\hbhardwaj\ML demo\clustering results\results.txt''', 'a+') as file:
        file.write(json.dumps(dict,indent=4))
        



    





