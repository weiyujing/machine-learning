from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.cluster import  KMeans
from sklearn.manifold import TSNE
import  matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import  RandomizedLogisticRegression as RLR
inputfile='D:\\abc3.csv'
outfile='D:\\temp.csv'
pd.set_option('display.max_rows',None)

k=6
iteration=100000
data=pd.read_csv(inputfile,index_col='ID')
#data_zs = 1.0*(data-data.mean())/data.std()
#print(np.isnan(data_zs))
model=KMeans(n_clusters=k,n_jobs=1,max_iter=iteration)
model.fit(data)
r1=pd.Series(model.labels_).value_counts()
r2=pd.DataFrame(model.cluster_centers_)
r=pd.concat([r2,r1],axis=1)
r.columns=list(data.columns)+[u'类别数目']
print(r)

r=pd.concat([data,pd.Series(model.labels_,index=data.index)],axis=1)
r.columns=list(data.columns)+[u'聚类类别']
print(r)

tsne = TSNE()
tsne.fit_transform(data)
tsne=pd.DataFrame(tsne.embedding_,index = data.index)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
d=tsne[r[u'聚类类别']==0]
plt.plot(d[0],d[1],'ro')
d=tsne[r[u'聚类类别']==1]
plt.plot(d[0],d[1],'go')
d=tsne[r[u'聚类类别']==2]
plt.plot(d[0],d[1],'b*')
d=tsne[r[u'聚类类别']==3]
plt.plot(d[0],d[1],'y*')
d=tsne[r[u'聚类类别']==4]
plt.plot(d[0],d[1],'r*')
d=tsne[r[u'聚类类别']==5]
plt.plot(d[0],d[1],'g*')
plt.show()



