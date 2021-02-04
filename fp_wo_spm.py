#since the data sets for this program have not been ran through spm
#velocities cannot be calculated.
import sys
'''sys.path.append('/Users/alechewitt/Desktop/faiss_project/faiss-master/faiss/python/swigfaiss.py')'''
#print(sys.path)

import faiss
import os
import numpy as np
import pandas as pd
import myFunctions1 as mf
from sympy import *
from sympy.vector import CoordSys3D
from sympy.utilities.lambdify import lambdify
import time
import faissFunctions as ff


#gives parallax corresponding to a distance


#print(sys.path)
print('making dataframe')
start=time.time()
df=pd.read_csv('/Users/alechewitt/Desktop/LBNLSULIresearch/DataSets/Sulidata_3_1.csv',
               dtype=float , skiprows=None, nrows=500)
df2=pd.read_csv('/Users/alechewitt/Desktop/LBNLSULIresearch/DataSets/Sulidata_3_1_ref.csv',
               dtype=float , skiprows=None, nrows=500)

print('no. columns ',len(df.columns))
end=time.time()-start
print('Done...',end , ' seconds')





#listing how many stars have a value of 0, 1, 2 for attribute 'phot_proc_mode'
phot_proc_mode=list(df['phot_proc_mode'])
num0=0; num1=0; num2=0; numnan=0
for n in range(len(phot_proc_mode)):
    if phot_proc_mode[n]==0:
        num0+=1
    if phot_proc_mode[n]==1:
        num1+=1
    if phot_proc_mode[n]==2:
        num2+=1
    if np.isnan(phot_proc_mode[n]):
        numnan+=1
ratio0=num0/len(phot_proc_mode);ratio1=num1/len(phot_proc_mode)
ratio2=num2/len(phot_proc_mode);rationan=numnan/len(phot_proc_mode)

#plotting below attribute, counting how many stars have value >2 and their ratio
aens=list(df['astrometric_excess_noise_sig'])
aenscount=0
for n in range(len(aens)):
    if aens[n]>2:
        aenscount+=1
    else:
        pass
aensratio=aenscount/len(aens)
#creates x values to plot aens
xaens=[]
for n in range(len(aens)):
    xaens.append(n)
aensbin=mf.bootstrap(0,100,.5,aens,xaens)
aensbinx=aensbin[0]
aensbiny=aensbin[2]

#count how many aens are equal to zero
aenscount1=0
for n in range(len(aens)):
    if aens[n]==0:
        aenscount1+=1

#added to apply const


print('creating vme dataframe df3')
start=time.time()
vmeind=[]
colnamdf2=df2.columns.values
for n in range(len(colnamdf2)):
    if colnamdf2[n][-3:]=='vme':
        vmeind.append(n)
df3=df2.iloc[:,vmeind]
end=time.time()-start
print('df3 done, ', end)

'''print('creating ve dataframe df3')
start=time.time()
vmeind=[]
colnamoe=[]
colnamdf2=df2.columns.values
for n in range(len(colnamdf2)):
    if colnamdf2[n][-10:]=='over_error':
        vmeind.append(n)
        colnamoe.append(colnamdf2[n])
df3=df2.iloc[:,vmeind]
end=time.time()-start
print('df3 done, ', end)

print('creating v dataframe df3')
colnamv=[]
for n in range(len(colnamoe)):
    colnamvtemp=colnamoe[n].replace('_over_error','')
    colnamv.append(colnamvtemp)
vind=[]
for n in range(len(colnamv)):
    for i in range(len(df2.columns.values)):
        if colnamv[n]==df2.columns.values[i]:
            vind.append(i)
df4=df2.iloc[:,vind]

#df3=pd.concat([df3,df4], axis=1)
print(np.shape(df3))
print(df3)
print(df3.columns)'''

    



#creates a file to list accepted features, ratios, etc.
ff.mkacptfile

print('before conversions ', len(df.columns))
print('Adding useful conversions to the dataframe')
start=time.time()
ff.usefulconv(df)
end=time.time()-start
print('Done...',end,' seconds')




print('Sympy...')
start=time.time()
calcpos=ff.calcpos(df)
rsh=calcpos[0]
rsheq=calcpos[1]
end=time.time()-start
print('Sympy Done...',end, ' seconds')




print('starting faiss')
#k nearest neighbors with rsh vector only, this picks out clusters in space
testrsh=rsh[:,:]
testrsht=np.transpose(testrsh).astype('float32',order='C')
rsh=testrsht
#redefining rsh and testrsht for testing purposes
#rsh=rtest
testrsht=rsh
rshx=rsh[:,0]; rshy=rsh[:,1]; rshz=rsh[:,2]
index3=faiss.IndexFlatL2(3)
print('testrsh ',index3.is_trained)
index3.add(testrsht)
D4,I4 = index3.search(testrsht,3)
print('shape D4 ',np.shape(D4))
print('Done...',end,' seconds')



#reattempt except this works for whole data set
D4f=D4.flatten()
I4f=I4.flatten()

D4I4f=[]
for n in range(len(D4f)):
    D4I4f.append([D4f[n],I4f[n]])
D4I4f.sort()
D4I4f1=[]
for n in range(len(D4I4f)):
    if D4I4f[n][0]!=0.0:
        D4I4f1.append(D4I4f[n])
#takes the closest N objects, indices
clind=[]
N=220
for n in range(int(np.floor(len(D4I4f1)/100))):
    clind.append(D4I4f1[n][1])
#turns indices to positions
rshclx=[]; rshcly=[]; rshclz=[]
for n in range(len(clind)):
    rshclx.append(rsh[clind[n],0])
    rshcly.append(rsh[clind[n],1])
    rshclz.append(rsh[clind[n],2])
    

#mf.plot3d(rshx,rshy,rshz,max(abs(min(rshx)),max(rshx)),max(abs(min(rshy)),max(rshy)),max(abs(min(rshz)),max(rshz)))
#mf.plot3d(rshcx,rshcy,rshcz,max(abs(min(rshx)),max(rshx)),max(abs(min(rshy)),max(rshy)),max(abs(min(rshz)),max(rshz)))

#finding N clusters and their x, y, z components of each star
'''Nclusters=ff.Nclusters(10,D4,I4,rsh)
rshclustsx=Nclusters[0]
rshclustsy=Nclusters[1]
rshclustsz=Nclusters[2]'''

#generate points of a circle thatx signifies bounds of galaxy
'''galr=25
delx=.01
ptscirc=ff.ptscirc(galr,delx,rhc)
ptsx=ptscirc[0]; ptsy=ptscirc[1]; ptsz=ptscirc[2]'''

#ff.dstvsneigh(rsh,3,100,2)


#Plots
#mf.plot3d(rshx,rshy,rshz,max(rshx),max(rshy),max(rshz))
#mf.plot3d(rshclustsx,rshclustsy,rshclustsz,max(rshx),max(rshy),max(rshz))
#mf.plot3d2(ptsx,ptsy,ptsz,rshx,rshy,rshz,galr,galr,galr)
minnind=ff.maxsim(rsh,3,100)
rshfoc=rsh[minnind]
rx=rshfoc[0]; ry=rshfoc[1]; rz=rshfoc[2]
#mf.plot3d(rx,ry,rz,max(rshx),max(rshy),max(rshz))

#ff.dstvsneigh(rsh,3,20,10)
#ff.dstvsneigh1(fw.rsh,3,20,10)
rsh2=rsh[:,0:2].astype('float32',order='C')

##dsvsn=ff.dstvsneigh1(df3,len(df3.iloc[1]),150,2)
#ff.dstvsneigh1(fw.df3,len(fw.df3.iloc[1]),150,2)
'''
#dsvsn[0]
numl=dsvsn[1]
print('numl ',np.shape(numl))
avdist=dsvsn[2]
print('avdist ', np.shape(avdist))
#df3 plays the role of r, in this case r=kn^(1/12)
#if we divide df3 by n^(1/12) it should approach a constant
#n to the 1 12th
n12=[]
for n in range(len(numl)):
    n12.append(numl[n]**(1/7))
#mf.plotStuff(fw.numl,fw.avdist,min(fw.numl),max(fw.numl),min(fw.avdist),max(fw.avdist))
##ff.dstvsneigh1deriv(numl,avdist)
numl=np.array(numl)
avdist=np.array(avdist)
n12=np.array(n12)
y=avdist/n12
##mf.plotStuff(numl,y,min(numl),max(numl),min(y),max(y))
#ff.dstvsneigh1(,len(fw.df3.iloc[1]),150,2)
#finding the average values of df3 columns
#ff.n_to_the_N(fw.numl,fw.avdist,1/
avg=[]
summ=[]
for n in range(len(df3.columns)):
    avgtemp=sum(abs(df3.iloc[:,n]))/len(df3.iloc[:,n])
    summtemp=sum(abs(df3.iloc[:,n]))
    avg.append(avgtemp)
    summ.append(summtemp)
'''

'''ff.plothistlog(df2,'phot_rp_mean_flux_over_error',min(df2['phot_rp_mean_flux_over_error']),max(df2['phot_rp_mean_flux_over_error']),xlbl='phot_rp_mean_flux_over_error')
ff.plothistlin(df2,'pmra_error',min(df2['pmra_error']),max(df2['pmra_error']),binwidth=.1,xlbl='pmra_error')
'''



#determining whether columns in the dataset depend on one another
#if cor>0 variables change in same direction, cor<0 variables change in opposite
#direction, cor=0 variables are not related.
from scipy.stats import pearsonr
cor,_=pearsonr(df3['pm_vme'],df3['pmra_vme'])
#list of correlations for df3, one for each combination of columns
cor_list=[]
for n in range(len(df3.columns)):
    for i in range(len(df3.columns)):
        if i!=n:
            cor,_=pearsonr(df3.iloc[:,n],df3.iloc[:,i])
            cor_list.append(cor)




#inverted faiss index
'''dimension = 3    # dimensions of each vector                         
n = 2000    # number of vectors                   
np.random.seed(1)             
v = rsh
nlist = 40  # number of clusters
quantiser = faiss.IndexFlatL2(dimension)  
index = faiss.IndexIVFFlat(quantiser, dimension, nlist,   faiss.METRIC_L2)
print(index.is_trained)   # False
index.train(v)  # train on the database vectors
print(index.ntotal)   # 0
index.add(v)   # add the vectors and update the index
print(index.is_trained)  # True
print(index.ntotal)   # 200
nprobe = 1  # find 2 most similar clusters
n_query = 4  
k = 55  # return 3 nearest neighbours
np.random.seed(0)   
query_vectors = np.random.random((n_query, dimension)).astype('float32')
d, i = index.search(query_vectors, k)'''

#turn df3 into a list
vecdf3=[]
for n in range(len(df3)):
    vecdf3.append(list(df3.iloc[n].values))
vecdf3=np.array(vecdf3).astype('float32')
dimension=len(vecdf3[0])

'''avdist,nlist=ff.av_dist_num_clusters(vecdf3,len(df3.iloc[1]),30)
mf.plotStuff(nlist,avdist,min(nlist),max(nlist),min(avdist),max(avdist))'''

###
'''vec_new=ff.return_n_clusters(vecdf3,len(vecdf3[0]),4)
centers=ff.return_n_centers(vec_new,len(vecdf3[0]))
sizes=ff.return_cluster_size(vec_new,centers)
print('before')
start=time.time()
foundk=ff.find_k(vecdf3,dimension,4)
end=time.time()-start
print('found k ',end)
start=time.time()
D,I=ff.IndexIVFFlat(vecdf3,len(vecdf3[0]),4,foundk)
end=time.time()-start
print('IVF ',end)
start=time.time()
cts=ff.count_negative_1(I)
end=time.time()-start
print('-1 counted ',end)
start=time.time()
I=ff.remove_value(I,-1)
end=time.time()-start
print('-1s removed ',end)
start=time.time()
I=ff.remove_duplicates(I,D,4)[1]
end=time.time()-start
print('duplicates removed ',end)
number_of_clusters=4
#calculating Dij, the distance between every possible combination of centers
vec_new=ff.return_n_clusters(vecdf3,len(vecdf3[0]),number_of_clusters)
centers=ff.return_n_centers(vec_new,len(vecdf3[0]))'''

'''Dij=[]
for i in range(len(centers)):
    Dijrow=[]
    for j in range(len(centers)):
            Dijtemp=np.linalg.norm(centers[i]-centers[j])
            Dijrow.append(Dijtemp)
    Dij.append(Dijrow)

#Sij= Dij/sqrt(Vi + Vj) i,j refers to cluster index, V=Sum(Dic^2), Dic is distance from center of cluster to a star in that cluster
#lets calculate Vi for each cluster
Vi=[]
for n in range(len(centers)):
    Dic=[]
    for j in range(len(vec_new[n])):
        Dictemp=np.linalg.norm(vec_new[n][j]-centers[n])
        Dic.append(Dictemp)
    V=sum(np.array(Dic)**2)
    Vi.append(V)

Sij=[]
for i in range(len(Vi)):
    for j in range(len(Vi)):
        Sij.append(Dij[i][j]/np.sqrt(Vi[i]+Vi[j]))'''
##
'''start=time.time()
av_sij, num_clusters=ff.av_sij_num_clusters(vecdf3,len(vecdf3[0]),30)
end=time.time()-start
print('av_sij ', end)

mf.plotStuff(num_clusters,av_sij,min(num_clusters),max(num_clusters),min(av_sij),max(av_sij))'''




'''index3=faiss.IndexFlatL2(3)
index3.is_trained
index3.add(rsh)
D4,I4 = index3.search(rsh,3)'''


#using faiss to get cluster vectors
number_of_clusters=5
numneigh=100
df3clust,centers,numstrs=ff.faiss_return_n_clusters(vecdf3,number_of_clusters,numneigh)
dimension=len(df3clust[0][0])
print(numstrs)

av_sij,num_clusters,Vitot,Dijtot,sumsij,Sijtot,nanstot,Sijdtot=ff.av_sij_num_clusters(vecdf3,dimension,30)
#mf.plotStuff(num_clusters,av_sij,min(num_clusters),max(num_clusters),min(av_sij),max(av_sij)+1)
