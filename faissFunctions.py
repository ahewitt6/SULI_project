import faiss
import os
import numpy as np
import pandas as pd
import myFunctions1 as mf
from sympy import *
from sympy.vector import CoordSys3D
from sympy.utilities.lambdify import lambdify
import time
import matplotlib.pyplot as plt

def ratioMeas(df):
    start=time.time()
    ratio_meas=[]
    rejected=[]
    counts=[]
    numcol=0
    for n in range(len(df.columns)):
        numcol+=1
        count=0
        rej_temp=[]
        for i in range(len(df)):
            if np.isnan(df.iloc[i][n]):
                rej_temp.append(df.iloc[i][n])
            else:
                count+=1
        rejected.append(rej_temp)
        num_measure_div_tot=count/len(df)
        ratio_meas.append(num_measure_div_tot)
        counts.append(count)
    end=time.time()-start
    return(ratio_meas,end,rejected,counts,numcol)

#gives the names of columns whose ratio_meas>ratio, indices, and the ratio of
#accepted features
def acceptedMeas(ratio,df,ratio_meas):
    start=time.time()
    namcol=[]
    namind=[]
    namcolrej=[]
    ratio_meas_ref=[]
    for n in range(len(ratio_meas)):
        if ratio_meas[n]>ratio:
            namcol.append(df.columns[n])
            namind.append(n)
            ratio_meas_ref.append(ratio_meas[n])
        else:
            namcolrej.append(df.columns[n])
    end=time.time()-start
    return(namcol,namind,ratio_meas_ref,end)

#delets old file and makes a new file with all accepted features
#an example of the feature and its ratio
def mkacptfile(df2):
    os.remove("acceptedColumns.txt")
    file1 = open("acceptedColumns.txt", "a")  # append mode 
    file1.write('Accepted Columns: ratio measured >0.95:' + "\n" + "\n") 

    for n in range(len(df2.columns)):
        file1.write(str(df2.columns[n]) + ' ; ' + 'Example measurement: ' + str(df2.iloc[1,n]) + ' ; ' +'ratio measured: ' + str(ratio_meas_ref[n]) + "\n")
    file1.close()

#adds some useful conversions to the dataframe
def usefulconv(df):
    start=time.time()
    dec_rad=df['dec']*(np.pi/180)
    df['dec (rad)']=dec_rad
    ra_rad=df['ra']*(np.pi/180)
    df['ra (rad)']=ra_rad
    l_rad=df['l']*(np.pi/180)
    df['l (rad)']=l_rad
    b_rad=df['b']*(np.pi/180)
    df['b (rad)']=b_rad
    parallax_as=df['parallax']/1000
    df['p_as']=parallax_as #parallax in arcseconds
    parallax_rad=df['parallax']*4.8481368*10**(-9) #mas to radians
    df['p (rad)']=parallax_rad
    distance=4.84814*10**(-6)*1/(np.tan(df['p (rad)']))
    df['dist_pc']=distance #parsecs
    df['dist_kpc']=distance/1000
    end=time.time()-start
    return(end)

#for a given measurement it returns sigma or the standard deviation
#input df and measurement name
def msig(df,measn):
    mean=sum(df[measn])/len(df)
    sig=np.sqrt(sum((df[measn]-mean)**2)/(len(df)-1))
    return(sig)

'''#adds useful quantities to the dataframe
#vme means (var-mean)/error
def usefulquant(df2):
    start=time.time()
    mean=[]
    for n in range(len(df2.columns)):
        notnan=[]
        for i in range(len(df2)):
            if np.isnan(df2.iloc[i,n])==False:
                notnan.append(df2.iloc[i,n])
        mean.append(sum(notnan)/len(notnan))
    cols=list(df2.columns)
    ra_vme=(df2.iloc[:,cols.index('ra')]-mean[cols.index('ra')])/df2.iloc[:,cols.index('ra_error')]
    df2['ra_vme']=ra_vme
    dec_vme=(df2.iloc[:,cols.index('dec')]-mean[cols.index('dec')])/df2.iloc[:,cols.index('dec_error')]
    df2['dec_vme']=dec_vme
    pmra_vme=(df2.iloc[:,cols.index('pmra')]-mean[cols.index('pmra')])/df2.iloc[:,cols.index('pmra_error')]
    df2['pmra_vme']=pmra_vme
    pmdec_vme=(df2.iloc[:,cols.index('pmdec')]-mean[cols.index('pmdec')])/df2.iloc[:,cols.index('pmdec_error')]
    df2['pmdec_vme']=pmdec_vme
    phot_g_mean_flux_vme=(df2.iloc[:,cols.index('phot_g_mean_flux')]-mean[cols.index('phot_g_mean_flux')])/df2.iloc[:,cols.index('phot_g_mean_flux_error')]
    df2['phot_g_mean_flux_vme']=phot_g_mean_flux_vme
    phot_bp_mean_flux_vme=(df2.iloc[:,cols.index('phot_bp_mean_flux')]-mean[cols.index('phot_bp_mean_flux')])/df2.iloc[:,cols.index('phot_bp_mean_flux_error')]
    df2['phot_bp_mean_flux_vme']=phot_bp_mean_flux_vme
    phot_rp_mean_flux_vme=(df2.iloc[:,cols.index('phot_rp_mean_flux')]-mean[cols.index('phot_rp_mean_flux')])/df2.iloc[:,cols.index('phot_rp_mean_flux_error')]
    df2['phot_rp_mean_flux_vme']=phot_rp_mean_flux_vme
    pmra_over_error=df2['pmra']/df2['pmra_error']
    df2['pmra_over_error']=pmra_over_error
    pmdec_over_error=df2['pmdec']/df2['pmdec_error']
    df2['pmdec_over_error']=pmdec_over_error
    end=time.time()-start
    print('adding useful quantities to df2 took ',end,' seconds')'''

#adds useful quantities to the dataframe
#vme means (var-mean)/sigma
def usefulquant(df2):
    start=time.time()
    mean=[]
    for n in range(len(df2.columns)):
        notnan=[]
        for i in range(len(df2)):
            if np.isnan(df2.iloc[i,n])==False:
                notnan.append(df2.iloc[i,n])
        mean.append(sum(notnan)/len(notnan))
    cols=list(df2.columns)
    #ra_sig=histwidth(df2,'ra')
    #ra_vme=(df2.iloc[:,cols.index('ra')]-mean[cols.index('ra')])/ra_sig
    #df2['ra_vme']=ra_vme
    #print('ra_vme shape ',len(ra_vme),' should be same as ', len(df2))
    
    #dec_sig=histwidth(df2,'dec')
    #dec_vme=(df2.iloc[:,cols.index('dec')]-mean[cols.index('dec')])/dec_sig
    #df2['dec_vme']=dec_vme
    pm_sig=histwidth(df2,'pm')
    pm_vme=(df2.iloc[:,cols.index('pm')]-mean[cols.index('pm')])/pm_sig
    df2['pm_vme']=pm_vme
    pmra_sig=histwidth(df2,'pmra')
    pmra_vme=(df2.iloc[:,cols.index('pmra')]-mean[cols.index('pmra')])/pmra_sig
    df2['pmra_vme']=pmra_vme
    pmdec_sig=histwidth(df2,'pmdec')
    pmdec_vme=(df2.iloc[:,cols.index('pmdec')]-mean[cols.index('pmdec')])/pmdec_sig
    df2['pmdec_vme']=pmdec_vme
    photg_sig=histwidth(df2,'phot_g_mean_flux')
    phot_g_mean_flux_vme=(df2.iloc[:,cols.index('phot_g_mean_flux')]-mean[cols.index('phot_g_mean_flux')])/photg_sig
    df2['phot_g_mean_flux_vme']=phot_g_mean_flux_vme
    photbp_sig=histwidth(df2,'phot_bp_mean_flux')
    phot_bp_mean_flux_vme=(df2.iloc[:,cols.index('phot_bp_mean_flux')]-mean[cols.index('phot_bp_mean_flux')])/photbp_sig
    df2['phot_bp_mean_flux_vme']=phot_bp_mean_flux_vme
    photrp_sig=histwidth(df2,'phot_rp_mean_flux')
    phot_rp_mean_flux_vme=(df2.iloc[:,cols.index('phot_rp_mean_flux')]-mean[cols.index('phot_rp_mean_flux')])/photrp_sig
    df2['phot_rp_mean_flux_vme']=phot_rp_mean_flux_vme
    pmra_over_error=df2['pmra']/df2['pmra_error']
    df2['pmra_over_error']=pmra_over_error
    pmdec_over_error=df2['pmdec']/df2['pmdec_error']
    df2['pmdec_over_error']=pmdec_over_error
    bp_rp_sig=histwidth(df2,'bp_rp')
    bp_rp_vme=(df2.iloc[:,cols.index('bp_rp')]-mean[cols.index('bp_rp')])/bp_rp_sig
    df2['bp_rp_vme']=bp_rp_vme
    bp_g_sig=histwidth(df2,'bp_g')
    bp_g_vme=(df2.iloc[:,cols.index('bp_g')]-mean[cols.index('bp_g')])/bp_g_sig
    df2['bp_g_vme']=bp_g_vme
    g_rp_sig=histwidth(df2,'g_rp')
    g_rp_vme=(df2.iloc[:,cols.index('g_rp')]-mean[cols.index('g_rp')])/g_rp_sig
    df2['g_rp_vme']=g_rp_vme
    
    
    end=time.time()-start
    print('adding useful quantities to df2 took ',end,' seconds')


#takes vec, dim, numneigh, as input and spits out the index corresponding
    #to the most similar region
def maxsim(vec,dim,numneigh):
    index3=faiss.IndexFlatL2(dim)
    index3.is_trained
    index3.add(vec)
    D4,I4 = index3.search(vec,numneigh)
    norms = []
    for i in range(len(D4)):
        norms.append(np.linalg.norm(D4[i]))
    normIndMin = np.argmin(norms)
    return(normIndMin)


#finds N clusters in the dataset and returns x, y, and z component of each star
#in the N clusters, each cluster has k stars
def Nclusters(N,D4,I4,rsh):
    clusters=[]
    D4=list(D4)
    I4=list(I4)
    for n in range(N):
        norms = []
        for i in range(len(D4)):
            norms.append(np.linalg.norm(D4[i]))
        normIndMin = np.argmin(norms)
        minNeighInd = I4[normIndMin]
        clustertemp=[]
        for j in range(len(minNeighInd)):
            clustertemp.append(rsh[minNeighInd[j]])
        clusters.append(clustertemp)
        #indices of I4 that share neighbors with minimum entry
        indMin=[]
        for i in range(len(I4)):
            I4sets = set(I4[i])
            if len(I4sets & set(I4[normIndMin]))!=0:
                indMin.append(i)
        #we want to take these entries out of I4 and D4 to find unique clusters
        print(len(D4))
        indMin.reverse()
        for i in range(len(indMin)):
            #D4=np.delete(D4,indMin[i])
            #I4=np.delete(D4,indMin[i])
            D4.pop(indMin[i])
            I4.pop(indMin[i])
        print('D4 length ',len(D4), ' stars')
    #lets store x, y, and z positions of each star for each cluster
    rshclustsx=[]; rshclustsy=[]; rshclustsz=[]
    for n in range(len(clusters)):
        clustertemp=clusters[n]
        for i in range(len(clustertemp)):
            clustertempstar=clustertemp[i]
            rshclustsx.append(clustertempstar[0])
            rshclustsy.append(clustertempstar[1])
            rshclustsz.append(clustertempstar[2])
    return(rshclustsx,rshclustsy,rshclustsz)

#generated points of a circle to give perspective on size of galaxy
def ptscirc(galr,delx,rhc):
    ptsx=[]; ptsy=[]; ptsz=[]
    N=int(2*galr/delx)
    for n in range(N+1):
        delx=.01
        x=-galr+delx*n
        rpos=np.array([[x],[np.sqrt(galr**2-x**2)],[0]])
        rneg=np.array([[x],[-np.sqrt(galr**2 - x**2)],[0]])
        scrpos=-rhc+rpos
        scrneg=-rhc+rneg
        ptsx.append(scrpos[0][0])
        ptsx.append(scrneg[0][0])
        ptsy.append(scrpos[1][0])
        ptsy.append(scrneg[1][0])
        ptsz.append(0)
        ptsz.append(0)
    return(ptsx,ptsy,ptsz)

#gives parallax associated with distance in pc
def dist_par(dist):
    a=4.84814*10**(-6)
    b=a*10**(-3)
    parlx=np.arctan(a/dist)/b
    return(parlx)

#gives distance associated with parallax in mas
def par_dist(par):
    a=4.84814*10**(-6)
    b=a*10**(-3)
    dist=a/np.tan(b*par)
    return(dist)

#gives ell/2 for a cube that is inside of the sphere, given r
def r_ell(r):
    ellov2=(1/np.sqrt(3))*r
    return(ellov2)

def ell_r(ellov2):
    r=(np.sqrt(3))*ellov2
    return(r)
#counts how many measurements there are for a given feature
#that is, how many are not NaN
def count(df,measurement_name):
    counts=0
    cols=list(df.columns)
    ind=cols.index(measurement_name)
    for n in range(len(df)):
        if np.isnan(df.iloc[n,ind])==False:
            counts+=1
    return(counts)
        

def plothist(df,measurement_name):
    meas=list(df[measurement_name])
    xmeas=[]
    for n in range(len(meas)):
        xmeas.append(n)
    delxbin=(max(meas)-min(meas))/20
    measbin=mf.bootstrap(min(meas)-abs(min(meas))/10,max(meas)+abs(max(meas))/10,delxbin,meas,xmeas)
    counts=count(df,measurement_name)
    measbinx=measbin[0]
    measbiny1=measbin[2]
    measbiny=list(np.array(measbin[2])/counts)
    mf.plotStuff(measbinx,measbiny,min(measbinx),max(measbinx)+max(measbinx)/10,min(measbiny)-abs(min(measbiny))/10,max(measbiny)+max(measbiny)/10,measurement_name,'Probability')  
    print(sum(measbiny),counts,sum(measbiny1))

#simpler version using matplotlib histogram
#plots histogram, x~log y~lin
def plothistlog(df,measurement_name,startx,endx,N=50,binwidth=1,xlbl='measurement',ylbl='Number of Stars'):
    maxx=max(df[measurement_name])
    minn=min(df[measurement_name])
    width=int(np.ceil((maxx-minn)/1000))
    #bin_list=list(range(int(np.floor(min(df[measurement_name]))),int(np.ceil(max(df[measurement_name]))),binwidth))
    #number of bins and bin list so that it looks like the bin widths are const.
    #on a logarithmic scale
    #N=20
    bin_list=[]
    for n in range(int(N+1)):
        meas_n=minn*(maxx/minn)**(n/N)
        bin_list.append(meas_n)
    plt.hist(df[measurement_name], bins = bin_list)
    plt.xscale('log')
    plt.xlim(startx,endx)
    plt.xlabel(xlbl,fontsize=20)
    plt.ylabel(ylbl,fontsize=20)
    plt.show()
#gives counts and bins of the histogram
def countsbins(df,measurement_name,N=50):
    maxx=max(df[measurement_name])
    minn=min(df[measurement_name])
    binwidth=(maxx-minn)/N
    bin_list=[]
    for n in range(int(N+1)):
        meas_n=minn+n*binwidth
        bin_list.append(meas_n)
    counts, bins, bars=plt.hist(df[measurement_name],bins=bin_list)
    return(counts,bins,bars)

#returns the half max width of the histogram
def histwidth(df,measurement_name):
    counts=countsbins(df,measurement_name)[0]
    bins=countsbins(df,measurement_name)[1]
    maxx=max(counts)
    #potential bins
    potbins=[]
    for n in range(len(counts)):
        if counts[n]>=maxx/2:
            potbins.append(bins[n])
            potbins.append(bins[n+1])
    width=max(potbins)-min(potbins)
    return(width)

#plots histogram using a linear scale for both x and y
def plothistlin(df,measurement_name,startx,endx,binwidth=1,xlbl='measurement',ylbl='Number of Stars'):
    #bin_list=list(range(int(np.floor(startx)),int(np.ceil(endx)),binwidth))
    maxx=max(df[measurement_name])
    minn=min(df[measurement_name])
    N=(maxx-minn)/binwidth
    bin_list=[]
    for n in range(int(N+1)):
        meas_n=minn+n*binwidth
        bin_list.append(meas_n)
    plt.hist(df[measurement_name], bins = bin_list)
    plt.xlim(startx,endx)
    plt.xlabel(xlbl,fontsize=20)
    plt.ylabel(ylbl,fontsize=20)
    plt.show()

#calculates rsh (position of star relative to sun) in galactic and equatorial given a dataframe
def calcpos(df):
    t=symbols('t')
    b=Function('b')(t)
    l=Function('l')(t)
    d=Function('d')(t)
    dec=Function('dec')(t)
    ra=Function('ra')(t)

    rshvar=Matrix([d*cos(b)*cos(l),d*cos(b)*sin(l),d*sin(b)])
    rshargs=[b,l,d]
    rsh1=lambdify(rshargs,rshvar)
    rsh2=rsh1(df['b (rad)'],df['l (rad)'],df['dist_kpc'])
    rsh=rsh2[:,0,:]
    #in equatorial
    rsheqvar=Matrix([d*cos(dec)*cos(ra),d*cos(dec)*sin(ra),d*sin(dec)])
    rsheqargs=[dec,ra,d]
    rsheq1=lambdify(rsheqargs,rsheqvar)
    rsheq2=rsh1(df['dec (rad)'],df['ra (rad)'],df['dist_kpc'])
    rsheq=rsheq2[:,0,:]
    rhc=np.array([[-8.34],[0],[.025]])
    return(rsh,rsheq)
    
#returns a plot of avg dist between neighbors vs number of neighbors
#vec is the list of vectors we are using, dim is the dimension of each vector
#itera is the max number of iterations, we iterate up to this point
#itera*inc is maximum number of neighbors
#inc is the increment of each iteration
def dstvsneigh(vec,dim,itera,inc):
    #number of neighbors, list
    numnl=[]
    #average distance tot, list
    avdistl=[]
    numneigh=0
    for i in range(itera):
        numneigh+=inc
        numnl.append(numneigh)
        index3=faiss.IndexFlatL2(dim)
        index3.is_trained
        index3.add(vec)
        D4,I4 = index3.search(vec,numneigh)
        avgdist=[]
        for n in range(len(D4)):
            avgdist.append(sum(D4[n])/len(D4[n]))
        avgdistTot=sum(avgdist)/len(avgdist)
        avdistl.append(avgdistTot)
    plott=mf.plotStuff(numnl,avdistl,min(numnl),max(numnl),min(avdistl),max(avdistl))
    return(plott)

#similar to dstvsneigh except whereas dstvsneigh uses all vectors of D4
#this one uses only one, corresponding to the region of max similarity
#could probably make this faster
def dstvsneigh1(vec,dim,itera,inc):
    if isinstance(vec,pd.DataFrame):
        vectemp=[]
        for n in range(len(vec)):
            vectemp.append(vec.iloc[n].values)
        vec=np.array(vectemp).astype('float32',order='C')
            
    else:
        pass
    print(np.shape(vec))
    maxneigh=inc*itera
    minnind=maxsim(vec,dim,maxneigh)
    avdist=[]
    #number of neighbors as a list
    numnl=[]
    numneigh=0
    for n in range(itera):
        numneigh+=inc
        numnl.append(numneigh)
        index3=faiss.IndexFlatL2(dim)
        index3.is_trained
        index3.add(vec)
        D4,I4 = index3.search(vec,numneigh)
        #avdistemp=np.linalg.norm(np.sqrt(D4[minnind]))
        #excludes first entry in D4[minnind] since its zero
        avdistemp=sum(np.sqrt(D4[minnind][1:]))/(len(D4[minnind])-1)
        avdist.append(avdistemp)
    plott=mf.plotStuff(numnl,avdist,min(numnl),max(numnl),min(avdist),max(avdist),xlbl='number of neighbors',ylbl='average distance (using vme)')
    return(plott,numnl,avdist)

#plots derivative of previous function
def dstvsneigh1deriv(numl,avdist):
    derivative=[]
    N=len(numl)
    for n in range(N-1):
        derivative.append((avdist[n+1]-avdist[n])/(numl[n+1]-numl[n]))
    numl=numl[0:N-1]
    print(len(numl),' ',len(derivative),' these should be the same')
    plott=mf.plotStuff(numl,derivative,min(numl),max(numl),min(derivative),max(derivative),xlbl='number of neighbors',ylbl='derivative of average distance')
    return(plott,numl,derivative)

def n_to_the_N(numl,avdist,N):
    nN=[]
    for n in range(len(numl)):
        nN.append(numl[n]**(N))
    avdist=np.array(avdist)
    nN=np.array(nN)
    y=avdist/nN
    plott=mf.plotStuff(numl,y,min(numl),max(numl),min(y),max(y),xlbl='number of neighbors', ylbl='(avg dist)/(num neigh)^(1/2) using vme')
    return(plott)

#ff.n_to_the_N(fw.numl,fw.avdist,1/
        


#faiss nearest neighbor
'''def faissnn(vec,dim,neigh)
    index3=faiss.IndexFlatL2(dim)
    #print('vec ',index3.is_trained)
    index3.add(vec)
    D4,I4 = index3.search(vec,neigh)
    return(D4,I4)'''


'''#brainstorming
maxnumneigh=itera*inc
numneigh=0
inc=10
for n in range(itera):
    numneigh+=inc
    
maxind=maxsim(vec,dim,maxnumneigh)
maxsimvec=vec[maxind]
D4,I4=faissnn(maxsimvec,dim,numneigh)
neighbors_added_indices=I4[-inc:]
neighbors_vec=vec[neighbors_added_indices]
D5,I5=faissnn(vec,dim,numneigh)
#nearest neighbor distances corresponding to the neighbors that were added
added_nearest_neighbor_distances=D5[neighbors_added_indices,1]
#needs more work and to be looked over thoroughly


D4[maxind]'''


def IndexFlatL2(vec,dim,numneigh):
    k=numneigh
    dimension=dim
    v=vec
    index2=faiss.IndexFlatL2(dimension)
    index2.is_trained
    index2.train(v)
    print(index2.is_trained)
    index2.add(v)
    D,I=index2.search(v,k)
    return(D,I)


def count_negative_1(I):
    count=0
    for n in range(len(I)):
        for j in range(len(I[0])):
            if I[n,j]==-1:
                count+=1
                break
    if count==len(I):
        #print('each entry contains a cluster')
        return(True)
    if count!=len(I):
        #print('Warning: there is not a negative one in each entry of I')
        #print('this means each entry does not contain a cluster')
        return(False)


#takes a list and removes all duplicates from it
def remove_value(I_list,value):
    nlist=[]
    ilist=[]
    I_list_copy=I_list.tolist().copy()
    for n in range(len(I_list)):
        for i in range(len(I_list[n])):
            if I_list[n,i]==value:
                I_list_copy[n].remove(value)
                nlist.append(n)
                ilist.append(i)
    return(I_list_copy)

#converts each entry in a list of lists into a set, and only keeps the unique ones, modifies corresponding ones in D
#nlist is the number of clusters, this is needed because in the end, len(I)=number_clusters
def remove_duplicates(I,D,nlist):
    Itemp=[]
    Dtemp=[]
    Itemp.append(I[0])
    Dtemp.append(D[0])
    for n in range(len(I)):
        #print(n)
        #print(len(Itemp))
        total_intersection=[]
        for i in range(len(Itemp)):
            intersection=len(set(Itemp[i]) & set(I[n]))
            total_intersection.append(intersection)
        if sum(total_intersection)==0:
            Itemp.append(I[n])
            Dtemp.append(D[n])
        if len(Itemp)==nlist:
            break
    return(Dtemp,Itemp)

def IndexIVFFlat(vec,dimension,number_of_clusters,k=40):
    nlist=number_of_clusters 
    v=vec                            
    quantiser = faiss.IndexFlatL2(dimension)  
    index = faiss.IndexIVFFlat(quantiser, dimension, nlist,   faiss.METRIC_L2)
    index.train(v)  # train on the database vectors
    index.add(v)   # add the vectors and update the index
    nprobe = 1  # find 2 most similar clusters
    n_query = 4  
    np.random.seed(0)   
    D, I = index.search(v, k)
    return(D,I)

#finds the number of neighbors required for I to contain -1 in every one of its lists
def find_k(vec,dimension,number_clusters):
    vec_length=len(vec)
    kinit=int(np.floor(vec_length/number_clusters))*2
    kinit1=int(np.floor(vec_length/number_clusters))
    print('start')
    for n in range(vec_length):
        print(n)
        kinit+=kinit1
        D,I=IndexIVFFlat(vec,dimension,number_clusters,kinit)
        stop=count_negative_1(I)
        if stop==True:
            break
    print('stop')
    kfinal=kinit
    return(kfinal)



def av_dist_num_clusters(vectorlist,dimension,number_clusters_max):
    maxnum=number_clusters_max
    v=vectorlist
    #maxnum=30
    #dimension=3
    avdist=[]; nlist=[]
    for n in range(1,maxnum):
        print('number of clusters: ',n)
        nlist.append(n)
        kneigh=find_k(v,dimension,n)
        D,I=IndexIVFFlat(v,dimension,n,kneigh)
        #count_negative_1(I)
        I=remove_value(I,-1)
        D=remove_value(D,3.4028234663852886e+38)
        D,I=remove_duplicates(I,D,nlist)    #removes duplicate lists in I, and corresponding lists in D
        Dflat=[]
        for n in range(len(D)):
            for i in range(len(D[n])):
                if D[n][i]!=0:
                    Dflat.append(D[n][i])
        avdistemptot=sum(Dflat)/len(Dflat)
        '''for n in range(len(D)):
            avdistemp=[]
            avdistemp.append(sum(D[n])/(len(D[n])-1))
        avdistemptot=sum(avdistemp)/len(avdistemp)'''
        avdist.append(avdistemptot)
    return(avdist,nlist)
        

def return_n_clusters(vector_list,dimension,number_of_clusters):
    v=vector_list
    nlist=number_of_clusters
    kneigh=find_k(v,dimension,nlist)
    D,I=IndexIVFFlat(v,dimension,nlist,kneigh)
    I=remove_value(I,-1)
    D,I=remove_duplicates(I,D,nlist)
    new_vector_list=[]
    for n in range(len(I)):
        new_vector_list.append(v[I[n]])
    return(new_vector_list)

#numneigh needs to be chosen so that every vector is sorted, non are left out
def faiss_return_n_clusters(vector_list,number_of_clusters,numneigh,niter=20,verbose=True):
    start=time.time()
    v=vector_list
    ncentroids = number_of_clusters
    dimension = v.shape[1]
    kmeans = faiss.Kmeans(dimension, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(v)
    quantiser = faiss.IndexFlatL2(dimension)  
    index1 = faiss.IndexIVFFlat(quantiser, dimension, ncentroids,   faiss.METRIC_L2)
    index1.train(v)  # train on the database vectors
    index1.add(v)
    centers=kmeans.centroids
    D,I=index1.search(kmeans.centroids,numneigh)
    I=remove_value(I,-1)
    number_of_stars=0
    for n in range(len(I)):
        number_of_stars+=len(I[n])
    vclusts=[]
    for n in range(len(I)):
        vtemp=v[I[n]]
        vclusts.append(vtemp)
    end=time.time()-start
    print('Created n clusters using faiss ',end,' seconds')
    #number of stars should equal dataset length, otherwise increase numneigh till it is
    return(vclusts,centers,number_of_stars)

#returns the centers of n clusters
def return_n_centers(new_vector_list,dimension):
    average_positions=[]
    for n in range(len(new_vector_list)):
        average_position=sum(new_vector_list[n])/len(new_vector_list[n])
        average_positions.append(average_position)
    return(average_positions)

def return_cluster_size(new_vector_list,average_positions):     #returns size of n clusters
    #distances measured from center of cluster
    xilist=[]
    for n in range(len(new_vector_list)):
        xis=[]
        for i in range(len(new_vector_list[n])):
            xi=np.linalg.norm(new_vector_list[n][i]-average_positions[n])
            xis.append(xi)
        xilist.append(xis)
    xirms=[]
    for n in range(len(xilist)):
        rms=np.sqrt(sum(xilist[n])/len(xilist[n]))
        xirms.append(rms)
    return(xirms)

def av_sij_num_clusters(vectorlist,dimension,number_clusters_max):
    maxnum=number_clusters_max
    v=vectorlist
    #maxnum=30
    #dimension=3
    av_Sij=[]; nlist=[]; Vitot=[]; Dijtot=[]; sumsijtot=[]; Sijtot=[]; nanstot=[]; Sijdtot=[]
    for n in range(12,maxnum):
        print('number of clusters: ',n)
        nlist.append(n)
        '''kneigh=find_k(v,dimension,n)
        D,I=IndexIVFFlat(v,dimension,n,kneigh)
        I=remove_value(I,-1)
        D=remove_value(D,3.4028234663852886e+38)
        D,I=remove_duplicates(I,D,nlist)'''    #removes duplicate lists in I, and corresponding lists in D
        kneigh=find_k(v,dimension,n)
        vec_new,centers,numstrs=faiss_return_n_clusters(v,n,kneigh)
        #centers=return_n_centers(vec_new,len(v[0]))
        Dij=[]
        for i in range(len(centers)):
            Dijrow=[]
            for j in range(len(centers)):
                    Dijtemp=np.linalg.norm(centers[i]-centers[j])
                    Dijrow.append(Dijtemp)
            Dij.append(Dijrow)
        Dijtot.append(Dij)

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
        Vitot.append(Vi)

        Sij=[]; nans=[]
        for i in range(len(Vi)):
            Sijrow=[]
            for j in range(i+1,len(Vi)):
                Sijtemp=Dij[i][j]/np.sqrt(Vi[i]+Vi[j])
                #Sijrow.append(Sijtemp)
                Sij.append(Dij[i][j]/np.sqrt(Vi[i]+Vi[j]))
                if np.isnan(Sijtemp):
                    nans.append([i,j])
            #Sij.append(Sijrow)
        Sijd=Sij
        Sij=np.array(Sij).flatten()
        Sijtot.append(Sij)
        av_Sij.append((sum(Sij)/(len(Vi)**2-len(Vi)))/2)
        sumsijtot.append(sum(Sij))
        Sijdtot.append(Sijd)
        nanstot.append(nans)
    return(av_Sij,nlist,Vitot,Dijtot,sumsijtot,Sijtot,nanstot,Sijdtot)
    





    
    
    
def av_sij_num_clusters(vectorlist,dimension,number_clusters_max):
    maxnum=number_clusters_max
    v=vectorlist
    #maxnum=30
    #dimension=3
    av_Sij=[]; nlist=[]; Vitot=[]; Dijtot=[]; sumsijtot=[]; Sijtot=[]; nanstot=[]; Sijdtot=[]
    for n in range(12,maxnum):
        print('number of clusters: ',n)
        nlist.append(n)
        '''kneigh=find_k(v,dimension,n)
        D,I=IndexIVFFlat(v,dimension,n,kneigh)
        I=remove_value(I,-1)
        D=remove_value(D,3.4028234663852886e+38)
        D,I=remove_duplicates(I,D,nlist)'''    #removes duplicate lists in I, and corresponding lists in D
        kneigh=find_k(v,dimension,n)
        vec_new,centers,numstrs=faiss_return_n_clusters(v,n,kneigh)
        #centers=return_n_centers(vec_new,len(v[0]))
        Dij=[]
        for i in range(len(centers)):
            Dijrow=[]
            for j in range(len(centers)):
                    Dijtemp=np.linalg.norm(centers[i]-centers[j])
                    Dijrow.append(Dijtemp)
            Dij.append(Dijrow)
        Dijtot.append(Dij)

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
        Vitot.append(Vi)

        Sij=[]; nans=[]
        for i in range(len(Vi)-1):
            Sijrow=[]
            for j in range(i+1,len(Vi)):
                Sijtemp=Dij[i][j]/np.sqrt(Vi[i]+Vi[j])
                #Sijrow.append(Sijtemp)
                Sij.append(Dij[i][j]/np.sqrt(Vi[i]+Vi[j]))
                if np.isnan(Sijtemp):
                    nans.append([i,j])
            #Sij.append(Sijrow)
        Sijd=Sij
        Sij=np.array(Sij).flatten()
        Sijtot.append(Sij)
        av_Sij.append((sum(Sij)/((len(Vi)**2-len(Vi))/2))/2)
        sumsijtot.append(sum(Sij))
        Sijdtot.append(Sijd)
        nanstot.append(nans)
    return(av_Sij,nlist,Vitot,Dijtot,sumsijtot,Sijtot,nanstot,Sijdtot)