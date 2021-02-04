import matplotlib.pyplot as plt
import numpy as np
import sys, os
import matplotlib.ticker as mticker
import matplotlib 

def bootstrap(start1,end1,delx1,Rp,vRp):
    N1=(end1-start1)/delx1
    bintot1=[]
    for n in range(0,int(np.ceil(N1))):
        rcbmagforbintot=[]
        for i in range(len(Rp)):
            if start1+(n)*delx1<Rp[i]<=start1+(n+1)*delx1:
                rcbmagforbintot.append(Rp[i])
        bintot1.append(len(rcbmagforbintot))
    #each element in vbin_all is an array representing the bin along with
    #the data in that bin
    vbin_all1=[]
    vuncbin_all1=[]
    rbin_all1=[]
    runcbin_all1=[]
    vavg1=[]
    rmagavg1=[]
    vcsuncmagavg1=[]
    runcmagavg1=[]
    numbin=[]; #number of points in each bin
    count=0
    for n in range(0,int(np.ceil(N1))):
        vbin1=[]
        rbin1=[]
        #vuncbin1=[]
        #runcbin1=[]
        for i in range(len(Rp)):
            if start1+(n)*delx1<Rp[i]<=start1+(n+1)*delx1 and bintot1[n]!=0:
                vbin1.append(vRp[i])
                rbin1.append(Rp[i])
                count+=1
                #vuncbin.append(vcsuncmagref22[i])
                #runcbin.append(runcmagref22[i])         
        if len(vbin1)!=0:
            vavg1.append(sum(vbin1[0:len(vbin1)])/bintot1[n])
            rmagavg1.append(sum(rbin1[0:len(rbin1)])/bintot1[n])
            #vcsuncmagavg22.append(sum(vuncbin[0:len(rbin)])/bintot[n])
            #runcmagavg22.append(sum(runcbin[0:len(rbin)])/bintot[n])
        vbin_all1.append(vbin1)
        rbin_all1.append(rbin1)
        if len(vbin1)!=0:
            numbin.append(len(vbin1))
        #vuncbin_all.append(vuncbin)
        #runcbin_all.append(runcbin)
    rbintot=[]; vbintot=[]
    for n in range(len(vbin_all1)):
        if len(vbin_all1[n])!=0 and len(rbin_all1[n])!=0:
            rbintot.append(rbin_all1[n])
            vbintot.append(vbin_all1[n])    
    numbin2=[]
    for l in range(len(numbin)):
        numbin2.append(numbin[l])
    return(rmagavg1,vavg1,numbin,numbin2,vbin_all1,rbin_all1,rbintot,vbintot)

def plotStuff(r,v,startx,endx,starty,endy,xlbl='x',ylbl='y'):
    binnedvvsrplot2=plt.plot(r,v,marker='o',linewidth=0,markersize=2)
    plt.xlim(startx,endx)
    plt.ylim(starty,endy)
    plt.xlabel(xlbl,fontsize=20)
    plt.ylabel(ylbl,fontsize=20)
    plt.show()

    
def getCircular(rcb,vcb,that, startx,endx,starty,endy,delx,smallv):
    rcbref=[]; vcbref=[]
    for n in range(len(vcb)):
        if abs(abs(np.linalg.norm(vcb[n]))-abs(np.dot(that[n],vcb[n])))/np.linalg.norm(vcb[n])<smallv:
            rcbref.append(rcb[n])
            vcbref.append(vcb[n])
    boot=bootstrap(startx,endx,delx,rcbref,vcbref)
    ravg=boot[0]; vavg=boot[1]
    return plotStuff(ravg,vavg,startx,endx,starty,endy)

def plotStuff2(r1,v1,r2,v2,startx,endx,starty,endy,vstd):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(r1,v1,marker='o',linewidth=0,markersize=3)
    ax.errorbar(r1,v1,vstd,fmt='none',elinewidth=2)
    ax.set_xlim(left=startx, right=endx)
    ax.set_xlabel("astrometric excess noise sig",fontsize=20)
    ax.set_ylim(bottom=starty, top=endy)
    ax.set_ylabel("number of stars",fontsize=20)
    ax=plt.twinx(ax=None)
    ax.set_yscale('log')
    ax.set_ylabel("Number of Stars",fontsize=20)
    ax.plot(r2,v2,marker='o',linewidth=0,markersize=3,color='r')
    plt.show()

    #ax.errorbar(r1,v1,vstd,fmt='none',elinewidth=.5)
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def sort(rmag,vmag):
    rmagc=rmag.copy(); vmagc=vmag.copy()
    rasc=[]; vasc=[]
    for n in range(len(rmag)):
        if n%2500==0:
            print("number sorted ",n)
        rasc.append(min(rmagc))
        rmind=np.argmin(rmagc)
        rmagc.remove(rmagc[rmind])
        vasc.append(vmagc[rmind])
        vmagc.remove(vmagc[rmind])
    return(rasc,vasc)


def bineq(rmag,vmag,Nstars):
    rbintot=[]; vbintot=[]; numbin=[]
    print("sorting")
    sort1=sort(rmag,vmag)
    print("done sorting")
    rasc=sort1[0]; vasc=sort1[1]
    rbintemp=[rasc[0]]; vbintemp=[vasc[0]]
    nd=[]; numbin2=[]
    for i in range(1,len(rasc)):
        if i%50000==0:
            print(i)
        n=0
        if len(rbintemp)<=Nstars and max(rbintemp)-min(rbintemp)<=1:
            rbintemp.append(rasc[i])
            vbintemp.append(vasc[i])
            n+=1
        elif len(rbintemp)<=Nstars and max(rbintemp)-min(rbintemp)>=1:
            rbintemp.append(rasc[i])
            vbintemp.append(vasc[i])
            n+=1
        elif len(rbintemp)>=Nstars and max(rbintemp)-min(rbintemp)<=1:
            rbintemp.append(rasc[i])
            vbintemp.append(vasc[i])
            n+=1
        elif len(rbintemp)>=Nstars and max(rbintemp)-min(rbintemp)>=1:
            rbintot.append(rbintemp)
            vbintot.append(vbintemp)
            numbin.append(len(rbintemp))
            numbin2.append(len(vbintemp))
            n+=1
            rbintemp=[rasc[i]]; vbintemp=[vasc[i]]
        nd.append(n)
    numbin.append(len(rbintemp))
    numbin22=[]
    for n in range(len(numbin)):
        numbin22.append((numbin[n]/max(numbin))*500)
        
    rbintot.append(rbintemp)
    vbintot.append(vbintemp)
    ravg=[]; vavg=[]
    for n in range(len(rbintot)):
        ravg.append(sum(rbintot[n])/len(rbintot[n]))
        vavg.append(sum(vbintot[n])/len(vbintot[n]))
    return(ravg,vavg,numbin,nd,numbin2,numbin22)
            
#equal size bins
def bineq2(rmag,vmag,Nstars):
    rbintot=[]; vbintot=[]; numbin=[]
    print("sorting")
    sort1=sort(rmag,vmag)
    print("done sorting")
    rasc=sort1[0]; vasc=sort1[1]
    rbintemp=[]; vbintemp=[]
    for n in range(len(rasc)):
        if len(rbintemp)<Nstars:
            rbintemp.append(rasc[n])
            vbintemp.append(vasc[n])
        elif len(rbintemp)==Nstars:
            rbintot.append(rbintemp)
            vbintot.append(vbintemp)
            numbin.append(len(rbintemp))
            rbintemp=[]; vbintemp=[]
        else:
            pass
    numbin2=[]
    for n in range(len(numbin)):
        numbin2.append((numbin[n]/max(numbin))*500)

    ravg=[]; vavg=[]
    for n in range(len(rbintot)):
        ravg.append(sum(rbintot[n])/len(rbintot[n]))
        vavg.append(sum(vbintot[n])/len(vbintot[n]))
    return(ravg,vavg,numbin,numbin2)

#actual bootstrap method for errors
import random
#rbintot is the array of arrays where each entry is a bin
#N is the number of resamplings
def booterr(rbintot,vbintot,N):
    vstd=[]
    for n in range(len(rbintot)):
        ravg=[]; vavg=[]
        for j in range(N):
            rsamp=[]; vsamp=[]
            for i in range(len(rbintot[n])):
                rsamp.append(random.choice(rbintot[n]))
                vsamp.append(random.choice(vbintot[n]))
            ravg.append(sum(rsamp)/len(rsamp))
            vavg.append(sum(vsamp)/len(vsamp))
        ravgtot=sum(ravg)/len(ravg)
        vavgtot=sum(vavg)/len(vavg)
        vstdin=[]
        for l in range(len(vavg)):
            vstdin.append((vavg[l]-vavgtot)**2)
        vstd.append(np.sqrt(sum(vstdin)/N))
    return(vstd)

def d3bin(startx,endx,starty,endy,delx,x1,y1,vmag):
    Nx=(endx-startx)/delx
    Ny=(endy-starty)/delx
    x1avg=[]; y1avg=[]; vmagavg=[]
    x1bin=[]; y1bin=[]; vmagbin=[]
    xi=startx-delx
    yi=starty-delx
    for i in range(int(np.floor(Nx))+1):
        xi+=delx
        for j in range(int(np.floor(Ny))+1):
            yi+=delx
            x1bintemp=[]; y1bintemp=[]; vmagbintemp=[];
            for n in range(len(x1)):
                if xi<x1[n]<=xi+delx and yi<y1[n]<=yi+delx:
                    x1bintemp.append(x1[n])
                    y1bintemp.append(y1[n])
                    vmagbintemp.append(vmag[n])
            x1bin.append(x1bintemp)
            y1bin.append(y1bintemp)
            vmagbin.append(vmagbintemp)

    for i in range(len(x1bin)):
        if len(x1bin[i])!=0:
            x1avg.append(sum(x1bin[i])/len(x1bin[i]))
            y1avg.append(sum(y1bin[i])/len(y1bin[i]))
            vmagavg.append(sum(vmagbin[i])/len(vmagbin[i]))
    return(xi,x1avg,y1avg,vmagavg,x1bin,y1bin,vmagbin,Nx,Ny)


        
        
            
                
        




def plot3d(x1avg,y1avg,vmagavg,xlim,ylim,vlim):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x =x1avg.copy()
    y =y1avg.copy()
    z =vmagavg.copy()



    ax.scatter(x, y, z, c='r', marker='o',s=.5)
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-ylim, ylim])
    ax.set_zlim([-vlim,vlim])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def plot3d2(x1,y1,z1,x2,y2,z2,xlim,ylim,vlim):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x =x1.copy()
    y =y1.copy()
    z =z1.copy()

    x2c=x2.copy()
    y2c=y2.copy()
    z2c=z2.copy()

    ax.scatter(x, y, z, c='r', marker='o',s=.5)
    ax.scatter(x2c,y2c,z2c,c='r', marker='o',s=.5)
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-ylim, ylim])
    ax.set_zlim([-vlim,vlim])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def plotStuffp(r1,v1,r2,v2,startx,endx,starty,endy):
	binnedvvsrplot2=plt.plot(r1,v1,marker='o',linewidth=0,markersize=.5)
	binnedvvsrplot3=plt.plot(r2,v2,marker='o',linewidth=0,markersize=.5)
	plt.xlim(startx,endx)
	plt.ylim(starty,endy)
	plt.xlabel("R (kpc)",fontsize=20)
	plt.ylabel("Vc (km/s)",fontsize=20)
	plt.show()

def plotStuffp1(r1,v1,r2,v2,r3,v3,startx,endx,starty,endy):
	binnedvvsrplot2=plt.plot(r1,v1,marker='o',linewidth=0,markersize=.5,color='blue')
	binnedvvsrplot3=plt.plot(r2,v2,marker='o',linewidth=0,markersize=.5,color='green')
	binnedvvsrplot4=plt.plot(r3,v3,marker='o',linewidth=0,markersize=.5,color='red')
	plt.xlim(startx,endx)
	plt.ylim(starty,endy)
	plt.xlabel("R from sun (my method) (kpc)",fontsize=20)
	plt.ylabel("R from sun (other methods) (kpc)",fontsize=20)
	plt.show()

#finds the nth max in an array such as (k,), does not affect input matrix
def nmax(arr,n):
    acop=arr.copy()
    for i in range(n-1):
        ind=np.argmax(acop)
        acop.pop(ind)
    return max(acop)
#returns index of nth max element, does not affect input matrix
def nmaxin(arr,n):
    acop=arr.copy()
    for i in range(n-1):
        ind=np.argmax(acop)
        acop.pop(ind)
    acopmax=max(acop)
    return(arr.index(acopmax))
    

#finds nth min in an array such as (k,), does not affect input matrix
#does not consider elements that are equal
def nmin(arr,n):
    acop=arr.copy()
    for i in range(n-1):
        ind=np.argmin(acop)
        acop.pop(ind)
    return min(acop)
#returns index of nth min element, does not affect input matrix
def nminin(arr,n):
    acop=arr.copy()
    for i in range(n-1):
        ind=np.argmin(acop)
        acop.pop(ind)
    acopmin=min(acop)
    return(arr.index(acopmin))

    
            
    
        
        
        
