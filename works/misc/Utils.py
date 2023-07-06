import sys, os, select
import socket
import math
import numpy as np
import random
import string
import copy
import scipy.spatial.distance as dist
import scipy.cluster.vq as vq
import time
import csv
from matplotlib import pyplot as plt
from numpy import array

hostname = socket.gethostname()

if "ubuntu" in hostname:
    import matplotlib.pyplot as plt

def findparamval(paramfilename,paramstr) :

    with open(paramfilename, 'r') as file:
        
        paramnum = file.read()

    paramnum =  paramnum.split("\n")

    ss = []

    for s in paramnum :

        if 0<=s.find("#") :

            s = s[:s.find("#")]

        ss.append(s)

    for s in ss :

        s = s.split()

        if len(s)==2 and s[0]==paramstr :

            if type(s[1])==str :

                return s[1]

            else :

                return float(s[1])
    

def loadbin(filename,N = None,npat = None,dtype = np.float32) :
    # Loads binary file (with header soon)

    if N==None :

        data = np.fromfile(filename,dtype)

    else :

        if npat==None : 

            data = np.fromfile(filename,dtype)

            if len(data)%N!=0 :
                print(len(data))
                raise (AssertionError("len(data) -- N mismatch"))

            npat = len(data)/N

        else :

            data = np.fromfile(filename,dtype,count = N * npat)
        
        npat = int(npat)

        N = int(N)

        data = data.reshape(npat,N)

    return np.array(data)


def rgb2hsv(rgb):
    
    r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0

    mx = max(r, g, b)
    mn = min(r, g, b)
    
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    
    return list(map(round, (h, s, v)))

def hex2rgb(c):
    if c[0] == '#':
        c = c.lstrip('#')
    lv = len(c)
    return ([int(c[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)])


def savebin(arr,filename) :

    #a = array(arr,type(arr[0,0]))
    a = array(arr,'float32')
    output_file = open(filename,'wb')
    a.tofile(output_file)
    output_file.close()


def loadcsv(filename) :
    datarows = []
    with open(filename,'rb') as csvfile:
        csvreader = csv.reader(csvfile) # ,delimiter=' ',quotechar='#')
        for row in csvreader:
            datarows.append(row)
    return np.array(datarows).astype(np.float32)


def loadandsaveMNISThid() :

#     trdata = loadcsv("train-20-30-10000.csv")
#     tedata = loadcsv("test-20-30-10000.csv")

    trdata = loadcsv("patt.train-classifier.csv")
    tedata = loadcsv("patt.test-classifier.csv")

    np.savetxt("mnist_trhid_60k.txt",trdata)
    np.savetxt("mnist_tehid_10k.txt",tedata)


"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

import struct

def loadMNIST(dataset = "training",what = "lbl",path = ".",savefname = None) :
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.

    But skipping img here ... and iterator as well.

    """

    if dataset=="training":
        if what=="all" or what=="both" :
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        elif what=="img" :
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            fname_lbl = None
        elif what=="lbl" :
            fname_img = None
            fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset=="testing":
        if what=="all" or what=="both" :
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
        elif what=="img" :
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            fname_lbl = None
        elif what=="lbl" :
            fname_img = None
            fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise(ValueError, "dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    flbl = None
    if fname_lbl!=None :
        with open(fname_lbl, 'rb') as flblfp:
            magic, num = struct.unpack(">II", flblfp.read(8))
            lbl = np.fromfile(flblfp, dtype=np.int8).astype(int)

        flbl = np.zeros((len(lbl),10))

        for i,x in zip(lbl,flbl) : x[i] = 1

    fimg = None
    if fname_img!=None :
        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        fimg = img.reshape(img.shape[0],img.shape[1]*img.shape[2]).astype(float)

        fimg = fimg/256.0

    if savefname!=None :
        if dataset=="training" :
            if fname_img!=None : savebin(fimg,savefname)
            if fname_lbl!=None : savebin(flbl,savefname)
        else :
            if fname_img!=None : savebin(fimg,savefname)
            if fname_lbl!=None : savebin(flbl,savefname)

    return fimg,flbl


def loadandsaveMNISTlbl() :

    trxlbl = loadMNIST("training")
    texlbl = loadMNIST("testing")

    np.savetxt("mnist_trlbl.txt",trxlbl)
    np.savetxt("mnist_telbl.txt",texlbl)


def loaddata(csvfile,savefile = None) :

    data = loadcsv(csvfile).astype(int)
    
    xlbl = np.zeros((len(data),10))

    for i,x in zip(data,xlbl) : x[i] = 1

    if savefile!=None : 
        np.savetxt(savefile,data)

    return xlbl


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


def getncore1(ipopnpart,hpopnpart,opopnpart = 1,useih = True,usehi = False,usehhi = False,usehhe = False,
              useho = True,usehno = False) :

    ncore = ipopnpart + hpopnpart + opopnpart
    if useih : ncore += ipopnpart * hpopnpart
    if usehi : ncore += ipopnpart + ipopnpart * hpopnpart
    if usehhi : ncore += hpopnpart * hpopnpart
    if usehhe : ncore += hpopnpart * hpopnpart
    if useho : ncore += hpopnpart * opopnpart
    if usehno : ncore += hpopnpart * opopnpart

    nnode = int(np.ceil(ncore/32.))

    return nnode,ncore


def getncore2(ipopnpart = 1,hpopnpart = 1) :

    nproc = ipopnpart + hpopnpart + 1;
    nproc += ipopnpart * hpopnpart;
    nproc += hpopnpart * 1;

    print("Minimum n:o cores = %d, n:o nodes = %d (maxnproc = %d)" % \
        (nproc,np.ceil(nproc/32.),np.ceil(nproc/32.)*32))


def factors(n):    
    result = set()
    for i in range(1, int(n ** 0.5) + 1):
        div, mod = divmod(n, i)
        if mod == 0:
            result |= {i, div}
    return sorted(result)


def checknproc(ipopnpart = 1,hpopnpart = 1,hhiprjp = True,hheprjp = True,avtrgain = 1) :

    nproc = ipopnpart + hpopnpart + 1
    nproc += ipopnpart * hpopnpart
    if hheprjp : nproc += hpopnpart * hpopnpart
    if hhiprjp : nproc += hpopnpart * hpopnpart
    nproc += hpopnpart * 1;
    if avtrgain!=0 :
        nproc += 1
        nproc += hpopnpart * 1

    print("Minimum n:o cores = %d and nodes = %d" % (nproc,np.ceil(nproc/32.)))


def bellrf(x,m,s) :

    x = (x - m)/s * (x - m)/s

    return np.exp(-x)


def intrfcode(data2,nint,kdx = 1,savefname = "") :

    xmin = np.min(data2,0)
    xmax = np.max(data2,0)

    nrow = data2.shape[0]
    ncol = data2.shape[1]

    rfdata2 = np.zeros((nrow,ncol*(nint+1)))

    for d in range(ncol) :

        dx = 1./nint

        x = (data2[:,d] - xmin[d])/(xmax[d] - xmin[d]);

        for b in range(nint+1) :

            rfdata2[:,d*(nint+1) + b] = bellrf(x,b*dx,kdx*dx)

    if savefname!="" : savebin(rfdata2,savefname)

    return rfdata2;



def imshow(data2,interpolation='none',aspect='auto',cmap = 'jet',figno = 1,vmin = None,vmax = None,clr = True) :

    modulename = 'matplotlib'
    if modulename not in sys.modules:
        print("Module has not been imported".format(modulename))
        return

    plt.figure(figno)

    if clr : plt.clf()

    plt.imshow(data2,interpolation = interpolation,aspect = aspect,cmap = cmap,vmin = vmin,vmax = vmax)

    
def plot(data,s1 = 0,sn = -1,figno = 1,clr = True) :

    if sn<0 : sn = len(data)

    plt.figure(figno)

    if clr : plt.clf()

    res = plt.plot(data[s1:sn])


def histo(data,nbin = 40,figno = 1) :

    plt.figure(figno)

    plt.clf()

    plt.hist(data.flatten(),bins = nbin);
