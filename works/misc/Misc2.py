from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import os
import time
import importlib
import numpy as np
from matplotlib import pyplot as plt

import Utils

def reload(file) :

    importlib.reload(file)
    

def domean(data,chunklen) :

    if len(data)%chunklen!=0 : raise AssertionError("Misc2::domean","len(data) -- chunklen mismatch")

    meandata = []

    for c in range(len(data)//chunklen) :

        meandata.append(np.mean(data[c*chunklen:(c+1)*chunklen],0))

    return np.array(meandata)
    

def dosgd(parfile = "mhcuparams1.par",datatype = "mnist",ext = ".log",max_iter = 25) :

    if Utils.findparamval(parfile,"H")!=None :
        H = int(Utils.findparamval(parfile,"H"))
    else :
        nrow = int(Utils.findparamval(parfile,"nrow"))
        ncol = int(Utils.findparamval(parfile,"ncol"))

        H = nrow * ncol

    M = int(Utils.findparamval(parfile,"M"))

    Mlbl = Utils.findparamval(parfile,"Mlbl")

    mlbl = 10

    if Mlbl!=None :
        print(Mlbl)
        mlbl = int(Mlbl)

    print("H = ",H," M = ",M," Mlbl = ",mlbl)

    N = H * M

    ntrpat = int(Utils.findparamval(parfile,"ntrpat"))
    ntetrpat = int(Utils.findparamval(parfile,"ntetrpat"))
    ntepat = int(Utils.findparamval(parfile,"ntepat"))

    # import os

    if ext==".log" :
        tract = Utils.loadbin("trhid.log",N)
        tetract = Utils.loadbin("tetrhid.log",N)
        teact = Utils.loadbin("tehid.log",N)
    elif ext==".dat" : 
        tract = Utils.loadbin("trhid.dat",N)
        tetract = Utils.loadbin("tetrhid.dat",N)
        teact = Utils.loadbin("tehid.dat",N)
    elif ext==".dat1" : 
        tract = Utils.loadbin("trhid.dat1",N)
        tetract = Utils.loadbin("tetrhid.dat1",N)
        teact = Utils.loadbin("tehid.dat1",N)
    elif ext==".dat2" : 
        tract = Utils.loadbin("trhid.dat2",N)
        tetract = Utils.loadbin("tetrhid.dat2",N)
        teact = Utils.loadbin("tehid.dat2",N)
    else :
        raise AssertionError("Misc2::dosgd","trhid/tetrhid/tehid files not found")
          
    print(tract.shape,tetract.shape,teact.shape)

    if len(tract)!=ntrpat :

        print("len(tract) = ",len(tract),"ntrpat = ",ntrpat);

        chunklen = len(tract)//ntrpat

        tract = domean(tract,chunklen)

        tetract = domean(tetract,chunklen)

        teact = domean(teact,chunklen)

        Utils.savebin(tract,"spktract.bin")

        Utils.savebin(tract,"spktetract.bin")

        Utils.savebin(teact,"spkteact.bin")

    tetract = tract[:ntetrpat]

    teact = teact[:ntepat]

    if datatype=="mnist" :
        trlbl = Utils.loadbin("../../../../../../Datasets/MNIST_Data/mnist_60k_trainlbl.bin",mlbl)[:len(tetract),:]
        telbl = Utils.loadbin("../../../../../../Datasets/MNIST_Data/mnist_10k_testlbl.bin",mlbl)[:len(teact),:]
    elif datatype=="cifar-10" :
        trlbl = Utils.loadbin("../../../../../../Datasets/CIFAR-10/cf10trlbls.bin",mlbl)[:len(tetract),:]
        telbl = Utils.loadbin("../../../../../../Datasets/CIFAR-10/cf10telbls.bin",mlbl)[:len(teact),:]
    else :
        trlbl = Utils.loadbin("trlbl.bin",mlbl)
        telbl = Utils.loadbin("telbl.bin",mlbl)

    trlbl = trlbl[:ntrpat]
    tetrlbl = trlbl[:ntetrpat]
    telbl = telbl[:ntepat]

    print("tract.shape = ",tract.shape,"tetract.shape = ",tetract.shape,", teact.shape = ",teact.shape)
    print("trlbl.shape = ",trlbl.shape,"tetrlbl.shape = ",tetrlbl.shape,"telbl.shape = ",telbl.shape)

    trY = list(map(np.argmax,trlbl))
    tetrY = list(map(np.argmax,tetrlbl))
    teY = list(map(np.argmax,telbl))

    print('Size of : tract = {} tetract = {} teact = {} trY = {} tetrY = {} teY = {}'.format \
        (len(tract),len(tetract),len(teact),len(trY),len(tetrY),len(teY)))

    start_time = time.time()

    clf = SGDClassifier(penalty="l2",max_iter=256,tol = 1e-3)
    clf.fit(tract,trY)

    prtetrY = clf.predict(tetract)
    prteY = clf.predict(teact)

    print ("Elapsed fit + predict time = %.3f sec" % (time.time() - start_time))

    print("tetrainacc = {:.1f} % testacc = {:.1f} %".format(
        100.*np.sum(prtetrY==tetrY)/float(len(tetract)),
        100.*np.sum(prteY==teY)/float(len(teact))))

    return 100.*np.sum(prtetrY==tetrY)/float(len(tetract)),100.*np.sum(prteY==teY)/float(len(teact))


def runsgd(paramfile = "pfashionL4hb.par") :

    import tensorflow as tf

    os.system("source activate tensorflow")

    logdir = os.getcwd() # setworkdir(dir,subdir)

    NEPOCH=1000
    BATCH_SIZE=256

    H = int(Utils.findparamval(paramfile,"Hin"));
    M = int(Utils.findparamval(paramfile,"Min"));

    Nh= H*M #36*25 # 529*25 #32*32*25 #36*100
    No=10    
    optimizer='adam'

    print(logdir)

    # train sgd+adam classifier
    
    h_train = np.fromfile("%s/trhid.dat"%logdir,dtype=np.float32).reshape((-1,Nh))
    h_test = np.fromfile("%s/tehid.dat"%logdir,dtype=np.float32).reshape((-1,Nh))
    #h_train = np.fromfile("../Data/cifarGMM_trainimg.bin",dtype=np.float32).reshape((-1,Nh))
    #h_test = np.fromfile("../Data/cifarGMM_testimg.bin",dtype=np.float32).reshape((-1,Nh))
    #h_train = np.fromfile("../ica/cifarICAuw_trainimg.bin",dtype=np.float32).reshape((-1,Nh))
    #h_test = np.fromfile("../ica/cifarICAuw_testimg.bin",dtype=np.float32).reshape((-1,Nh))
    print ("\nhpop:",h_train.shape,h_test.shape)
    
    # o_train = np.fromfile("../Data/cifar_trainlbl.bin",dtype=np.float32).reshape((-1,No))[:h_train.shape[0]]
    # o_test = np.fromfile("../Data/cifar_testlbl.bin",dtype=np.float32).reshape((-1,No))[:h_test.shape[0]]

    o_train = np.fromfile("trlbl.dat",dtype=np.float32).reshape((-1,No))[:h_train.shape[0]]
    o_test = np.fromfile("telbl.dat",dtype=np.float32).reshape((-1,No))[:h_test.shape[0]]
    
    print (h_train.shape, h_test.shape, o_train.shape, o_test.shape)

    # building the network
    hiddens = tf.keras.layers.Input((Nh), name='hidden')
    outputs = tf.keras.layers.Dense((No), activation='softmax', name="output")(hiddens)
    
    # train classifier        
    classifier = tf.keras.Model(inputs=hiddens, outputs=outputs)

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    classifier.compile(optimizer=optimizer,
                       loss="categorical_crossentropy",
                       metrics=['categorical_accuracy']
    )
    
    classifier.summary()
        
    history = classifier.fit(h_train, o_train,
                             batch_size=BATCH_SIZE,
                             epochs=NEPOCH,
                             shuffle=True,
                             validation_split=0.2,
                             callbacks=[callback]
    )
    
    tr_loss, tr_acc = classifier.evaluate(h_train, o_train, verbose=2)
    #o_predtrain = classifier.predict(h_train)
    #tr_correct = np.argmax(o_predtrain,axis=1)==np.argmax(o_train,axis=1)
    #del tr_loss,tr_acc,o_predtrain,tr_correct,h_train,o_train
    
    te_loss, te_acc = classifier.evaluate(h_test, o_test, verbose=2)
    #o_predtest = classifier.predict(h_test)
    #te_correct = np.argmax(o_predtest,axis=1)==np.argmax(o_test,axis=1)
    

def sem(data) :

    return np.std(data)/np.sqrt(len(data))


def runnsgd(N,codestr,parfile = "mhcuparams1.par",datatype = "mnist",max_iter = 25) :

    import os

    trcorrs = []

    tecorrs = []

    for n in range(N) :

        print("n = ",n)

        if "beskow" in Utils.hostname :

            os.system("srun -n 1 --unbuffered " + codestr)

        else :

            os.system(codestr)

        trcorr,tecorr = dosgd(parfile,datatype,max_iter)

        trcorrs.append(trcorr)

        tecorrs.append(tecorr)

    print("mean trainacc = {:.2f} ({:,.2f}) % mean testacc = {:.2f} ({:,.2f})  %". \
          format(np.mean(trcorrs),sem(trcorrs),np.mean(tecorrs),sem(tecorrs)))

    return trcorrs,tecorrs


def kmeans(X,K) :

    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)

    return kmeans


def patsdist(X,metric = 'euclidean') :

    return squareform(pdist(X,metric = metric))
    

def patsdist(X1,X2,metric = 'euclidean') :

    D = squareform(pdist(np.vstack((X1,X2)),metric = metric))

    return D[len(X1):,:len(X2)]
    

def plotmisc1(parfile = "mhcuparams1.par",figno = 1) :

    H = int(Utils.findparamval(parfile,"H"))

    M = int(Utils.findparamval(parfile,"M"))

    nepoc = int(Utils.findparamval(parfile,"nepoc"))

    misc = Utils.loadbin("misc.log",784*H)

    miscx = misc.reshape(nepoc,H,784)

    sumisc = []

    for epoc in range(nepoc) :

        sumisc.append(np.sum(miscx[epoc,:,:]))

    sm = np.array(sumisc)

    plt.figure(figno)

    plt.clf()

    plt.plot(sm)
    

def plotrf1(parfile = "mhcuparams1.par",figno = 1,dur = 0.5,imside = 28) :

    H = int(Utils.findparamval(parfile,"H"))

    M = int(Utils.findparamval(parfile,"M"))

    N = H * M

    wij1 = Utils.loadbin("wij1.bin",N)

    vmin = np.min(wij1)

    vmax = np.max(wij1)

    print('vmin = {} vmax = {}'.format(vmin,vmax))

    plt.figure(figno)

    plt.clf()

    for r in range(0,wij1.shape[1],100) :

        plt.imshow(wij1[:,r].reshape(28,28),interpolation='none',aspect='auto',cmap = 'jet',vmin = vmin,vmax = vmax)

        plt.show()

        plt.pause(dur)


# Analysing cifar-10 images

def mean(imgs,rgb = 0) :

# imgs dim (nimg,32,32,3)

    meanimg = np.zeros((32,32))

    if rgb==None :

        for r in range(32) :

            for c in range(32) :

                meanimg[r,c] = np.mean(imgs[:,r,c])

    else :

        for r in range(32) :

            for c in range(32) :

                meanimg[r,c] = np.mean(imgs[:,r,c,rgb])

    return meanimg


def var(imgs,rgb = 0) :

# imgs dim (nimg,32,32,3)

    varimg = np.zeros((32,32))

    if rgb==None :

        for r in range(32) :

            for c in range(32) :

                varimg[r,c] = np.var(imgs[:,r,c])

    else :

        for r in range(32) :

            for c in range(32) :

                varimg[r,c] = np.var(imgs[:,r,c,rgb])

    return varimg


def entimgs(imgs) :

# imgs dim (nimg,32,32)

    logimgs = np.log(imgs + 1e-12)

    return -np.sum(np.multiply(imgs,logimgs),0).reshape(32,32)


def entpats(pats) :

    logpats = np.log(pats + 1e-12)

    return -np.sum(np.multiply(pats,logpats),0)


def varscale(imgs,rgb) :

    varimg = var(imgs,rgb)

    varscimgs = np.zeros((len(imgs),32,32))

    if rgb==None :

        for i in range(len(imgs)) :

            varscimgs[i] = np.divide(imgs[i,:,:],varimg)

    else :

        for i in range(len(imgs)) :

            varscimgs[i] = np.divide(imgs[i,:,:,rgb],varimg)

    return varscimgs
    

def meanscale(imgs,rgb) :

    meanimg = mean(imgs,rgb)

    meanscimgs = np.zeros((len(imgs),32,32))

    if rgb==None :

        for i in range(len(imgs)) :

            meanscimgs[i] = np.divide(imgs[i,:,:],meanimg)

    else :

        for i in range(len(imgs)) :

            meanscimgs[i] = np.divide(imgs[i,:,:,rgb],meanimg)

    return meanscimgs
    

def meanvarscale(imgs,rgb) :

    meanscimgs = meanscale(imgs,rgb)

    meanvarscimgs = varscale(meanscimgs,None)

    meanvarmeanscimgs = meanscale(meanvarscimgs,None)

    return meanvarmeanscimgs


def myzca(X) :

    # samples in rows

    cov = np.cov(X,rowvar=False)

    U,S,V = np.linalg.svd(cov)
     
    epsilon = 0.1
    
    X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X.T).T

    X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())

    return X_ZCA_rescaled


def dim(X) :

    if len(X.shape)==1 or X.shape[0]==1 : return 1

    return len(X.shape)
    

def ndata(X) :

    if len(X.shape)==1 : return X.shape[0]

    return X.shape[0] * X.shape[1]


def npixel(X) :

    if dim(X)==1 and ndata(X)%3!=0 : raise AssertionError("npixel::Not proper color data")

    if len(X.shape)==1 : return len(X)/3

    return X.shape[0] * X.shape[1]/3


def cutout(X,r0 = 0,rn = None,c0 = 0,cn = None) :

    if dim(X)==1 :
        
        cpixel = int(np.sqrt(npixel(X)))

        rpixel = int(np.ceil(npixel(X)/cpixel))

        if cpixel * rpixel != npixel(X) : raise AssertionError("cutout::cpixel*rpixel -- npixel mismatch")

        if rn==None : rn = rpixel
        if cn==None : cn = cpixel

        if r0<0 : raise AssertionError("cutout::r0<0")
        if r0+rn>npixel(X) : raise AssertionError("cutout::r0+rn>ndata(X)")

        Xc = np.zeros((rn,3 * cn))

        print(r0,r0+rn)

        for r in range(r0,r0+rn) :

            p0 = 3 * (r * cpixel +  c0)

            pn = 3 * (r * cpixel + c0 + cn)

            print(r-r0,p0,pn,pn-p0)

            Xc[r-r0] = X[p0:pn]

        return Xc


    elif dim(X)==2 :

        cpixel = npixel(X[0])
        rpixel = len(X)

        if rn==None : rn = rpixel
        if cn==None : cn = cpixel

        if r0<0 : raise AssertionError("cutout::r0<0")
        if r0+rn>rpixel : raise AssertionError("cutout::r0+rn>rpixel")
        if c0<0 : raise AssertionError("cutout::c0<0")
        if c0+cn>cpixel : raise AssertionError("cutout::c0+cn>cpixel")

        return X[r0:r0+rn,c0*3:(c0+cn)*3]

    elif dim(X)==3 :
        
        cpixel = X.shape[1]
        rpixel = X.shape[0]

        if X.shape[2]!=3 : raise AssertionError("cutout::Illegal image depth")

        if rn==None : rn = rpixel
        if cn==None : cn = cpixel

        if r0<0 : raise AssertionError("cutout::r0<0")
        if r0+rn>rpixel : raise AssertionError("cutout::r0+rn>rpixel")
        if c0<0 : raise AssertionError("cutout::c0<0")
        if c0+cn>cpixel : raise AssertionError("cutout::c0+cn>cpixel")

        return X[r0:r0+rn,c0:c0+cn,:]

    else : raise AssertionError("cutout::Illegal dimension of input")
        

def showrgbimage(X,cpixel = None,pixelsize = 0.2,figno = 1):

    if dim(X)==1 :

        if cpixel==None :

            cpixel = int(np.sqrt(npixel(X)))

        rpixel = int(np.ceil(npixel(X)/cpixel))

        if cpixel * rpixel != npixel(X) : raise AssertionError("showrgbimage::cpixel -- rpixel mismatch")

        # print(rpixel,cpixel)

        X = X.reshape(rpixel,cpixel,3)

    elif dim(X)==2 :

        if cpixel!=None :

            rpixel = int(np.ceil(npixel(X)/cpixel))

            print(npixel(X),rpixel,cpixel)

            if cpixel * rpixel != npixel(X) : raise AssertionError("showrgbimage::cpixel -- rpixel mismatch")

        else :

            cpixel = int(X.shape[1]/3)

            rpixel = X.shape[0]

        X = X.reshape(rpixel,cpixel,3)

    elif dim(X)==3 :

        rpixel = X.shape[0]

        cpixel = X.shape[1]
        
    else : raise AssertionError("Illegal dimension of input")

    figxsize = pixelsize * cpixel

    figysize = pixelsize * rpixel * 3/4

    # print("cpixel = {} rpixel = {} figxsize = {} figysize = {}".format(cpixel,rpixel,figxsize,figysize))

    if figno<=0 : return

    plt.figure(figno,figsize=(figxsize,figysize))

    plt.gcf().set_size_inches(figxsize,figysize)

    plt.axis('off')

    plt.imshow(X)

    plt.show()


def composeimgs(imgs,nrow = 28,ncol = 28,figno = 1,cmap = 'jet') :

    NROW = int(np.sqrt(len(imgs)) + 0.5)

    NCOL = NROW + int(NROW*NROW<len(imgs))

    # print(len(imgs),NROW,NCOL)

    cnrow = NROW * nrow + NROW

    cncol = NCOL * ncol + NCOL

    compimgs = np.zeros((cnrow,cncol))

    from matplotlib import pyplot as plt

    for R in range(NROW) :

        for C in range(NCOL) :

            imgno = R * NCOL + C

            if len(imgs)<=imgno : break

            img = imgs[imgno].reshape(nrow,ncol)
            
            compimgs[R*nrow + R:(R+1)*nrow + R,C*ncol + C:(C+1)*ncol + C] = img

    Utils.imshow(compimgs,interpolation='none',aspect='auto',cmap = cmap,figno=figno)
