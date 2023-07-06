import sys, os, select
import math
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
import random
import string
from matplotlib import pyplot as plt
import matplotlib.animation as animation
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
import Utils
import csv
from ast import literal_eval
import pandas as pd

sims = 10
if sims==0:
    dorun=False
else:
    dorun=True


PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'

H1 = int(Utils.findparamval(PATH+"olflangmain1.par","H"))
M1 = int(Utils.findparamval(PATH+"olflangmain1.par","M"))
H2 = int(Utils.findparamval(PATH+"olflangmain1.par","H2"))
M2 = int(Utils.findparamval(PATH+"olflangmain1.par","M2"))
N1 = H1*M1
N2 = H2*M2


os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')


def main(parfilename=PATH+"olflangmain1.par"):


    if dorun:
        for x in range(sims):
            print("\n\n Simulation number: {}".format(x+1))
            os.system("./olflangmain1")
            if x==0:
                Wij11 = Utils.loadbin("Wij11.bin",N1,N1) #default size = (simulation steps * N*N, ) #Wij11
                Wij12 = Utils.loadbin("Wij12.bin",N1,N2)
                Wij21 = Utils.loadbin("Wij21.bin",N2,N1)
                Wij22 = Utils.loadbin("Wij22.bin",N2,N2)
                Bj11 = Utils.loadbin("Bj11.bin")
                Bj22 = Utils.loadbin("Bj22.bin")
                Bj12 = Utils.loadbin("Bj12.bin")
                Bj21 = Utils.loadbin("Bj21.bin")

                if sims==1:
                    Wij11 = np.reshape(Wij11,(1,N1,N1))
                    Wij12 = np.reshape(Wij12,(1,N1,N2))
                    Wij21 = np.reshape(Wij21,(1,N2,N1))
                    Wij22 = np.reshape(Wij22,(1,N2,N2))
                    Bj11 = Bj11.reshape(1,Bj11.shape[0])
                    Bj22 = Bj22.reshape(1,Bj22.shape[0])
                    Bj21 = Bj21.reshape(1,Bj21.shape[0])
                    Bj12 = Bj12.reshape(1,Bj12.shape[0])
            else:
                if x==1:
                    Wij11 = np.stack((Wij11,Utils.loadbin("Wij11.bin",N1,N1)))
                    Wij12 = np.stack((Wij12,Utils.loadbin("Wij12.bin",N1,N2)))
                    Wij21 = np.stack((Wij21,Utils.loadbin("Wij21.bin",N2,N1)))
                    Wij22 = np.stack((Wij22,Utils.loadbin("Wij22.bin",N2,N2)))
                    Bj11 = np.stack((Bj11,Utils.loadbin("Bj11.bin")))
                    Bj22 = np.stack((Bj22,Utils.loadbin("Bj22.bin")))
                    Bj21 = np.stack((Bj21,Utils.loadbin("Bj21.bin")))
                    Bj12 = np.stack((Bj12,Utils.loadbin("Bj12.bin")))
                else:
                    Wij11 = np.concatenate((Wij11,Utils.loadbin("Wij11.bin",N1,N1)[None]),axis=0)
                    Wij12 = np.concatenate((Wij12,Utils.loadbin("Wij12.bin",N1,N2)[None]),axis=0)
                    Wij21 = np.concatenate((Wij21,Utils.loadbin("Wij21.bin",N2,N1)[None]),axis=0)
                    Wij22 = np.concatenate((Wij22,Utils.loadbin("Wij22.bin",N2,N2)[None]),axis=0)
                    Bj11 = np.concatenate((Bj11,Utils.loadbin("Bj11.bin")[None]),axis=0)
                    Bj22 = np.concatenate((Bj22,Utils.loadbin("Bj22.bin")[None]),axis=0)
                    Bj12 = np.concatenate((Bj12,Utils.loadbin("Bj12.bin")[None]),axis=0)
                    Bj21 = np.concatenate((Bj21,Utils.loadbin("Bj21.bin")[None]),axis=0)

    else:
        Wij11 = Utils.loadbin("Wij11.bin",N1,N1)
        Wij11 = Wij11.reshape(1,Wij11.shape[0],Wij11.shape[1])
        Wij22 = Utils.loadbin("Wij22.bin",N2,N2)
        Wij22 = Wij22.reshape(1,Wij22.shape[0],Wij22.shape[1])
        Wij12 = Utils.loadbin("Wij12.bin",N1,N2)
        Wij12 = Wij12.reshape(1,Wij12.shape[0],Wij12.shape[1])
        Wij21 = Utils.loadbin("Wij21.bin",N2,N1)
        Wij21 = Wij21.reshape(1,Wij21.shape[0],Wij21.shape[1])
        Bj11 = Utils.loadbin("Bj11.bin")
        Bj22 = Utils.loadbin("Bj22.bin")
        Bj21 = Utils.loadbin("Bj21.bin")
        Bj12 = Utils.loadbin("Bj12.bin")
        Bj11 = Bj11.reshape(1,Bj11.shape[0])
        Bj22 = Bj22.reshape(1,Bj22.shape[0])
        Bj21 = Bj21.reshape(1,Bj21.shape[0])
        Bj12 = Bj12.reshape(1,Bj12.shape[0])



    Wij11mean = Wij11.mean(axis=0)
    Wij22mean = Wij22.mean(axis=0)
    Wij21mean = Wij21.mean(axis=0)
    Wij12mean = Wij12.mean(axis=0)

    Bj11mean = Bj11.mean(axis=0)
    Bj22mean = Bj22.mean(axis=0)
    Bj21mean = Bj21.mean(axis=0)
    Bj12mean = Bj12.mean(axis=0)
    print(Wij11[:,2,2])
    
    # Utils.savebin(Wij11mean.flatten(),"Wij11pre_si_2clusters_withFam.bin")
    # Utils.savebin(Wij22mean.flatten(),"Wij22pre_si_2clusters_withFam.bin")
    # Utils.savebin(Wij21mean.flatten(),"Wij21pre_si_2clusters_withFam.bin")
    # Utils.savebin(Wij12mean.flatten(),"Wij12pre_si_2clusters_withFam.bin")
    # Utils.savebin(Bj11mean.flatten(),"Bj11pre_si_2clusters_withFam.bin")
    # Utils.savebin(Bj22mean.flatten(),"Bj22pre_si_2clusters_withFam.bin")
    # Utils.savebin(Bj12mean.flatten(),"Bj12pre_si_2clusters_withFam.bin")
    # Utils.savebin(Bj21mean.flatten(),"Bj21pre_si_2clusters_withFam.bin")

    Utils.savebin(Wij11mean.flatten(),"Wij11pre_si_4clusters_GasolineSingleAssoc.bin")
    Utils.savebin(Wij22mean.flatten(),"Wij22pre_si_4clusters_GasolineSingleAssoc.bin")
    Utils.savebin(Wij21mean.flatten(),"Wij21pre_si_4clusters_GasolineSingleAssoc.bin")
    Utils.savebin(Wij12mean.flatten(),"Wij12pre_si_4clusters_GasolineSingleAssoc.bin")
    Utils.savebin(Bj11mean.flatten(),"Bj11pre_si_4clusters_GasolineSingleAssoc.bin")
    Utils.savebin(Bj22mean.flatten(),"Bj22pre_si_4clusters_GasolineSingleAssoc.bin")
    Utils.savebin(Bj12mean.flatten(),"Bj12pre_si_4clusters_GasolineSingleAssoc.bin")
    Utils.savebin(Bj21mean.flatten(),"Bj21pre_si_4clusters_GasolineSingleAssoc.bin")

    # Utils.savebin(Wij11mean.flatten(),"Wij11pre_correctdescs_maxassocs4_OrthoLang.bin")
    # Utils.savebin(Wij22mean.flatten(),"Wij22pre_correctdescs_maxassocs4_OrthoLang.bin")
    # Utils.savebin(Wij21mean.flatten(),"Wij21pre_correctdescs_maxassocs4_OrthoLang.bin")
    # Utils.savebin(Wij12mean.flatten(),"Wij12pre_correctdescs_maxassocs4_OrthoLang.bin")
    # Utils.savebin(Bj11mean.flatten(),"Bj11pre_correctdescs_maxassocs4_OrthoLang.bin")
    # Utils.savebin(Bj22mean.flatten(),"Bj22pre_correctdescs_maxassocs4_OrthoLang.bin")
    # Utils.savebin(Bj12mean.flatten(),"Bj12pre_correctdescs_maxassocs4_OrthoLang.bin")
    # Utils.savebin(Bj21mean.flatten(),"Bj21pre_correctdescs_maxassocs4_OrthoLang.bin")
main()