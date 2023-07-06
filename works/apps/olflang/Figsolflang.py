import sys, os, select
import csv
import math
import pandas as pd
import numpy as np
import random
import string
import json
import seaborn as sns
from scipy.spatial import distance
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker as tck
import matplotlib.patches as mpatches
from collections import Counter
from scipy.stats import pearsonr as corr
import colorsys
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import Utils

PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'
buildPATH = '/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/'
figPATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/DualNet/Diagnostics/OrthoPats/'

kelly_colors = ['#BE0032', #Red
                '#F3C300', #Mustard
                '#8b4513', #saddlebrown
                '#F38400', #Peach
                '#A1CAF1', #Baby Blue
                '#C2B280', #Grunge greenish brown
                '#848482', #Grey
                '#008856', #Green
                '#E68FAC', #Pink
                '#0067A5', #Blue
                '#F99379', #Light Pink
                '#604E97', #Violet
                '#F6A600', #Orange
                '#B3446C', #Wine Stain red
                '#DCD300', #Slime yellowish green
                '#ff0000', #Maroon
                '#8DB600', #Parrot Green
                '#654522', #Brown
                '#E25822', #Dark Pink
                '#2B3D26', #Dark NAvy Blue
                '#F2F3F4', #Off White/Greyish White
                '#222222' #Black
                ]

def cosdist(pat,act) :

    bothpats = np.vstack((pat,act))

    return (1 - squareform(pdist(bothpats,metric='cosine')))[0,1]


def empatdist1(dorun = True,parfilename = "olflangmain1.par",figno = 1,clr = True,field = "act", r0 = 0,
               r1 = None) :


    if dorun : os.system("./olflangmain1")



    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))

    N = H * M

    if field=="inp" :

        data1 = Utils.loadbin("inp1.log",N)
        data2 = Utils.loadbin("inp2.log",N)

    elif field=="dsup" :

        data1 = Utils.loadbin("dsup1.log",N)
        data2 = Utils.loadbin("dsup2.log",N)

    elif field=="act" :

        data1 = Utils.loadbin("act1.log",N)  # N x 960 (number of simulation steps) array
        data2 = Utils.loadbin("act2.log",N)

    else : raise AssertionError("No such field")

    if len(data1)!=len(data2) : raise AssertionError("Illegal len(data1)!=len(data2)")

    if r1==None : r1 = max(len(data1),len(data2))

    trpats1 = Utils.loadbin("trpats1.bin",N)

    trpats2 = Utils.loadbin("trpats2.bin",N)

    trpats = trpats2 # np.concatenate((trpats1,trpats2))

    X1 = []

    for dat in data2 :

        x1 = []

        for trpat in trpats :

            x1.append(cosdist(trpat,dat))

        X1.append(x1)

    return X1



def convolve(data,H,M):
    N = H*M
    convolved_data = np.zeros(shape=(data.shape[0],M))
    for i in range(N):
        j = i%M
        convolved_data[:,j] += data[:,i]/H

    return convolved_data




# def plotter(dorun=False):

#     N = 606#np.loadtxt('binpats_sub1.txt')[1]
#     M = 2
#     MCmin = 360 #Make sure even number so it doesnt cut a hypercolumn in between
#     HCshow = 10

#     MCcenter = 360

#     descriptors = pd.read_csv('DescriptorHCs.csv',squeeze=True,index_col=0)
#     os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

#     if dorun :
#        os.system("./olflangmain1")

#     data1 = Utils.loadbin("act1.log",N)
#     MCrange = np.arange(int(MCcenter-HCshow*M)(/2),int(MCmin+HCshow*M))
#     #MCrange = np.arange(int(MCcenter-(HCshow/2)*M),int(MCmin+(HCshow/2)*M))
#     print(MCrange)
#     HCrange = [int(math.floor(x/2)) for x in MCrange[::2]]
#     HClabels = descriptors.loc[HCrange].tolist()
#     data1 = data1[:,MCrange]
#     data1 = data1.T

#     #print(HCrange)
#     colormap1 = plt.imshow(data1,interpolation='none',aspect='auto',cmap = 'jet' )
#     #plt.gca().autoscale(False)
#     #plt.gca().axis([0,data1.shape[1],MCmin,MCmin+HCshow*M])

#     cbar = plt.colorbar(colormap1)
#     yticklocations = np.arange(0,len(MCrange),2) + 0.5
#     plt.gca().set_yticks(yticklocations)
#     plt.gca().set_yticklabels(HClabels)
#     for i in range(1,data1.shape[0],2):
#         plt.axhline(y=i+.5,color='k')

#     def mjrFormatter(x, pos):
#         return int(math.floor(x/2))
#         #return(HCrange[x])

#     #plt.ylim(MCmin,MCmin+HCshow*M)
#     print(plt.gca().get_yticks())
#     plt.gcf().set_size_inches(20, 10)
#     #plt.gca().set_yticklabels(HCrange)
#     #plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(mjrFormatter))
#     plt.show()
    
def plotter(dorun=False):
    H = 303
    M = 2
    N = M*H
    thresh = 0.8       #threshold to count as active
    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

    if dorun :
       os.system("./olflangmain1")

    data1 = Utils.loadbin("act1.log",N).T

    print(data1.shape)

    for timestep in range(data1.shape[1]):
        act = data1[:,timestep]

        if timestep==0:
            act_thresholded = np.array([(act[i:i+2]>0.8).astype(int) for i in range(0,len(act),2)])
            act_thresholded = np.array([1 if np.array_equal(i,[1,0]) else 0 for i in act_thresholded]) 
        else:
            cur_act = np.array([(act[i:i+2]>0.8).astype(int) for i in range(0,len(act),2)])
            cur_act = np.array([1 if np.array_equal(i,[1,0]) else 0 for i in cur_act]) 
            act_thresholded = np.column_stack((act_thresholded,cur_act))


    colormap1 = plt.imshow(act_thresholded,interpolation='none',aspect='auto',cmap = plt.cm.binary)
    plt.title("LTM #1",fontsize = 14)
    cbar = plt.colorbar(colormap1)
    plt.xlabel("Timesteps",fontsize=16)
    plt.ylabel("HC",fontsize=16)
    plt.xticks(fontsize = 16)
    cbar.ax.tick_params(labelsize=14) 

    plt.gcf().set_size_inches(20, 10)

    plt.show()




def actplot(dorun = False,HCshow=20,MCmin = 540,showparams=False, parfilename = PATH+"olflangmain1.par",mode = "img",figno = 1,clr = True,
            field = "act",r0 = 0,r1 = None,c0 = 0,c1 = None,stride = 1) :

    
    os.chdir(buildPATH)
    if dorun :
        os.system("./olflangmain1")

    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))
    N =  H * M

    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))
    N2 =  H2 * M2


    igain = float(Utils.findparamval(parfilename,"igain"))
    again = float(Utils.findparamval(parfilename,"again"))
    taum= float(Utils.findparamval(parfilename,"taum"))
    taua= float(Utils.findparamval(parfilename,"taua"))
    adgain = float(Utils.findparamval(parfilename,"adgain"))


    taup = float(Utils.findparamval(parfilename,"taup"))
    taucond = float(Utils.findparamval(parfilename,"taucond"))

    recuwgain = float(Utils.findparamval(parfilename,"recuwgain"))
    assowgain = float(Utils.findparamval(parfilename,"assowgain"))
    bgain = float(Utils.findparamval(parfilename,"bgain"))

    etrnrep = int(Utils.findparamval(parfilename,"etrnrep"))

    nmean = float(Utils.findparamval(parfilename,"nmean"))
    namp = float(Utils.findparamval(parfilename,"namp"))

    #for printing important param at the right side of the image
    textstr = 'H = %d\nM = %d\n\nigain = %.2f\nagain = %.2f\ntaum = %.4f\ntaua = %.4f\nadgain = %.1f\n\ntaup = %.1f\ntaucond = %.3f\n\nrecuwgain = %.2f\nassowgain = %.2f\nbgain = %.2f\n\netrnrep = %d\n\nnmean = %.3f\nnamp = %.3f'%(H, M, igain, again, taum, taua, adgain, taup, taucond,recuwgain,assowgain,bgain,etrnrep, nmean, namp)

    text_file = open("simstages.txt", "r")
    simstages = [int(line) for line in text_file]
    text_file.close()



    if field=="inp" :

        data1 = Utils.loadbin("inp1.log",N)
        data2 = Utils.loadbin("inp2.log",N2)

    elif field=="dsup" :

        data1 = Utils.loadbin("dsup1.log",N)
        data2 = Utils.loadbin("dsup2.log",N2)

    elif field=="expdsup" :
        data1 = Utils.loadbin("expdsup1.log",N)
        data2 = Utils.loadbin("expdsup2.log",N2)

    elif field=="act" :

        data1 = Utils.loadbin("act1.log",N)
        data2 = Utils.loadbin("act2.log",N2)

    elif field=="ada" :

        data1 = Utils.loadbin("ada1.log",N)
        data2 = Utils.loadbin("ada2.log",N2)

    elif field == "bwsup" :

        data1 = Utils.loadbin("bwsup1.log",N)
        data2 = Utils.loadbin("bwsup2.log",N2)

    elif field == "expdsup" :

        data1 = Utils.loadbin("expdsup1.log",N)
        data2 = Utils.loadbin("expdsup2.log",N2)

    if HCshow>0:
        data1 = data1[:,0:HCshow*M]
        data2 = data2[:,0:HCshow*M2]

    else : raise AssertionError("Illegal field")




    if len(data1)!=len(data2) : raise AssertionError("Illegal len(data1)!=len(data2)")

    if r1==None : r1 = len(data1)

    if c1==None : c1 = max(len(data1[0]),len(data2[0]))

    if figno<1 : return

    if clr : plt.clf()

    data1 = data1.T
    data2 = data2.T

    

    if mode=="img" :

         ax1 = plt.subplot(2,1,1)
         colormap1 = ax1.imshow(data1[r0:r1, c0:],interpolation='none',aspect='auto',cmap = 'jet')
         plt.title("LTM #1",fontsize = 14)
         cbar = plt.colorbar(colormap1,ax=ax1)
         plt.xlabel("Timesteps",fontsize=16)
         plt.ylabel("MCs",fontsize=16)
         plt.xticks(fontsize = 16)
         plt.yticks(fontsize = 16)
         cbar.ax.tick_params(labelsize=14) 
    else :

        ax1.plot(data1[c0:c1:stride,r0:r1])


    if mode=="img" :
        ax2 = plt.subplot(2,1,2)
        colormap2 = ax2.imshow(data2[r0:r1,c0:],interpolation='none',aspect='auto',cmap = 'jet')
        plt.title("LTM #2", fontsize = 14)
        cbar = plt.colorbar(colormap2,ax=ax2)
        plt.xlabel("Timesteps",fontsize=16)
        plt.ylabel("MCs",fontsize=16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        cbar.ax.tick_params(labelsize=14) 
    else :

        ax2.plot(data2[c0:c1:stride,r0:r1])

    plt.subplots_adjust(hspace=0.3)
    plt.gcf().set_size_inches(20, 10)

    if showparams:
        plt.figtext(0.82,0.2, textstr, fontsize=14)

    plt.show()

def wdist(dorun=False):
     if dorun:
        os.system("./olflangmain1")

     N = 303

     os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

     w11 = Utils.loadbin("Wij11.bin",N)

     plt.gcf().set_size_inches(8,5)
     plt.hist(w11.flatten(),bins=100)
     plt.yscale('log')
     plt.ylabel('Count',size=14)
     plt.title('Wij11')
     plt.xlabel('Weights',size=14)
     plt.savefig(PATH+'weight_hist_igain2,5_adgain0',dpi=300)
     plt.show()

def pos(MCpre,MCpos,N):
    return N*MCpre+MCpos

def bias_dist(dorun=False):
    if dorun:
        os.system("./olflangmain1")

    N = 303

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

    w11 = Utils.loadbin("Bj11.bin",N)

    plt.gcf().set_size_inches(8,5)
    plt.hist(w11.flatten(),bins=100)
    plt.yscale('log')
    plt.ylabel('Count',size=14)
    plt.title('Bj11')
    plt.xlabel('Bias',size=14)
    plt.savefig(PATH+'bias_hist_igain1_adgain0',dpi=300)
    plt.show()

def Ptrace_vs_act(dorun=False):

    if dorun:
        os.system("./olflangmain1")

    N = 303

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),constrained_layout=True)

    pi11 = Utils.loadbin("Pi11.log",N)
    pj11 = Utils.loadbin("Pj11.log",N)
    pij11 = Utils.loadbin("Pij11.log",N*N)

    pre_node = 200
    post_node1 = 225
    post_node2 = 275
    ij_pos1 = pos(pre_node,post_node1,N)
    ij_pos2 = pos(pre_node,post_node2,N)

    start_timestep = 35
    end_timestep = 75

    duration = np.arange(start_timestep,end_timestep)

    linestyles = ["--","--","--","-.","-.","-","-"]

    datapoints = [
        pi11[duration,pre_node],
        pi11[duration,post_node1],
        pi11[duration,post_node2],
        pj11[duration,post_node1],
        pj11[duration,post_node2],
        pij11[duration,ij_pos1],
        pij11[duration,ij_pos2]
        ]

    legend = [
        'Pi Unit '+str(pre_node),
        'Pj Unit '+str(post_node1),
        'Pi Unit '+str(post_node2),
        'Pj Unit '+str(post_node1),
        'Pj Unit '+str(post_node2),
        'Pij Units '+str(pre_node)+'-'+str(post_node1),
        'Pij Units '+str(pre_node)+'-'+str(post_node2),

             ]

    for i in range(len(datapoints)):
        if linestyles[i] == "-":
            linewidth = 2
        else:
            linewidth = 1
        ax1.plot(duration,datapoints[i],linestyles[i],label=legend[i],linewidth=linewidth)
        ax1.margins(x=0)

    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.title.set_text("P-traces")
    #ax1.set_ylabel('Pj',size=16)
    ax1.legend(loc=4)

    def mjrFormatter(x, pos):
    #relabel y axis labels from 1 so MC labels dont start from 0
        return int(x+pre_node-5)

    # wij11 = Utils.loadbin("Wij11.log",N*N)
    # datapoints = [wij11[duration,pos(pre_node,pre_node,N)],
    #             wij11[duration,pos(post_node1,post_node1,N)],
    #             wij11[duration,pos(post_node2,post_node2,N)],
    #             wij11[duration,pos(pre_node,post_node1,N)],
    #             wij11[duration,pos(pre_node,post_node2,N)]
    #             ]

    # linestyles = ['--','--','--','-','-']

    # legend = [
    #         'Wij Units '+str(pre_node)+'-'+str(pre_node),
    #         'Wij Units '+str(post_node1)+'-'+str(post_node1),
    #         'Wij Units '+str(post_node2)+'-'+str(post_node2),
    #         'Wij Units '+str(pre_node)+'-'+str(post_node1),
    #         'Wij Units '+str(pre_node)+'-'+str(post_node2)
    # ]

    # for i in range(len(datapoints)):
    #     ax2.plot(duration,datapoints[i],linestyles[i],label=legend[i])
    
    # ax2.margins(x=0)
    # ax2.title.set_text("Local weights")
    # ax2.set_ylabel("Wij11",fontsize=16)
    # ax2.legend(loc=4)

    Bj11 = Utils.loadbin("Bj11.log",N)
    datapoints = [Bj11[duration,pre_node],
                Bj11[duration,post_node1],
                Bj11[duration,post_node2],
                ]

    linestyles = ['--','--','--','-','-']

    legend = [
            'Bj Units '+str(pre_node)+'-'+str(pre_node),
            'Bj Units '+str(post_node1)+'-'+str(post_node1),
            'Bj Units '+str(post_node2)+'-'+str(post_node2),
            'Bj Units '+str(pre_node)+'-'+str(post_node1),
            'Bj Units '+str(pre_node)+'-'+str(post_node2)
    ]

    for i in range(len(datapoints)):
        ax2.plot(duration,datapoints[i],linestyles[i],label=legend[i])
    
    ax2.margins(x=0)
    ax2.title.set_text("Bias")
    ax2.set_ylabel("Bj11",fontsize=16)
    ax2.legend(loc=4)

    data1 = Utils.loadbin("act1.log",N)
    data1 = data1.T
    print(data1.shape)
    colormap1 = ax3.imshow(data1[pre_node-5:post_node2+5, duration],interpolation='none',aspect='auto',cmap = 'jet')

    ax3.title.set_text("LTM #1 Activity")
    ax3.set_ylabel('Unit',fontsize=16)
    ax3.set_yticks(np.arange(5,post_node2-pre_node+6,25))
    ax3.yaxis.set_major_formatter(tck.FuncFormatter(mjrFormatter))

    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    plt.colorbar(colormap1,ax=ax3)

    #plt.savefig(PATH+'Ptrace_vs_act_Bj11_200_225_275_Wgain10',dpi=300)
    plt.show()



def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def datasize(filename) :

    if not os.path.isfile(filename) : return 0

    return os.path.getsize(filename)


def doplot(filename,C,ax,r0 = 0,r1 = None,c0 = 0,stride = 1) :

    if datasize(filename)>0 :
        data = Utils.loadbin(filename,C)
        if r1==None : r1 = len(data)
        ax.plot(data[r0:r1,0::stride])




def doimshow(filename,R,C,ax,idx = -1) :

    if datasize(filename)>0 :
        
        data = Utils.loadbin(filename,R*C)
        colormap1 = ax.imshow(data[0:R, 0:],interpolation='none',aspect='auto',cmap = 'jet')
        ax.imshow(data[idx].reshape(R,C),interpolation='none',aspect='auto',cmap = 'jet')
        plt.colorbar(colormap1,ax=ax)

def bwplot(dorun = False,parfilename = PATH+"olflangmain1.par",figno = 1,clr = True, r0 = 0,r1 = None) :


    if dorun :
       os.system("./olflangmain1")

    
    #N = int(np.loadtxt('binpats_sub1.txt')[1])
    
    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N = H * M
    N2 = H2*M2

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

    if figno<1 : return

    if clr : plt.clf()

    #Weights vs time
    # ax1 = plt.subplot(1,1,1)
    # doplot("Wij11.log",N*N,ax1,stride=10)
    # plt.title("Wij11")

    # ax2 = plt.subplot(1,2,2)
    # doplot("Wij22.log",N*N,ax2)
    # plt.title("Wij22")


    # #####Final Weights
    ax1 = plt.subplot(2,2,1)
    doimshow("Wij11.bin",N,N,ax1)
    plt.title("Wij11",fontsize=16)
    ax1.set_xlabel("Post-synaptic MC")
    ax1.set_ylabel("Pre-synaptic MC")

    ax2 = plt.subplot(2,2,2)
    doimshow("Wij22.bin",N2,N2,ax2)
    plt.title("Wij22",fontsize=16)

    ax3 = plt.subplot(2,2,3)
    doimshow("Wij21.bin",N2,N,ax3)
    plt.title("Wij21",fontsize=16)
    # ax3.set_xlabel("MC")
    # ax3.set_ylabel("HC")

    ax4 = plt.subplot(2,2,4)
    doimshow("Wij12.bin",N,N2,ax4)
    plt.title("Wij12",fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    #Bias plot
    # ax1 = plt.subplot(2,2,1)
    # doimshow("Bj11.bin",H,M,ax1)
    # plt.title("Bj11")
    # plt.xlabel("Minicolumn No")
    # plt.ylabel("Hypercolumn No")
    
    # ax2 = plt.subplot(2,2,2)
    # doimshow("Bj22.bin",H2,M2,ax2)
    # plt.title("Bj22")
    
    # ax3 = plt.subplot(2,2,3)
    # doimshow("Bj21.bin",H2,M,ax3)
    # plt.title("Bj21")
    
    # ax4 = plt.subplot(2,2,4)
    # doimshow("Bj12.bin",H,M2,ax4)
    # plt.title("Bj12")


    # ax1 = plt.subplot(2,2,1)
    # ax1.plot(within_wij11)
    # ax1.set_ylabel("Within pats")
    # ax1.set_title("Wij11")

    # ax2 = plt.subplot(2,2,2)
    # ax2.plot(within_wij22)
    # ax2.set_ylabel("Within pats")
    # ax2.set_title("Wij11")

    # ax3 = plt.subplot(2,2,3)
    # ax3.plot(between_wij11)
    # ax3.set_ylabel("Between pats")
    # ax3.set_title("Wij11")

    # ax4 = plt.subplot(2,2,4)
    # ax4.plot(between_wij11)
    # ax4.set_ylabel("Between pats")
    # ax4.set_title("Wij11")





    plt.gcf().set_size_inches(18, 13.5)
    #plt.savefig(figPATH+'Weights_Mat_ltm1FullNorm_ltm2HalfNorm.png',dpi=400)
    plt.show()


def plot_within_between_weights(parfilename=PATH+'olflangmain1.par',savef=0):


    linestyles = [
            'solid',
            'dashed',
            'dashdot',
         ((0, (2, 1))), #'dotted'
         #((0, (5, 1))), #'thick dashed',        
         # ((0, (2.5, 1))), #'medium dashed',                
         # ((0, (1, 1))),   #'thin dashed',        
         ((0, (2, 1, 1, 1, 1, 1))), #'dashdotdotted',         
         ]

    ODORS_en = ['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Terpentine',
            'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish']
    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N = H * M
    N2 = H2*M2

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
    trpats1 = Utils.loadbin("trpats1.bin",N)       
    trpats2 = Utils.loadbin("trpats2.bin",N2)    

    Wij11 = Utils.loadbin("Wij11.bin",N,N)
    Wij22 = Utils.loadbin("Wij22.bin",N,N)
    Wij21 = Utils.loadbin("Wij21.bin",N2,N)
    Wij12 = Utils.loadbin("Wij12.bin",N,N2)


    # fig,(ax1,ax2,ax3,ax4) = plt.subplots(2,2,figsize=(10,6))
    fig = plt.figure(figsize=(10,8))

    within_wij12,between_wij12 = get_assoc_w(Wij12,trpats1,trpats2)
    within_wij21,between_wij21 = get_assoc_w(Wij21,trpats2,trpats1)

    within_wij11,between_wij11 = get_weights_over_pattern(Wij11,trpats1)
    within_wij22,between_wij22 = get_weights_over_pattern(Wij22,trpats2)

    # within_wij11pre,between_wij11pre = get_weights_over_pattern(Wij11pre,trpats1)
    # within_wij22pre,between_wij22pre = get_weights_over_pattern(Wij22pre,trpats2)

    def connectpoints(y1,y2,x,c,ax):
        ax.plot([x,x],[y1,y2],color=c)

    ax1 = fig.add_subplot(221)
    ax1.plot(np.arange(trpats1.shape[0]),within_wij11.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='r')
    ax1.plot(np.arange(trpats1.shape[0]),between_wij11.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='b')
    ax1.set_xticks(np.arange(trpats1.shape[0]))
    for i in range(len(within_wij11.mean(axis=1))):
        connectpoints(within_wij11.mean(axis=1)[i],between_wij11.mean(axis=1)[i],i,kelly_colors[i],ax1)
    # ax1.set_ylabel("Within pats")
    yabs_max = abs(max(ax1.get_ylim(), key=abs))
    yabs_max = abs(max(ax1.get_ylim(), key=abs))
    ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax1.set_title("Wij11")

    ax2 = fig.add_subplot(222)
    ax2.plot(np.arange(trpats2.shape[0]),within_wij22.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='r')
    ax2.plot(np.arange(trpats2.shape[0]),between_wij22.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='b')
    ax2.set_xticks(np.arange(trpats1.shape[0]))
    for i in range(len(within_wij22.mean(axis=1))):
        connectpoints(within_wij22.mean(axis=1)[i],between_wij22.mean(axis=1)[i],i,kelly_colors[i],ax2)
    yabs_max = abs(max(ax2.get_ylim(), key=abs))
    yabs_max = abs(max(ax2.get_ylim(), key=abs))
    ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax2.set_title("Wij22")

    ax3 = fig.add_subplot(223)
    ax3.plot(np.arange(trpats1.shape[0]),within_wij21.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='r')
    ax3.plot(np.arange(trpats1.shape[0]),between_wij21.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='b')
    ax3.set_xticks(np.arange(trpats1.shape[0]))
    for i in range(len(within_wij21.mean(axis=1))):
        connectpoints(within_wij21.mean(axis=1)[i],between_wij21.mean(axis=1)[i],i,kelly_colors[i],ax3)
    yabs_max = abs(max(ax3.get_ylim(), key=abs))
    yabs_max = abs(max(ax3.get_ylim(), key=abs))
    ax3.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax3.set_title("Wij21")

    ax4 = fig.add_subplot(224)
    ax4.plot(np.arange(trpats2.shape[0]),within_wij12.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='r')
    ax4.plot(np.arange(trpats2.shape[0]),between_wij12.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='b')
    ax4.set_xticks(np.arange(trpats1.shape[0]))
    for i in range(len(within_wij12.mean(axis=1))):
        connectpoints(within_wij12.mean(axis=1)[i],between_wij12.mean(axis=1)[i],i,kelly_colors[i],ax4)
    yabs_max = abs(max(ax4.get_ylim(), key=abs))
    yabs_max = abs(max(ax4.get_ylim(), key=abs))
    ax4.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax4.set_title("Wij12")

    l = []
    l_labels = []
    for i in range(trpats1.shape[0]):
        l.append(Line2D([0], [0], color=kelly_colors[i], lw=4, ls=linestyles[int(i/trpats1.shape[0])]))
        l_labels.append('{}'.format(ODORS_en[i]))   ####NOTE: Need to change this when more patterns are used in lang net

    ax4.legend(l,l_labels,ncol=1,loc='center left',bbox_to_anchor=(1, 0.7),title='Pats',title_fontsize=10,fontsize=8)
    figPATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/DualNet/'
    if savef:
        plt.savefig(figPATH+'MeanPatWeights_recuweights',dpi=400)

    plt.show()


def plot_weights_over_time():
    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N = H * M
    N2 = H2*M2

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
    trpats1 = Utils.loadbin("trpats1.bin",N)       
    trpats2 = Utils.loadbin("trpats2.bin",N2)    

    Wij11 = Utils.loadbin("Wij11.bin",N,N)
    Wij22 = Utils.loadbin("Wij22.bin",N2,N2)

    within_wij11,between_wij11 = get_weights_over_pattern(Wij11,trpats1)
    within_wij22,between_wij22 = get_weights_over_pattern(Wij22,trpats2)

    ax1 = plt.subplot(1,2,1)
    ax1.plot(np.arange(trpats1.shape[0]),within_wij11.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='r')
    ax1.plot(np.arange(trpats1.shape[0]),between_wij11.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='b')
    # ax1.set_ylabel("Within pats")
    yabs_max = abs(max(ax1.get_ylim(), key=abs))
    yabs_max = abs(max(ax1.get_ylim(), key=abs))
    ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax1.set_title("Wij11")

    ax2 = plt.subplot(1,2,2)
    ax2.plot(np.arange(trpats1.shape[0]),within_wij22.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='r')
    ax2.plot(np.arange(trpats1.shape[0]),between_wij22.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='b')
    yabs_max = abs(max(ax2.get_ylim(), key=abs))
    yabs_max = abs(max(ax2.get_ylim(), key=abs))
    ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax2.set_title("Wij22")

    plt.show()

def get_weights_over_pattern(wij, patterns):
    
    # 10, 90?
    within  = np.zeros((patterns.shape[0],patterns.shape[1]))
    between = np.zeros((patterns.shape[0], wij.shape[0]*wij.shape[0]))
        
    
    for p in range(patterns.shape[0]):
        
        # get non zero values
        pat  = patterns[p]
        indx = np.nonzero(pat)[0]
        c1, c2 = 0, 0
        for i in indx:
            for j in range(patterns.shape[1]):
                if j in indx: 
                    within[p,c1] = wij[i,j] 
                    c1 += 1
                else:
                    between[p,c2] = wij[i,j] 
                    c2 += 1
            
    return within[:,:c1], between[:,:c2]    

def get_assoc_w(wij, pats1, pats2):
    
    # 10, 90?
    associated  = np.zeros((pats1.shape[0],pats1.shape[1]))
    unassociated = np.zeros((pats1.shape[0], wij.shape[0]*wij.shape[0]))
        
    
    for p in range(pats1.shape[0]):
        
        # get non zero values
        pat1  = pats1[p]
        pat2 = pats2[p]
        pat1_indx = np.nonzero(pat1)[0]
        pat2_indx = np.nonzero(pat2)[0]
        c1, c2 = 0, 0
        for i in pat1_indx:
            for j in range(pats1.shape[1]):
                if j in pat2_indx: 

                    # print("pat1: {} pat1_idx: {} pat2_idx: {}, w: {}".format(p,i,j,wij[i,j]))
                    associated[p,c1] = wij[i,j] 
                    c1 += 1
                else:
                    unassociated[p,c2] = wij[i,j] 
                    c2 += 1
            
    return associated[:,:c1], unassociated[:,:c2]    

def plot_patternwise_bias(parfilename=PATH+"olflangmain1.par"):
    
    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N = H * M
    N2 = H2*M2

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

    trpats = Utils.loadbin("trpats1.bin",N)

    trpats2 = Utils.loadbin("trpats2.bin",N2)

    Bj11 = Utils.loadbin("Bj11.bin",H,M)
    Bj22 = Utils.loadbin("Bj22.bin",H2,M2)
    Bj12 = Utils.loadbin("Bj12.bin",H2,M2)
    Bj21 = Utils.loadbin("Bj21.bin",H,M)

    Bj21 = Bj21.flatten()
    Bj12 = Bj12.flatten()
    Bj11 = Bj11.flatten()
    Bj22 = Bj22.flatten()

    p1 = np.zeros([trpats.shape[0],H])
    p2 = np.zeros([trpats2.shape[0],H2])

    for i,pat in enumerate(trpats): 
        p1[i] = np.where(pat>0)[0] 

    for i,pat in enumerate(trpats2): 
        p2[i] = np.where(pat>0)[0] 


    patwise_Bj11,patwise_Bj21 = np.zeros(p1.shape),np.zeros(p1.shape)
    patwise_Bj22,patwise_Bj12 = np.zeros(p2.shape),np.zeros(p2.shape)


    for i,pat in enumerate(p1):
        patwise_Bj11[i] = Bj11[pat.astype(int)]
        patwise_Bj21[i] = Bj21[pat.astype(int)]

    for i,pat in enumerate(p2):
        patwise_Bj22[i] = Bj22[pat.astype(int)]
        patwise_Bj12[i] = Bj12[pat.astype(int)]


    fig,ax= plt.subplots(2,1,figsize=(15,8))

    width = 0.4
    ax[0].bar(np.arange(patwise_Bj11.shape[0])-width/2,patwise_Bj11.mean(axis=1),width=width,color='tab:blue')
    ax[0].bar(np.arange(patwise_Bj21.shape[0])+width/2,patwise_Bj21.mean(axis=1),width=width,color='tab:red')
    ax[0].set_title('LTM1')
    ax[1].bar(np.arange(patwise_Bj22.shape[0])-width/2,patwise_Bj22.mean(axis=1),width=width,color='tab:blue')
    ax[1].bar(np.arange(patwise_Bj12.shape[0])+width/2,patwise_Bj12.mean(axis=1),width=width,color='tab:red')
    ax[1].set_title('LTM2')

    plt.show()

def plot_bias_vs_unitoverlap(parfilename = PATH+"olflangmain1.par"):
    '''
        Show the distribution of biases of units vs number of times a unit is activated during training (How much is it shared)
    '''
    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N = H * M
    N2 = H2*M2

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

    trpats = Utils.loadbin("trpats1.bin",N)

    trpats2 = Utils.loadbin("trpats2.bin",N2)

    Bj11 = Utils.loadbin("Bj11.bin",H,M)
    Bj22 = Utils.loadbin("Bj22.bin",H2,M2)
    Bj12 = Utils.loadbin("Bj12.bin",H2,M2)
    Bj21 = Utils.loadbin("Bj21.bin",H,M)

    Bj21 = Bj21.flatten()
    Bj12 = Bj12.flatten()

    p1 = np.zeros(trpats.shape[0]*H) 
    p2 = np.zeros(trpats2.shape[0]*H2)

    for i in range(trpats.shape[0]): 
        p1[i*H:(i+1)*H] = np.where(trpats[i] > 0)[0] 

    for i in range(trpats2.shape[0]): 
        p2[i*H2:(i+1)*H2] = np.where(trpats2[i] > 0)[0] 


    fig,ax=plt.subplots(1,2,figsize=(15,8),sharex=True)

    ax[0].hist(p1,bins=H*M,range=(0,H*M)) 
    ax[1].hist(p2,bins=H2*M2,range=(0,H2*M2))

    ax0_twin = ax[0].twinx()
    ax1_twin = ax[1].twinx()

    ax0_twin.plot(np.arange(N),Bj21,linestyle='None',markersize = 10.0,marker='o',mfc='tab:red') 
    ax1_twin.plot(np.arange(N2),Bj12,linestyle='None',markersize = 10.0,marker='o',mfc='tab:red')

    plt.show()








def plot_patoverlap_distribution(parfilename = PATH+"olflangmain1.par"):


    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))

    N = H * M

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
    trpats1 = Utils.loadbin("trpats1.bin",N)


    active_units = np.nonzero(trpats1)[1].reshape(trpats1.shape[0],H)
    #active_units = active_units%H

    results = np.zeros(active_units.shape)
    
    # count overlaps for each pattern in each Hc
    for col in range(active_units.shape[1]):
        unique, counts = np.unique(active_units[:,col], return_counts=True)
        for pi in range(active_units.shape[0]):
            indx = np.where(unique==active_units[pi,col])[0][0]
            results[pi,col] = counts[indx] - 1  # count other patterns (removing the current one)


    fig,ax = plt.subplots(int(trpats1.shape[0]/2),2, figsize=(6,10), sharex='all', sharey='all') 
    x = [i for i in range(15)]
    for i in range(active_units.shape[0]):
        u, c = np.unique(results[i], return_counts=True)
        y = [c[np.where(u==i)[0][0]] if i in u else 0 for i in x]
        ax[i%(8)][int(i/8)].bar(x=x,height=y)
    plt.show()

def pat_simmat(parfilename = PATH+"olflangmain1.par",langnet_pats_fname='',odnet_pats_fname='',mode='simmat'):
    '''
        Look at similarity between patterns
    '''
    H1 = int(Utils.findparamval(parfilename,"H"))
    M1 = int(Utils.findparamval(parfilename,"M"))

    N1 = H1 * M1

    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N2 = H2 * M2
    ODORS_en = ['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Terpentine',
            'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish']

    # descs_en = ODORS_en
    # descs_en = ['Gasoline', 'Turpentine', 'Shoe.Polish', 'Leather', 'Soap'  , 'Perfume' , 'Cinnamon' , 'Spice' , 'Vanilla' , 'Peppermint', 'Mentol', 'Mint',
    #             'Banana', 'Fruit', 'Apple', 'Lemon','Orange','Licorice', 'Anise', 'Pine.Needle', 'Garlic', 'Onion', 'Disgusting' , 'Coffee', 'Chocolate', 'Flower',
    #             'Clove', 'Dentist', 'Pineapple',    'Rose', 'Mushroom', 'Champinjon', 'Fish', 'Herring', 'Seafood']

    # descs_en = ['Gasoline', 'Leather', 'Cinnamon', 'Mint', 'Banana', 'Lemon', 'Licorice',
    #    'Shoe.Polish', 'Terpentine', 'Pine.Needle', 'Garlic', 'Onion', 'Coffee',
    #    'Chocolate', 'Apple', 'Fruit', 'Dentist', 'Spice', 'Clove',
    #    'Perfume', 'Rose', 'Flower', 'Mushroom', 'Fish']

    #si_4clusters
    descs_en = ['gasoline', 'turpentine', 'shoe polish', 'leather', 'soap', 'perfume', 'cinnamon', 'spice', 'vanilla', 'mint', 'banana',
    'fruit', 'lemon', 'orange', 'licorice', 'anise', 'pine needles', 'garlic', 'onion', 'disgusting', 'coffee', 'chocolate', 'apple',
    'flower', 'clove', 'dentist', 'pineapple', 'caramel', 'rose', 'mushroom', 'fish', 'herring', 'shellfish']

    #Si_4clusters_FilteredSNACK DATA
    # descs_en = ['bensin', 'terpentin', 'tjära', 'läder', 'tvål', 'parfym', 'skokräm', 'kanel', 'vanilj', 'blomma', 
    #             'mint', 'banan', 'frukt', 'citron', 'apelsin', 'citrus', 'lakrits', 'anis', 'krydda', 'tallbarr', 
    #             'vitlök', 'lök', 'kaffe', 'choklad', 'äpple', 'nejlika', 'kryddnejlika', 'tandläkare', 'ananas', 
    #             'jordgubbe', 'karamell', 'ros', 'svamp', 'fisk', 'sill', 'skaldjur']

    #si_2clusters
   #  descs_en = ['bensin', 'terpentin', 'skokräm', 'läder', 'tvål', 'parfym', 'krydda',
   # 'gummi', 'kemisk', 'kanel', 'vanilj', 'blomma', 'pepparmint', 'mentol',
   # 'mint', 'polkagris', 'banan', 'frukt', 'äpple', 'päron', 'citron',
   # 'apelsin', 'citrus', 'lakrits', 'anis', 'tallbarr', 'fernissa',
   # 'målarfärg', 'vitlök', 'lök', 'äcklig', 'kaffe', 'choklad', 'nejlika',
   # 'tandläkare', 'ananas', 'karamell', 'ros', 'svamp', 'champinjon',
   # 'fisk', 'sill', 'skaldjur', 'illa']

    ##Correct descs
    # descs_en = ['bensin', 'petroleum', 'bensinmack', 'diesel', 'läder', 
    # 'skokräm', 'skinn', 'kanel', 'kanelbulle', 'pepparmint', 'mint', 'mentol', 
    # 'banan', 'skumbanan', 'citron', 'citrus', 'lime', 'citronmeliss', 'lakrits', 'anis', 
    # 'salmiak', 'saltlakrits', 'terpentin', 'fernissa', 'målarfärg', 'lösningsmedel', 'vitlök', 
    # 'lök', 'stekt.lök', 'purjolök', 'kaffe', 'kaffesump', 'snabbkaffe', 'kaffeböna', 'äpple', 'tandläkare', 
    # 'nejlika', 'kryddnejlika', 'sjukhus', 'ananas', 'ros', 'rosenvatten', 'rosenolja', 'svamp', 'champinjon', 
    # 'kantarell', 'mögelsvamp', 'fisk', 'sill', 'skaldjur', 'räka']

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
    trpats1 = Utils.loadbin(buildPATH+"trpats1.bin",N1)
    trpats2 = Utils.loadbin(buildPATH+"trpats2.bin",N2)

    p1 = np.zeros(trpats1.shape[0]*H1) 
    p2 = np.zeros(trpats2.shape[0]*H2)

    for i in range(trpats1.shape[0]): 
        p1[i*H1:(i+1)*H1] = np.where(trpats1[i] > 0)[0]

    for i in range(trpats2.shape[0]): 
        p2[i*H2:(i+1)*H2] = np.where(trpats2[i] > 0)[0]

    p1=p1.reshape(trpats1.shape[0],H1)
    p2=p2.reshape(trpats2.shape[0],H2)

    if langnet_pats_fname:
        fname1 = "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/"+langnet_pats_fname
        with open(fname1) as f:
            p1 = json.load(f)
        p1 = np.array(p1)
        print('Loading '+langnet_pats_fname)
    else:
        print('Loading trpats1.bin')

    if odnet_pats_fname:
        fname2 = "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/"+odnet_pats_fname
        with open(fname2) as f:
            p2 = json.load(f)
        p2 = np.array(p2)
        print('Loading '+odnet_pats_fname)
    else:
        print('Loading trpats2.bin')


    # fname2 = "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_npat16_emax20_H15_M15_dampen_factor2.json"          
    # with open(fname1) as f:
    #     p1 = json.load(f)  

    # with open(fname2) as f:
    #     p2 = json.load(f)  

    # p1 = np.array(p1)
    # p2 = np.array(p2)

    if mode=='simmat':
        # path = '/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/'
        # filename = path+'w2v_topdescs05thresh_D0_npat24_emax06_H20_M20_v2_sigmoidTransformed(beta=5 gamma=10).json'
        # with open(filename) as f:
        #     p1 = json.load(f)
        # p1 = np.array(p1)

        # filename = path+'odors_D0_npat16_emax06_H20_M20_v2_sigmoidTransformed(beta=5 gamma=11).json'
        # with open(filename) as f:
        #     p2 = json.load(f)
        # p2 = np.array(p2)    
            
        p1_simmat = np.zeros([p1.shape[0],p1.shape[0]])
        for i in range(p1.shape[0]):
            for j in range(p1.shape[0]):
                p1_simmat[i,j]=np.count_nonzero(p1[i]==p1[j])

        p2_simmat = np.zeros([p2.shape[0],p2.shape[0]])
        for i in range(p2.shape[0]):
            for j in range(p2.shape[0]):
                p2_simmat[i,j]=np.count_nonzero(p2[i]==p2[j])

        fig,ax1 = plt.subplots(1,1,figsize=(20,10))
        plot1 = sns.heatmap(p1_simmat,ax=ax1,annot=True,xticklabels=descs_en,yticklabels=descs_en)
        plt.xticks(rotation=75)
        ax1.set_title("LTM1 (lang) pats")
        fig.tight_layout()
        plt.show()

        fig,ax2 = plt.subplots(1,1,figsize=(10,10))
        plot2 = sns.heatmap(p2_simmat,ax=ax2,annot=True,xticklabels=ODORS_en,yticklabels=ODORS_en)
        ax2.set_title("LTM2 (od) pats")
        plt.show()

    if mode=='simmat_w2v':
        p1_simmat = np.zeros([p1.shape[0],p1.shape[0]])
        for i in range(p1.shape[0]):
            for j in range(p1.shape[0]):
                print(i,j)
                p1_simmat[i,j]=np.count_nonzero(p1[i]==p1[j])

        w2v_distmat = pd.read_csv('/home/rohan/Documents/Olfaction/W2V/BloggModel_CriticalWords_similarity_sorted(SNACK).csv',index_col=0)

        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,10))

        plot1 = sns.heatmap(p1_simmat,ax=ax1,annot=True,xticklabels=ODORS_en,yticklabels=ODORS_en)
        ax1.set_title("LTM1 (lang) pats")

        plot2 = sns.heatmap(w2v_distmat,ax=ax2)
        ax2.set_title("W2V Cos Sim")

    if mode=='unit_overlaps':
        unit_overlaps1 = np.zeros(p1.shape)
        unit_overlaps2 = np.zeros(p2.shape)


        for i,pat in enumerate(p1):
            for j,unit in enumerate(pat): 
                 unit_overlaps1[i,j] = np.count_nonzero(p1[:,j]==unit)

        for i,pat in enumerate(p2):
            for j,unit in enumerate(pat): 
                 unit_overlaps2[i,j] = np.count_nonzero(p2[:,j]==unit)


        fig,ax = plt.subplots(8,7,figsize=(15,15))
        for i in range(8):
            for j in range(7):
                if i*8+j>=p1.shape[0]:
                    ax[i][j].axis('off')
                    continue
                ax[i][j].bar(np.arange(len(p1[0])),unit_overlaps1[i*8+j])
                ax[i][j].set_title("pat {}".format((i*8+j)+1),size=12)
                #ax[i][j].set_ylim([0,16])

        fig.suptitle('LTM1 (lang) pats')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.77)

        fig2,ax2 = plt.subplots(8,2,figsize=(6,15))
        for i in range(8):
            for j in range(2):
                ax2[i][j].bar(np.arange(len(p2[0])),unit_overlaps2[j*8+i])
                ax2[i][j].set_title("pat {}".format((j*8+i)+1))
                ax2[i][j].set_ylim([0,16])

        fig2.suptitle('LTM2 (od) pats')
        fig2.tight_layout()
        fig2.subplots_adjust(hspace=0.77)
        plt.show()

    if mode=='overlap_histogram':

        p1_simmat = np.zeros([p1.shape[0],p1.shape[0]])
        for i in range(p1.shape[0]):
            for j in range(p1.shape[0]):
                p1_simmat[i,j]=np.count_nonzero(p1[i]==p1[j])

        p2_simmat = np.zeros([p2.shape[0],p2.shape[0]])
        for i in range(p2.shape[0]):
            for j in range(p2.shape[0]):
                    p2_simmat[i,j]=np.count_nonzero(p2[i]==p2[j])

        p1_squareform = p1_simmat.max()-distance.squareform(p1_simmat.max()-p1_simmat)
        p2_squareform = p2_simmat.max()-distance.squareform(p2_simmat.max()-p2_simmat) 

        p1_overlap_counts = np.unique(p1_squareform,return_counts=True)
        p2_overlap_counts = np.unique(p2_squareform,return_counts=True)


        print(p1_overlap_counts)

        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,16),sharex=True)
        # ax1.hist(p1_simmat.flatten(),align='left',color='tab:blue',linewidth=0.75,edgecolor='k')
        # ax2.hist(p2_simmat.flatten(),bins=np.linspace(0,H2+1,H2),align='left',color='tab:orange',linewidth=0.75,edgecolor='k')
        # fig.supxlabel('Overlaps',size=14)
        # #ax2.set_xlabel('Overlaps')
        # # ax1.set_xticks(np.arange(1,H1+1))
        # # ax1.set_xticklabels(np.arange(1,H1+1))
        # # ax2.set_xticks(np.arange(H2))
        # # ax2.set_xticklabels(np.arange(H2))
        # ax1.set_ylabel('Count',size=14)
        # ax2.set_ylabel('Count',size=14)

        ax1.bar(p1_overlap_counts[0],p1_overlap_counts[1],color='tab:blue')
        ax2.bar(p2_overlap_counts[0],p2_overlap_counts[1],color='tab:orange')

        fig.supxlabel('Overlapping Units',size=14)
        ax1.set_xticks(np.arange(0,p1_overlap_counts[0].max()+1))
        #ax2.set_xticks(np.arange(0,p2_overlap_counts[0].max()+1))
        ax1.set_ylabel('Count',size=14)
        ax2.set_ylabel('Count',size=14)
        ax1.set_title('Lang Pats (LTM1), H: {}, M:{}'.format(H1,M1),size=16)
        ax2.set_title('Od Pats (LTM2), H:{}, M:{}'.format(H2,M2),size=16)

        ax1.axvline(x=p1_squareform.mean(),color='tab:red', linestyle='dashed', linewidth=1.5,label='mean')
        ax2.axvline(x=p2_squareform.mean(),color='tab:red', linestyle='dashed', linewidth=1.5,label='mean')
        ax1.set_xlim([-0.5,p1_overlap_counts[0].max()+1])
        # ax2.set_xlim([-0.5,p2_overlap_counts[0].max()+1])
        ax1.tick_params(labelsize=14)
        ax2.tick_params(labelsize=14)
        ax1.legend()
        ax2.legend()
        print(p1_squareform.mean(),p2_squareform.mean())
        
        plt.subplots_adjust(hspace=0.21,bottom=0.06)
        #plt.tight_layout()
        #plt.savefig(figPATH+'Pat_Overlap_Distribution',dpi=400)
        plt.show()


    elif mode=='odorpats_overlapdist':
        odor_overlaps = [[] for i in range(p2.shape[0])]

        for i,pat1 in enumerate(p2):
            for j,pat2 in enumerate(p2):
                if i == j:
                    continue
                else:

                    odor_overlaps[i].append(np.count_nonzero(p2[i]==p2[j]))

        fig,ax = plt.subplots(4,4,figsize=(12,12))
        bins = np.arange(10) - 0.5
        for i in range(4):
            for j in range(4):
                od = 4*i+j
                ax[i][j].hist(odor_overlaps[od],bins,range=(0,10),edgecolor='k')
                ax[i][j].set_xticks(np.arange(10))
                ax[i][j].set_title(ODORS_en[od])
                ax[i][j].set_ylim([0,16])
                ax[i][j].set_xlim([0,10])

        fig.suptitle('LTM2 (od) pats')
        fig.tight_layout()
        #fig2.subplots_adjust(hspace=0.77)

        plt.show()

    elif mode=='langpats_overlapdist':
        lang_overlaps = [[] for i in range(p1.shape[0])]
        
        for i,pat1 in enumerate(p1):
            for j,pat2 in enumerate(p1):
                if i == j:
                    continue
                else:

                    lang_overlaps[i].append(np.count_nonzero(p1[i]==p1[j]))

        fig,ax = plt.subplots(8,8,figsize=(12,12))
        bins = np.arange(10) - 0.5

        for i in range(8):
            for j in range(8):
                od = 8*i+j

                if od>=p1.shape[0]:
                    ax[i][j].axis('off')
                    continue

                if od==40:
                    print(descs_en[od],p1.shape[0])

                ax[i][j].hist(lang_overlaps[od],bins,range=(0,10),edgecolor='k')
                ax[i][j].set_xticks(np.arange(10))
                ax[i][j].set_title(descs_en[od])
                ax[i][j].set_ylim([0,p1.shape[0]])
                ax[i][j].set_xlim([-0.5,10])

        fig.suptitle('LTM1 (lang) pats')
        fig.tight_layout()
        #fig2.subplots_adjust(hspace=0.77)

        plt.show()


def combinedattractor_distmat(parfilename=PATH+'olflangmain1.par'):
    '''
     Combine ltm1 and ltm2 attractors and show distance matrix
     Works only when npats in both nets is same
    '''
    H = int(Utils.findparamval(parfilename,"H"))
    M = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N = H * M
    N2 = H2*M2

    ODORS_en = ['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Turpentine',
            'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish']

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

    trpats1 = Utils.loadbin("trpats1.bin",N)

    trpats2 = Utils.loadbin("trpats2.bin",N2)

    combined_pats = np.concatenate([trpats1,trpats2],axis=1)

    simmat = np.zeros([trpats1.shape[0],trpats2.shape[0]])


    for i in range(trpats1.shape[0]):
        for j in range(trpats2.shape[0]):
            simmat[i][j] = np.dot(combined_pats[i],combined_pats[j]) / (np.linalg.norm(combined_pats[i])*np.linalg.norm(combined_pats[j]))

    fig,ax1 = plt.subplots(1,1,figsize=(8,8))
    
    plot1 = sns.heatmap(simmat,ax=ax1,xticklabels=ODORS_en,yticklabels=ODORS_en)
    
    ax1.set_title("Combined label-percept pats Cos Sim")
    plt.xticks(rotation=45)

    plt.show()

def plot_wdist(savef=0,savefname='',parfilename=PATH+'olflangmain1.par',figPATH='/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/DualNet/16od_16labs/taum'):
    '''
    Plot distribution of weights (exc and inh)
    '''
    H1 = int(Utils.findparamval(parfilename,"H"))
    M1 = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N = H1 * M1
    N2 = H2*M2

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
    trpats1 = Utils.loadbin("trpats1.bin",N)       
    trpats2 = Utils.loadbin("trpats2.bin",N2)    

    Wij11 = Utils.loadbin("Wij11.bin",N,N)
    Wij12 = Utils.loadbin("Wij12.bin",N,N2)
    Wij22 = Utils.loadbin("Wij22.bin",N2,N2)
    Wij21 = Utils.loadbin("Wij21.bin",N2,N)

    p1 = np.zeros(trpats1.shape[0]*H1) 
    p2 = np.zeros(trpats2.shape[0]*H2)

    for i in range(trpats1.shape[0]): 
        p1[i*H1:(i+1)*H1] = np.where(trpats1[i] > 0)[0]

    for i in range(trpats2.shape[0]): 
        p2[i*H2:(i+1)*H2] = np.where(trpats2[i] > 0)[0]

    p1=p1.reshape(trpats1.shape[0],H1)
    p2=p2.reshape(trpats2.shape[0],H2)

    w11_exc,w22_exc,w12_exc,w21_exc = [],[],[],[]
    w11_inh,w22_inh,w12_inh,w21_inh = [],[],[],[]

    for i in range(trpats1.shape[0]):
        pat1 = p1[i]
        for j in pat1.astype(int):
            for k in range(trpats1.shape[1]):
                if k in pat1:
                    w11_exc.append(Wij11[j,k])
                else:
                    w11_inh.append(Wij11[j,k])

    for i in range(trpats2.shape[0]):
        pat2 = p2[i]
        for j in pat2.astype(int):
            for k in range(trpats2.shape[1]):
                if k in pat2:
                    w22_exc.append(Wij22[j,k])
                else:
                    w22_inh.append(Wij22[j,k])


    #patstat = np.loadtxt('/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_16od_16descs.txt')
    # patstat = np.loadtxt('/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_topdescs5e-01thresh.txt')
    for i,(ltm1_pat,ltm2_pat) in enumerate(patstat.astype(int)):
        pat1 = p1[ltm1_pat]
        pat2 = p2[ltm2_pat]

        for j in pat1.astype(int):
            for k in range(trpats2.shape[1]):
                if k in pat2: 

                    # print("pat1: {} pat1_idx: {} pat2_idx: {}, w: {}".format(p,i,j,wij[i,j]))
                    w12_exc.append(Wij12[j,k])                     
                else:
                    w12_inh.append(Wij12[j,k]) 

    for i,(ltm1_pat,ltm2_pat) in enumerate(patstat.astype(int)):
        pat1 = p1[ltm1_pat]
        pat2 = p2[ltm2_pat]

        for j in pat2.astype(int):
            for k in range(trpats1.shape[1]):
                if k in pat1: 

                    # print("pat1: {} pat1_idx: {} pat2_idx: {}, w: {}".format(p,i,j,wij[i,j]))
                    w21_exc.append(Wij21[j,k])                     
                else:
                    w21_inh.append(Wij21[j,k]) 


    w11_exc = np.array(w11_exc)
    w21_exc = np.array(w21_exc)
    w12_exc = np.array(w12_exc)
    w22_exc = np.array(w22_exc)
    w11_inh = np.array(w11_inh)
    w21_inh = np.array(w21_inh)
    w12_inh = np.array(w12_inh)
    w22_inh = np.array(w22_inh)

    fig,ax = plt.subplots(2,2,figsize=(8,8))

    ax[0,0].hist(w11_exc,bins=50,alpha=0.5,label='Exc',color='tab:red',density=True,stacked=True)
    ax[0,0].hist(w11_inh,bins=50,alpha=0.5,label='Inh',color='tab:blue',density=True,stacked=True)  
    ax[0,1].hist(w22_exc,bins=50,alpha=0.5,label='Exc',color='tab:red',density=True,stacked=True)
    ax[0,1].hist(w22_inh,bins=50,alpha=0.5,label='Inh',color='tab:blue',density=True,stacked=True) 
    ax[1,0].hist(w12_exc,bins=50,alpha=0.5,label='Exc',color='tab:red',density=True,stacked=True)
    ax[1,0].hist(w12_inh,bins=50,alpha=0.5,label='Inh',color='tab:blue',density=True,stacked=True) 
    ax[1,1].hist(w21_exc,bins=50,alpha=0.5,label='Exc',color='tab:red',density=True,stacked=True)
    ax[1,1].hist(w21_inh,bins=50,alpha=0.5,label='Inh',color='tab:blue',density=True,stacked=True) 

    ax[0,0].set_title('Wij11')
    ax[0,1].set_title('Wij22')
    ax[1,0].set_title('Wij12')
    ax[1,1].set_title('Wij21')
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    # plt.show()
    if savef:
        plt.savefig(savefname,dpi=300)
    else:
        plt.show()
    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/')

def generate_alternatives(parfilename = PATH+"olflangmain1.par"):
    '''
        For the 16 odors, generate 4 alternatives (3 wrong, 1 correct) from the other 16 odors
    '''

    H1 = int(Utils.findparamval(parfilename,"H"))
    M1 = int(Utils.findparamval(parfilename,"M"))

    N1 = H1 * M1

    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N2 = H2 * M2
    ODORS_en = ['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Terpentine',
            'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish']
    # descs_en = ['Gasoline', 'Turpentine', 'Shoe.Polish', 'Leather', 'Soap'  , 'Perfume' , 'Cinnamon' , 'Spice' , 'Vanilla' , 'Peppermint', 'Mentol', 'Mint',
    #             'Banana', 'Fruit', 'Apple', 'Lemon','Orange','Licorice', 'Anise', 'Pine.Needle', 'Garlic', 'Onion', 'Disgusting' , 'Coffee', 'Chocolate', 'Flower',
    #             'Clove', 'Dentist', 'Pineapple',    'Rose', 'Mushroom', 'Champinjon', 'Fish', 'Herring', 'Seafood']

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
    trpats1 = Utils.loadbin("trpats1.bin",N1)
    trpats2 = Utils.loadbin("trpats2.bin",N2)

    p1 = np.zeros(trpats1.shape[0]*H1) 
    p2 = np.zeros(trpats2.shape[0]*H2)

    for i in range(trpats1.shape[0]): 
        p1[i*H1:(i+1)*H1] = np.where(trpats1[i] > 0)[0]

    for i in range(trpats2.shape[0]): 
        p2[i*H2:(i+1)*H2] = np.where(trpats2[i] > 0)[0]

    p1=p1.reshape(trpats1.shape[0],H1)
    p2=p2.reshape(trpats2.shape[0],H2)
    p1_simmat = np.zeros([p1.shape[0],p1.shape[0]])
    for i in range(p1.shape[0]):
        for j in range(p1.shape[0]):
            p1_simmat[i,j]=np.count_nonzero(p1[i]==p1[j])

    p2_simmat = np.zeros([p2.shape[0],p2.shape[0]])
    for i in range(p2.shape[0]):
        for j in range(p2.shape[0]):
            p2_simmat[i,j]=np.count_nonzero(p2[i]==p2[j])

    for i,overlaps in enumerate(p1_simmat):
        od = ODORS_en[i]
        #least_overlaps = np.where(overlaps<=20/100*H1)[0] #select least sim alternative from under 15% overlap labels
        mid_overlaps = np.where(((overlaps>20/100*H1)&(overlaps<=30/100*H1)))[0]
        #most_overlaps = np.where(((overlaps>30/100*H1)&(overlaps<=60/100*H1)))[0]
        self_ = i

        #print(od,least_overlaps,mid_overlaps,most_overlaps)
        #sample 1 randomly from least,mid and most overlaps
        least = overlaps.tolist().index(overlaps.min())#np.random.choice(least_overlaps)
        mid = np.random.choice(mid_overlaps)
        most = overlaps.argsort()[-2] #np.random.choice(most_overlaps)

        #print('{}: {},{},{},{}'.format(ODORS_en[i],ODORS_en[least],ODORS_en[mid],ODORS_en[most],ODORS_en[self_]))
        print([self_,least,mid,most])

def plot_heats(d1,d2,lab1,lab2):
    '''
        plot heatmaps for 2 distance matrices to compare
    '''
    fig,ax = plt.subplots(1,2)
    sns.heatmap(d1,ax=ax[0],annot=True,annot_kws={"size": 8})
    sns.heatmap(d2,ax=ax[1],annot=True,annot_kws={"size": 8})
    ax[0].set_title(lab1)
    ax[1].set_title(lab2)
    plt.show()

def plot_dists(d1,d2,lab1,lab2):
    '''
        plot distributions fo 2 distance matrices to compare
    '''
    d1 = d1.to_numpy()[np.triu_indices_from(d1,k=1)]   
    d2 = d2.to_numpy()[np.triu_indices_from(d2,k=1)]   
    df = pd.DataFrame(zip(d1,d2),columns = [lab1,lab2])
    sns.displot(data=df,bins=50)
    plt.show()


def visualise_associations(savef=0):
    '''
        Visualise associations between the two networks
    '''
    #fname = PATH+"patstat_16od_16descs.txt"   ###Change when using different assoc file
    a = patstat #np.loadtxt(fname)

    if a.shape[1]==3:
        a = np.delete(a,2,1)

    ODORS_en = ['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Turpentine',
            'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish']

    #descs for 0.5 thresh in extract_top_descs
    descs_en1 = ['Gasoline', 'Leather', 'Cinnamon', 'Mint', 'Banana', 'Lemon', 'Licorice',
         'Shoe.Polish', 'Terpentine', 'Pine.Needle', 'Garlic', 'Onion', 'Coffee',
         'Chocolate', 'Apple', 'Fruit', 'Dentist', 'Spice', 'Clove',
         'Perfume', 'Rose', 'Flower', 'Mushroom', 'Fish']

    #descs for 0.33 thresh
    descs2 = ['bensin', 'läder', 'tvål', 'parfym', 'kanel', 'mint', 'banan',
       'frukt', 'äpple', 'citron', 'lakrits', 'terpentin', 'skokräm',
       'tallbarr', 'vitlök', 'lök', 'kaffe', 'choklad', 'nejlika',
       'krydda', 'tandläkare', 'ros', 'blomma', 'svamp', 'fisk', 'sill',
       'skaldjur']

    descs_en2 = ['Gasoline','Leather','Soap','Perfume','Cinnamon','Mint','Banana','Fruit','Apple','Lemon','Licorice',
                'Turperntine','Shoe.Polish','Pine.Needle','Garlic','Onion','Coffee','Chocolate','Clove','Spice','Dentist',
                'Rose','Flower','Mushroom','Fish','Herring','Seafood']

    #si_4clusters
    descs_en3 = ['gasoline', 'turpentine', 'shoe polish', 'leather', 'soap', 'perfume', 'cinnamon', 'spice', 'vanilla', 'mint', 'banana',
    'fruit', 'lemon', 'orange', 'licorice', 'anise', 'pine needles', 'garlic', 'onion', 'disgusting', 'coffee', 'chocolate', 'apple',
    'flower', 'clove', 'dentist', 'pineapple', 'caramel', 'rose', 'mushroom', 'fish', 'herring', 'shellfish']


    #si_2clusters
    descs_en4 = ['bensin', 'terpentin', 'skokräm', 'läder', 'tvål', 'parfym', 'krydda',
       'gummi', 'kemisk', 'kanel', 'vanilj', 'blomma', 'pepparmint', 'mentol',
       'mint', 'polkagris', 'banan', 'frukt', 'äpple', 'päron', 'citron',
       'apelsin', 'citrus', 'lakrits', 'anis', 'tallbarr', 'fernissa',
       'målarfärg', 'vitlök', 'lök', 'äcklig', 'kaffe', 'choklad', 'nejlika',
       'tandläkare', 'ananas', 'karamell', 'ros', 'svamp', 'champinjon',
       'fisk', 'sill', 'skaldjur', 'illa']

    ##correct_descs_maxassoc4
    descs_en5 = ['bensin', 'petroleum', 'bensinmack', 'diesel', 'läder', 
    'skokräm', 'skinn', 'kanel', 'kanelbulle', 'pepparmint', 'mint', 'mentol', 
    'banan', 'skumbanan', 'citron', 'citrus', 'lime', 'citronmeliss', 'lakrits', 'anis', 
    'salmiak', 'saltlakrits', 'terpentin', 'fernissa', 'målarfärg', 'lösningsmedel', 'vitlök', 
    'lök', 'stekt.lök', 'purjolök', 'kaffe', 'kaffesump', 'snabbkaffe', 'kaffeböna', 'äpple', 'tandläkare', 
    'nejlika', 'kryddnejlika', 'sjukhus', 'ananas', 'ros', 'rosenvatten', 'rosenolja', 'svamp', 'champinjon', 
    'kantarell', 'mögelsvamp', 'fisk', 'sill', 'skaldjur', 'räka']
    

    assocs_dict={}
    for key,value in a:
        if value not in assocs_dict:
            assocs_dict[int(value)]=[int(key)]
        else:
            assocs_dict[int(value)].append(int(key))

    print(assocs_dict)
    fig,ax= plt.subplots(figsize=(20,8))
    gap = 1
    for i,(key,vals) in enumerate(assocs_dict.items()):
        x = np.mean(vals)
        indent = len(a[:,1])/5
        ax.scatter(key+indent,1,color=kelly_colors[key],s=200,alpha=0.6)
        ax.annotate(ODORS_en[key],xy=(key+indent,1.1),rotation=45,rotation_mode='anchor',size=14)
        for v in vals:
            #print('key: {}, v:{}'.format(key,v))
            ax.scatter(v,0,color=kelly_colors[key],s=200,alpha=0.6)
            ax.plot([v,key+indent],[0,1],color=kelly_colors[key],alpha=0.6)
            ax.annotate(descs_en3[v],xy=(v,-.1),rotation=-45,rotation_mode='anchor',size=14)

    ax.annotate('ODORS',xy=(a[:,1].mean()+indent,1.5),weight='heavy',size=16,ha='center')
    ax.annotate('DESCRIPTORS',xy = (a[:,1].mean()+indent,-.5),weight='heavy',size=16,ha='center')
    ax.set_ylim(-1,2)
    ax.set_xlim(-1,(np.array(a).max()+1))
    plt.axis('off')
    fig.tight_layout()
    plt.show()

def plot_multi_association_weights(odor=''):
    '''
    Plot weight distribution between shared associations
    '''
    parfilename=PATH+'olflangmain1.par'
    H1 = int(Utils.findparamval(parfilename,"H"))
    M1 = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N1 = H1 * M1
    N2 = H2*M2

    ODORS_en = np.array(['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Turpentine',
            'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish'])

    descs_en = ['Gasoline', 'Leather', 'Cinnamon', 'Mint', 'Banana', 'Lemon', 'Licorice',
         'Shoe.Polish', 'Terpentine', 'Pine.Needle', 'Garlic', 'Onion', 'Coffee',
         'Chocolate', 'Apple', 'Fruit', 'Dentist', 'Spice', 'Clove',
         'Perfume', 'Rose', 'Flower', 'Mushroom', 'Fish']

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
    trpats1 = Utils.loadbin("trpats1.bin",N1)       
    trpats2 = Utils.loadbin("trpats2.bin",N2)    

    p1 = np.zeros(trpats1.shape[0]*H1).astype(int) 
    p2 = np.zeros(trpats2.shape[0]*H2).astype(int)

    for i in range(trpats1.shape[0]): 
        p1[i*H1:(i+1)*H1] = np.where(trpats1[i] > 0)[0]

    for i in range(trpats2.shape[0]): 
        p2[i*H2:(i+1)*H2] = np.where(trpats2[i] > 0)[0]

    p1=p1.reshape(trpats1.shape[0],H1)
    p2=p2.reshape(trpats2.shape[0],H2)


    Wij11 = Utils.loadbin("Wij11.bin",N1,N1)
    Wij22 = Utils.loadbin("Wij22.bin",N2,N2)
    Wij21 = Utils.loadbin("Wij21.bin",N2,N1)
    Wij12 = Utils.loadbin("Wij12.bin",N1,N2)

    Bj11 = Utils.loadbin("Bj11.bin")
    Bj22 = Utils.loadbin("Bj22.bin")
    Bj12 = Utils.loadbin("Bj12.bin")
    Bj21 = Utils.loadbin("Bj21.bin")

    def get_excitatory_weights(W,descpat,odpat,net='21'):
        wlist = []
        if net == '11':
            pat1_id = descpat
            pat2_id = descpat
            pat1 = p1
            pat2 = p1
        elif net == '22':
            pat1_id = odpat
            pat2_id = odpat
            pat1 = p2
            pat2 = p2   
        elif net == '12':
            pat1_id = descpat
            pat2_id = odpat
            pat1 = p1
            pat2 = p2 
        elif net == '21':
            pat1_id = odpat
            pat2_id = descpat
            pat1 = p2
            pat2 = p1  

        for i,pre_mc in enumerate(pat1[pat1_id]):
            for j,post_mc in enumerate(pat2[pat2_id]):
                wlist.append(W[pre_mc,post_mc])

        return (np.array(wlist))

    def get_bias_distribution(pat1,post_net='2'):
        if pat1.ndim == 1:
            if post_net == '2':
                blist = [Bj12[unit] for unit in pat1]
            else:
                blist = [Bj21[unit] for unit in pat1]
        else:
            if post_net == '1':
                blist = [[Bj11[unit] for unit in units] for units in pat1]
            elif post_net == '2':
                blist = [[Bj22[unit] for unit in units] for units in pat1]
        return blist

    fname = PATH+'patstat_topdescs5e-1thresh.txt'   ###Change when using different assoc file
    assocs = np.loadtxt(fname).astype(int)    
    assocs_pd = pd.Series(assocs[:,1],index=assocs[:,0])

    if odor: #plot for specific odor
        oid = np.where(ODORS_en==odor)[0][0]

        if type(assocs_pd[oid]) == np.int64:
            associated_descs = [assocs_pd[oid]]
        else:
            associated_descs = assocs_pd[oid].to_list()

        fig,ax = plt.subplots(1,1,figsize=(15,8))

        fig2,ax2 = plt.subplots(1,1,figsize=(15,8))

        assoc_colors = ['tab:blue','tab:green','tab:orange','tab:red']
        for i,desc_id in enumerate(associated_descs):
            w = get_excitatory_weights(Wij21,desc_id,oid,net='21')
            b = get_bias_distribution(p2[oid],post_net='2')
            sns.histplot(w,bins=50,color = assoc_colors[i],alpha=0.7,label = descs_en[desc_id],ax=ax)
            sns.histplot(b,bins=50,color = assoc_colors[i],alpha=0.7,label = descs_en[desc_id],ax=ax2)

        plt.legend()
        fig.suptitle('Weight_Distribution')
        fig2.suptitle('Bias Distribution')
        plt.show()


    else:

        #Most number of od to lang associations
        max_assoc = Counter(assocs[:,0]).most_common()[0][1] 

        assocwise_weights = [[] for i in range(max_assoc)]

        assocwise_bias_ltm2 = [[] for i in range(max_assoc)]

        for oid in range(len(ODORS_en)):
            if type(assocs_pd[oid]) == np.int64:
                associated_descs = [assocs_pd[oid]]
            else:
                associated_descs = assocs_pd[oid].to_list()

            nassocs = len(associated_descs)
            print(nassocs,oid)
            b2 = get_bias_distribution(p2[oid],post_net='2')
            assocwise_bias_ltm2[nassocs-1].extend(b2)
            for i,desc_id in enumerate(associated_descs):
                w = get_excitatory_weights(Wij21,desc_id,oid,'21')
                


                assocwise_weights[nassocs-1].extend(w)

        print(len(assocwise_bias_ltm2[0]))

        assoc_colors = ['tab:blue','tab:green','tab:orange','tab:red']

        fig,ax = plt.subplots(1,1,figsize=(15,8))
        fig2,ax2 = plt.subplots(1,1,figsize=(15,8))
        for i in range(max_assoc):
            sns.histplot(assocwise_weights[i],color=assoc_colors[i],bins=75,alpha=0.8,label = '{} shared assocs'.format(i+1),ax=ax)
            ax.axvline(np.array(assocwise_weights[i]).mean(), color=assoc_colors[i], linestyle='dashed', linewidth=1)
            sns.histplot(assocwise_bias_ltm2[i],color=assoc_colors[i],bins=75,alpha=0.8,label = '{} shared assocs'.format(i+1),ax=ax2)

        # post_net = '2'
        # patwise_Bj = np.array(get_bias_distribution(p2,post_net=post_net))
        

        # for i in range(patwise_Bj.shape[0]):
        #     if post_net == '1':
        #         sns.histplot(patwise_Bj[i],bins=50,color = kelly_colors[assocs[np.where(assocs[:,1]==i)[0][0],0]],alpha=0.7,label = descs_en[i],ax=ax2)
        #     else:
        #         sns.histplot(patwise_Bj[i],bins=50,color = kelly_colors[i],alpha=0.7,label = ODORS_en[i],ax=ax2,kde=True)
        fig.suptitle('Associative Weight Distribution')
        fig2.suptitle('Bias Distribution')
        ax.legend()
        ax2.legend()
        plt.show()


def get_colors(c,assocs,delta=10):
    '''
        helper function to take hex color of odor, convert to hsv and change V by delta % to color associations
    ''' 
    rgb = Utils.hex2rgb(c)
    hsv = colorsys.rgb_to_hsv(rgb[0]/255,rgb[1]/255,rgb[2]/255)


    cols = [colorsys.hsv_to_rgb(hsv[0],hsv[1],hsv[2])]
    v = hsv[0]

    if assocs == 0:
        return([hsv[0],hsv[1],v-(delta*v/100)])

    for i in range(assocs):
        v -= delta*v/100
        # col = [hsv[0],hsv[1],v]
        col = [v,hsv[1],hsv[2]]
        col = list(colorsys.hsv_to_rgb(col[0],col[1],col[2]))
        
        cols.append(col)


    return cols

def plot_weights_assocwise(mode='group_by_nassocs'):
    '''
    plot weights by number of associations or for each odor
    '''
    parfilename=PATH+'olflangmain1.par'
    H1 = int(Utils.findparamval(parfilename,"H"))
    M1 = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N1 = H1 * M1
    N2 = H2*M2



    if patstat.shape[1]==2:
        patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2'])
    else:
        patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2','trn_effort'])

    ############ REMEMBER TO CHANGE DESCS WHEN NEEDED
    descs_en = ['gasoline', 'turpentine', 'shoe polish', 'leather', 'soap', 'perfume', 'cinnamon', 'spice', 'vanilla', 'mint', 'banana',
    'fruit', 'lemon', 'orange', 'licorice', 'anise', 'pine needles', 'garlic', 'onion', 'disgusting', 'coffee', 'chocolate', 'apple',
    'flower', 'clove', 'dentist', 'pineapple', 'caramel', 'rose', 'mushroom', 'fish', 'herring', 'shellfish']


    #Si_4clusters_FilteredSNACK DATA
    # descs_en = ['bensin', 'terpentin', 'tjära', 'läder', 'tvål', 'parfym', 'skokräm', 'kanel', 'vanilj', 'blomma', 
    #             'mint', 'banan', 'frukt', 'citron', 'apelsin', 'citrus', 'lakrits', 'anis', 'krydda', 'tallbarr', 
    #             'vitlök', 'lök', 'kaffe', 'choklad', 'äpple', 'nejlika', 'kryddnejlika', 'tandläkare', 'ananas', 
    #             'jordgubbe', 'karamell', 'ros', 'svamp', 'fisk', 'sill', 'skaldjur']

    # descs_en = ['bensin', 'terpentin', 'skokräm', 'läder', 'tvål', 'parfym', 'krydda',
    #    'gummi', 'kemisk', 'kanel', 'vanilj', 'blomma', 'pepparmint', 'mentol',
    #    'mint', 'polkagris', 'banan', 'frukt', 'äpple', 'päron', 'citron',
    #    'apelsin', 'citrus', 'lakrits', 'anis', 'tallbarr', 'fernissa',
    #    'målarfärg', 'vitlök', 'lök', 'äcklig', 'kaffe', 'choklad', 'nejlika',
    #    'tandläkare', 'ananas', 'karamell', 'ros', 'svamp', 'champinjon',
    #    'fisk', 'sill', 'skaldjur', 'illa']

    # descs_en = ['bensin', 'petroleum', 'bensinmack', 'diesel', 'läder', 
    # 'skokräm', 'skinn', 'kanel', 'kanelbulle', 'pepparmint', 'mint', 'mentol', 
    # 'banan', 'skumbanan', 'citron', 'citrus', 'lime', 'citronmeliss', 'lakrits', 'anis', 
    # 'salmiak', 'saltlakrits', 'terpentin', 'fernissa', 'målarfärg', 'lösningsmedel', 'vitlök', 
    # 'lök', 'stekt.lök', 'purjolök', 'kaffe', 'kaffesump', 'snabbkaffe', 'kaffeböna', 'äpple', 'tandläkare', 
    # 'nejlika', 'kryddnejlika', 'sjukhus', 'ananas', 'ros', 'rosenvatten', 'rosenolja', 'svamp', 'champinjon', 
    # 'kantarell', 'mögelsvamp', 'fisk', 'sill', 'skaldjur', 'räka']

    ODORS_en = np.array(['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Turpentine',
            'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish'])

    # descs_en = ODORS_en

    os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
    trpats1 = Utils.loadbin(buildPATH+"trpats1.bin",N1)       
    trpats2 = Utils.loadbin(buildPATH+"trpats2.bin",N2)    

    # w11 = Utils.loadbin(buildPATH+"Wij11pre_si_4clusters.bin",N1,N1)
    # w22 = Utils.loadbin(buildPATH+"Wij22pre_si_4clusters.bin",N2,N2)
    # w21 = Utils.loadbin(buildPATH+"Wij21pre_si_4clusters.bin",N2,N1)
    # w12 = Utils.loadbin(buildPATH+"Wij12pre_si_4clusters.bin",N1,N2)
    # b22 = Utils.loadbin(buildPATH+"Bj22pre_si_4clusters.bin")
    # b21 = Utils.loadbin(buildPATH+"Bj21pre_si_4clusters.bin")
    # b11 = Utils.loadbin(buildPATH+"Bj11pre_si_4clusters.bin")
    # b12 = Utils.loadbin(buildPATH+"Bj12pre_si_4clusters.bin")

    w11 = Utils.loadbin(buildPATH+"Wij11.bin",N1,N1)
    w22 = Utils.loadbin(buildPATH+"Wij22.bin",N2,N2)
    w21 = Utils.loadbin(buildPATH+"Wij21.bin",N1,N2)
    w12 = Utils.loadbin(buildPATH+"Wij12.bin",N2,N1)

    b22 = Utils.loadbin(buildPATH+"Bj22.bin")
    b21 = Utils.loadbin(buildPATH+"Bj21.bin")
    b11 = Utils.loadbin(buildPATH+"Bj11.bin")
    b12 = Utils.loadbin(buildPATH+"Bj12.bin")

    ###binarize graded inputs (until I find a better way to think of attractor weights for graded patterns)
    if np.any(((trpats2 >0) & (trpats2 <1))):
        trpats2 = np.where(trpats2>0.4,1,0)


    p1 = np.zeros(trpats1.shape[0]*H1).astype(int) 
    p2 = np.zeros(trpats2.shape[0]*H2).astype(int)

    for i in range(trpats1.shape[0]): 
        p1[i*H1:(i+1)*H1] = np.where(trpats1[i] > 0)[0]

    for i in range(trpats2.shape[0]): 
        p2[i*H2:(i+1)*H2] = np.where(trpats2[i] > 0)[0]

    p1=p1.reshape(trpats1.shape[0],H1)
    p2=p2.reshape(trpats2.shape[0],H2)

    print(p1.shape)
    ndescs = patstat_pd.LTM1.max()+1
    nods = patstat_pd.LTM2.max()+1
    ####Get mean similarity of all descs and od patterns with other patterns
    p1_meansim = np.zeros(ndescs)
    for i in range(ndescs):
        curpat  = p1[i]
        for j in range(ndescs):
            if j!=i:
                p1_meansim[i] += np.count_nonzero(p1[i]==p1[j])/(ndescs-1)

    p2_meansim = np.zeros(nods)
    for i in range(nods):
        curpat  = p2[i]
        for j in range(nods):
            if j!=i:
                p2_meansim[i] += np.count_nonzero(p2[i]==p2[j])/(nods-1)



    if mode == 'assocw_group_by_nassocs':

        ltm1_nassocs = patstat_pd.LTM1.value_counts().sort_index()
        ltm2_nassocs = patstat_pd.LTM2.value_counts().sort_index()
        print(ltm1_nassocs.unique())
        print(ltm2_nassocs.unique())

        w12_nassocwise_attr_weights = [[] for i in range(ltm1_nassocs.max())]        
        w21_nassocwise_attr_weights = [[] for i in range(ltm2_nassocs.max())]
        for i,n in enumerate(ltm2_nassocs.unique()):
            print('\n\tOd->Lang Nassoc: {} \n'.format(n))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(ltm2_nassocs[ltm2_nassocs==n].index),['LTM1','LTM2']].to_numpy()

            for desc,od in od_desc_pairs:
                print('Desc: {}   Od: {}'.format(desc,od))
                langpat = p1[desc]
                odpat = p2[od]

                for pre in odpat:
                    for post in langpat:
                                w21_nassocwise_attr_weights[n-1].append(w21[pre,post])

        for i,n in enumerate(ltm1_nassocs.unique()):
            print('\n\tLang->Od Nassoc: {} \n'.format(n))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM1.isin(ltm1_nassocs[ltm1_nassocs==n].index),['LTM1','LTM2']].to_numpy()

            for desc,od in od_desc_pairs:
                print('Desc: {}   Od: {}'.format(desc,od))
                langpat = p1[desc]
                odpat = p2[od]

                for pre in langpat:
                    for post in odpat:
                            w12_nassocwise_attr_weights[n-1].append(w12[pre,post])


        plot_colors = ['tomato','forestgreen','royalblue','gold']
        bins1 = np.histogram([item for sublist in w12_nassocwise_attr_weights for item in sublist],bins=150)[1]
        bins2 = np.histogram([item for sublist in w21_nassocwise_attr_weights for item in sublist],bins=150)[1]

        fig,ax = plt.subplots(1,2,figsize=(15,8),sharey=True)
        # for i,n in enumerate(nassocs.unique()):  ##for i in range(nassocs.max()-1,-1,-1)
        #     print('len {} assoc_weights: {}'.format(n,len(nassocwise_attr_weights[i])))
        #     sns.histplot(nassocwise_attr_weights[i],stat='probability', kde='True', bins=bins, color=plot_colors[i],label='{} Associations'.format(n),ax=ax)
        # ax.legend()

        x1 = np.arange(ltm1_nassocs.max())
        x2 = np.arange(ltm2_nassocs.max())
        # labels1 = ['{} Associations'.format(i+1) for i in range(ltm1_nassocs.max())]
        # labels2 = ['{} Associations'.format(i+1) for i in range(ltm2_nassocs.max())]
        labels1 = ['{} Associations \n (npats: {})'.format(i+1,patstat_pd.loc[patstat_pd.LTM1.isin(ltm1_nassocs[ltm1_nassocs==i+1].index),'LTM1'].unique().size) for i in range(ltm1_nassocs.max())]
        labels2 = ['{} Associations \n (npats: {})'.format(i+1,patstat_pd.loc[patstat_pd.LTM2.isin(ltm2_nassocs[ltm2_nassocs==i+1].index),'LTM2'].unique().size) for i in range(ltm2_nassocs.max())]

        sns.boxplot(data=w12_nassocwise_attr_weights,palette=plot_colors,ax=ax[0])
        sns.boxplot(data=w21_nassocwise_attr_weights,palette=plot_colors,ax=ax[1])
        means1 = [np.mean(j) for j in w12_nassocwise_attr_weights]
        means2 = [np.mean(j) for j in w21_nassocwise_attr_weights]
        ax[0].scatter(x1,means1,marker='o',color='white',s=100)
        # for i in range(len(x1)):
        #     ax[0].text(x1[i],means1[i],means1[i])
        ax[0].set_xticks(x1)
        ax[0].set_xticklabels(labels1)

        ax[1].scatter(x2,means2,marker='o',color='white',s=100)
        # for i in range(len(x2)):
        #     ax[1].text(x2[i],means2[i],means2[i])
        ax[1].set_xticks(x2)
        ax[1].set_xticklabels(labels2)

        ax[0].set_xlabel('Grouped by (Lang->Od Associations)',size=14)
        ax[1].set_xlabel('Grouped by (Od->Lang Associations)',size=14)
        ax[0].set_title('AssocWeights (Lang->Od)')
        ax[1].set_title('AssocWeights (Od->Lang)')
        ax[0].set_ylabel('Weights',size=14)
            
        plt.show()

    elif mode=='w21_group_by_odor':

        assocwise_attr_weights = [[] for i in range(16)]
        for i in range(16):
            print('\n\tOdor: {} \n'.format(i+1))
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2==i,['LTM1','LTM2']].to_numpy()
            for desc,od in od_desc_pairs:
                langpat = p1[desc]
                odpat = p2[od]

                for pre in odpat:
                    for post in langpat:
                        assocwise_attr_weights[i].append(w21[pre,post])

        bins = np.histogram([item for sublist in assocwise_attr_weights for item in sublist],bins=150)[1]
        fig,ax = plt.subplots(1,1,figsize=(15,8))
        # for i in range(16):
        #     sns.histplot(assocwise_attr_weights[i],stat='count', kde='True', bins=bins, color=kelly_colors[i],label=ODORS_en[i],ax=ax)

        sns.boxplot(data=assocwise_attr_weights,palette=kelly_colors,ax=ax)
        means = [np.mean(j) for j in assocwise_attr_weights]
        ax.scatter(np.arange(nods),means,marker='o',color='grey',s=50)
        ax.set_xticks(np.arange(nods))
        ax.set_xticklabels(ODORS_en,rotation=45)
            
        #ax.set_xlabel('Weights',size=14)
        ax.set_ylabel('Weight',size=14)
            
        plt.show()

    elif mode=='w21_group_by_indivassocs':

        assocwise_attr_weights = [[] for i in range(patstat_pd.shape[0])]
        counter = 0
        nassocs = patstat_pd.LTM2.value_counts().sort_index()

        for i in range(trpats2.shape[0]):
            print('\n\tOdor: {} \n'.format(i+1))
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2==i,['LTM1','LTM2']].to_numpy()
            for desc,od in od_desc_pairs:
                langpat = p1[desc]
                odpat = p2[od]

                for pre in odpat:
                    for post in langpat:
                        assocwise_attr_weights[counter].append(w21[pre,post])
                counter+=1





        hist,bins = np.histogram([item for sublist in assocwise_attr_weights for item in sublist],bins=150)
        fig,ax = plt.subplots(1,1,figsize=(15,8))
        counter = 0
        for od in range(trpats2.shape[0]):
            na = nassocs[od]
            od_assocs = patstat_pd.loc[patstat_pd.LTM2==od,'LTM1'].values
            col = get_colors(kelly_colors[od],na)
            # print(col)

            #####Need to fix labelling of distriubtions. Counter should not be used in descs_en labelling
            for i in range(na):
                print(od,od_assocs[i],ODORS_en[od],descs_en[od_assocs[i]])

                sns.histplot(assocwise_attr_weights[counter],stat='count', kde='True', bins=bins, color=col[i],label='{}-{}'.format(ODORS_en[od],descs_en[od_assocs[i]]),ax=ax)

                ax.axvline(np.mean(assocwise_attr_weights[counter]), color=col[i], linestyle='dashed', linewidth=1)
                ax.annotate('{}-{}'.format(ODORS_en[od],descs_en[od_assocs[i]]),xy=(np.mean(assocwise_attr_weights[counter]),1),xycoords=('data','axes fraction'),c=col[i],rotation=45,annotation_clip=False)
                counter+=1

            
        ax.set_xlabel('Weights',size=14)
        ax.set_ylabel('Count',size=14)
        fig.tight_layout()   
        plt.show()


    elif mode=='bias_group_by_nassocs':

        b2 = (b22+b12)/2
        b1 = (b21+b11)/2

        nassocs = patstat_pd.LTM2.value_counts().sort_index()
        nassocs_ltm1 = nassocs #patstat_pd.LTM1.value_counts().sort_index()

        ##Are all descs unique or there is sharing between odors
        if nassocs_ltm1.max()==1:
            uniquedesc_flag = 1
        else:
            uniquedesc_flag = 0

        if uniquedesc_flag == 1:
            nassocwise_attr_od_biases = [[] for i in range(nassocs.max())]
            nassocwise_attr_lang_biases = [[] for i in range(nassocs.max())]

            odors_added = []    ###To avoid repetitions in bias appending
            for i in range(nassocs.max()):
                print('\n\tNassoc: {} \n'.format(i+1))
                ### Get pairs of odor descriptors from odors that have the same no of assocs
                od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(nassocs[nassocs==i+1].index),['LTM1','LTM2']].to_numpy()

                for desc,od in od_desc_pairs:
                    print('Desc: {}   Od: {}'.format(desc,od))
                    langpat = p1[desc]
                    odpat = p2[od]
                    if not od in odors_added:
                        print('Desc: {}   Od: {}'.format(desc,od))
                        nassocwise_attr_od_biases[i].extend(b2[odpat])
                        nassocwise_attr_lang_biases[i].extend(b1[langpat])
                        odors_added.append(od)
                    else:
                        nassocwise_attr_lang_biases[i].extend(b1[langpat])

        else:
            nassocwise_attr_od_biases = [[] for i in range(nassocs.max())]
            nassocwise_attr_lang_biases = [[] for i in range(nassocs_ltm1.max())]

            odors_added = []    ###To avoid repetitions in bias appending
            for i in range(nassocs.max()):
                print('\n\tNassoc: {} \n'.format(i+1))
                ### Get pairs of odor descriptors from odors that have the same no of assocs
                od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(nassocs[nassocs==i+1].index),['LTM1','LTM2']].to_numpy()

                for desc,od in od_desc_pairs:
                    print('Desc: {}   Od: {}'.format(desc,od))
                    odpat = p2[od]
                    if not od in odors_added:
                        nassocwise_attr_od_biases[i].extend(b2[odpat])
                        odors_added.append(od)

            descs_added = []
            for i in range(nassocs_ltm1.max()):
                print('\n\tNassoc: {} \n'.format(i+1))
                ### Get pairs of odor descriptors from odors that have the same no of assocs
                od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(nassocs_ltm1[nassocs_ltm1==i+1].index),['LTM1','LTM2']].to_numpy()
                # print(od_desc_pairs.shape[0])
                for desc,od in od_desc_pairs:
                    print('Desc: {}   Od: {}'.format(descs_en[desc],ODORS_en[od]))
                    descpat = p1[desc]
                    if not desc in descs_added:
                        if len(np.where(b1[descpat]>-4)[0])>0 and i==0:
                            print(desc,od)
                        nassocwise_attr_lang_biases[i].extend(b1[descpat])
                        descs_added.append(desc)



        b1_flat = [item for sublist in nassocwise_attr_lang_biases for item in sublist]    
        b2_flat = [item for sublist in nassocwise_attr_od_biases for item in sublist]

        plot_colors = ['tomato','forestgreen','royalblue','gold']
        bins1 = np.histogram(b1_flat,bins=50)[1]
        bins2 = np.histogram(b2_flat,bins=50)[1]
        fig,ax = plt.subplots(1,2,figsize=(15,8),sharey=True)
        x1 = np.arange(nassocs_ltm1.max())
        x2 = np.arange(nassocs.max())
        # for i in range(nassocs.max()):
        #     print(patstat_pd.loc[patstat_pd.LTM2.isin(nassocs[nassocs==i+1].index),['LTM1','LTM2']])
        labels_langnet = ['{} Associations \n (npats: {})'.format(i+1,patstat_pd.loc[patstat_pd.LTM2.isin(nassocs_ltm1[nassocs_ltm1==i+1].index),'LTM1'].unique().size) for i in range(nassocs_ltm1.max())]
        labels_odnet = ['{} Associations \n (npats: {})'.format(i+1,patstat_pd.loc[patstat_pd.LTM2.isin(nassocs[nassocs==i+1].index),'LTM2'].unique().size) for i in range(nassocs.max())]
        # for i in range(nassocs.max()-1,-1,-1):
            # sns.histplot(nassocwise_attr_lang_biases[i],stat='probability', kde=False, bins=bins1, color=plot_colors[i],label='{} Associations'.format(i+1),ax=ax[0])
            # sns.histplot(nassocwise_attr_od_biases[i],stat='probability', kde=False, bins=bins2, color=plot_colors[i],label='{} Associations'.format(i+1),ax=ax[1])

        sns.violinplot(data=nassocwise_attr_lang_biases,ax=ax[0])
        sns.violinplot(data=nassocwise_attr_od_biases,ax=ax[1])

        means1 = [np.mean(j) for j in nassocwise_attr_lang_biases]
        means2 = [np.mean(j) for j in nassocwise_attr_od_biases]

        print('LangBiases_Mean: ',means1,'\n OdBiases_mean: ',means2)
        print(np.mean([item for sublist in nassocwise_attr_od_biases for item in sublist]))
        ax[0].scatter(x1,means1,marker='o',color='white',s=100)
        ax[1].scatter(x2,means2,marker='o',color='white',s=100)
        ax[0].set_xticks(x1)
        ax[0].set_xticklabels(labels_langnet)
        ax[1].set_xticks(x2)
        ax[1].set_xticklabels(labels_odnet)

        # ax[0].legend()
        # ax[1].legend()     
        # ax[0].set_xlabel('Weights',size=14)
        ax[0].set_ylabel('Bias',size=14)
        # ax[0].set_ylim([-6,-2])
        # ax[1].set_xlabel('Bias',size=14)
        if not uniquedesc_flag:
            ax[0].set_xlabel('LTM1 (Lang) -> LTM2 (Od) Associations')
            ax[1].set_xlabel('LTM2 (Od) -> LTM1 (Lang) Associations')

        ax[0].set_title('Lang Net Biases')
        ax[1].set_title('Odor Net Biases')
        plt.show()

    elif mode=='bias_indiv_odors':


        b2 = (b22+b12)/2

        od_biases = [[] for i in range(nods)]

        odors_added = []    ###To avoid repetitions in bias appending
        for i in range(nods):
            print('Odor: {}'.format(i+1))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2==i,['LTM1','LTM2']].to_numpy()
            odpat = p2[i]
            od_biases[i].extend(b2[odpat])


        # od_biases = np.array(od_biases)
        biases_mean = [np.mean(x) for x in od_biases]
        biases_std = [np.mean(x) for x in od_biases] 


        print(biases_mean)
        print(p2_meansim)

        # r,pval = corr(biases_mean,p2_meansim)
        # print(r,pval)
        plot_colors = ['tomato','forestgreen','royalblue','gold']
        bias_idx = np.argsort(biases_mean)

        # fig,(ax) = plt.subplots(1,1,figsize=(18,10),sharex=True)
        # for i,od in enumerate(bias_idx):
        #     nassocs = patstat_pd.loc[patstat_pd.LTM2==od,'LTM1'].shape[0]
        #     #print(od,nassocs)
        #     ax.bar(i,biases_mean[od],yerr=biases_std[od],color=plot_colors[nassocs-1])

        # patches = []
        # for i in range(len(plot_colors)):
        #     patches.append(mpatches.Patch(color=plot_colors[i],label='{} Associations'.format(i+1)))



        # ax.set_xticks(np.arange(nods))
        # ax.set_xticklabels(np.array(ODORS_en)[bias_idx],rotation=45) 
        # ax.set_xlabel('Odors',size=14)
        # ax.set_ylabel('Bias',size=14)
        # ax.xaxis.grid(True)
        # ax.legend(handles=patches,bbox_to_anchor=(1,0.67))

        # ax2.plot(p2_meansim[bias_idx]) 
        # ax2.set_ylabel('Mean Pattern Similarity \n(Overlapping HCs)',size=14)
        # ax2.set_title('Correlation R: {:.3f}, p: {:.3f}'.format(r,pval))
        # ax2.set_xticks(np.arange(nods))
        # ax2.set_xticklabels(np.array(ODORS_en)[bias_idx],rotation=45)  

 
        b2_flat = [item for sublist in od_biases for item in sublist]

        bins = np.histogram(b2_flat,bins=100)[1]
        fig,ax = plt.subplots(1,1,figsize=(15,8))
        # for i in range(trpats2.shape[0]):
        #     sns.histplot(od_biases[i],stat='count', kde=True,bins=bins, color=kelly_colors[i],label=ODORS_en[i],ax=ax,alpha=0.7)

        sns.violinplot(data=od_biases,palette=kelly_colors,ax=ax)
        #x_min, x_max = ax.get_xlim()
        # for i in range(trpats2.shape[0]):
        #     xs = np.linspace(x_min, x_max, 200)
        #     shape, location, scale = stats.lognorm.fit(od_biases[i])
        #     ax.plot(xs, stats.lognorm.pdf(xs, s=shape, loc=location, scale=scale), color=kelly_colors[i], ls=':')
        ax.set_xticks(np.arange(nods))
        ax.set_xticklabels(ODORS_en,rotation=45)    
        ax.set_ylabel('Biases',size=14)
        
        for i in range(nods):
            print(max(od_biases[i]),min(od_biases[i]))


        plt.show()

    elif mode=='bias_indiv_descs':


        b1 =  (b11+b21)/2

        desc_biases = [[] for i in range(ndescs)]
        print(ndescs,nods)
        print(desc_biases[-1])
        for i in range(ndescs):
            # print('Desc: {}'.format(descs_en[i]))
            descpat = p1[i]
            if i == 15:
                #print(desc_biases[i])
                print(b1[descpat])
            desc_biases[i].extend(b1[descpat])

        biases_mean = [np.mean(x) for x in desc_biases]
        biases_std = [np.std(x) for x in desc_biases]

        print(desc_biases[0])

        r,pval = corr(biases_mean,p1_meansim)
        print(r,pval)

        bias_idx = np.argsort(biases_mean)

        b1_flat = [item for sublist in desc_biases for item in sublist]

        bins = np.histogram(b1_flat,bins=100)[1]
        fig,ax = plt.subplots(1,1,figsize=(15,8))


        # for i in range(trpats1.shape[0]):
        #     if i>=len(kelly_colors):
        #         color = get_colors(kelly_colors[i%len(kelly_colors)],0)
        #     else:
        #         color = kelly_colors[i]
        #     sns.histplot(desc_biases[i],stat='count', kde=True,bins=bins, color=color,label=descs_en[i],ax=ax,alpha=0.7)
        
        sns.violinplot(data=desc_biases,palette=kelly_colors,ax=ax)
        ax.set_xticks(np.arange(ndescs))
        ax.set_xticklabels(descs_en,rotation=45)
        # ax.legend(bbox_to_anchor=(1,1.1))     
        # ax.set_xlabel('Biases',size=14)
        # ax.set_ylabel('Count',size=14)


        # fig,(ax,ax2) = plt.subplots(2,1,figsize=(15,8),sharex=True)

        # for i,desc in enumerate(bias_idx):
        #     associated_odors = patstat_pd.loc[patstat_pd.LTM1==desc,'LTM2']


        #     for j,od in enumerate(associated_odors.values):
        #         print(od,associated_odors.size)
        #         height = biases_mean[desc]/associated_odors.size
        #         if j==0: #bottom
        #             if associated_odors.size == 1:
        #                 ax.bar(i,biases_mean[desc],yerr=biases_std[desc],color=kelly_colors[od])
        #             else:
        #                 ax.bar(i,height,color=kelly_colors[od])
        #         elif j==associated_odors.size-1:
        #             ax.bar(i,height,yerr=biases_std[desc],bottom = j*height, color=kelly_colors[od])
        #         else:
        #             ax.bar(i,height,bottom = j*height, color=kelly_colors[od])

        # ##### create legend for odors and colors
        # patches = []
        # for i in range(nods):
        #     patches.append(mpatches.Patch(color=kelly_colors[i],label=ODORS_en[i]))


        # # ax2 = ax.twinx()
        # # ax2.plot(np.array(descs_en)[bias_idx],p1_meansim[bias_idx])

        # ax.set_xticks(np.arange(ndescs))
        # ax.set_xticklabels(np.array(descs_en)[bias_idx],rotation=45) 
        # ax.set_xlabel('Descriptors',size=14)
        # ax.set_ylabel('Bias',size=14)
        # ax.xaxis.grid(True)
        # ax.legend(handles=patches,bbox_to_anchor=(1,0.67))

        # ax2.plot(p1_meansim[bias_idx]) 
        # ax2.set_ylabel('Mean Pattern Similarity \n(Overlapping HCs)',size=14)
        # ax2.set_title('Correlation R: {:.3f}, p: {:.3f}'.format(r,pval))
        # ax2.set_xticks(np.arange(ndescs))
        # ax2.set_xticklabels(np.array(descs_en)[bias_idx],rotation=45) 

        plt.show()

    elif mode=='biasdist_bothnets':

       b2 = (b22+b12)/2

       od_biases = [[] for i in range(nods)]

       odors_added = []    ###To avoid repetitions in bias appending
       for i in range(nods):
            print('Odor: {}'.format(i+1))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2==i,['LTM1','LTM2']].to_numpy()
            odpat = p2[i]
            od_biases[i].extend(b2[odpat])

       od_biases = np.array(od_biases)
       biases_std = od_biases.std(axis=1)
       b2_flat = [item for sublist in od_biases for item in sublist]

       bins2 = np.histogram(b2_flat,bins=50)[1]
       

       b1 =  (b11+b21)/2

       desc_biases = [[] for i in range(ndescs)]

       for i in range(ndescs):
            print('Desc: {}'.format(descs_en[i]))
            descpat = p1[i]
            desc_biases[i].extend(b1[descpat])

       desc_biases = np.array(desc_biases)
       biases_std = desc_biases.std(axis=1)
       b1_flat = [item for sublist in desc_biases for item in sublist]

       bins1 = np.histogram(b1_flat,bins=50)[1]  

       fig,ax = plt.subplots(2,1,figsize=(10,8),sharex=True) 
       sns.histplot(b1_flat,stat='count', kde=True,bins=bins1, color='tab:blue',ax=ax[0])
       sns.histplot(b2_flat,stat='count', kde=True,bins=bins2, color='tab:orange',ax=ax[1])
       ax[0].set_title('LTM1 Bias Distribution')
       ax[1].set_title('LTM2 Bias Distribution')
       ax[0].set_ylabel('Count')
       ax[1].set_ylabel('Count')
       ax[0].axvline(x=np.mean(b1_flat),linestyle='--',linewidth=1,color='grey')
       ax[1].axvline(x=np.mean(b2_flat),linestyle='--',linewidth=1,color='grey')
       xmin = np.min(b1_flat+b2_flat)-0.5
       xmax = np.max(b1_flat+b2_flat)+0.5
       ax[0].set_xlim([xmin,xmax])
       ax[1].set_xlim([xmin,xmax])
       plt.show()

    elif mode=='recuwdist_bothnets':

       w11_exc=[]
       w11_inh= [] 

       for pat in p1:
            for j in pat.astype(int):
                for k in range(trpats1.shape[1]): #Over every unit
                    if k in pat:
                        w11_exc.append(w11[j,k])
                    else:
                        w11_inh.append(w11[j,k])


       w22_exc=[]
       w22_inh= [] 

       for pat in p2:
            for j in pat.astype(int):
                for k in range(trpats2.shape[1]): #Over every unit
                    if k in pat:
                        w22_exc.append(w22[j,k])
                    else:
                        w22_inh.append(w22[j,k])


       w11_all = w11_exc+w11_inh
       bins1 = np.histogram(w11_all,bins=75)[1]
       
       w22_all = w22_exc+w22_inh
       bins2 = np.histogram(w22_all,bins=75)[1]
 

       fig,ax = plt.subplots(2,1,figsize=(10,8),sharex=True) 
       sns.histplot(w11_exc,stat='probability', kde=False,bins=bins1, color='tab:blue',ax=ax[0],label='Excitatory Connections')
       sns.histplot(w11_inh,stat='probability', kde=False,bins=bins1, color='tab:red',ax=ax[0],label='Inhibitory Connections')
       sns.histplot(w22_exc,stat='probability', kde=False,bins=bins2, color='tab:blue',ax=ax[1],label='Excitatory Connections')
       sns.histplot(w22_inh,stat='probability', kde=False,bins=bins2, color='tab:red',ax=ax[1],label='Inhibitory Connections')

       ax[0].set_title('LTM1 Recurrent Weight Distribution')
       ax[1].set_title('LTM2 Recurrent Weight Distribution')
       ax[0].set_ylabel('Density')
       ax[1].set_ylabel('Density')

       ax[0].axvline(x=np.mean(w11_exc),linestyle='--',linewidth=1,color='tab:blue')
       ax[0].axvline(x=np.mean(w11_inh),linestyle='--',linewidth=1,color='tab:red')
       ax[1].axvline(x=np.mean(w22_exc),linestyle='--',linewidth=1,color='tab:blue')
       ax[1].axvline(x=np.mean(w22_inh),linestyle='--',linewidth=1,color='tab:red')

       xmin = np.min(w11_all+w22_all)-0.5
       xmax = np.max(w11_all+w22_all)+0.5
       ax[0].set_xlim([xmin,xmax])
       ax[1].set_xlim([xmin,xmax])
       ax[0].set_ylim([0,1])
       ax[1].set_ylim([0,1])
       ax[0].legend(loc='upper left')
       ax[1].legend(loc='upper left')
       plt.show()


    elif mode=='recurrentw_odorwise':
        odorwise_recurrent_weights = [[] for i in range(nods)]

        for i in range(nods):
            print('\n\tOdor: {} \n'.format(i+1))
            odpat = p2[i]
            for pre in odpat:
                for post in odpat:
                        odorwise_recurrent_weights[i].append(w22[pre,post])


        odorwise_recurrent_weights = np.array(odorwise_recurrent_weights)
        recuw_mean = odorwise_recurrent_weights.mean(axis=1)
        recuw_std = odorwise_recurrent_weights.std(axis=1)
        recuw_idx = np.argsort(recuw_mean)


        # r,pval = corr(recuw_mean,p2_meansim)
        # print(r,pval)

        bins = np.histogram([item for sublist in odorwise_recurrent_weights for item in sublist],bins=150)[1]
        fig,ax = plt.subplots(1,1,figsize=(15,8))
        # for i in range(trpats2.shape[0]):
        #     sns.histplot(odorwise_recurrent_weights[i],stat='count', kde='True', bins=bins, color=kelly_colors[i],label=ODORS_en[i],ax=ax)
        #     ax.axvline(np.mean(odorwise_recurrent_weights[i]),linestyle='--',color=kelly_colors[i])

        sns.violinplot(data=odorwise_recurrent_weights.T,palette=kelly_colors,label=ODORS_en,ax=ax)
        ax.set_xticks(np.arange(trpats2.shape[0]))
        ax.set_xticklabels(ODORS_en,rotation=45)
        # ax.set_xlabel('Weights',size=14)
        # ax.set_ylabel('Count',size=14)

        # fig,(ax) = plt.subplots(1,1,figsize=(15,10),sharex=True)
        # for i,od in enumerate(recuw_idx):
        #         ax.bar(i,recuw_mean[od], yerr=recuw_std[od], color=kelly_colors[od])


        # ##### create legend for odors and colors
        # patches = []
        # for i in range(nods):
        #     patches.append(mpatches.Patch(color=kelly_colors[i],label=ODORS_en[i]))



        # ax.set_xticks(np.arange(nods))
        # ax.set_xticklabels(np.array(ODORS_en)[recuw_idx],rotation=45) 
        # ax.set_xlabel('Odors',size=14)
        # ax.set_ylabel('Mean recurrent weight',size=14)
        # ax.xaxis.grid(True)
        # ax.legend(handles=patches,bbox_to_anchor=(1,0.67))

        # ax2.plot(p2_meansim[recuw_idx]) 
        # ax2.set_ylabel('Mean Pattern Similarity \n(Overlapping HCs)',size=14)
        # ax2.set_title('Correlation R: {:.3f}, p: {:.3f}'.format(r,pval))
        # ax2.set_xticks(np.arange(nods))
        # ax2.set_xticklabels(np.array(ODORS_en)[recuw_idx],rotation=45)   
        plt.show()


    elif mode=='odornet_recurrentw_nassocwise':

        nassocs = patstat_pd.LTM2.value_counts().sort_index()

        nassocwise_recu_weights = [[] for i in range(nassocs.max())]
        for i in range(nassocs.max()):
            print('\n\tNassoc: {} \n'.format(i+1))
            ### Get odors with same no of associations
            nassoc_ods = nassocs[nassocs==i+1].index
            for od in nassoc_ods:
                odpat = p2[od]
                for pre in odpat:
                    for post in odpat:
                        nassocwise_recu_weights[i].append(w22[pre,post])




        plot_colors = ['tomato','forestgreen','royalblue','gold']
        bins = np.histogram([item for sublist in nassocwise_recu_weights for item in sublist],bins=150)[1]
        fig,ax = plt.subplots(1,1,figsize=(15,8))
        for i in range(nassocs.max()-1,-1,-1):
            print('len {} assoc_weights: {}'.format(i+1,len(nassocwise_recu_weights[i])))
            sns.histplot(nassocwise_recu_weights[i],stat='probability', kde='True', bins=bins, color=plot_colors[i],label='{} Associations'.format(i+1),ax=ax)
            
        ax.legend()
            
        ax.set_xlabel('Weights',size=14)
        ax.set_ylabel('Count',size=14)
            
        plt.show()

    elif mode=='recurrentw_descwise':

        descwise_recurrent_weights = [[] for i in range(ndescs)]
        for i in range(ndescs):
            print('\nDesc: {} '.format(i+1))
            descpat = p1[i]
            for pre in descpat:
                for post in descpat:
                        descwise_recurrent_weights[i].append(w11[pre,post])

        # bins = np.histogram([item for sublist in odorwise_recurrent_weights for item in sublist],bins=150)[1]
        # fig,ax = plt.subplots(1,1,figsize=(15,8))
        # for i in range(16):
        #     sns.histplot(odorwise_recurrent_weights[i],stat='count', kde='True', bins=bins, color=kelly_colors[i],label=ODORS_en[i],ax=ax)

        # ax.legend()
            
        # ax.set_xlabel('Weights',size=14)
        # ax.set_ylabel('Count',size=14)
            
        # plt.show()

        descwise_recurrent_weights = np.array(descwise_recurrent_weights)
        recuw_mean = descwise_recurrent_weights.mean(axis=1)
        recuw_std = descwise_recurrent_weights.std(axis=1)
        recuw_idx = np.argsort(recuw_mean)
        fig,(ax) = plt.subplots(1,1,figsize=(15,10),sharex=True)



        r,pval = corr(recuw_mean,p1_meansim)
        print(r,pval)

        ####BAR PLOT
        # for i,desc in enumerate(recuw_idx):
        #     associated_odors = patstat_pd.loc[patstat_pd.LTM1==desc,'LTM2']


        #     for j,od in enumerate(associated_odors.values):
        #         #print(od,associated_odors.size)
        #         height = recuw_mean[desc]/associated_odors.size
        #         if j==0: #bottom
        #             if associated_odors.size == 1:
        #                 ax.bar(i,recuw_mean[desc],yerr=recuw_std[desc],color=kelly_colors[od])
        #             else:
        #                 ax.bar(i,height,color=kelly_colors[od])
        #         elif j==associated_odors.size-1:
        #             ax.bar(i,height,yerr=recuw_std[desc],bottom = j*height, color=kelly_colors[od])
        #         else:
        #             ax.bar(i,height,bottom = j*height, color=kelly_colors[od])

        # ##### create legend for odors and colors
        # patches = []
        # for i in range(nods):
        #     patches.append(mpatches.Patch(color=kelly_colors[i],label=ODORS_en[i]))



        # ax.set_xticks(np.arange(ndescs))
        # ax.set_xticklabels(np.array(descs_en)[recuw_idx],rotation=45) 
        # ax.set_xlabel('Descriptors',size=14)
        # ax.set_ylabel('Mean recurrent weight',size=14)
        # ax.xaxis.grid(True)
        # ax.legend(handles=patches,bbox_to_anchor=(1,0.67))

        # ax2.plot(p1_meansim[recuw_idx]) 
        # ax2.set_ylabel('Mean Pattern Similarity \n(Overlapping HCs)',size=14)
        # ax2.set_title('Correlation R: {:.3f}, p: {:.3f}'.format(r,pval))
        # ax2.set_xticks(np.arange(ndescs))
        # ax2.set_xticklabels(np.array(descs_en)[recuw_idx],rotation=45)   

        ####VIOLIN/BOX PLOT
        sns.boxplot(data=descwise_recurrent_weights.T,palette=kelly_colors,ax=ax)
        #ax2 =ax.twinx()
        ax.plot(np.arange(ndescs),p1_meansim,marker='o',color='tab:red',label='Mean Overlap')
        ax.legend()
        ax.set_xticks(np.arange(ndescs))
        ax.set_xticklabels(np.array(descs_en),rotation=45) 
        ax.set_xlabel('Descriptors',size=14)
        ax.set_ylabel('Weights',size=14)
        plt.show()
        
    elif mode=='recurrentw_descs_nassocwise':

        nassocs = patstat_pd.LTM2.value_counts().sort_index()
        nassocwise_recuw = [[] for i in range(nassocs.max())]

        odors_added = []    ###To avoid repetitions in bias appending
        for i in range(nassocs.max()):
            print('\n\tNassoc: {} \n'.format(i+1))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(nassocs[nassocs==i+1].index),['LTM1','LTM2']].to_numpy()

            for desc,od in od_desc_pairs:
                print('Desc: {}   Od: {}'.format(desc,od))
                odpat = p2[od]
                if not od in odors_added:
                    nassocwise_attr_od_biases[i].extend(b2[odpat])
                    odors_added.append(od)

        descs_added = []
        for i in range(nassocs_ltm1.max()):
            print('\n\tNassoc: {} \n'.format(i+1))
            ### Get pairs of odor descriptors from odors that have the same no of assocs
            od_desc_pairs = patstat_pd.loc[patstat_pd.LTM2.isin(nassocs_ltm1[nassocs_ltm1==i+1].index),['LTM1','LTM2']].to_numpy()
            # print(od_desc_pairs.shape[0])
            for desc,od in od_desc_pairs:
                print('Desc: {}   Od: {}'.format(descs_en[desc],ODORS_en[od]))
                descpat = p1[desc]
                if not desc in descs_added:
                    if len(np.where(b1[descpat]>-4)[0])>0 and i==0:
                        print(desc,od)
                    nassocwise_attr_lang_biases[i].extend(b1[descpat])
                    descs_added.append(desc)


    elif mode=='assocw_particular_odor':

        for od in range(1):
            assocs = patstat_pd.loc[patstat_pd.LTM2==od,'LTM1'].values
            c = get_colors(kelly_colors[od],len(assocs))
            
            w21_assocwise = [[] for i in range(len(assocs))]
            for i,desc in enumerate(assocs):
                odpat = p2[od]
                descpat = p1[desc]
                for pre in odpat:
                    for post in descpat:
                        w21_assocwise[i].append(w21[pre,post])

            w12_assocwise = [[] for i in range(len(assocs))]
            for i,desc in enumerate(assocs):
                odpat = p2[od]
                descpat = p1[desc]
                for pre in descpat:
                    for post in odpat:
                        w12_assocwise[i].append(w12[pre,post])


            fig,(ax,ax2) = plt.subplots(1,2,figsize=(15,10))
            colors = ['tomato','forestgreen','royalblue','gold']
            bins = np.histogram([item for sublist in w21_assocwise for item in sublist],bins=100)[1]
            labels = []
            for i,desc in enumerate(assocs):
            #   sns.histplot(w21_assocwise[i],bins=bins,label='{}-{} assoc'.format(ODORS_en[od],descs_en[desc]),ax=ax,color=c[i])
                labels.append('{}-\n{} assoc'.format(ODORS_en[od],descs_en[desc]))
            sns.violinplot(data=w21_assocwise,ax=ax,palette=c)
            means = [np.mean(x) for x in w21_assocwise]
            ax.scatter(np.arange(len(means)), means, color='white',s=100)
            ax.set_xticks(np.arange(len(w12_assocwise)))
            ax.set_xticklabels(labels,rotation=0,size=14)
            ax.set_title('Odor -> Language Net',size=14)
            # ax.set_xlim([0,0.5])
            # ax.legend()

            bins = np.histogram([item for sublist in w12_assocwise for item in sublist],bins=100)[1]
            labels = []
            for i,desc in enumerate(assocs):
            #         sns.histplot(w12_assocwise[i],bins=bins,label='{}-{} assoc'.format(descs_en[desc],ODORS_en[od]),ax=ax2,color=c[i])
                labels.append('{}-\n{} assoc'.format(descs_en[desc],ODORS_en[od]))
            means = [np.mean(x) for x in w12_assocwise]
            sns.violinplot(data=w12_assocwise,ax=ax2,palette=c)
            ax2.scatter(np.arange(len(means)), means, color='white',s=100)
            ax2.set_xticks(np.arange(len(w21_assocwise)))
            ax2.set_xticklabels(labels,rotation=0,size=14)
            # ax2.set_xlim([0,0.5])
            ax2.set_title('Language -> Odor Net',size=14)
            # ax2.legend()
            fig.tight_layout()
            # plt.savefig('/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/DualNet/Network15x15/Semantization/SimpsonIndex_4Clusters/AddRemove_Descriptors/Fish3Desc->2Desc/AssocW_IndivOdors/'f'{ODORS_en[od]}_AssocWeights')
            plt.show()



def plot_Ptraces():
    parfilename=PATH+'olflangmain1.par'
    H1 = int(Utils.findparamval(parfilename,"H"))
    M1 = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N1 = H1 * M1
    N2 = H2*M2

    def pos(MCpre,MCpos,N):
        return N*MCpre+MCpos

    # pi11 = Utils.loadbin(buildPATH+"Pi11.log",H1*N1) #shape of pi is (timesteps,H*M)
    # pi11 = np.reshape(pi11,(pi11.shape[0],H1,N1)) #Pi is stored as (timestep, target hypercolumn, source unit) -> (timestep,H,N)
    # pj11 = Utils.loadbin(buildPATH+"Pj11.log",N1)
    # pij11 = Utils.loadbin(buildPATH+"Pij11.log",N1*N1)

    pi22 = Utils.loadbin(buildPATH+"Pi22.log",H2*N2) #shape of pi is (timesteps,H*M)
    pi22 = np.reshape(pi22,(pi22.shape[0],H2,N2)) #Pi is stored as (timestep, target hypercolumn, source unit) -> (timestep,H,N)
    
    pj22 = Utils.loadbin(buildPATH+"Pj22.log",N2)
    pij22 = Utils.loadbin(buildPATH+"Pij22.log",N2*N2)

    ltm2_mc_pre = 3
    ltm2_mc_pos = 3
    ltm2_pos = pos(ltm2_mc_pre,ltm2_mc_pos,N2)

    calculated_wij = np.log(pij22[-1,ltm2_pos]/(pi22[-1,0,ltm2_mc_pre]*pj22[-1,ltm2_mc_pos]))
    print(calculated_wij)
    fig,(ax1) = plt.subplots(1,1,figsize=(15,8))
    duration = np.arange(0,pj22.shape[0],1)
    linestyles = ["-","-","-","--","--","--","-.","-.","-.",":"] # - for Pi, -- for Pj, -. for Pij
    datapoints = [
      #pi[:,1,MC_pre1],
      pi22[:,0,ltm2_mc_pre],
      pj22[:,ltm2_mc_pos],
      pij22[:,ltm2_pos],
      ]
    legend = [
      #'Pi LTM2 HC'+str(MC_pre1/M)+' MC'+str(MC_pre1%M),
      'Pi MC{}'.format(ltm2_mc_pre),
      'Pj MC{}'.format(ltm2_mc_pre),
      'Pij MC{}-MC{}'.format(ltm2_mc_pre,ltm2_mc_pos)    
           ]

    for i in range(len(datapoints)):
        #ax1.plot(duration,pi[:,0,MC_pre1],linestyles[0], duration,pj[:,pos1],linestyles[1], duration,pj[:,pos2],linestyles[1], duration,pij[:,pos1],linestyles[2] duration,pij[:,pos2],linestyles[2])
          if linestyles[i] == "-.":
              linewidth = 2
          else:
              linewidth = 1
          ax1.plot(duration,datapoints[i],linestyles[i],label=legend[i],linewidth=linewidth)
          ax1.margins(x=0)


    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.title.set_text("P-traces")
    ax1.legend(loc=4)
    plt.show()



def plot_energy():
    parfilename=PATH+'olflangmain1.par'
    H1 = int(Utils.findparamval(parfilename,"H"))
    M1 = int(Utils.findparamval(parfilename,"M"))
    H2 = int(Utils.findparamval(parfilename,"H2"))
    M2 = int(Utils.findparamval(parfilename,"M2"))

    N1 = H1 * M1
    N2 = H2*M2
    H_en1 = Utils.loadbin(buildPATH+"H_en1.log",N2).T [:,9800:]
    H_en2 = Utils.loadbin(buildPATH+"H_en2.log",N2).T[:,9800:]
    H_en1 = -1*H_en1.sum(axis=0) 
    H_en2 = -1*H_en2.sum(axis=0) 
    
    fig,ax = plt.subplots(2,1,figsize=(20,8),sharex=True)
    ax[0].plot(H_en1,color='tab:blue')
    ax[1].plot(H_en2,color='tab:orange')

    ax[0].set_title('LTM1 Energy')
    ax[1].set_title('LTM2 Energy')
    plt.gcf().tight_layout() 
    plt.show()                                                                                                 



def run():
    
    # bwplot()
    #plot_bias_vs_unitoverlap()
    #plot_within_between_weights(savef=0)
    # pat_simmat(mode = 'simmat')
    #plot_patoverlap_distribution()
    #plot_patternwise_bias()
    #combinedattractor_distmat()
    #actplot()
    #plot_wdist(savef=0)
    #generate_alternatives()
    visualise_associations()
    #plot_multi_association_weights('')
    # plot_weights_assocwise('assocw_particular_odor')
    #plot_energy()
    #plot_Ptraces()

patstat = np.loadtxt('/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_si_nclusters4_topdescs.txt')
# patstat = np.loadtxt('/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_16od_16descs.txt')
run()