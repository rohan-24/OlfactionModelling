import sys, os, select
import csv
import math
import pandas as pd
import numpy as np
import random
import string
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker as tck
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import Utils

PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'
buildPATH  = '/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/'
parfilename = PATH+'olflangmain1.par'

H1 = int(Utils.findparamval(parfilename,"H"))
M1 = int(Utils.findparamval(parfilename,"M"))

H2 = int(Utils.findparamval(parfilename,"H2"))
M2 = int(Utils.findparamval(parfilename,"M2"))
nstep = int(Utils.findparamval(parfilename,"nstep"))
recall_nstep = int(Utils.findparamval(parfilename,"recallnstep"))
recall_ngap = int(Utils.findparamval(parfilename,"recallngap"))
cueHCs = int(Utils.findparamval(parfilename,"cueHCs"))
partial_mode = Utils.findparamval(parfilename,"partial_mode")
cues = Utils.loadbin(buildPATH+"cues.bin").astype(int)
#### NOTE: CHANGE PATSTAT FILENAME IF CHANGED IN MAIN SIM ############
assocs = np.loadtxt('/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_16od_16descs.txt',dtype='int64')

N1 = H1 * M1
N2 = H2 * M2


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

def plot_assoc_weights():
	'''
	Plot weights for each association
	'''


	trpats1 = Utils.loadbin(buildPATH+"trpats1.bin",N1)
	trpats2 = Utils.loadbin(buildPATH+"trpats2.bin",N2)

	#### Get active units ID in each HC for each pattern
	p1 = np.zeros([trpats1.shape[0],H1],dtype='int64') 
	p2 = np.zeros([trpats2.shape[0],H2],dtype='int64')

	for i in range(trpats1.shape[0]): 
		p1[i] = np.where(trpats1[i] > 0)[0] 

	for i in range(trpats2.shape[0]): 
		p2[i] = np.where(trpats2[i] > 0)[0] 

	w21 = Utils.loadbin(buildPATH+'Wij21.bin',N2,N1) 

	assoc_w = [[] for i in range(assocs.shape[0])] #np.zeros([assocs.shape[0]])
	for i,(ltm1_pat_id,ltm2_pat_id) in enumerate(assocs):
		#Get associated patterns
		pat1, pat2 = p1[ltm1_pat_id], p2[ltm2_pat_id]
		for mc_pre in pat2:
			for mc_post in pat1:
				assoc_w[i].append(w21[mc_pre,mc_post])

	assoc_w = np.array(assoc_w)
	
	print(assoc_w.mean(axis=1))
	fig,ax = plt.subplots(1,1,figsize=(15,6))
	for i,(ltm1_pat_id,ltm2_pat_id) in enumerate(assocs):
		ax.hist(assoc_w[i],50,color=kelly_colors[i],label='pat{}-pat{}')

	plt.show()


           





def run():
	plot_assoc_weights()
run()