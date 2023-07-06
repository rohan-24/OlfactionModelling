'''
This script analyses recall behaviour in different ways for the two network model.
'''

import sys, os, select
import csv
import math
import pandas as pd
import numpy as np
import random
import string
import json
from matplotlib import pyplot as plt
from matplotlib import ticker as tck
import matplotlib.colors as mcolors
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import Utils

PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'
buildPATH = '/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/' 
#figPATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/W2V_GreedyAlg/DistortedPats/'
dorun=False
parfilename = "olflangmain1.par"
H1 = int(Utils.findparamval(PATH+parfilename,"H"))
M1 = int(Utils.findparamval(PATH+parfilename,"M"))
recallnstep = int(Utils.findparamval(PATH+parfilename,"recallnstep"))
recallngap = int(Utils.findparamval(PATH+parfilename,"recallngap"))

N1 = H1*M1


kelly_colors = ['#BE0032', #Red
				'#F3C300', #Mustard
				'#875692', #Lavender
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
				'#882D17', #Maroon
				'#8DB600', #Parrot Green
				'#654522', #Brown
				'#E25822', #Dark Pink
				'#2B3D26', #Dark NAvy Blue
				'#F2F3F4', #Off White/Greyish White
				'#222222' #Black
				]


def calc_recallwinner():
	'''
	get a list containing winners at every time step of recall phase based on smallest cosine distance between input patterns and recall activity
	'''

	trpats1 = Utils.loadbin(buildPATH+"trpats1.bin",N1)
	data1 = Utils.loadbin(buildPATH+"act1.log",N1).T 	#(units,timestep)

	text_file = open(buildPATH+"simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()
	encoding_steps = simstages[:simstages.index(-1)]
	recall_timelogs = simstages[simstages.index(-2)+1:]
	recall_start_time = recall_timelogs[0]

	recall_score = np.zeros((trpats1.shape[0],data1.shape[1]-recall_start_time))
	winner = np.zeros(data1.shape[1]-recall_start_time)
	recall_thresh  = 0.01

	for i,timestep in enumerate(range(recall_start_time, data1.shape[1])):
		act = data1[:,timestep]
		for j,pat in enumerate(trpats1):
			recall_score[j,i] = 1 - np.dot(act,pat) / (np.linalg.norm(act)*np.linalg.norm(pat))
		
		#print(np.min(recall_score[:,i]))
		if np.min(recall_score[:,i])<recall_thresh:
			winner[i] = np.argmin(recall_score[:,i]) 
		else:
			winner[i] = None

	#Get duration of activation of each pattern and threshold 
	#winner_pd = pd.Series(winner).value_counts() 

	#return(winner_pd.index.astype(int).tolist())
	return(winner)


def plot_assocmat(winners):
	'''
		Plot a matrix showing the target odors on one axis and associated distractors on other axis. Compare recall based assocmat with actual assocmat.
		Note that this works only when all odors are being cued
	'''
	assocs = np.loadtxt("/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_top3snackdescs.txt")
	# assocs = np.loadtxt("/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_n3.txt")
	cues = Utils.loadbin("/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/cues.bin")  
	trpats1 = Utils.loadbin(buildPATH+"trpats1.bin",N1)
	trpats2 = Utils.loadbin(buildPATH+"trpats2.bin",N1)

	true_assocmat = np.zeros([int(assocs[:,1].max()+1),int(assocs[:,0].max()+1)]) # odors x descriptors
	recall_assocmat = np.zeros([int(assocs[:,1].max()+1),int(assocs[:,0].max()+1)]) # odors x descriptors
	for x in assocs:
		true_assocmat[int(x[1]),int(x[0])] = 1 


	for pat_idx in range(trpats2.shape[0]):
		cue = int(cues[pat_idx])
		recall_state = winners[:,cue*(recallnstep+recallngap):(cue+1)*(recallnstep+recallngap)]
		active,counts = np.unique(recall_state,return_counts=True)
		# print(cue,recall_state)
		for i,act in enumerate(active):
			if np.isnan(act):
				continue
			else:
				recall_assocmat[cue,int(act)] = counts[i]/(winners.shape[0]*(recallnstep+recallngap))

	fig,(ax,ax2) = plt.subplots(1,2,figsize=(15,8))
	ax.imshow(true_assocmat,interpolation='none', aspect='auto',cmap = 'RdBu')
	ax.set_xlabel('Descriptors',size=14)
	ax.set_ylabel('Odors',size=14)
	ax.set_title('Trained associations',size=14)

	#print(recall_assocmat.min(),recall_assocmat.max())
	cmap0 = mcolors.LinearSegmentedColormap.from_list('', ['white', 'darkblue'])
	ax2.imshow(recall_assocmat,interpolation='none', aspect='auto',cmap = 'RdBu', alpha = 0.6)
	ax2.set_xlabel('Descriptors',size=14)
	ax2.set_ylabel('Odors',size=14)
	ax2.set_title('Recall',size=14)

	plt.show()

def run():
	# winner = calc_recallwinner()
	# print(winner.shape)
	# plot_assocmat(winner)

	sims = 10
	winners = [[] for i in range(sims)]
	
	for i in range(sims):
		#Run Sim
		os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
		os.system("./olflangmain1 ")
		winners[i] = calc_recallwinner()

		#Plot Act Plot
		os.chdir(PATH)
		#os.system("python3 modular_actplot.py "+str(cueHCs))
		
	winners = np.array(winners)
	print(winners[:,0:5])
	plot_assocmat(winners)

	'''
	TODO: Aggregate assocmat over multiple simulations

	Overlap in associations. How to parametrize it? Look at snack data

	'''

run()