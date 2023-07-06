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
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import Utils

PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'

dorun=False
H = 303
M = 2
N = M*H
encoding_thresh = 0.8       # activity threshold to count as active during encoding
recall_thresh = 0.8			# activity threshold to count as active during encoding   IT is set this high because of high background activity level since we have 2 MCs
parfilename = "olflangmain1.par"

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

def calc_recallwinner(data1,trpats,recall_start_time):
	'''
	get a list containing winners at every time step of recall phase based on smallest cosine distance between input patterns and recall activity
	'''

	trpats = Utils.loadbin("trpats1.bin",N)
	data1 = Utils.loadbin("act1.log",N).T 	#(units,timestep)


	text_file = open("simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()
	encoding_steps = simstages[:simstages.index(-1)]
	recall_timelogs = simstages[simstages.index(-1)+1:]
	recall_start_time = recall_timelogs[0]


	recall_score = np.zeros((trpats.shape[0],data1.shape[1]-recall_start_time))
	winner = np.zeros(data1.shape[1]-recall_start_time)
	recall_thresh  = 0.01

	for i,timestep in enumerate(range(recall_start_time, data1.shape[1])):
		act = data1[:,timestep]
		for j,pat in enumerate(trpats):
			recall_score[j,i] = 1 - np.dot(act,pat) / (np.linalg.norm(act)*np.linalg.norm(pat))
		
		if np.min(recall_score[:,i])<recall_thresh:
			winner[i] = np.argmin(recall_score[:,i]) 
		else:
			winner[i] = None

	#Get duration of activation of each pattern and threshold 
	winner_pd = pd.Series(winner).value_counts() 
	print(winner_pd)
	return(winner)

def attractor_wdist():
	H = int(Utils.findparamval(parfilename,"H"))
	M = int(Utils.findparamval(parfilename,"M"))
	nstep = int(Utils.findparamval(parfilename,"nstep"))
	cueHCs = int(Utils.findparamval(parfilename,"cueHCs"))

	N = H * M

	os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

	# trpats = Utils.loadbin("trpats1.bin",N)
	with open('/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/multi_D0_npat16_emax05.json','r') as f:
		pats = json.load(f)

	wij11 = Utils.loadbin('Wij11.bin')

	attractor_weights = [[] for i in range(pats.shape[0])]

	for i,pat in enumerate(pats):


def run():
	
	winners = [[] for i in range(1,10)]
	for cueHCs in range(1,10):
		os.system("./olflangmain1 "+str(cueHCs))

		winners[cueHCs-1] = calc_recallwinner()







run()