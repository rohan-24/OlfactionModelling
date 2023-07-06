import sys, os, select
import csv
import math
import pandas as pd
import numpy as np
import random
import string
from matplotlib import pyplot as plt
from matplotlib import ticker as tck
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import Utils

PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'
figPATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/W2V_GreedyAlg/DistortedPats/'

H = 303
M = 2
N = M*H
encoding_thresh = 0.8       # activity threshold to count as active during encoding
recall_thresh = 0.8			# activity threshold to count as active during encoding   IT is set this high because of high background activity level since we have 2 MCs
parfilename = "olflangmain1.par"
colors_hex = [
	"0xC10020", # Vivid Red
	"0xFFB300", # Vivid Yellow
	"0x803E75", # Strong Purple
	"0xFF6800", # Vivid Orange
	"0xA6BDD7", # Very Light Blue
	"0xCEA262", # Grayish Yellow
	"0x817066", # Medium Gray

	# The following don't work well for people with defective color vision
	"0x007D34", # Vivid Green
	"0xF6768E", # Strong Purplish Pink
	"0x00538A", # Strong Blue
	"0xFF7A5C", # Strong Yellowish Pink
	"0x53377A", # Strong Violet
	"0xFF8E00", # Vivid Orange Yellow
	"0xB32851", # Strong Purplish Red
	"0xF4C800", # Vivid Greenish Yellow
	"0x7F180D", # Strong Reddish Brown
	"0x93AA00", # Vivid Yellowish Green
	"0x593315", # Deep Yellowish Brown
	"0xF13A13", # Vivid Reddish Orange
	"0x232C16", # Dark Olive Green
	]

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
	check recall at each time step in recall cue phase and see if it matches any pattern
	'''

	cos_dist = np.zeros((trpats.shape[0],data1.shape[1]-recall_start_time))
	winner = np.zeros(data1.shape[1]-recall_start_time)
	recall_thresh  = 1e-6 #0.01

	for i,timestep in enumerate(range(recall_start_time, data1.shape[1])):
		act = data1[:,timestep]
		for j,pat in enumerate(trpats):
			cos_dist[j,i] = 1 - np.dot(act,pat) / (np.linalg.norm(act)*np.linalg.norm(pat))
			
		if np.min(cos_dist[:,i])<recall_thresh:
			winner[i] = np.argmin(recall_score[:,i]) 
			#print(winner[i],np.min(recall_score[:,i]))
		else:
			winner[i] = None

	return(winner,cos_dist)

def check_state(data1,trpats,cue_timesteps,recall_nstep,cos_dist):
	'''
	check if recall is spurious
	'''
	for i,t in enumerate(range(cue_timesteps[0],data1.shape[1])):
		act = data1[:,t]

		


def main(argv):


	H = int(Utils.findparamval(parfilename,"H"))
	M = int(Utils.findparamval(parfilename,"M"))
	nstep = int(Utils.findparamval(parfilename,"nstep"))
	recall_nstep = int(Utils.findparamval(parfilename,"recallnstep"))
	
	if len(argv)>1:
		cueHCs = int(argv[1])
	else:
		cueHCs = int(Utils.findparamval(parfilename,"cueHCs"))

	N = H * M

	os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

	trpats = Utils.loadbin("trpats1.bin",N)
	data1 = Utils.loadbin("act1.log",N).T 	#(units,timestep)


	text_file = open("simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()

	text_file = open("patorder.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	patorder = [int(line) for line in text_file]
	text_file.close()

	encoding_steps = simstages[:simstages.index(-1)]
	recall_timelogs = simstages[simstages.index(-2)+1:]
	recall_start_time = recall_timelogs[0]

	recall_cue_steps = [x for i,x in enumerate(recall_timelogs) if i%2==0]
	npats = 16

	recall_npats = 16
	# simstages.pop(0)
	# training_points = []
	# for i,t in enumerate(npats):
	# 	training_points.append(t[i])
	# 	simstages.pop(i)

	# recall_point = simstages[0]
	recall_behaviour = [] #0 if no recall, Pattern id if recall, across recall phase

	#Recall score is based on cosine distance between recall activity at each time step and all training patterns
	winner,_ = calc_recallwinner(data1,trpats,recall_start_time)


	print(recall_start_time,recall_cue_steps)


	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,8), gridspec_kw={'height_ratios': [1, 5],'hspace': 0.05},sharex = True)

	colormap1 = ax2.imshow(data1,interpolation='none',aspect='auto',cmap = plt.cm.binary)
	#cbar = plt.colorbar(colormap1)
	ax2.set_xlabel("Timesteps",fontsize=16)
	ax2.set_ylabel("MCs",fontsize=16)
	for i in range(1,H):
		if i == cueHCs:
			#ax2.axhline(y=i*M-1,xmax=encoding_steps[-1],color='tab:blue')
			ax2.axhline(y=i*M-1,color='tab:orange')
			ax2.plot((0, recall_start_time), (i*M-1, i*M-1), 'tab:blue')
			ax2.plot((recall_start_time, data1.shape[1]), (i*M-1, i*M-1), 'tab:orange')
		else:
			ax2.axhline(y=i*M-1,color='tab:blue')

	color_counter = 0
	for i,step in enumerate(encoding_steps):
		# if i==len(encoding_steps)-1:
		# 	ax2.axvline(x=step,color='k',linestyle='-',linewidth=3)
		if i%2==0: #Start of gap phase
			ax1.scatter(x=np.arange(step-nstep,step),y=np.ones(nstep), c = kelly_colors[patorder[color_counter]],s=20)
			color_counter += 1
			if color_counter==trpats.shape[0]:
				color_counter = 0

	for i,step in enumerate(recall_timelogs[:-1]):
		if i==0:	#Start of recall phase
			plt.axvline(x=step,color='k',linestyle='-',linewidth=3)
		elif i!=0 and i%2==0:	#Start of Cue Phase
			plt.axvline(x=step,color='k',linestyle=':')
		else:	#Start of Gap Phase
			# plt.axvline(x=step,color='k',linestyle='--')
			pass


	col_list = [kelly_colors[int(i)] if i >=0 else 'w' for i in winner]
	ax1.scatter(np.arange(recall_start_time,data1.shape[1]),y = np.ones(data1.shape[1]-recall_start_time), c=col_list,s=20)
	ax1.set_ylim(0.8,2)
	ax1.axis('off')
	ax2.set_ylim(N,0)
	ax2.set_xlim(0,data1.shape[1])

	#plt.savefig(figPATH+'ActPlot_npats16+cueHCs_'+str(cueHCs),dpi=400)
	#plt.savefig(figPATH+'ActPlot_npats16_distortHCs8',dpi=400)
	plt.show()


if __name__ == "__main__":
   main(sys.argv[:])