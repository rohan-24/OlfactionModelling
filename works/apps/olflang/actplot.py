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

dorun=False
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


def check_recall(act,pats):
	'''
	check recall at each time step in recall cue phase and see if it matches any pattern
	'''

	for i,pat in enumerate(pats):	

		pat =  np.array([(pat[j:j+2]).astype(int) for j in range(0,len(pat),2)]) 
		pat = np.array([1 if np.array_equal(j,[1,0]) else 0 for j in pat])
		if np.array_equal(act,pat):
			return kelly_colors[i+1]
		else: 
			x = 0

	return(kelly_colors[x])


def run():


	igain = float(Utils.findparamval(parfilename,"igain"))
	again = float(Utils.findparamval(parfilename,"again"))
	taum= float(Utils.findparamval(parfilename,"taum"))
	taua= float(Utils.findparamval(parfilename,"taua"))
	adgain = float(Utils.findparamval(parfilename,"adgain"))


	taup = float(Utils.findparamval(parfilename,"taup"))
	taucond = float(Utils.findparamval(parfilename,"taucond"))
	recuwgain = float(Utils.findparamval(parfilename,"recuwgain"))
	bgain = float(Utils.findparamval(parfilename,"bgain"))
	bwgain = float(Utils.findparamval(parfilename,"bwgain"))
	
	nstep = int(Utils.findparamval(parfilename,"nstep"))
	ngap = int(Utils.findparamval(parfilename,"ngap"))
	etrnrep = int(Utils.findparamval(parfilename,"etrnrep"))

	nmean = float(Utils.findparamval(parfilename,"nmean"))
	namp = float(Utils.findparamval(parfilename,"namp"))
	recallnsteps = int(Utils.findparamval(parfilename,"recallnstep"))
	textstr = 'H = %d\nM = %d\n\nigain = %.2f\nagain = %.2f\ntaum = %.4f\ntaua = %.4f\nadgain = %.1f\n\ntaup = %.1f\ntaucond = %.2f\n\nrecuwgain = %.2f\nbgain = %.2f\nbwgain = %.1f\n\netrnrep = %d\n\nnmean = %.1f\nnamp = %.1f'%(H, M, igain, again, taum, taua, adgain, taup, taucond,recuwgain,bgain,bwgain,etrnrep,nmean, namp)

	# etrnrep = np.arange(1,etrnrep+1)
	os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

	trpats = Utils.loadbin("trpats1.bin",N)
	#print(trpats.shape)

	if dorun :
	   os.system("./olflangmain1")

	data1 = Utils.loadbin("act1.log",N).T


	text_file = open("simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()

	#recall_cue_steps = simstages[3:]
	encoding_steps = simstages[:simstages.index(-1)]
	recall_timelogs = simstages[simstages.index(-1)+1:]
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
	print(data1.shape)
	for timestep in range(data1.shape[1]):
		act = data1[:,timestep]

		if timestep==0:
			act_thresholded = np.array([(act[i:i+2]>0.8).astype(int) for i in range(0,len(act),2)])
			act_thresholded = np.array([1 if np.array_equal(i,[1,0]) else 0 for i in act_thresholded]) 
		else:
			if timestep>= recall_start_time: #simstages[3]:#simstages[-1]-ngap*recall_npats:	#If in recall phase
				cur_act = np.array([(act[i:i+2]>recall_thresh).astype(int) for i in range(0,len(act),2)])
				cur_act = np.array([1 if np.array_equal(i,[1,0]) else 0 for i in cur_act]) 
				recall_behaviour.append(check_recall(cur_act,trpats))
			else:
				cur_act = np.array([(act[i:i+2]>encoding_thresh).astype(int) for i in range(0,len(act),2)])
				cur_act = np.array([1 if np.array_equal(i,[1,0]) else 0 for i in cur_act]) 


			act_thresholded = np.column_stack((act_thresholded,cur_act))

		

	#recall_behaviour = ['g' if i==1 else 'r' for i in recall_behaviour]



	colormap1 = plt.imshow(act_thresholded,interpolation='none',aspect='auto',cmap = plt.cm.binary)
	plt.title("LTM #1",fontsize = 14)
	#cbar = plt.colorbar(colormap1)
	plt.xlabel("Timesteps",fontsize=16)
	plt.ylabel("HC",fontsize=16)

	plt.xticks(fontsize = 16)

	color_counter = 0
	for i,step in enumerate(encoding_steps):
		if i==len(encoding_steps)-1:
			plt.axvline(x=step,color='k',linestyle='-',linewidth=3)
		if i%2==0: #Start of gap phase
			plt.axvline(x=step,color='k',linestyle='--')
			plt.scatter(x=np.arange(step-recallnsteps,step),y=np.ones(recallnsteps)*-20, c = kelly_colors[color_counter+1])
			color_counter+=1
		else:	#Start of cue phase
			plt.axvline(x=step,color='k',linestyle=':')

	for i,step in enumerate(recall_timelogs[:-1]):
		if i==0:	#Start of recall phase
			plt.axvline(x=step,color='k',linestyle='-',linewidth=3)
		elif i!=0 and i%2==0:	#Start of Cue Phase
			plt.axvline(x=step,color='k',linestyle=':')
		else:	#Start of Gap Phase
			plt.axvline(x=step,color='k',linestyle='--')


	for i,col in enumerate(recall_behaviour):
		plt.scatter(x=recall_start_time+i,y= -20,c=col,s=5)

	#cbar.ax.tick_params(labelsize=14) 
	
	plt.gcf().set_size_inches(20, 10)
	# plt.gcf().tight_layout()
	plt.gcf().subplots_adjust(
	left=0.06,
	right=0.825
	)
	plt.figtext(0.88,0.25, textstr, fontsize=14)

	plt.show()


run()