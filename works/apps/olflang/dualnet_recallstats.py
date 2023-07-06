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
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib import ticker as tck
from scipy.stats import sem
import Figsolflang
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import Utils

PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'
buildPATH  = '/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/'
figPATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/DualNet/NeuralCodingWorkshop2023/OnlyOdorLabels/'

parfilename = PATH+"olflangmain1.par"
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
ODORS_en = ['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Terpentine',
			'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish']

H1 = int(Utils.findparamval(parfilename,"H"))
M1 = int(Utils.findparamval(parfilename,"M"))

H2 = int(Utils.findparamval(parfilename,"H2"))
M2 = int(Utils.findparamval(parfilename,"M2"))
nstep = int(Utils.findparamval(parfilename,"nstep"))
recall_nstep = int(Utils.findparamval(parfilename,"recallnstep"))
recall_ngap = int(Utils.findparamval(parfilename,"recallngap"))
cueHCs = int(Utils.findparamval(parfilename,"cueHCs"))
distortHCs1 = int(Utils.findparamval(parfilename,"distortHCs1"))
distortHCs2 = int(Utils.findparamval(parfilename,"distortHCs2"))
partial_mode = Utils.findparamval(parfilename,"partial_mode")
recall_thresh  = float(Utils.findparamval(parfilename,"recall_thresh"))
assoc_cspeed = float(Utils.findparamval(parfilename,"assoc_cspeed"))
assoc_dist  = float(Utils.findparamval(parfilename,"assoc_dist"))
delay = int(assoc_dist/assoc_cspeed/1e-3)

linestyles = [
		'solid',
		'dashed',
		'dashdot',
     ((0, (2, 1))), #'dotted'
     #((0, (5, 1))), #'thick dashed',        
     # ((0, (2.5, 1))),	#'medium dashed',                
     # ((0, (1, 1))),	#'thin dashed',        
     ((0, (2, 1, 1, 1, 1, 1))),	#'dashdotdotted',         
     ]

# if distortHCs>0:
# 	distortHCs_ltm1_list = Utils.loadbin(buildPATH+'distortHCs_ltm1.bin',distortHCs) 
# 	distortHCs_ltm2_list = Utils.loadbin(buildPATH+'distortHCs_ltm2.bin',distortHCs)


cues = Utils.loadbin(buildPATH+"cues.bin").astype(int)
#### NOTE: CHANGE PATSTAT FILENAME IF CHANGED IN MAIN SIM ############
patstat = np.loadtxt('/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_16od_16descs.txt').astype(int)

N1 = H1 * M1
N2 = H2 * M2

cued_net = Utils.findparamval(parfilename,"cued_net")

sims = 20
combined_attractor=False	#Check recall based on combined ltm1 and ltm2 attractor

def check_cuenet(trpats,data,recall_start_time,cued_net = 'LTM1'):
	'''
		Check the state of the cues in the cue network. Returns a list containing colors for the state.
		Red: Pattern not in training patterns.
		Green:  Correct cue pattern
		Orange: Another pattern from the training patterns
	'''

	patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2'])


	recall_score = np.zeros((trpats.shape[0],data.shape[1]-recall_start_time))
	winner = np.zeros(data.shape[1]-recall_start_time)
	recall_colors = np.empty(winner.shape,dtype='object')


	for i,timestep in enumerate(range(recall_start_time, data.shape[1])):
		act = data[:,timestep].astype(float)

		cue_number = int(i/(recall_ngap+recall_nstep))
		cue = cues[cue_number]


		if cued_net == 'LTM1':
			cue_assoc = patstat_pd.loc[patstat_pd.LTM2 == cue, 'LTM1'].tolist()
			H = H2
		elif cued_net == 'LTM2':
			cue_assoc = patstat_pd.loc[patstat_pd.LTM1 == cue, 'LTM2'].tolist()
			H = H1




		for j,pat in enumerate(trpats):
			recall_score[j,i] = 1 - np.dot(act,pat) / (np.linalg.norm(act)*np.linalg.norm(pat))


		#print(np.min(recall_score[:,i]))
		if np.min(recall_score[:,i])<recall_thresh and (np.mean(act.reshape(-1,H),axis=1).mean()>1e-3):
			winner[i] = np.argmin(recall_score[:,i]) 
			if winner[i] in cue_assoc:
				recall_colors[i] = 'tab:green'
			else:
				recall_colors[i] = 'tab:orange'
		else:
			winner[i] = None
			recall_colors[i] = 'tab:red'

	#Get duration of activation of each pattern and threshold 
	#winner_pd = pd.Series(winner).value_counts() 

	#return(winner_pd.index.astype(int).tolist())
	return(winner,recall_colors,recall_score)

def calc_recallwinner(cued_trpats,cue_trpats,cued_data,cue_data,recall_start_time,cued_net = 'LTM1'):
	'''
	get a list containing winners at every time step of recall phase based on smallest cosine distance between input patterns and recall activity
	'''

	patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2'])


	recall_score = np.zeros((cued_trpats.shape[0],cued_data.shape[1]-recall_start_time))
	cue_recall_score = np.zeros((cue_trpats.shape[0],cue_data.shape[1]-recall_start_time))
	winner = np.zeros(cued_data.shape[1]-recall_start_time)
	recall_colors = np.empty(winner.shape,dtype='object')
	true_cues = []	#Stores the actual pattern or closest pattern in the cue network
	if cued_net == 'LTM1':	
		H_cued = H1
		H_cue = H2
	elif cued_net == 'LTM2':
		H_cued = H2
		H_cue = H1
	for i,timestep in enumerate(range(recall_start_time, cued_data.shape[1])):
		cued_act = cued_data[:,timestep].astype(float)
		cue_act = cue_data[:,timestep].astype(float)

		if combined_attractor:
			combined_pats = np.concatenate([cue_trpats,cued_trpats],axis=1)
			combined_act = np.concatenate([cue_act,cued_act])
			for j,pat in enumerate(combined_pats):
				recall_score[j,i] = 1 - np.dot(combined_act,pat) / (np.linalg.norm(combined_act)*np.linalg.norm(pat)) 

		else:
			for j,pat in enumerate(cue_trpats):
				cue_recall_score[j,i] = 1 - np.dot(cue_act,pat) / (np.linalg.norm(cue_act)*np.linalg.norm(pat))

			#if np.min(cue_recall_score[:,i])<recall_thresh: 
			for j,pat in enumerate(cued_trpats):
				recall_score[j,i] = 1 - np.dot(cued_act,pat) / (np.linalg.norm(cued_act)*np.linalg.norm(pat))


		#print(np.min(recall_score[:,i]))
		if np.min(recall_score[:,i])<recall_thresh and ((np.mean(cued_act.reshape(-1,H_cued),axis=1).mean()>1e-3) and (np.mean(cue_act.reshape(-1,H_cue),axis=1).mean()>1e-3)):
		#if np.min(recall_score[:,i])<recall_thresh:
			winner[i] = np.argmin(recall_score[:,i]) 
		else:
			winner[i] = None

	#Get duration of activation of each pattern and threshold 
	#winner_pd = pd.Series(winner).value_counts() 

	#return(winner_pd.index.astype(int).tolist())
	return(winner,recall_colors,recall_score,true_cues)

def confmat_metrics(conf_mat,cue_net,distortHCs,savef=0,doplot=0):
	'''
		Calculate performance metrics for a given confusion matrix
	'''

	#True Positives
	tp = np.diag(conf_mat)

	#false positives -> for each odor category, how often a label was recalled when it was not cued
	fp = np.sum(conf_mat, axis=1) - tp

	#false negatives -> for each odor category, how often was a wrong prediction made when an odor is cued
	fn = np.sum(conf_mat,axis=0) - tp


	#Accuracy -> single value; Overall accuracy of recall
	accuracy = conf_mat.trace()/16#conf_mat.sum()



	#Precision -> category wise; out of all the times an odor label was retrieved, how often was it right
	precision = np.zeros(len(tp))
	for i in range(len(tp)):
		if tp[i]+fp[i] == 0 or np.isnan(tp[i]+fp[i]):
			precision[i] = 0
		else:
			precision[i] = tp[i]/(tp[i]+fp[i])

	#Recall -> category wise; for each odor cue, how often do you get the label right
	recall = np.zeros(len(tp))
	for i in range(len(tp)):
		if tp[i]+fn[i] == 0 or np.isnan(tp[i]+fn[i]):
			recall[i] = 0
		else:
			recall[i] = tp[i]/(tp[i]+fn[i])



	if doplot:
		#Plot recall, precision bar plot
		fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,8))

		for i in range(len(precision)):
			ax1.bar(i,1-precision[i],color=kelly_colors[i])	#1-precision = false discovery rate
			ax2.bar(i,1-recall[i],color=kelly_colors[i])	#1-recall = miss rate

		ax1.set_xticks(np.arange(len(precision)))
		ax1.set_xticklabels(ODORS_en,rotation=45)
		ax2.set_xticks(np.arange(len(precision)))
		ax2.set_xticklabels(ODORS_en,rotation=45)
		ax1.set_ylim(0,1)
		ax2.set_ylim(0,1)
		plt.subplots_adjust(hspace=0.57)
		ax1.set_title('False labelling rate')
		ax2.set_title('Miss rate')
		#fig.suptitle(cue_net+' Performance Metrics')
		fig.suptitle(' Performance Metrics')

		if savef:
			plt.savefig(figPATH+'ConfMatMetrics_HalfNorm_BidirectionalAsso_OrthoLangPats_sims{}_ltm2distortHCs{}'.format(sims,distortHCs2),dpi=300)
			metrics_dict = {'Accuracy':accuracy,'Precision':precision,'Recall':recall}
			with open(figPATH+'ConfMatMetrics_HalfNorm_BidirectionalAsso_OrthoLangPats_sims{}_ltm2distortHCs{}.txt'.format(sims,distortHCs2), 'w') as f:
				print(metrics_dict, file=f)

		plt.show()
	return(accuracy,1-precision,1-recall)


def confusion_mat(trpats1,trpats2,winners,winners2,recall_start_time,single_sim = 0,savef=0):

	if cued_net=='LTM1':
		cue_net = 'LTM2'
		conf_mat = np.zeros([trpats1.shape[0],trpats2.shape[0]])
		cued_pats = trpats1
	elif cued_net=='LTM2':
		cue_net = 'LTM1'
		conf_mat = np.zeros([trpats2.shape[0],trpats1.shape[0]])
		cued_pats = trpats2
	else: raise Exception('confusion_mat(): Invalid cued_net')

	if single_sim:	#If we get confusion for individual runs
		cued_recall_state,cue_recall_state = np.empty([recall_nstep+recall_ngap]),np.empty([recall_nstep+recall_ngap])

		for pat_idx in range(cued_pats.shape[0]):
			cued_recall_state = winners[cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap):(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
			active,counts = np.unique(cued_recall_state,return_counts=True)

			cue_recall_state = winners2[cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap)+recall_nstep:(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
			active_cue,counts_cue = np.unique(cue_recall_state,return_counts=True)

			for i, act in enumerate(active):
				if np.isnan(act):
					continue
				else:
					#print(act,act_cued)
					conf_mat[int(act),cues[pat_idx]] = counts[i]/(recall_nstep+recall_ngap)	#Generates based on intended cue vs cued response

		accuracy,_,_ = confmat_metrics(conf_mat,cue_net,distortHCs2,doplot=0,savef=0)

		return(accuracy)

	else:
		cued_recall_state,cue_recall_state = np.empty([sims,(recall_nstep+recall_ngap)]),np.empty([sims,(recall_nstep+recall_ngap)])

		for pat_idx in range(cued_pats.shape[0]):
			cued_recall_state = winners[:,cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap):(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
			active,counts = np.unique(cued_recall_state,return_counts=True)

			# cue_recall_state = winners2[:,cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap)+recall_nstep:(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
			# active_cue,counts_cue = np.unique(cue_recall_state,return_counts=True)

			####Check the way I calculate confusion. Is this correct?

			#print(active_cue,active)
			# if pat_idx == 0:
			# 	# print(cued_recall_state.shape,cued_recall_state)
			# 	# print(cue_recall_state.shape,cue_recall_state)
			# 	print(active_cue,active)
			# 	print(counts,counts_cue)
			#for i,(act,act_cued) in enumerate(zip(active,active_cue)):
			#print(active)
			for i, act in enumerate(active):
				if np.isnan(act):
					continue
				else:
					#print(act,act_cued)
					conf_mat[int(act),cues[pat_idx]] = counts[i]/(sims*(recall_nstep+recall_ngap))	#Generates based on intended cue vs cued response
					#conf_mat[int(act),int(act_cued)] += counts[i]/(sims*(recall_nstep+recall_ngap))	#Generates based on active pattern in cueing net vs cued response

	#print('Conf mat sum: {}, Conf mat trace: {}, Accuracy: {}'.format(conf_mat.sum(),conf_mat.trace(),conf_mat.trace()/conf_mat.sum()))
	
	accuracy,precision,recall = confmat_metrics(conf_mat,cue_net,distortHCs2,savef=0)
	fig,ax = plt.subplots(1,1,figsize=(8,8))
	plot = ax.imshow(conf_mat,interpolation='none', aspect='auto',cmap = plt.cm.binary,vmin=0,vmax=1)
	cbar = plt.colorbar(plot,ax=ax)
	if combined_attractor:
		ax.set_xlabel('Cues',size=14)
		ax.set_ylabel('Retrieved',size=14)
	else:
		ax.set_xlabel('Cued Odor',size=14)
		ax.set_ylabel('Recalled Label',size=14)
	ax.set_yticks(np.arange(len(ODORS_en)))
	ax.set_yticklabels(ODORS_en)
	ax.set_xticks(np.arange(len(ODORS_en)))
	ax.set_xticklabels(ODORS_en,rotation=45)
	# ax.set_title('Accuracy: {:.2f}'.format(accuracy))
	fig.tight_layout()
	#plt.savefig(figPATH+'ConfMat_cueHC{}'.format(cueHCs),dpi=300)
	if savef:
		if combined_attractor:
			plt.savefig(figPATH+'ConfMat_FullNorm_BidirectionalAsso_sims{}_ltm2distortHCs{}'.format(sims,distortHCs2),dpi=300)
		else:
			np.savetxt(figPATH+'ConfMat_sims{}'.format(sims),conf_mat)
			plt.savefig(figPATH+'ConfMat_plot_sims{}'.format(sims),dpi=300)

	plt.show()

def confusion_mat_indivnets(trpats1,trpats2,winners1,winners2,recall_score1,recall_score2,true_cues,single_sim=0,savef=0):
	'''
	Plot confusion matrix for both nets (pattern completion tests)
	Works if cue sequence is same in both nets
	'''

	conf_mat1 = np.zeros([trpats1.shape[0],trpats1.shape[0]])
	conf_mat2 = np.zeros([trpats2.shape[0],trpats2.shape[0]])

	if single_sim:
		recall_state1,recall_state2 = np.empty([(recall_nstep+recall_ngap)]),np.empty([(recall_nstep+recall_ngap)])

		for pat_idx in range(trpats1.shape[0]):
			recall_state1 = winners1[cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap):(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
			#current_true_cues = true_cues[cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap):(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
			active,counts = np.unique(recall_state1,return_counts=True)

			for i,act in enumerate(active):
				if np.isnan(act):
					continue
				else:
					conf_mat1[int(act),cues[pat_idx]] = counts[i]/(recall_nstep+recall_ngap)

		for pat_idx in range(trpats2.shape[0]):
			recall_state2 = winners2[cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap):(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
			active,counts = np.unique(recall_state2,return_counts=True)

			for i,act in enumerate(active):
				if np.isnan(act):
					continue
				else:
					conf_mat2[int(act),cues[pat_idx]] = counts[i]/(recall_nstep+recall_ngap)

		accuracy1,precision1,recall1 = confmat_metrics(conf_mat1,'LTM1',distortHCs1,savef=0,doplot=0)
		accuracy2,precision2,recall2 = confmat_metrics(conf_mat2,'LTM2',distortHCs2,savef=0,doplot=0)

		return(accuracy1,accuracy2)

	else:
		recall_state1,recall_state2 = np.empty([sims,(recall_nstep+recall_ngap)]),np.empty([sims,(recall_nstep+recall_ngap)])
		for pat_idx in range(trpats1.shape[0]):
			recall_state1 = winners1[:,cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap):(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
			#current_true_cues = true_cues[:,cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap):(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
			active,counts = np.unique(recall_state1,return_counts=True)

			for i,act in enumerate(active):
				if np.isnan(act):
					continue
				else:
					conf_mat1[int(act),cues[pat_idx]] = counts[i]/(sims*(recall_nstep+recall_ngap))

			# cued_cue_pairs = np.stack((recall_state1,current_true_cues),axis=-1).reshape(-1,2) #((sims*(recall nstep+ngap)) x 2 Matrix containing cued-cue pair for each timestep in each simulation)
			# for (cued,cue) in cued_cue_pairs:
			# 	if np.isnan(cued):
			# 		continue
			# 	else:
			# 		conf_mat1[int(cued),int(cue)] += 1/((sims*(recall_nstep+recall_ngap)))


		for pat_idx in range(trpats2.shape[0]):
			recall_state2 = winners2[:,cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap):(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
			active,counts = np.unique(recall_state2,return_counts=True)

			for i,act in enumerate(active):
				if np.isnan(act):
					continue
				else:
					conf_mat2[int(act),cues[pat_idx]] = counts[i]/(sims*(recall_nstep+recall_ngap))

		accuracy1,precision1,recall1 = confmat_metrics(conf_mat1,'LTM1',distortHCs1,savef=0,doplot=0)
		accuracy2,precision2,recall2 = confmat_metrics(conf_mat2,'LTM2',distortHCs2,savef=0,doplot=0)

	fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,8))
	ax1.imshow(conf_mat1,interpolation='none', aspect='auto',cmap = plt.cm.binary)
	ax1.set_xlabel('Cue',size=14)
	ax1.set_ylabel('Retrieved',size=14)
	ax1.set_yticks(np.arange(len(ODORS_en)))
	ax1.set_yticklabels(ODORS_en)
	ax1.set_xticks(np.arange(len(ODORS_en)))
	ax1.set_xticklabels(ODORS_en,rotation=45)
	ax1.set_title('LTM 1 (lang) (Accuracy: {:.2f})'.format(accuracy1))
	#ax1.set_title('LTM 1 (lang)')

	ax2.imshow(conf_mat2,interpolation='none', aspect='auto',cmap = plt.cm.binary)
	ax2.set_xlabel('Cue',size=14)
	ax2.set_ylabel('Retrieved',size=14)
	ax2.set_yticks(np.arange(len(ODORS_en)))
	ax2.set_yticklabels(ODORS_en)
	ax2.set_xticks(np.arange(len(ODORS_en)))
	ax2.set_title('(Accuracy: {:.2f})'.format(accuracy2))
	ax2.set_xticklabels(ODORS_en,rotation=45)
	ax2.set_title('LTM 2 (od) (Accuracy: {:.2f})'.format(accuracy2))
	#ax2.set_title('LTM 2 (od)')
	fig.tight_layout()

	if savef:
		plt.savefig(figPATH+'ConfMat_indivnets_Asso12gain1-0_LTM1distortHCs{}_LTM2distortHCs{}_sims{}_LTM1H{}M{}'.format(distortHCs1,distortHCs2,sims,H1,M1),dpi=300)

	plt.show()

def BWplot(Wij11,Wij22,Wij21,Wij12,trpats1,trpats2,savef=0):
	fig = plt.figure(figsize=(15,8))

	within_wij12,between_wij12 = get_assoc_w(Wij12,trpats1,trpats2,wflag='w12')
	within_wij21,between_wij21 = get_assoc_w(Wij21,trpats1,trpats2,wflag='w21')

	within_wij11,between_wij11 = get_weights_over_pattern(Wij11,trpats1)
	within_wij22,between_wij22 = get_weights_over_pattern(Wij22,trpats2)

	def connectpoints(y1,y2,x,c,ax):
		ax.plot([x,x],[y1,y2],color=c)

	ax1 = fig.add_subplot(221)
	ax1.plot(np.arange(trpats1.shape[0]),within_wij12.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='r')
	ax1.plot(np.arange(trpats1.shape[0]),between_wij12.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='b')
	ax1.set_xticks(np.arange(trpats1.shape[0]))
	for i in range(len(within_wij12.mean(axis=1))):
		connectpoints(within_wij12.mean(axis=1)[i],between_wij12.mean(axis=1)[i],i,kelly_colors[i],ax1)
	# ax1.set_ylabel("Within pats")
	yabs_max = abs(max(ax1.get_ylim(), key=abs))
	yabs_max = abs(max(ax1.get_ylim(), key=abs))
	ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
	ax1.set_title("Wij12")

	ax2 = fig.add_subplot(222)
	ax2.plot(np.arange(trpats1.shape[0]),within_wij21.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='r')
	ax2.plot(np.arange(trpats2.shape[0]),between_wij21.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='b')
	ax2.set_xticks(np.arange(trpats1.shape[0]))
	for i in range(len(within_wij21.mean(axis=1))):
		connectpoints(within_wij21.mean(axis=1)[i],between_wij21.mean(axis=1)[i],i,kelly_colors[i],ax2)
	yabs_max = abs(max(ax2.get_ylim(), key=abs))
	yabs_max = abs(max(ax2.get_ylim(), key=abs))
	ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
	ax2.set_title("Wij21")

	ax3 = fig.add_subplot(223)
	ax3.plot(np.arange(trpats1.shape[0]),within_wij11.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='r')
	ax3.plot(np.arange(trpats1.shape[0]),between_wij11.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='b')
	ax3.set_xticks(np.arange(trpats1.shape[0]))
	for i in range(len(within_wij11.mean(axis=1))):
		connectpoints(within_wij11.mean(axis=1)[i],between_wij11.mean(axis=1)[i],i,kelly_colors[i],ax3)
	yabs_max = abs(max(ax3.get_ylim(), key=abs))
	yabs_max = abs(max(ax3.get_ylim(), key=abs))
	ax3.set_ylim(ymin=-yabs_max, ymax=yabs_max)
	ax3.set_title("Wij11")

	ax4 = fig.add_subplot(224)
	ax4.plot(np.arange(trpats1.shape[0]),within_wij22.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='r')
	ax4.plot(np.arange(trpats1.shape[0]),between_wij22.mean(axis=1),linestyle='None',markersize = 10.0,marker='o',mfc='b')
	ax4.set_xticks(np.arange(trpats1.shape[0]))
	for i in range(len(within_wij22.mean(axis=1))):
		connectpoints(within_wij22.mean(axis=1)[i],between_wij22.mean(axis=1)[i],i,kelly_colors[i],ax4)
	yabs_max = abs(max(ax4.get_ylim(), key=abs))
	yabs_max = abs(max(ax4.get_ylim(), key=abs))
	ax4.set_ylim(ymin=-yabs_max, ymax=yabs_max)
	ax4.set_title("Wij22")

	l = []
	l_labels = []
	for i in range(trpats1.shape[0]):
		l.append(Line2D([0], [0], color=kelly_colors[i], lw=4, ls=linestyles[int(i/trpats1.shape[0])]))
		l_labels.append('{}'.format(ODORS_en[i]))	####NOTE: Need to change this when more patterns are used in lang net

	ax4.legend(l,l_labels,ncol=1,loc='center left',bbox_to_anchor=(1, 0.7),title='Pats',title_fontsize=10,fontsize=8)

	if savef:
		plt.savefig(figPATH+'PatWeights_FullNorm_BidirectionalAsso_sims{}_ltm2distortHCs{}'.format(sims,distortHCs2),dpi=300)

	plt.show()


def get_weights_over_pattern(wij, patterns):
	
	# 10, 90?
	within  = np.zeros((patterns.shape[0],wij.shape[0]*wij.shape[0]))
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
	
def get_assoc_w(wij, pats1, pats2,wflag):
	
	# 10, 90?

	if wflag=='w12':
		pre_pats = pats1
		post_pats = pats2
		
	elif wflag=='w21':
		pre_pats = pats2
		post_pats = pats1

	else:
		raise Exception('get_assoc_w(): Invalid wflag!')
	associated  = np.zeros((patstat.shape[0],N1*N1))
	unassociated = np.zeros((patstat.shape[0], N1*N1))
	# associated  = np.zeros((pats1.shape[0],pats1.shape[1]))
	# unassociated = np.zeros((pats1.shape[0], N1*N1))
		
	print(wij.shape)
	for p in range(patstat.shape[0]):
		
		# get non zero values

		pre_indx = np.nonzero(pre_pats[p])[0]
		post_indx = np.nonzero(post_pats[p])[0]
		c1, c2 = 0, 0


		for i in pre_indx:
			for j in range(post_pats.shape[1]):
				if j in post_indx: 

					# print("pat1: {} pat1_idx: {} pat2_idx: {}, w: {}".format(p,i,j,wij[i,j]))
					associated[p,c1] = wij[i,j] 
					c1 += 1
				else:
					unassociated[p,c2] = wij[i,j] 
					c2 += 1
			
	return associated[:,:c1], unassociated[:,:c2]   


def run_sims():
	# winner = calc_recallwinner()
	# print(winner.shape)
	# plot_assocmat(winner)

	
	winners_cued = [[] for i in range(sims)]
	winners_cues = [[] for i in range(sims)]
	true_cues = [[] for i in range(sims)]
	#accuracy = np.zeros(sims)
	accuracy = np.zeros([sims,2])	#Use when calculating accuracy for separate nets
	recall_score_cued,recall_score_cues = [],[]
	#os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')


	if sims == 0:
			os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
			Wij11 = Utils.loadbin("Wij11.bin",N1,N1) #default size = (simulation steps * N*N, ) #Wij11
			Wij12 = Utils.loadbin("Wij12.bin",N2,N1)
			Wij21 = Utils.loadbin("Wij21.bin",N1,N2)
			Wij22 = Utils.loadbin("Wij22.bin",N2,N2)
			trpats = Utils.loadbin("trpats1.bin",N1)
			trpats2 = Utils.loadbin("trpats2.bin",N2)

			data1 = Utils.loadbin("act1.log",N1).T  #(units,timestep)
			data2 = Utils.loadbin("act2.log",N2).T 

			text_file = open("simstages.txt", "r")
			simstages = [int(line) for line in text_file]
			text_file.close()
			recall_timelogs = simstages[simstages.index(-2)+1:]
			recall_start_time = recall_timelogs[0]
			if cued_net == 'LTM1':
				cue_net = 'LTM2'
				cue_pats = trpats2 #Utils.loadbin("cuepats.bin",N2)
				cued_pats = trpats
				cue_data = data2 #Network that is cueing
				cued_data = data1 #Network being cued

			elif cued_net == 'LTM2':
				cue_net = 'LTM1'
				cue_pats = trpats#Utils.loadbin("cuepats.bin",N1)
				cued_pats = trpats2
				cue_data = data1
				cued_data = data2
			os.chdir(PATH)

	for i in range(sims):
		#Run Sim
		os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
		print('\n\nSimulation No {}\n\n'.format(i+1))
		os.system("./olflangmain1 ")
		################ INIT PARAMS######################################
		trpats = Utils.loadbin("trpats1.bin",N1)
		trpats2 = Utils.loadbin("trpats2.bin",N2)
		data1 = Utils.loadbin("act1.log",N1).T  #(units,timestep)
		data2 = Utils.loadbin("act2.log",N2).T 

		text_file = open("simstages.txt", "r")
		#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
		simstages = [int(line) for line in text_file]
		text_file.close()
		recall_timelogs = simstages[simstages.index(-2)+1:]
		recall_start_time = recall_timelogs[0]
		if cued_net == 'LTM1':
			cue_net = 'LTM2'
			cue_pats = trpats2 #Utils.loadbin("cuepats.bin",N2)
			cued_pats = trpats
			cue_data = data2 #Network that is cueing
			cued_data = data1 #Network being cued

		elif cued_net == 'LTM2':
			cue_net = 'LTM1'
			cue_pats = trpats#Utils.loadbin("cuepats.bin",N1)
			cued_pats = trpats2
			cue_data = data1
			cued_data = data2
		###################################################################
		# if i == 0:

			# Wij11 = Utils.loadbin("Wij11.bin",N1,N1) #default size = (simulation steps * N*N, ) #Wij11
			# Wij12 = Utils.loadbin("Wij12.bin",N2,N1)
			# Wij21 = Utils.loadbin("Wij21.bin",N1,N2)
			# Wij22 = Utils.loadbin("Wij22.bin",N2,N2)

			# if sims==1:
				# Wij11 = np.reshape(Wij11,(1,N1,N1))
				# Wij12 = np.reshape(Wij12,(1,N2,N1))
				# Wij21 = np.reshape(Wij21,(1,N1,N2))
				# Wij22 = np.reshape(Wij22,(1,N2,N2))

		# else:
		# 	if i==1:
		# 		Wij11 = np.stack((Wij11,Utils.loadbin("Wij11.bin",N1,N1)))
		# 		Wij12 = np.stack((Wij12,Utils.loadbin("Wij12.bin",N2,N1)))
		# 		Wij21 = np.stack((Wij21,Utils.loadbin("Wij21.bin",N1,N2)))
		# 		Wij22 = np.stack((Wij22,Utils.loadbin("Wij22.bin",N2,N2)))

		# 	else:
		# 		Wij11 = np.concatenate((Wij11,Utils.loadbin("Wij11.bin",N1,N1)[None]),axis=0)
		# 		Wij12 = np.concatenate((Wij12,Utils.loadbin("Wij12.bin",N2,N1)[None]),axis=0)
		# 		Wij21 = np.concatenate((Wij21,Utils.loadbin("Wij21.bin",N1,N2)[None]),axis=0)
		# 		Wij22 = np.concatenate((Wij22,Utils.loadbin("Wij22.bin",N2,N2)[None]),axis=0)

		### Get recall data for cue and cued network separately
		winners_cued[i],_,recall_score_cued_i,true_cues[i] = calc_recallwinner(cued_pats,cue_pats,cued_data,cue_data,recall_start_time,cued_net)  #hardcoded LTM1 as cued net. Need to generalise for future
		winners_cues[i],_,recall_score_cues_i = check_cuenet(cue_pats,cue_data,recall_start_time,cue_net)

		accuracy[i] = confusion_mat(trpats,trpats2,winners_cued[i],winners_cues[i],recall_start_time,single_sim=1)
		#accuracy[i] = confusion_mat_indivnets(trpats,trpats2,winners_cued[i],winners_cues[i],recall_score_cued_i,recall_score_cues_i,true_cues[i],single_sim=1,savef=0)

		recall_score_cued.append(recall_score_cued_i)
		recall_score_cues.append(recall_score_cues_i)
		#Plot Act Plot
		os.chdir(PATH)

		#os.system("python3 modular_actplot.py "+str(cueHCs))
	# os.system("python3 dualnet_actplot.py")
	print(accuracy.mean(),accuracy.std(),'\n',accuracy)

	# np.savetxt(figPATH+'sims{}_Accuracy_FullNorm_BidirectionalAsso_ltm2distortHCs{}.txt'.format(sims,distortHCs2), accuracy,  fmt='%1.2f',delimiter=',')
	# if sims == 0:
	# 	Wij11mean = Wij11
	# 	Wij22mean = Wij22
	# 	Wij21mean = Wij21
	# 	Wij12mean = Wij12
	# else:
	# 	os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
	# 	Wij11 = Utils.loadbin("Wij11.bin",N1,N1) #default size = (simulation steps * N*N, ) #Wij11
	# 	Wij12 = Utils.loadbin("Wij12.bin",N2,N1)
	# 	Wij21 = Utils.loadbin("Wij21.bin",N1,N2)
	# 	Wij22 = Utils.loadbin("Wij22.bin",N2,N2)
	# 	os.chdir(PATH)
	# else:
	# 	Wij11mean = Wij11.mean(axis=0)
	# 	Wij22mean = Wij22.mean(axis=0)
	# 	Wij21mean = Wij21.mean(axis=0)
	# 	Wij12mean = Wij12.mean(axis=0)

	winners_cued = np.array(winners_cued)
	winners_cues = np.array(winners_cues)
	true_cues = np.array(true_cues)
	recall_score_cued = np.array(recall_score_cued)
	recall_score_cues = np.array(recall_score_cues)

	confusion_mat(trpats,trpats2,winners_cued,winners_cues,recall_start_time,savef=1)
	#confusion_mat_indivnets(trpats,trpats2,winners_cued,winners_cues,recall_score_cued,recall_score_cues,true_cues,savef=1)
	# BWplot(Wij11,Wij22,Wij21,Wij12,trpats,trpats2,savef=1)
	#Figsolflang.plot_wdist(savef=0,savefname=figPATH+'WDist_sims{}_HalfNorm_delay{}_NoAsso21_OrthoLangPats_ltm2distortHCs{}'.format(sims,delay,distortHCs2))
	#plot_assocmat(winners)


def accuracyplot():

	alla_mean_acc = np.zeros(4)
	alla_sem_acc = np.zeros(4) 
	alla_std_acc  = np.zeros(4)
	asso21_mean_acc  = np.zeros(4)
	asso21_sem_acc = np.zeros(4)
	asso21_std_acc = np.zeros(4)
	noasso_mean_acc = np.zeros([4,2])
	noasso_sem_acc = np.zeros([4,2])
	noasso_std_acc = np.zeros([4,2])

	alla_acc = np.zeros([4,sims])
	asso21_acc = np.zeros([4,sims])
	noasso_acc = np.zeros([4,sims,2])
	#Get bidirectional accuracies across noise levels
	fpath = figPATH+'BidirectionalAsso/'
	for i,distort in enumerate([0,2,4,6]):
		fname = 'sims30_Accuracy_bidirectionalAsso_LTM1distortHCs0_LTM2distortHCs{}.txt'.format(distort)
		alla_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		alla_mean_acc[i] = np.mean(alla_acc[i])
		alla_sem_acc[i] = sem(alla_acc[i])
		alla_std_acc[i] = np.std(alla_acc[i])

	#Get asso21 accuracies:
	fpath = figPATH+'Asso21/'
	for i,distort in enumerate([0,2,4,6]):
		fname = 'sims30_Accuracy_Asso21_LTM1distortHCs0_LTM2distortHCs{}.txt'.format(distort)
		asso21_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		asso21_mean_acc[i] = np.mean(asso21_acc[i])
		asso21_sem_acc[i] = sem(asso21_acc[i])
		asso21_std_acc[i] = np.std(asso21_acc[i])

	#Get no asso accuracies:
	fpath = figPATH+'NoAsso/'
	for i,distort in enumerate([0,2,4,6]):
		fname = 'sims30_Accuracy_NoAsso_LTM1distortHCs{}_LTM2distortHCs{}.txt'.format(distort,distort)
		noasso_acc[i] = np.loadtxt(fpath+fname,delimiter=',')
		noasso_mean_acc[i] = np.mean(noasso_acc[i],axis=0)
		noasso_sem_acc[i] = sem(noasso_acc[i],axis=0)
		noasso_std_acc[i] = np.std(noasso_acc[i],axis=0)


	x = np.arange(4)
	width = 0.1
	fig,ax = plt.subplots(1,1,figsize=(15,8))
	rects1 = ax.bar(x - width, alla_mean_acc, width,yerr=alla_sem_acc, label='Bidirectional Associations')
	rects2 = ax.bar(x, asso21_mean_acc, width,yerr=asso21_sem_acc, label='od -> lang Association')
	rects3 = ax.bar(x + width, noasso_mean_acc[:,0], width, yerr= noasso_sem_acc[:,0],label='Lang Net (Local Only)',color='tab:red')
	rects4 = ax.bar(x + width*2, noasso_mean_acc[:,1], width, yerr= noasso_sem_acc[:,1],label='Od Net (Local Only)',color='tab:red',alpha=0.7)
	plt.xticks(x,['No Distortions','2HCs','4HCs','6HCs'])
	plt.ylabel('Average Accuracy ({} sims)'.format(sims))
	plt.legend()
	plt.show()

def accuracyplot2():
	'''
	varying asso12 gain [0,0.1,0.2,0.4,0.5]
	asso21 gain is maintained at 0.5
	'''
	asso12_0_0_mean_acc  = np.zeros(6)
	# asso12_0_1_mean_acc  = np.zeros(4)
	# asso12_0_2_mean_acc  = np.zeros(4)
	# asso12_0_4_mean_acc  = np.zeros(4)
	asso12_0_5_mean_acc  = np.zeros(6)

	asso12_0_0_sem_acc  = np.zeros(6)
	# asso12_0_1_sem_acc  = np.zeros(4)
	# asso12_0_2_sem_acc  = np.zeros(4)
	# asso12_0_4_sem_acc  = np.zeros(4)
	asso12_0_5_sem_acc  = np.zeros(6)

	asso12_0_0_acc = np.zeros([6,sims])
	# asso12_0_1_acc = np.zeros([4,sims])
	# asso12_0_2_acc = np.zeros([4,sims])
	# asso12_0_4_acc = np.zeros([4,sims])
	asso12_0_5_acc = np.zeros([6,sims])

	#Get bidirectional accuracies across noise levels
	fpath = figPATH+'Asso12_0.0_H20M20/'
	for i,distort in enumerate([0,4,8,10,12,16]):
		fname = 'sims30_Accuracy_RandomLangPats.txt'
		asso12_0_0_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		asso12_0_0_mean_acc[i] = np.mean(asso12_0_0_acc[i])
		asso12_0_0_sem_acc[i] = sem(asso12_0_0_acc[i])

	# #Get asso21 accuracies:
	# fpath = figPATH+'Asso12_0.1_H10M10/'
	# for i,distort in enumerate([0,2,4,6]):
	# 	fname = 'sims30_Accuracy_Asso12gain0-1_LTM1distortHCs0_LTM2distortHCs{}.txt'.format(distort)
	# 	asso12_0_1_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
	# 	asso12_0_1_mean_acc[i] = np.mean(asso12_0_1_acc[i])
	# 	asso12_0_1_sem_acc[i] = sem(asso12_0_1_acc[i])

	# #Get no asso accuracies:
	# fpath = figPATH+'Asso12_0.2_H10M10/'
	# for i,distort in enumerate([0,2,4,6]):
	# 	fname = 'sims30_Accuracy_Asso12gain0-2_LTM1distortHCs0_LTM2distortHCs{}.txt'.format(distort)
	# 	asso12_0_2_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
	# 	asso12_0_2_mean_acc[i] = np.mean(asso12_0_2_acc[i])
	# 	asso12_0_2_sem_acc[i] = sem(asso12_0_2_acc[i])

	# fpath = figPATH+'Asso12_0.4_H10M10/'
	# for i,distort in enumerate([0,2,4,6]):
	# 	fname = 'sims30_Accuracy_Asso12gain0-4_LTM1distortHCs0_LTM2distortHCs{}.txt'.format(distort)
	# 	asso12_0_4_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
	# 	asso12_0_4_mean_acc[i] = np.mean(asso12_0_4_acc[i])
	# 	asso12_0_4_sem_acc[i] = sem(asso12_0_4_acc[i])

	fpath = figPATH+'Asso12_0.5_H20M20/'
	for i,distort in enumerate([0,4,8,10,12,16]):
		fname = 'sims30_Accuracy_Asso12gain1-0_LTM1distortHCs0_LTM2distortHCs{}.txt'.format(distort)
		asso12_0_5_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		asso12_0_5_mean_acc[i] = np.mean(asso12_0_5_acc[i])
		asso12_0_5_sem_acc[i] = sem(asso12_0_5_acc[i])


	x = np.arange(6)
	width = 0.1
	fig,ax = plt.subplots(1,1,figsize=(15,8))
	rects1 = ax.bar(x - width/2, asso12_0_0_mean_acc, width,yerr=asso12_0_0_sem_acc, label='Lang -> Od gain = 0.0')
	# rects2 = ax.bar(x - width, asso12_0_1_mean_acc, width,yerr=asso12_0_1_sem_acc, label='Lang -> Od gain = 0.1')
	# rects3 = ax.bar(x, asso12_0_2_mean_acc, width,yerr=asso12_0_2_sem_acc, label='Lang -> Od gain = 0.2')
	# rects4 = ax.bar(x + width, asso12_0_4_mean_acc, width, yerr= asso12_0_4_sem_acc,label='Lang -> Od gain = 0.4')
	rects5 = ax.bar(x + width/2, asso12_0_5_mean_acc, width, yerr= asso12_0_5_sem_acc,label='Lang -> Od gain = 0.5')
	plt.xticks(x,['No Distortions','20% HCs','40% HCs','50% HCs','60% HCs','80% HCs'])
	plt.ylabel('Average Accuracy ({} sims)'.format(sims))
	plt.legend()
	plt.show()

def accuracyplot3():

	fpath = figPATH+'CombinedAttractorTest/'
	for i,distort in enumerate([10]):
		fname = 'sims30_Accuracy_Asso12gain0-0_LTM1distortHCs0_LTM2distortHCs{}.txt'.format(distort)
		asso12_0_0_acc = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		asso12_0_0_mean_acc = np.mean(asso12_0_0_acc)
		asso12_0_0_sem_acc = sem(asso12_0_0_acc)

	fpath = figPATH+'CombinedAttractorTest/'
	for i,distort in enumerate([10]):
		fname = 'sims30_Accuracy_Asso12gain0-5_LTM1distortHCs0_LTM2distortHCs{}.txt'.format(distort)
		asso12_0_5_acc = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		asso12_0_5_mean_acc = np.mean(asso12_0_5_acc)
		asso12_0_5_sem_acc = sem(asso12_0_5_acc)

	fpath = figPATH+'CombinedAttractorTest/'
	for i,distort in enumerate([10]):
		fname = 'sims30_Accuracy_Asso12gain1-0_LTM1distortHCs0_LTM2distortHCs{}.txt'.format(distort)
		asso12_1_0_acc = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		asso12_1_0_mean_acc = np.mean(asso12_1_0_acc)
		asso12_1_0_sem_acc = sem(asso12_1_0_acc)


	x = np.arange(3)
	width = 0.1
	fig,ax = plt.subplots(1,1,figsize=(15,8))
	# rects5 = ax.bar(x - width, asso12_0_0_mean_acc, width, yerr= asso12_0_0_sem_acc,label='Lang -> Od gain = 0.0')
	# rects5 = ax.bar(x, asso12_0_5_mean_acc, width, yerr= asso12_0_5_sem_acc,label='Lang -> Od gain = 0.5')
	# rects5 = ax.bar(x + width, asso12_1_0_mean_acc, width, yerr= asso12_1_0_sem_acc,label='Lang -> Od gain = 1.0')
	rects = ax.bar(x,[asso12_0_0_mean_acc,asso12_0_5_mean_acc,asso12_1_0_mean_acc],yerr=[asso12_0_0_sem_acc,asso12_0_5_sem_acc,asso12_1_0_sem_acc])
	plt.xticks(x,[0.0,0.5,1.0])
	plt.ylabel('Average Accuracy ({} sims)'.format(sims),size=14)
	plt.xlabel('Lang->Od wgain',size=14)
	plt.title('50% HCs Distorted')
	#plt.legend()
	plt.show()

def accuracyplot4():
	'''
		Ortho vs Random lang pats
	'''
	ortho_alla_mean_acc = np.zeros(4)
	random_alla_mean_acc = np.zeros(4)
	ortho_asso21_mean_acc = np.zeros(4)
	random_asso21_mean_acc = np.zeros(4)

	ortho_alla_sem_acc = np.zeros(4)
	random_alla_sem_acc = np.zeros(4)
	ortho_asso21_sem_acc = np.zeros(4)
	random_asso21_sem_acc = np.zeros(4)

	ortho_alla_acc = np.zeros([4,sims])
	random_alla_acc = np.zeros([4,sims])
	ortho_asso21_acc = np.zeros([4,sims])
	random_asso21_acc = np.zeros([4,sims])

	#Get bidirectional accuracies across noise levels
	fpath = figPATH+'Ortho/'
	for i,distort in enumerate([0,4,8,10]):
		fname = 'sims30_Accuracy_BidirectionalAsso_OrthoLangPats_ltm2distortHCs{}.txt'.format(distort)
		ortho_alla_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		print(ortho_alla_acc[i])
		ortho_alla_mean_acc[i] = np.mean(ortho_alla_acc[i])
		ortho_alla_sem_acc[i] = sem(ortho_alla_acc[i])

	for i,distort in enumerate([0,4,8,10]):
		fname = 'sims30_Accuracy_NoAsso12_OrthoLangPats_ltm2distortHCs{}.txt'.format(distort)
		ortho_asso21_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		ortho_asso21_mean_acc[i] = np.mean(ortho_asso21_acc[i])
		ortho_asso21_sem_acc[i] = sem(ortho_asso21_acc[i])

	#Get asso21 accuracies:
	fpath = figPATH+'Random/'
	for i,distort in enumerate([0,4,8,10]):
		fname = 'sims30_Accuracy_BidirectionalAsso_RandomLangPats_ltm2distortHCs{}.txt'.format(distort)
		random_alla_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		random_alla_mean_acc[i] = np.mean(random_alla_acc[i])
		random_alla_sem_acc[i] = sem(random_alla_acc[i])

	for i,distort in enumerate([0,4,8,10]):
		fname = 'sims30_Accuracy_NoAsso12_RandomLangPats_ltm2distortHCs{}.txt'.format(distort)
		random_asso21_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		random_asso21_mean_acc[i] = np.mean(random_asso21_acc[i])
		random_asso21_sem_acc[i] = sem(random_asso21_acc[i])


	x = np.arange(4)
	width = 0.1
	#print(ortho_alla_mean_acc)
	fig,ax = plt.subplots(1,1,figsize=(15,8))
	rects1 = ax.bar(x - width, ortho_alla_mean_acc, width,yerr=ortho_alla_sem_acc, label='Ortho Bidirectional Associations',color='tab:blue')
	rects2 = ax.bar(x, ortho_asso21_mean_acc, width,yerr=ortho_asso21_sem_acc, label='Ortho od -> lang Associations',color='tab:blue',alpha=0.7)
	rects3 = ax.bar(x + width, random_alla_mean_acc, width, yerr= random_alla_sem_acc,label='Random Bidirectional Associations',color='tab:red')
	rects4 = ax.bar(x + width*2, random_asso21_mean_acc, width, yerr= random_asso21_sem_acc,label='Random od -> lang Associations',color='tab:red',alpha=0.7)
	plt.xticks(x,['No Distortions','20%','40%','50%'])
	plt.ylabel('Average Accuracy ({} sims)'.format(sims))
	plt.legend(bbox_to_anchor=(1.0, 1.15))
	plt.show()

def accuracyplot5():
	'''
		Half Norm W2V pats 40% and 50% distortions, bidirectional vs od to lang associations
	'''
	w2v_alla_mean_acc = np.zeros(2)
	w2v_asso21_mean_acc = np.zeros(2)

	w2v_alla_sem_acc = np.zeros(2)
	w2v_asso21_sem_acc = np.zeros(2)

	w2v_alla_acc = np.zeros([2,sims])
	w2v_asso21_acc = np.zeros([2,sims])

	#Get bidirectional accuracies across noise levels
	fpath = figPATH
	for i,distort in enumerate([8,10]):
		fname = 'sims30_Accuracy_HalfNorm_BidirectionalAsso_OrthoLangPatstxt_ltm2distortHCs{}'.format(distort)
		w2v_alla_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		w2v_alla_mean_acc[i] = np.mean(w2v_alla_acc[i])
		w2v_alla_sem_acc[i] = sem(w2v_alla_acc[i])

	for i,distort in enumerate([8,10]):
		fname = 'sims30_Accuracy_HalfNorm_NoAsso12_OrthoLangPatstxt_ltm2distortHCs{}'.format(distort)
		w2v_asso21_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		w2v_asso21_mean_acc[i] = np.mean(w2v_asso21_acc[i])
		w2v_asso21_sem_acc[i] = sem(w2v_asso21_acc[i])



	x = np.arange(2)
	width = 0.1
	#print(ortho_alla_mean_acc)
	fig,ax = plt.subplots(1,1,figsize=(15,8))
	rects1 = ax.bar(x - width/2, w2v_alla_mean_acc, width,yerr=w2v_alla_sem_acc, label='W2V Bidirectional Associations',color='tab:blue')
	rects4 = ax.bar(x + width/2, w2v_asso21_mean_acc, width, yerr= w2v_asso21_sem_acc,label='W2V od -> lang Associations',color='tab:red')
	plt.xticks(x,['40%','50%'])
	plt.ylabel('Average Accuracy ({} sims)'.format(sims))
	plt.legend(bbox_to_anchor=(1.0, 1.15))
	plt.show()

def accuracyplot6():
	'''
		Half Norm, 40% Distortion, W2V lang pats, delays: 5ms,10ms,20ms, bidirectional vs unidirectional (od->lang) associations
	'''
	w2v_alla_mean_acc = np.zeros(3)
	w2v_asso21_mean_acc = np.zeros(3)

	w2v_alla_sem_acc = np.zeros(3)
	w2v_asso21_sem_acc = np.zeros(3)

	w2v_alla_acc = np.zeros([3,sims])
	w2v_asso21_acc = np.zeros([3,sims])

	fpath = figPATH+'W2V/'
	for i,dl in enumerate([5,10,20]):
		fname = 'sims20_Accuracy_HalfNorm_delay{}_BidirectionalAsso_W2VLangPats_ltm2distortHCs8.txt'.format(dl)
		w2v_alla_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		
		w2v_alla_mean_acc[i] = np.mean(w2v_alla_acc[i])
		w2v_alla_sem_acc[i] = sem(w2v_alla_acc[i])

	for i,dl in enumerate([5,10,20]):
		fname = 'sims20_Accuracy_HalfNorm_delay{}_NoAsso12_W2VLangPats_ltm2distortHCs8.txt'.format(dl)
		w2v_asso21_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		w2v_asso21_mean_acc[i] = np.mean(w2v_asso21_acc[i])
		w2v_asso21_sem_acc[i] = sem(w2v_asso21_acc[i])




	x = np.arange(3)
	width = 0.1
	print(w2v_alla_mean_acc)
	fig,ax = plt.subplots(1,1,figsize=(15,8))
	rects1 = ax.bar(x - width/2, w2v_alla_mean_acc, width,yerr=w2v_alla_sem_acc, label='Bidirectional Associations',color='tab:blue')
	rects2 = ax.bar(x + width/2, w2v_asso21_mean_acc, width,yerr=w2v_asso21_sem_acc, label='od -> lang Associations',color='tab:red',alpha=0.7)
	plt.xticks(x,['5','10','20'])
	plt.xlabel('Delay (ms)')
	plt.ylabel('Average Accuracy ({} sims)'.format(sims))
	plt.legend(bbox_to_anchor=(1.0, 1.15))
	plt.show()

def accuracyplot7():
	'''
		Half Norm vs Full Norm BWgain=0 recall stim, 0,20%,40% distortions bidirectional vs unidirectional
	'''
	half_alla_mean_acc = np.zeros(3)
	half_asso21_mean_acc = np.zeros(3)

	half_alla_sem_acc = np.zeros(3)
	half_asso21_sem_acc = np.zeros(3)

	half_alla_acc = np.zeros([3,sims])
	half_asso21_acc = np.zeros([3,sims])

	full_alla_mean_acc = np.zeros(3)
	full_asso21_mean_acc = np.zeros(3)

	full_alla_sem_acc = np.zeros(3)
	full_asso21_sem_acc = np.zeros(3)

	full_alla_acc = np.zeros([3,sims])
	full_asso21_acc = np.zeros([3,sims])

	#BidirectionalAsso Full Norm
	fpath = figPATH
	for i,distort in enumerate([0,4,8]):
		fname = 'FullNorm/sims20_Accuracy_FullNorm_BidirectionalAsso_ltm2distortHCs{}.txt'.format(distort)
		full_alla_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		full_alla_mean_acc[i] = np.mean(full_alla_acc[i])
		full_alla_sem_acc[i] = sem(full_alla_acc[i])

	#Asso21 Full Norm
	for i,distort in enumerate([0,4,8]):
		fname = 'FullNorm/sims20_Accuracy_FullNorm_Asso21Only_ltm2distortHCs{}.txt'.format(distort)
		full_asso21_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		full_asso21_mean_acc[i] = np.mean(full_asso21_acc[i])
		full_asso21_sem_acc[i] = sem(full_asso21_acc[i])


	#BidirectionalAsso HALF Norm
	fpath = figPATH
	for i,distort in enumerate([0,4,8]):
		fname = 'HalfNorm/sims20_Accuracy_HalfNorm_BidirectionalAsso_ltm2distortHCs{}.txt'.format(distort)
		half_alla_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		half_alla_mean_acc[i] = np.mean(half_alla_acc[i])
		half_alla_sem_acc[i] = sem(half_alla_acc[i])

	#Asso21 HALF Norm
	for i,distort in enumerate([0,4,8]):
		fname = 'HalfNorm/sims20_Accuracy_HalfNorm_Asso21Only_ltm2distortHCs{}.txt'.format(distort)
		half_asso21_acc[i] = np.loadtxt(fpath+fname,delimiter=',')[:,0]
		half_asso21_mean_acc[i] = np.mean(half_asso21_acc[i])
		half_asso21_sem_acc[i] = sem(half_asso21_acc[i])



	x = np.arange(3)
	width = 0.1
	#print(ortho_alla_mean_acc)
	fig,ax = plt.subplots(1,1,figsize=(15,8))
	rects1 = ax.bar(x - width, full_alla_mean_acc, width,yerr=full_alla_sem_acc, label=' FullNorm Bidirectional Associations',color='tab:blue',edgecolor='k')
	rects4 = ax.bar(x, full_asso21_mean_acc, width, yerr= full_asso21_sem_acc,label='FullNorm od -> lang Associations',color='tab:blue',hatch='/',edgecolor='k')
	rects1 = ax.bar(x + width, half_alla_mean_acc, width,yerr=half_alla_sem_acc, label='HalfNorm Bidirectional Associations',color='tab:red',edgecolor='k')
	rects4 = ax.bar(x + width*2, half_asso21_mean_acc, width, yerr= half_asso21_sem_acc,label='HalfNorm od -> lang Associations',color='tab:red',hatch='/',edgecolor='k')
	plt.xticks(x,['0%','20%','40%'])
	plt.ylabel('Average Accuracy ({} sims)'.format(sims))
	plt.legend(bbox_to_anchor=(1.0, 1.15))
	plt.show()


run_sims()
#accuracyplot()
#accuracyplot2()
#accuracyplot3()
#accuracyplot4()
#accuracyplot5()
#accuracyplot6()
#accuracyplot7()