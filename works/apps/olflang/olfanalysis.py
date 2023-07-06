import sys, os, select
import csv
import math
import pandas as pd
import numpy as np
import random
import string
from scipy.spatial.distance import cosine
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib import ticker as tck
import seaborn as sns
from scipy import stats
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import Utils

PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'
buildPATH  = '/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/'
figPATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/DualNet/16od_16labs/ReplayActivity(FreeRecall)/'

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

linestyles = [
		'solid',
		'dashed',
		'dashdot',
		'dotted',
	  ((0, (2, 1))), #'dotted'
	  #((0, (5, 1))), #'thick dashed',        
	  # ((0, (2.5, 1))), #'medium dashed',                
	  # ((0, (1, 1))),   #'thin dashed',        
	  ((0, (2, 1, 1, 1, 1, 1))),  #'dashdotdotted',         
	  ]
lines = ['—','- - - -','- . - . ','. . . . ']
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
etrnrep = float(Utils.findparamval(parfilename,"etrnrep"))

assoc_cspeed = float(Utils.findparamval(parfilename,"assoc_cspeed"))
assoc_dist  = float(Utils.findparamval(parfilename,"assoc_dist"))
delay = int(assoc_dist/assoc_cspeed/1e-3)

if distortHCs2>0:
	#distortHCs_ltm1_list = Utils.loadbin(buildPATH+'distortHCs_ltm1.bin',distortHCs) 
	distortHCs_ltm2_list = Utils.loadbin(buildPATH+'distortHCs_ltm2.bin',distortHCs2)

if distortHCs1>0:
	distortHCs_ltm1_list = Utils.loadbin(buildPATH+'distortHCs_ltm1.bin',distortHCs1)

cues = Utils.loadbin(buildPATH+"cues.bin").astype(int)
alternatives = Utils.loadbin(buildPATH+"alternatives.bin").astype(int)
alternatives = alternatives.reshape(-1,4)
#### NOTE: CHANGE PATSTAT FILENAME IF CHANGED IN MAIN SIM ############
patstat_fname = 'patstat_16od_16descs.txt'   #patstat_16od_16descs.txt #patstat_top3snackdescs.txt
patstat = np.loadtxt(PATH+patstat_fname).astype(int)
patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2'])


ODORS_en = ['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Turpentine',
				'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish']


if patstat_fname == 'patstat_16od_16descs.txt':
#16 odor labels
	descs_en = ODORS_en

elif patstat_fname == 'patstat_top3snackdescs.txt':
#Top 3 Descs  list
	descs_en = ['Gasoline', 'Terpentine', 'Shoe.Polish', 'Leather', 'Soap'  , 'Perfume' , 'Cinnamon' , 'Spice' , 'Vanilla' , 'Peppermint', 'Mentol', 'Mint',
				'Banana', 'Fruit', 'Apple', 'Lemon','Orange','Licorice', 'Anise', 'Pine.Needle', 'Garlic', 'Onion', 'Disgusting' , 'Coffee', 'Chocolate', 'Flower',
				'Clove', 'Dentist', 'Pineapple', 'Rose', 'Mushroom', 'Champinjon', 'Fish', 'Herring', 'Seafood']

elif patstat_fname == 'patstat_topdescs5e-01thresh.txt':

	descs_en = ['Gasoline', 'Leather', 'Cinnamon', 'Mint', 'Banana', 'Lemon', 'Licorice',
		 'Shoe.Polish', 'Terpentine', 'Pine.Needle', 'Garlic', 'Onion', 'Coffee',
		 'Chocolate', 'Apple', 'Fruit', 'Dentist', 'Spice', 'Clove',
		 'Perfume', 'Rose', 'Flower', 'Mushroom', 'Fish']


N1 = H1 * M1
N2 = H2 * M2

cued_net = Utils.findparamval(parfilename,"cued_net")

def get_dist(distmat,pat1,pat2):
	if type(distmat) == pd.core.frame.DataFrame:
		distmat = distmat.to_numpy()
	return distmat[pat1,pat2]

def generate_distmat(pats,metric='hamming'):
	'''
	Create a distance matrix based on patterns
	'''
	distmat = np.zeros([pats.shape[0],pats.shape[0]])

	if metric == 'cosine':
		for i,pat1 in enumerate(pats):
			for j,pat2 in enumerate(pats):
				distmat[i,j] = cosine(pat1,pat2) 

	elif metric == 'hamming':
		for i,pat1 in enumerate(pats):
			for j,pat2 in enumerate(pats):
				distmat[i,j] = np.count_nonzero(pat1!=pat2)/pat1.shape[0] 

	elif metric == 'w2v':
		simmat = pd.read_csv('/home/rohan/Documents/Olfaction/W2V/BloggModel_CriticalWords_similarity_sorted(SNACK).csv',index_col=0)
		distmat = 1-simmat
		distmat = distmat.to_numpy()

	elif metric == 'psychophysical':
		simmat = pd.read_csv('/home/rohan/Documents/Olfaction/Odor_similarity_data/Odors_MeanRatedSimilarity.csv',index_col=0)
		distmat = 1-simmat
		distmat = distmat.to_numpy()

	return distmat

def calc_active_pats(act,trpats,recall_start_time = 0):
	'''
		Calculate active pattern during recall at every time step
	'''
	recall_score = np.zeros((trpats.shape[0],act.shape[1]-recall_start_time))
	winner = np.zeros(act.shape[1]-recall_start_time)
	for i,timestep in enumerate(range(recall_start_time, act.shape[1])):
		cur_act = act[:,timestep].astype(float)
		for j,pat in enumerate(trpats):
			recall_score[j,i] = 1 - np.dot(cur_act,pat) / (np.linalg.norm(cur_act)*np.linalg.norm(pat))

		if np.min(recall_score[:,i])<recall_thresh: #and (np.mean(act.reshape(-1,H),axis=1).mean()>1e-3):
			winner[i] = np.argmin(recall_score[:,i]) 
		else:
			winner[i] =None

	return winner

def analyse_freerecall_sequence(net='LTM1'):
	'''
		Analyze sequence of replayed patterns during free recall wrt distance
	'''
	if net == 'LTM1':
		H = H1
		M = M1
		N = N1
		simmat = pd.read_csv('/home/rohan/Documents/Olfaction/W2V/BloggModel_CriticalWords_similarity_sorted(SNACK).csv',index_col=0)
		distmat = 1-simmat
		print(distmat)
		distmat = distmat.to_numpy()
		labels = descs_en
		net_name = 'Lang'
		act = Utils.loadbin(buildPATH+'act1.log'.format(str(net)),N).T
		trpats = Utils.loadbin(buildPATH+'trpats1.bin'.format(str(net)),N)
		metric = 'w2v'

	elif net == 'LTM2':
		H = H2
		M = M2
		N = N2
		simmat = pd.read_csv('/home/rohan/Documents/Olfaction/Odor_similarity_data/Odors_MeanRatedSimilarity.csv',index_col=0)
		distmat = 1-simmat
		distmat = distmat.to_numpy()
		labels = ODORS_en
		net_name = 'Od'
		act = Utils.loadbin(buildPATH+'act1.log'.format(str(net)),N).T
		trpats = Utils.loadbin(buildPATH+'trpats1.bin'.format(str(net)),N)
		metric = 'od ratings'

	elif net == 'combined':
		H = H1+H2
		M = M1+M2
		N = H*M

		trpats1 = Utils.loadbin(buildPATH+'trpats1.bin',N1)
		trpats2 = Utils.loadbin(buildPATH+'trpats2.bin',N2) 
		trpats = np.concatenate([trpats1,trpats2],axis=1)
		act1 = Utils.loadbin(buildPATH+'act1.log',N1).T
		act2 = Utils.loadbin(buildPATH+'act2.log',N2).T
		act = np.concatenate([act1,act2],axis=0)

		print(trpats.shape)
		print(act.shape)

		distmat = generate_distmat(trpats)
		labels = [x+'-'+y for (x,y) in zip(descs_en,ODORS_en)]
		net_name = 'combined'
		metric = 'hamming'
		print(distmat)

	text_file = open(buildPATH+"simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()
	recall_timelogs = simstages[simstages.index(-2)+1:]
	recall_start_time = recall_timelogs[0]

	#Get active pattern at each time step
	winners = calc_active_pats(act,trpats,recall_start_time)

	winners = winners[~np.isnan(winners)]

	curr = winners[0]
	recall_seq = [int(curr)]	#sequence of replayed items
	for i,x in enumerate(winners[1:]):
		if x == curr:
			continue
		else:
			recall_seq.append(int(x))
			curr = x

	print(recall_seq)

	curr = recall_seq[0]
	distances = []
	for i,x in enumerate(recall_seq[1:]):
		print(descs_en[curr],descs_en[x])
		distances.append(get_dist(distmat,curr,x))
		curr = x


	fig,ax = plt.subplots(figsize=(15,8))
	#ax.plot(distances,marker='o')

	x = np.arange(1,len(distances)+1)
	x = x*8

	bars = ax.bar(x,distances,color='k',alpha=0.75)

	x_start = np.array([plt.getp(item, 'x') for item in bars])
	x_end   = x_start+[plt.getp(item, 'width') for item in bars]

	print(x_start)
	#Need to find better way to position rectangles
	rect_width = 2
	rect_height = 0.01
	rect_starty = 0.05
	for i,xpos in enumerate(x):
		rect_startx = xpos - x[0]/1.8

		pat = recall_seq[i]
		rect = patches.Rectangle((rect_startx, rect_starty), rect_width, rect_height, linewidth=1, facecolor=kelly_colors[pat])	

		# if counter%2 == 0:
		# 	ax.annotate(text = labels[pat],xy=(rect_startx,0.4),fontsize=8,rotation=90)
		# else:
		ax.annotate(text = labels[pat],xy=(rect_startx,0.06),fontsize=8,rotation=90)
		ax.add_patch(rect)

		if i == len(x)-1:
			rect_startx = xpos + x[0]/1.8
			pat = recall_seq[i+1]
			rect = patches.Rectangle((rect_startx, rect_starty), rect_width, rect_height, linewidth=1, facecolor=kelly_colors[pat])	
			ax.annotate(text = labels[pat],xy=(rect_startx,0.06),fontsize=8,rotation=90)
			ax.add_patch(rect)


	ax.tick_params(labelbottom=False) 
	ax.set_ylabel('Distance (metric = {})'.format(metric),size=14) 
	if net=='combined':
		ax.set_title('Combined Nets Replay Activity'.format(net, net_name))
	else:
		ax.set_title('{} ({}) Replay Activity'.format(net, net_name))

	ax.axhline(distmat[distmat!=0].mean(),linestyle = '--',color='purple')
	# ax.annotate('Mean Distance',xy = (-1,distmat.to_numpy()[distmat!=0].mean()),fontsize=8)

	ax.axhline(distmat[distmat!=0].min(),linestyle = '--',color='tab:blue')
	# ax.annotate('Min Distance',xy = (-1,distmat.to_numpy()[distmat!=0].min()),fontsize=8)

	ax.axhline(distmat[distmat!=0].max(),linestyle = '--',color='tab:red')
	# ax.annotate('Max Distance',xy = (-1,distmat.to_numpy()[distmat!=0].max()),fontsize=8)

	ax.text(1.02, distmat[distmat!=0].mean(), "Mean Distance", color = 'purple' , transform=ax.get_yaxis_transform())
	ax.text(1.02, distmat[distmat!=0].min(), "Min Distance", color = 'tab:blue' , transform=ax.get_yaxis_transform())
	ax.text(1.02, distmat[distmat!=0].max(), "Max Distance", color = 'tab:red' , transform=ax.get_yaxis_transform())

	plt.show()

def analyse_recall_sequence(recall_act,pats,net='LTM1',ax = None,plot_title = '',col = 0,distmetric='hamming'):

	#Get active pattern at each time step
	winners = calc_active_pats(recall_act,pats)

	winners = winners[~np.isnan(winners)]

	distmat = generate_distmat(pats,metric=distmetric)

	if net == 'LTM1':
		labels = descs_en
	elif net == 'LTM2':
		labels = ODORS_en
	elif net == 'combined':
		#labels = [x+'-'+y for (x,y) in zip(descs_en,ODORS_en)]
		labels = ODORS_en
	curr = winners[0]
	recall_seq = [int(curr)]	#sequence of replayed items
	for i,x in enumerate(winners[1:]):
		if x == curr:
			continue
		else:
			recall_seq.append(int(x))
			curr = x

	print(recall_seq)

	curr = recall_seq[0]
	distances = []
	for i,x in enumerate(recall_seq[1:]):
		print(descs_en[curr],descs_en[x])
		distances.append(get_dist(distmat,curr,x))
		curr = x


	if not ax:
		fig,ax = plt.subplots(figsize=(15,8))
	#ax.plot(distances,marker='o')

	x = np.arange(1,len(distances)+1)
	x = x*8

	bars = ax.bar(x,distances,color='k',alpha=0.75)

	x_start = np.array([plt.getp(item, 'x') for item in bars])
	x_end   = x_start+[plt.getp(item, 'width') for item in bars]

	print(x_start)
	#Need to find better way to position rectangles
	rect_width = 2
	rect_height = 0.05
	rect_starty = distmat[distmat!=0].mean()
	for i,xpos in enumerate(x):
		rect_startx = xpos - x[0]/1.8

		pat = recall_seq[i]
		rect = patches.Rectangle((rect_startx, rect_starty), rect_width, rect_height, linewidth=1, facecolor=kelly_colors[pat])	

		# if counter%2 == 0:
		# 	ax.annotate(text = labels[pat],xy=(rect_startx,0.4),fontsize=8,rotation=90)
		# else:
		ax.annotate(text = labels[pat],xy=(rect_startx,distmat[distmat!=0].mean()+0.1*distmat[distmat!=0].mean()),fontsize=10,rotation=0)
		ax.add_patch(rect)

		if i == len(x)-1:
			rect_startx = xpos + x[0]/1.8
			pat = recall_seq[i+1]
			rect = patches.Rectangle((rect_startx, rect_starty), rect_width, rect_height, linewidth=1, facecolor=kelly_colors[pat])	
			ax.annotate(text = labels[pat],xy=(rect_startx, distmat[distmat!=0].mean()+0.1*distmat[distmat!=0].mean()),fontsize=10,rotation=0)
			ax.add_patch(rect)


	ax.tick_params(labelbottom=False) 
	ax.set_title(plot_title)


	ax.axhline(distmat[distmat!=0].mean(),linestyle = '--',color='purple')
	# ax.annotate('Mean Distance',xy = (-1,distmat.to_numpy()[distmat!=0].mean()),fontsize=8)

	ax.axhline(distmat[distmat!=0].min(),linestyle = '--',color='tab:blue')
	# ax.annotate('Min Distance',xy = (-1,distmat.to_numpy()[distmat!=0].min()),fontsize=8)

	ax.axhline(distmat[distmat!=0].max(),linestyle = '--',color='tab:red')
	# ax.annotate('Max Distance',xy = (-1,distmat.to_numpy()[distmat!=0].max()),fontsize=8)

	if col > 0:
		ax.text(1.02, distmat[distmat!=0].mean(), "Mean Distance", color = 'purple' , transform=ax.get_yaxis_transform())
		ax.text(1.02, distmat[distmat!=0].min(), "Min Distance", color = 'tab:blue' , transform=ax.get_yaxis_transform())
		ax.text(1.02, distmat[distmat!=0].max(), "Max Distance", color = 'tab:red' , transform=ax.get_yaxis_transform())

	#plt.show()


def analyse_cuedrecall_sequence(distmetric = 'hamming'):


	act1 = Utils.loadbin(buildPATH+'act1.log',N1).T
	trpats1 = Utils.loadbin(buildPATH+'trpats1.bin',N1)
	act2 = Utils.loadbin(buildPATH+'act2.log',N2).T
	trpats2 = Utils.loadbin(buildPATH+'trpats2.bin',N2)

	trpats = np.concatenate([trpats1,trpats2],axis=1)
	act = np.concatenate([act1,act2],axis=0)

	cues = Utils.loadbin(buildPATH+'cues.bin')

	text_file = open(buildPATH+"simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()
	recall_timelogs = simstages[simstages.index(-2)+1:]
	recall_start_time = recall_timelogs[0]

	start = recall_start_time
	if cues.shape[0]<=8:
		rows = cues.shape[0]
		cols = 1
	else:
		rows = 8
		cols = 2
	fig,ax = plt.subplots(rows,cols,figsize=(15,20))

	rcounter = 0
	ccounter = 0
	for i,t in enumerate(recall_timelogs[2::2]):
		if rcounter>7:
			rcounter = 0
			ccounter = 1

		if cues.shape[0] == 1:
				analyse_recall_sequence(act[:,start:t],trpats,net = 'combined',ax=ax,plot_title='Cue = {}'.format(descs_en[int(cues[i])]),distmetric = distmetric)
		elif cues.shape[0]>1 and cues.shape[0]<=8:
				print(int(cues[i]))
				analyse_recall_sequence(act[:,start:t],trpats,net = 'combined',ax=ax[rcounter],plot_title='Cue = {}'.format(descs_en[int(cues[i])]),distmetric = distmetric)
		else:
				analyse_recall_sequence(act[:,start:t],trpats,net = 'combined',ax=ax[rcounter][ccounter],plot_title='Cue = {}'.format(descs_en[int(cues[i])]),col = ccounter,distmetric = distmetric)
		start = t

		rcounter +=1
	plt.subplots_adjust(hspace=0.45)
	fig.supylabel('Distance (metric: {})'.format(distmetric))
	plt.show()

def analyze_sumsupport(plot_mode = 'dsup',net='indiv_nets'):
	'''
	Analyze sum of support for each pattern during recall
	can also be used to show sum of adaptations/pattern wise
	'''

	mode1 = Utils.loadbin(buildPATH+'{}1.log'.format(plot_mode),N1).T
	trpats1 = Utils.loadbin(buildPATH+'trpats1.bin',N1)
	mode2 = Utils.loadbin(buildPATH+'{}2.log'.format(plot_mode),N2).T
	trpats2 = Utils.loadbin(buildPATH+'trpats2.bin',N2)

	#trpats = np.concatenate([trpats1,trpats2],axis=1)
	#mode = np.concatenate([mode1,mode2],axis=0)

	H = H1+H2
	p1 = np.zeros(trpats1.shape[0]*H1) 
	p2 = np.zeros(trpats2.shape[0]*H2)
	p_combined = np.zeros(trpats2.shape[0]*H)

	for i in range(trpats1.shape[0]): 
		p1[i*H1:(i+1)*H1] = np.where(trpats1[i] > 0)[0]

	for i in range(trpats2.shape[0]): 
		p2[i*H2:(i+1)*H2] = np.where(trpats2[i] > 0)[0]

	for i in range(trpats.shape[0]): 
		p_combined[i*H:(i+1)*H] = np.where(trpats[i] > 0)[0]

	p1=p1.reshape(trpats1.shape[0],H1)
	p2=p2.reshape(trpats2.shape[0],H2)
	p_combined=p_combined.reshape(trpats.shape[0],H)

	text_file = open(buildPATH+"simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()
	recall_timelogs = simstages[simstages.index(-2)+1:]
	recall_start_time = recall_timelogs[0]


	supsum1 = np.zeros([mode1[:,recall_start_time:].shape[1],trpats1.shape[0]]).T
	supsum2 = np.zeros([mode2[:,recall_start_time:].shape[1],trpats2.shape[0]]).T
	supsum_combined = np.zeros([mode[:,recall_start_time:].shape[1],trpats.shape[0]]).T

	print(supsum1.shape)
	if net == 'indiv_nets':
		for i,t in enumerate(range(recall_start_time,mode1.shape[1])):
			for p in range(p1.shape[0]):
				supsum1[p,i] = mode1[p1[p].astype(int),t].sum()

		for i,t in enumerate(range(recall_start_time,mode2.shape[1])):
			for p in range(p2.shape[0]):
				supsum2[p,i] = mode2[p2[p].astype(int),t].sum()


		act1 = Utils.loadbin(buildPATH+'act1.log',N1).T
		act2 = Utils.loadbin(buildPATH+'act2.log',N2).T
		fig,ax = plt.subplots(4,1,figsize=(20,8),sharex=True)

		colormap1 = ax[0].imshow(act1[:,recall_start_time:],interpolation='none',aspect='auto',cmap = 'jet')
		#cbar1 = plt.colorbar(colormap1,ax=ax[0]) 

		colormap2 = ax[1].imshow(act2[:,recall_start_time:],interpolation='none',aspect='auto',cmap = 'jet')
		#cbar2 = plt.colorbar(colormap2,ax=ax[1]) 

		for i in range(supsum1.shape[0]):
			ax[2].plot(supsum1[i],color=kelly_colors[i],label=descs_en[i])


		for i in range(supsum2.shape[0]):
			ax[3].plot(supsum2[i],color=kelly_colors[i],label=ODORS_en[i])



		plt.xlabel('Time',size=14)
		ax[2].set_ylabel("$sumsup_{LTM1}$",size=14)
		ax[3].set_ylabel("$sumsup_{LTM2}$",size=14)
		ax[2].legend(bbox_to_anchor=(1.1, 1),loc=1)
		# ax[3].legend(bbox_to_anchor=(1.1, 1.05))
		ax[0].set_title('Act LTM1')
		ax[1].set_title('Act LTM2')
		ax[0].set_ylabel('Unit',size=14)
		ax[1].set_ylabel('Unit',size=14)

		ax[2].axhline(y=0,linestyle='--',c='grey',lw=0.5)
		ax[3].axhline(y=0,linestyle='--',c='grey',lw=0.5)

		ax[0].axvline(x=recall_timelogs[1]-recall_start_time,linestyle=':',c='grey')
		ax[1].axvline(x=recall_timelogs[1]-recall_start_time,linestyle=':',c='grey')
		ax[2].axvline(x=recall_timelogs[1]-recall_start_time,linestyle=':',c='grey')
		ax[3].axvline(x=recall_timelogs[1]-recall_start_time,linestyle=':',c='grey')
		plt.show()

	if net == ' combined':
		for t in range(recall_start_time,dsup.shape[1]):
			for i in range(p_combined.shape[0]):
				supsum_combined[i,t] = dsup2[p_combined[i].astype(int),t].sum()

		fig,ax = plt.subplots(1,1,figsize=(20,8))
		for i in range(supsum_combined.shape[1]):
			ax.plot(supsum_combined[i],color=kelly_colors[i])



def get_weightsbetween_pats(pat1,pat2):
	'''
		Get the weights between 2 pats
	'''
	pass

def main():
	#analyse_freerecall_sequence(net='LTM1')
	#analyse_cuedrecall_sequence(distmetric='w2v')
	analyze_sumsupport('dsup')
if __name__ == "__main__":
	main()
