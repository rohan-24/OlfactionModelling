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
import seaborn as sns
from scipy import stats,spatial
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import Utils

PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'
buildPATH  = '/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/'
figPATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/DualNet/Network15x15/Semantization/SimpsonIndex_4Clusters/OdorCueNoise/'

parfilename = PATH+"olflangmain1.par"
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

mokole_colors = [
'#2f4f4f',  #darkslategray
'#800000',  #maroon
'#191970',  #midnightblue
'#006400',  #darkgreen
'#bdb76b',  #darkkhaki
'#ff0000',  #red
'#ffa500',  #orange
'#ffff00',  #yellow
'#0000cd',  #mediumblue
'#00ff00',  #lime
'#00fa9a',  #mediumspringgreen
'#00ffff',  #aqua
'#ff00ff',  #fuchsia
'#6495ed',  #cornflower
'#ff1493',  #deeppink
'#ffc0cb',  #pink
]




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
	  ((0, (5, 1))), #'thick dashed',        
	  ((0, (2.5, 1))), #'medium dashed',                
	  ((0, (1, 1))),   #'thin dashed',        
	  ((0, (2, 1, 1, 1, 1, 1))),  #'dashdotdotted',    
	  ((0, (2, 2, 2, 2, 2, 2))),  #'dashdotdotted',     
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

use_intensity = int(Utils.findparamval(parfilename,"use_intensity"))
if distortHCs2>0:
	#distortHCs_ltm1_list = Utils.loadbin(buildPATH+'distortHCs_ltm1.bin',distortHCs) 
	distort = Utils.loadbin(buildPATH+'distortHCs_ltm2.bin').astype(int)
	split_ids = np.where(distort==-1)[0]
	distortHCs_ltm2_list = []
	for i,idx in enumerate(split_ids):
		if i == 0:
			distortHCs_ltm2_list.append(distort[:idx])
		elif i == len(split_ids)-1:
			distortHCs_ltm2_list.append(distort[split_ids[i-1]+1:idx])
		else:
			distortHCs_ltm2_list.append(distort[split_ids[i-1]+1:split_ids[i]])

	print(distortHCs_ltm2_list)

if distortHCs1>0:
	distortHCs_ltm1_list = Utils.loadbin(buildPATH+'distortHCs_ltm1.bin',distortHCs1)

cues = Utils.loadbin(buildPATH+"cues.bin").astype(int)
alternatives = Utils.loadbin(buildPATH+"alternatives.bin").astype(int)
alternatives = alternatives.reshape(-1,4)
#### NOTE: CHANGE PATSTAT FILENAME IF CHANGED IN MAIN SIM ############
patstat_fname = 'patstat_si_nclusters4_topdescs.txt'   #patstat_16od_16descs.txt #patstat_top3snackdescs.txt #patstat_topdescs5e-01thresh.txt #patstat_topdescs3e-01thresh #patstat_si_nclusters4_topdescs.txt
patstat = np.loadtxt(PATH+patstat_fname).astype(int)

if patstat.shape[1]==2:
	patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2'])
else:
	patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2','trn_effort'])


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

elif patstat_fname == 'patstat_topdescs3e-01thresh.txt':
	
	descs_en = ['Gasoline','Leather','Soap','Perfume','Cinnamon','Mint','Banana','Fruit','Apple','Lemon','Licorice',
				'Turperntine','Shoe.Polish','Pine.Needle','Garlic','Onion','Coffee','Chocolate','Clove','Spice','Dentist',
				'Rose','Flower','Mushroom','Fish','Herring','Seafood']

elif patstat_fname == 'patstat_si_nclusters4_topdescs.txt':

	descs_en = ['gasoline', 'turpentine', 'shoe polish', 'leather', 'soap', 'perfume', 'cinnamon', 'spice', 'vanilla', 'mint', 'banana',
	'fruit', 'lemon', 'orange', 'licorice', 'anise', 'pine needles', 'garlic', 'onion', 'disgusting', 'coffee', 'chocolate', 'apple',
	'flower', 'clove', 'dentist', 'pineapple', 'caramel', 'rose', 'mushroom', 'fish', 'herring', 'shellfish']

elif patstat_fname == 'patstat_si_nclusters2_topdescs.txt':
	descs_en = ['bensin', 'terpentin', 'skokräm', 'läder', 'tvål', 'parfym', 'krydda',
       'gummi', 'kemisk', 'kanel', 'vanilj', 'blomma', 'pepparmint', 'mentol',
       'mint', 'polkagris', 'banan', 'frukt', 'äpple', 'päron', 'citron',
       'apelsin', 'citrus', 'lakrits', 'anis', 'tallbarr', 'fernissa',
       'målarfärg', 'vitlök', 'lök', 'äcklig', 'kaffe', 'choklad', 'nejlika',
       'tandläkare', 'ananas', 'karamell', 'ros', 'svamp', 'champinjon',
       'fisk', 'sill', 'skaldjur', 'illa']

elif patstat_fname == 'patstat_correctdescs_maxassocs4.txt':
	descs_en = ['bensin', 'petroleum', 'bensinmack', 'diesel', 'läder', 
	'skokräm', 'skinn', 'kanel', 'kanelbulle', 'pepparmint', 'mint', 'mentol', 
	'banan', 'skumbanan', 'citron', 'citrus', 'lime', 'citronmeliss', 'lakrits', 'anis', 
	'salmiak', 'saltlakrits', 'terpentin', 'fernissa', 'målarfärg', 'lösningsmedel', 'vitlök', 
	'lök', 'stekt.lök', 'purjolök', 'kaffe', 'kaffesump', 'snabbkaffe', 'kaffeböna', 'äpple', 'tandläkare', 
	'nejlika', 'kryddnejlika', 'sjukhus', 'ananas', 'ros', 'rosenvatten', 'rosenolja', 'svamp', 'champinjon', 
	'kantarell', 'mögelsvamp', 'fisk', 'sill', 'skaldjur', 'räka']

elif patstat_fname == 'patstat_si_nclusters4_topdescs_Age&MMSEFiltered.txt':
	descs_en = ['bensin', 'terpentin', 'tjära', 'läder', 'tvål', 'parfym', 'skokräm', 'kanel', 'vanilj', 'blomma', 
				'mint', 'banan', 'frukt', 'citron', 'apelsin', 'citrus', 'lakrits', 'anis', 'krydda', 'tallbarr', 
				'vitlök', 'lök', 'kaffe', 'choklad', 'äpple', 'nejlika', 'kryddnejlika', 'tandläkare', 'ananas', 
				'jordgubbe', 'karamell', 'ros', 'svamp', 'fisk', 'sill', 'skaldjur']


	####### Top 3 Descs  for reference
	# Bensin ['bensin' 'terpentin' 'skokräm']
	# Läder ['läder' 'tvål' 'parfym']
	# Kanel ['kanel' 'krydda' 'vanilj']
	# Pepparmint ['mint' 'mentol' 'pepparmint']
	# Banan ['banan' 'frukt' 'äpple']
	# Citron ['citron' 'apelsin' 'frukt']
	# Lakrits ['lakrits' 'krydda' 'anis']
	# Terpentin ['skokräm' 'terpentin' 'tallbarr']
	# Vitlök ['vitlök' 'lök' 'äcklig']
	# Kaffe ['kaffe' 'choklad' 'krydda']
	# Äpple ['äpple' 'frukt' 'blomma']
	# Kryddnejlika ['tandläkare' 'krydda' 'nejlika']
	# Ananas ['frukt' 'blomma' 'ananas']
	# Ros ['parfym' 'ros' 'blomma']
	# Svamp ['svamp' 'äcklig' 'champinjon']
	# Fisk ['fisk' 'sill' 'skaldjur']





N1 = H1 * M1
N2 = H2 * M2

cued_net = Utils.findparamval(parfilename,"cued_net")

def calc_recallwinner(cued_trpats,cue_trpats,cued_data,cue_data,recall_start_time,cued_net = 'LTM1'):
	'''
	get a list containing winners at every time step of recall phase based on smallest cosine distance between input patterns and recall activity
	'''


	recall_score = np.zeros((cued_trpats.shape[0],cued_data.shape[1]-recall_start_time))
	cue_recall_score = np.zeros((cue_trpats.shape[0],cue_data.shape[1]-recall_start_time))
	winner = np.zeros(cued_data.shape[1]-recall_start_time)
	recall_colors = np.empty(winner.shape,dtype='object')

	for i,timestep in enumerate(range(recall_start_time, cued_data.shape[1])):
		cued_act = cued_data[:,timestep].astype(float)
		cue_act = cue_data[:,timestep].astype(float)

		# for j,pat in enumerate(cue_trpats):
		# 	cue_recall_score[j,i] = 1 - np.dot(cue_act,pat) / (np.linalg.norm(cue_act)*np.linalg.norm(pat))

		#if np.min(cue_recall_score[:,i])<recall_thresh: 
		# cue_id = int(i/(recall_ngap+recall_nstep))
		# input_cue = cues[cue_id]
		# cuenet_closestpat = np.argmin(cue_recall_score[:,i]) 


		cue_number = int(i/(recall_ngap+recall_nstep))
		cue = cues[cue_number]


		if cued_net == 'LTM1':
			#cue_assoc = patstat_pd.loc[patstat_pd.LTM2 == cuenet_closestpat, 'LTM1'].tolist()
			H = H1
		elif cued_net == 'LTM2':
			#cue_assoc = patstat_pd.loc[patstat_pd.LTM1 == cuenet_closestpat, 'LTM2'].tolist()
			H = H2




		for j,pat in enumerate(cued_trpats):
			if (np.linalg.norm(cued_act) == 0):
				recall_score[j,i] = np.nan
			else:
				recall_score[j,i] = 1 - np.dot(cued_act,pat) / (np.linalg.norm(cued_act)*np.linalg.norm(pat))



		#print(np.min(recall_score[:,i]))
		# if np.min(recall_score[:,i])<recall_thresh:
		#  winner[i] = np.argmin(recall_score[:,i]) 
		#  if winner[i] == input_cue:
		#     recall_colors[i] = 'tab:green'
		#     if ODORS_en[input_cue] == 'Garlic':
		#        print('input_cue:{},winner: {}'.format(input_cue,ODORS_en[int(winner[i])]))
		#  elif (winner[i] in cue_assoc) and (winner[i] != input_cue):
		#     recall_colors[i] = '#96901D'
		#  else:
		#     recall_colors[i] = 'tab:orange'
		# else:
		#  winner[i] = None
		#  recall_colors[i] = 'tab:red'


		if np.min(recall_score[:,i])<recall_thresh and (np.mean(cued_act.reshape(-1,H),axis=1).mean()>1e-3):
			winner[i] = np.argmin(recall_score[:,i]) 

			####NOTE Color coding assumes that number of language patterns is greater than or equal to number of odor patterns
			if cued_net == 'LTM1':
				assocs = patstat_pd.loc[patstat_pd.LTM1==int(winner[i]),'LTM2'].values
				if cue in assocs:
					recall_colors[i] = kelly_colors[cue]
				else:
					recall_colors[i] = kelly_colors[assocs[0]]

			else:
				recall_colors[i] = kelly_colors[patstat_pd.loc[patstat_pd.LTM2==int(winner[i]),'LTM1'].values[0]]#kelly_colors[int(winner[i])]
		else:
			winner[i] = None
			recall_colors[i] = 'white'

	#Get duration of activation of each pattern and threshold 
	#winner_pd = pd.Series(winner).value_counts() 

	#return(winner_pd.index.astype(int).tolist())
	
	return(winner,recall_colors,recall_score)


def check_cuenet(cuepats,data,recall_start_time,cue_net = 'LTM2'):
	'''
		Check the state of the cues in the cue network. Returns a list containing colors for the state.
		Red: Pattern not in training patterns.
		Green:  Correct cue pattern
		Orange: Another pattern from the training patterns
	'''

	recall_score = np.zeros((cuepats.shape[0],data.shape[1]-recall_start_time))
	winner = np.zeros(data.shape[1]-recall_start_time)
	recall_colors = np.empty(winner.shape,dtype='object')

	for i,timestep in enumerate(range(recall_start_time, data.shape[1])):
		act = data[:,timestep].astype(float)

		# cue_number = int(i/(recall_ngap+recall_nstep))
		# cue = cues[cue_number]

		# true_cue_pat = cuepats[cue]

		for j,pat in enumerate(cuepats):
			if (np.linalg.norm(act) == 0):
				recall_score[j,i] = np.nan
			else:
				recall_score[j,i] = 1 - np.dot(act,pat) / (np.linalg.norm(act)*np.linalg.norm(pat)) #Cosine distance



		# if np.min(recall_score[:,i])<recall_thresh:
		#  winner[i] = np.argmin(recall_score[:,i]) 
		#  if winner[i] == cue:
		#     recall_colors[i] = 'tab:green'
		#  else:
		#     recall_colors[i] = 'tab:orange'
		# else:
		#  winner[i] = None
		#  recall_colors[i] = 'tab:red'

		if cued_net=='LTM1':
			H = H2
		elif cued_net=='LTM2':
			H = H1
		if np.min(recall_score[:,i])<recall_thresh and (np.mean(act.reshape(-1,H),axis=1).mean()>1e-3):
			winner[i] = np.argmin(recall_score[:,i]) 
			if cued_net == 'LTM1':
				recall_colors[i] = kelly_colors[int(winner[i])]
			else:
				recall_colors[i] = kelly_colors[patstat_pd.loc[patstat_pd.LTM1==int(winner[i]),'LTM2'].values[0]]
		else:
			winner[i] = None
			recall_colors[i] = 'white'


	return(winner,recall_colors,recall_score)


def plot_recall_full(data1,data2):

	fig,(ax1,ax2) = plt.subplots(2,1,figsize=(15,8),sharex=True,sharey=True)

	colormap1 = ax1.imshow(data1,interpolation='none',aspect='auto',cmap = plt.cm.binary)
	cbar = plt.colorbar(colormap1,ax=ax1) 
	ax1.set_xlabel("Timesteps",fontsize=16)
	ax1.set_title("LTM #1 (lang)",fontsize = 14)
	ax1.set_ylabel("MCs",fontsize=16)
	for i in range(1,H1):
			ax1.axhline(y=i*M1-1,color='tab:blue')


	colormap2 = ax2.imshow(data2,interpolation='none',aspect='auto',cmap = plt.cm.binary)
	cbar2 = plt.colorbar(colormap2,ax=ax2) 

	ax2.set_xlabel("Timesteps",fontsize=16)
	ax2.set_title("LTM #2 (od)",fontsize = 14)
	ax2.set_ylabel("MCs",fontsize=16)
	for i in range(1,H2):
			ax2.axhline(y=i*M2-1,color='tab:blue')

	plt.subplots_adjust(hspace=0.6)

	plt.show()

def shade_silentHCs(ax,fc='tab:orange',phase='recall',recall_start_time=0):
	'''
		Shade silent LTM2 HCs based on odor intensity partial patterns
	'''
	odor_seq = patstat_pd['LTM2'].to_numpy()
	od_intensity_noise = np.loadtxt(PATH+'od_intensity_noise_20HCs_6e-01max_noise.txt').astype(int)
	od_intensity_noise = pd.DataFrame(od_intensity_noise,columns=['Odor','Noise'])
	if phase == 'training':
		partial_pats = Utils.loadbin(buildPATH+"trnphase_partialpats.bin",N2) 
	elif phase == 'recall':
		partial_pats = Utils.loadbin(buildPATH+"recallphase_partialpats.bin",N2)


	HCs = np.arange(H2)

	noisy_cue_counter = 0
	for cue in cues:

		if od_intensity_noise.loc[od_intensity_noise.Odor==cue,'Noise'].values[0] == 0:
			continue
		else:
			start_x = recall_start_time+cue*(recall_ngap+recall_nstep)
			end_x = start_x+(recall_nstep+recall_ngap)
			pat = partial_pats[noisy_cue_counter]
			noisy_cue_counter += 1
			pat_activeHCs = np.where(pat != 0)[0]
			pat_activeHCs = (pat_activeHCs/H2).astype(int)
			pat_silentHCs = list(set(HCs)-set(pat_activeHCs))
			print(pat_activeHCs)
			for h in pat_silentHCs:
				start_y = h*M2
				end_y = (h+1)*M2
				rect = patches.Rectangle((start_x, start_y-.5), end_x-start_x, end_y-start_y-.5, linewidth=1, edgecolor='none', facecolor=fc,alpha=0.2)
				ax.add_patch(rect)




def shade_cuedHCs(cuedHCs_lists,data,recall_start_time,MC,ax,fc='tab:green'):
	
	for pat in range(len(cuedHCs_lists)):
		start_x = recall_start_time+pat*(recall_ngap+recall_nstep)
		end_x = start_x+(recall_nstep+recall_ngap)
		for h in cuedHCs_lists[pat]:
			start_y = h*MC
			end_y = (h+1)*MC
			rect = patches.Rectangle((start_x, start_y-.5), end_x-start_x, end_y-start_y-.5, linewidth=1, edgecolor='none', facecolor=fc,alpha=0.2)
			ax.add_patch(rect)

def get_cuedHCs(net='LTM1'):
	if net=='LTM1':
		cuepats = Utils.loadbin("cues1.bin",N1)
		HC = H1
		MC = M1
	elif net=='LTM2':
		cuepats = Utils.loadbin("cuepats.bin",N2)
		HC = H2
		MC = M2

	# cuedHCs_list = np.zeros([cuepats.shape[0],cueHCs],dtype='int64')
	cuedHCs_list = [[] for i in range(cuepats.shape[0])]
	for pat in range(cuepats.shape[0]):
		counter = 0
		for i in range(cuepats.shape[1]):
			if cuepats[pat][i] == 1:
				
				#cuedHCs_list[pat][counter] = int(i/HC)
				cuedHCs_list[pat].append(int(i/MC))
				counter+=1
		
	return (cuedHCs_list)  

def plot_studylag(net='LTM1'):
	'''
		Display a distribution of lag based on recall replay activity. Useful to analyse sequences.
		Currently set up to work only if non random encoding order
	'''
	os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

	if net == 'LTM1':
		trpats = Utils.loadbin("trpats1.bin",N1)
		act = Utils.loadbin("act1.log",N1).T   #(units,timestep)

	elif net == 'LTM2':
		trpats = Utils.loadbin("trpats2.bin",N2)
		act = Utils.loadbin("act2.log",N2).T   #(units,timestep)

	text_file = open("simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()

	encoding_order = np.arange(trpats.shape[0])

	if simstages[0] == -1 or -1 not in simstages:   #preloaded
		encoding_end = 0
	else:
		encoding_steps = simstages[:simstages.index(-1)]
		encoding_end = encoding_steps[-1]
	recall_timelogs = simstages[simstages.index(-2)+1:]
	recall_start_time = recall_timelogs[0]

	winners,ltm1_col_list,ltm1_recall_score = check_cuenet(trpats,act,recall_start_time,cue_net=net)

	winners = winners[~np.isnan(winners)]

	curr = winners[0]
	recall_seq = [int(curr)]	#sequence of replayed items
	for i,x in enumerate(winners[1:]):
		if x == curr:
		  continue
		else:
		  recall_seq.append(int(x))
		  curr = x

	curr = recall_seq[0]
	lag = []
	for i,x in enumerate(recall_seq[1:]):
		curr_ind = np.where(encoding_order==curr)[0][0]
		next_ind = np.where(encoding_order==x)[0][0]

		lag.append(next_ind-curr_ind)
		curr = x

	x,height = np.unique(lag,return_counts=True)
	print(np.min(lag),np.max(lag))

	nmean  = float(Utils.findparamval(parfilename,"nmean"))
	namp  = float(Utils.findparamval(parfilename,"namp"))
	plt.figure(figsize=(8,8))
	plt.bar(x,height)
	plt.xlabel('Lag',size=14)
	plt.ylabel('Count',size=14)
	#g = sns.displot(lag,bins=np.max(lag)-np.min(lag)+1)
	plt.xticks(np.arange(np.min(lag),np.max(lag)+1,1))
	plt.title(net+' Lag Distribution')
	plt.tight_layout()
	plt.savefig(figPATH+'{}_LagDistribution_npats{}_nmean{}_namp{}'.format(net,trpats.shape[0],nmean,namp).replace('.',','))
	plt.show()

def analyze_replay(net='LTM1'):
	'''
		Analyze replay activity during free recall 
	'''
	os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

	if net == 'LTM1':
		trpats = Utils.loadbin("trpats1.bin",N1)
		act = Utils.loadbin("act1.log",N1).T   #(units,timestep)
		H = H1
		M = M1
	elif net == 'LTM2':
		trpats = Utils.loadbin("trpats2.bin",N2)
		act = Utils.loadbin("act2.log",N2).T   #(units,timestep)
		H = H2
		M = M2
	elif net == 'combined':
		trpats1 = Utils.loadbin("trpats1.bin",N1)
		trpats2 = Utils.loadbin("trpats2.bin",N2)
		trpats = np.concatenate([trpats1,trpats2],axis=1)

	text_file = open("simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()

	if simstages[0] == -1 or -1 not in simstages:   #preloaded
		encoding_end = 0
	else:
		encoding_steps = simstages[:simstages.index(-1)]
		encoding_end = encoding_steps[-1]
	recall_timelogs = simstages[simstages.index(-2)+1:]
	recall_start_time = recall_timelogs[0]

	winners,ltm1_col_list,ltm1_recall_score = check_cuenet(trpats,act,recall_start_time,cue_net=net)

	winners = winners[~np.isnan(winners)]

	replayed_pats,replay_duration = np.unique(winners,return_counts=True)

	p = np.zeros(trpats.shape[0]*H) 
	for i in range(trpats.shape[0]): 
		p[i*H:(i+1)*H] = np.where(trpats[i] > 0)[0]
	p=p.reshape(trpats.shape[0],H)

	p_simmat = np.zeros([p.shape[0],p.shape[0]])
	for i in range(p.shape[0]):
			for j in range(p.shape[0]):
				 p_simmat[i,j]=np.count_nonzero(p[i]==p[j])

	mean_overlaps = np.zeros(p_simmat.shape[0])
	for i,row in enumerate(p_simmat):
		mean_overlaps[i] = row[row!=20].mean()

	

	mean_overlaps = mean_overlaps[replayed_pats.astype(int)]

	replay_duration = (replay_duration - replay_duration.mean())/replay_duration.std()
	mean_overlaps = (mean_overlaps - mean_overlaps.mean())/mean_overlaps.std()

	print(replay_duration,mean_overlaps)	
	fig,ax = plt.subplots(1,1,figsize=[8,8])

	print(mean_overlaps.shape,replay_duration.shape)
	width = 0.5
	x = replayed_pats
	# ax.bar(x-width/2,mean_overlaps[replayed_pats.astype(int)],label='mean overlap',width=width)
	# ax.bar(x+width/2,replay_duration,label='replay duration',width=width)

	slope, intercept, r, p, std_err = stats.linregress(mean_overlaps,replay_duration)

	print(intercept)
	def line_eq(x):
		return slope*x+intercept
	mymodel = list(map(line_eq, mean_overlaps))       

	for i,(rp,rd,mo) in enumerate(zip(replayed_pats,replay_duration,mean_overlaps)):
		plt.scatter(mo,rd,color = kelly_colors[int(rp)])
	plt.plot(mean_overlaps,mymodel,label='y = {:.2f}x + {:.3f} | r = {:.2f}, p = {:.3f}'.format(slope,intercept,r,p))
	plt.xlabel("$Z_{Mean Overlap}$",size=14)
	plt.ylabel("$Z_{Replay Duration}$",size=14)
	plt.title(net)
	plt.legend()
	#plt.savefig(figPATH+'{}_overlap_vs_duration_regplot'.format(net))
	plt.show()

def plot_single_net_states(mode='patwise'):


	os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

	trpats = Utils.loadbin("trpats1.bin",N1)
	act = Utils.loadbin("act1.log",N1).T   #(units,timestep)
	dsup = Utils.loadbin("dsup1.log",N1).T 
	ada = Utils.loadbin("ada1.log",N1).T 
	bwsup = Utils.loadbin("bwsup1.log",N1).T
	inp = Utils.loadbin("inp1.log",N1).T

	text_file = open("simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()


	if simstages[0] == -1 or -1 not in simstages:   #preloaded
		encoding_end = 0
	else:
		encoding_steps = simstages[:simstages.index(-1)]
		encoding_end = encoding_steps[-1]
	recall_timelogs = simstages[simstages.index(-2)+1:]
	recall_start_time = recall_timelogs[0]


 
	plot_recall_start_time = recall_start_time 
	plot_start = 0

	net = 'LTM1'
	pats = trpats
	labels = descs_en

	winners,ltm1_col_list,ltm1_recall_score = check_cuenet(pats,act,recall_start_time,cue_net=net)

	replayed_pats = np.unique(winners[~np.isnan(winners)])
	replayed_pats_count = replayed_pats.shape[0]
	print('No of items retrieved: {}'.format(replayed_pats.shape[0]))
	print(replayed_pats)
	fig = plt.figure(figsize=(20,12))

	if mode == 'state': #shows dynamics as imshow 
		gs = gridspec.GridSpec(ncols=1,nrows=4,hspace=0.5,height_ratios = [2.5,5,5,5],right = 0.9,bottom=0.1,left=0.1)
		gs_topright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.9,bottom=0.56,top=0.73,right=0.95)
		gs_centerright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.9,bottom=0.33,top=0.49,right=0.95)
		gs_bottomright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.9,bottom=0.1,top=0.25,right=0.95)

		ax_ltm1act = fig.add_subplot(gs[1,0])  #Main LTM1 Act plot
		ax_ltm1CosDistLog = fig.add_subplot(gs[0,0],sharex=ax_ltm1act) #Cosine Distance over time between patterns and acitivation state
		ax_ltm1dsup = fig.add_subplot(gs[2,0]) #LTM1 dsup plot
		ax_ltm1ada = fig.add_subplot(gs[3,0])  #LTM1 ada plot
		ax_ltm1act_cbar = fig.add_subplot(gs_topright[0,0],sharey=ax_ltm1act)
		ax_ltm1dsup_cbar = fig.add_subplot(gs_centerright[0,0],sharey=ax_ltm1dsup)
		ax_ltm1ada_cbar = fig.add_subplot(gs_bottomright[0,0],sharey=ax_ltm1ada)

		ax_ltm1act_cbar.axis('off')
		ax_ltm1ada_cbar.axis('off')
		ax_ltm1dsup_cbar.axis('off')
		x = np.arange(plot_recall_start_time,plot_recall_start_time+len(winners))  

		min_err_pats_ltm1 = np.argsort(np.mean(ltm1_recall_score,axis=1))
		ax_ltm1CosDistLog.set_ylim(0,1)
		ax_ltm1CosDistLog.axhline(y=recall_thresh,linestyle='--',c='k',linewidth=0.7)
		ax_ltm1CosDistLog.annotate('cos dist thresh = {}'.format(recall_thresh),xy = (nstep,recall_thresh+0.05),xytext=(nstep,recall_thresh+0.05),fontsize=10)
		for i,ltm1_pat in enumerate(min_err_pats_ltm1):
			ltm1_colors = kelly_colors[ltm1_pat]
			ax_ltm1CosDistLog.plot(x,ltm1_recall_score[ltm1_pat],c=ltm1_colors,linewidth=1.5)#linestyles[int(cued_pat_id/cued_pats.shape[0])])
		
		# ax_ltm1CosDistLog.axis('off')
		ax_ltm1CosDistLog.set_title('Cos Dist')


		colormap1 = ax_ltm1act.imshow(act,interpolation='none',aspect='auto',cmap = plt.cm.binary,vmin=0,vmax=1)
		cbar = plt.colorbar(colormap1,ax=ax_ltm1act_cbar,location='right',pad=0.01)
		ax_ltm1act.set_ylim(N1,-1)
		ax_ltm1act.set_title('Act')
		for i in range(1,H1):
			ax_ltm1act.axhline(y=i*M1-0.5,color='k',linewidth=0.5,alpha=0.5)
		ax_ltm1act.axvline(x=plot_recall_start_time,color='k')
		# ax_ltm1act.set_ylabel("LTM1 \n\n MCs",fontsize=16)

		colormap2 = ax_ltm1dsup.imshow(dsup,interpolation='none',aspect='auto',cmap = 'jet')
		cbar2 = plt.colorbar(colormap2,ax=ax_ltm1dsup_cbar,location='right',pad=0.01)
		ax_ltm1dsup.set_title('Dsup')
		for i in range(1,H1):
			ax_ltm1dsup.axhline(y=i*M1-0.5,color='k',linewidth=0.5,alpha=0.5)
		ax_ltm1dsup.axvline(x=plot_recall_start_time,color='k')
		ax_ltm1dsup.set_ylim(N1,-1)

		colormap3 = ax_ltm1ada.imshow(ada,interpolation='none',aspect='auto',cmap = 'jet')
		cbar3 = plt.colorbar(colormap3,ax=ax_ltm1ada_cbar,location='right',pad=0.01)
		ax_ltm1ada.set_title('Ada')
		for i in range(1,H1):
			ax_ltm1ada.axhline(y=i*M1-0.5,color='k',linewidth=0.5,alpha=0.5)
		ax_ltm1ada.axvline(x=plot_recall_start_time,color='k')
		ax_ltm1ada.set_ylim(N1,-1)

		#plt.savefig(figPATH+'singleNet_stateplot_overlappingpats',dpi=400)

	elif mode == 'patwise':
		gs = gridspec.GridSpec(ncols=1,nrows=3,hspace=0.5,height_ratios = [5,5,5],right = 0.9,bottom=0.1,left=0.1)
		# gs_topright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.9,bottom=0.56,top=0.73,right=0.95)
		# gs_centerright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.9,bottom=0.33,top=0.49,right=0.95)
		# gs_bottomright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.9,bottom=0.1,top=0.25,right=0.95)

		ax_ltm1act = fig.add_subplot(gs[0,0])  #LTM1 Act plot
		#ax_ltm1bwsup = fig.add_subplot(gs[1,0]) #LTM1 Bwsup plot
		ax_ltm1dsup = fig.add_subplot(gs[1,0]) #LTM1 dsup plot
		ax_ltm1ada = fig.add_subplot(gs[2,0])  #LTM1 ada plot

		#ax_ltm1inp = fig.add_subplot(gs[4,0])  #LTM1 inp plot
		# ax_ltm1act_cbar = fig.add_subplot(gs_topright[0,0],sharey=ax_ltm1act)
		# ax_ltm1dsup_cbar = fig.add_subplot(gs_centerright[0,0],sharey=ax_ltm1dsup)
		# ax_ltm1ada_cbar = fig.add_subplot(gs_bottomright[0,0],sharey=ax_ltm1ada)

		# ax_ltm1act_cbar.axis('off')
		# ax_ltm1ada_cbar.axis('off')
		# ax_ltm1dsup_cbar.axis('off')
		x = np.arange(plot_recall_start_time,plot_recall_start_time+len(winners))  
		ax_ltm1act.get_shared_x_axes().join(ax_ltm1act, ax_ltm1dsup,ax_ltm1ada)


		if not replayed_pats.size:
			replayed_pats = [0,14]
		for i, pat_id in enumerate(replayed_pats):
			pat_id = int(pat_id)

			pat = trpats[pat_id]

			active_inds = np.where(pat==1)[0]
			# plot0 = ax_ltm1bwsup.plot(bwsup[active_inds].mean(axis=0),color=kelly_colors[pat_id])
			plot1 = ax_ltm1act.plot(act[active_inds].mean(axis=0),color=kelly_colors[pat_id])
			plot2 = ax_ltm1dsup.plot(dsup[active_inds].mean(axis=0),color=kelly_colors[pat_id])
			plot3 = ax_ltm1ada.plot(ada[active_inds].mean(axis=0),color=kelly_colors[pat_id])
			# plot4 = ax_ltm1inp.plot(inp[active_inds].mean(axis=0),color=kelly_colors[pat_id])
		fig.suptitle('No of items replayed: {}'.format(int(replayed_pats_count)))
		ax_ltm1act.set_title('Act')
		# ax_ltm1bwsup.set_title('Bwsup')
		ax_ltm1dsup.set_title('Dsup')
		ax_ltm1ada.set_title('Ada')
		# ax_ltm1inp.set_title('Inp')

		again  = float(Utils.findparamval(parfilename,"again"))
		igain  = float(Utils.findparamval(parfilename,"igain"))
		taua  = float(Utils.findparamval(parfilename,"taua"))
		adgain  = float(Utils.findparamval(parfilename,"adgain"))
		recuwgain  = float(Utils.findparamval(parfilename,"recuwgain"))

		plt.savefig(figPATH+'singleNet_state_patwise_npats{}_again{}_igain{}_adgain{}_recuwgain{}_inp1e2'.format(trpats.shape[0],again,igain,adgain,recuwgain).replace('.',','),dpi=400)

	plt.show()

def free_recall(argv,savef=0):
	if len(argv)>1:
		plot_mode = argv[1]  #act,inp,dsup,expdsup

	else:
		plot_mode = "act"


	os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')

	trpats = Utils.loadbin(buildPATH+"trpats1.bin",N1)
	trpats2 = Utils.loadbin(buildPATH+"trpats2.bin",N2)
	data1 = Utils.loadbin(plot_mode+"1.log",N1).T   #(units,timestep)
	data2 = Utils.loadbin(plot_mode+"2.log",N2).T 

	text_file = open("simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()

	if  (plot_mode == 'full'):
		plot_recall_full(data1,data2)
	else:

		
		if simstages[0] == -1 or -1 not in simstages:   #preloaded
			encoding_end = 0
		else:
			encoding_steps = simstages[:simstages.index(-1)]
			encoding_end = encoding_steps[-1]
		recall_timelogs = simstages[simstages.index(-2)+1:]
		recall_start_time = recall_timelogs[0]


		plot_data1 = data1 
		plot_data2 = data2
		plot_recall_start_time = recall_start_time 
		plot_start = 0

		cued_net = 'LTM1'
		cue_net = 'LTM2'
		cue_pats = trpats2 #Utils.loadbin("cuepats.bin",N2)
		cued_pats = trpats
		cue_data = data2 #Network that is cueing
		cued_data = data1 #Network being cued
		cue_labels = ODORS_en
		cued_labels = descs_en

		winners,ltm1_col_list,ltm1_recall_score = calc_recallwinner(cued_pats,cue_pats,cued_data,cue_data,recall_start_time,cued_net=cued_net)
		_,ltm2_col_list,ltm2_recall_score = check_cuenet(cue_pats,cue_data,recall_start_time,cue_net=cue_net)

		fig = plt.figure(figsize=(16,12))
		gs_top= gridspec.GridSpec(ncols=1,nrows=2,hspace=0.05,height_ratios = [2,6],right = 0.9,bottom=0.55,left=0.1)
		gs_bottom = gridspec.GridSpec(ncols=1,nrows=2,wspace=0.05,height_ratios = [2,6],top=0.43,right=0.9,left=0.1)
		gs_topright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.91,bottom=0.55,top=0.785,right=0.95) # ncols=1,nrows=1,wspace=0.05,left=0.91,bottom=0.1,top=0.67,right=0.95
		gs_bottomright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.91,top=0.315,right=0.95)
		# gs_bottom = gridspec.GridSpec(3, 1)
		ax_ltm1main = fig.add_subplot(gs_top[1,0])   #Main LTM1 Act plot
		ax_ltm1CosDistLog = fig.add_subplot(gs_top[0,0],sharex=ax_ltm1main)  #Continuous Recall score
		ax_ltm2main = fig.add_subplot(gs_bottom[1,0])   #LTM2 Act plot
		ax_ltm2CosDistLog = fig.add_subplot(gs_bottom[0,0],sharex=ax_ltm2main)  #Continuous Recall score
		ax_ltm1InpDist = fig.add_subplot(gs_topright[0,0],sharey=ax_ltm1main)   #LTM1 unit distribution
		ax_ltm2InpDist = fig.add_subplot(gs_bottomright[0,0],sharey=ax_ltm2main)      

		x = np.arange(plot_recall_start_time,plot_recall_start_time+len(winners))  





		min_err_pats_ltm1 = np.argsort(np.mean(ltm1_recall_score,axis=1))#[:4]  #Get top 3 pats with lowest average error through recall time slot
		min_err_pats_ltm2 = np.argsort(np.mean(ltm2_recall_score,axis=1))#[:4]


		ax_ltm1CosDistLog.set_ylim(0,1)
		ax_ltm2CosDistLog.set_ylim(0,1)
		ax_ltm1CosDistLog.axhline(y=recall_thresh,linestyle='--',c='k',linewidth=0.7)
		ax_ltm2CosDistLog.axhline(y=recall_thresh,linestyle='--',c='k',linewidth=0.7)
		ax_ltm1CosDistLog.annotate('cos dist = {}'.format(recall_thresh),xy = (nstep,recall_thresh+0.05),xytext=(nstep,recall_thresh+0.05),fontsize=10)
		ax_ltm2CosDistLog.annotate('cos dist = {}'.format(recall_thresh),xy = (nstep,recall_thresh+0.05),xytext=(nstep,recall_thresh+0.05),fontsize=10)
		for i,(ltm1_pat,ltm2_pat) in enumerate(zip(min_err_pats_ltm1,min_err_pats_ltm2)):
			ltm1_colors = kelly_colors[ltm1_pat]
			ltm2_colors = kelly_colors[ltm2_pat]
			ax_ltm1CosDistLog.plot(x,ltm1_recall_score[ltm1_pat],c=ltm1_colors,linewidth=1.5)#linestyles[int(cued_pat_id/cued_pats.shape[0])])
			ax_ltm2CosDistLog.plot(x,ltm2_recall_score[ltm1_pat],c=ltm2_colors,linewidth=1.5)


			#label lines
			ax_ltm1CosDistLog.annotate(cued_labels[ltm1_pat],xy=(x[0]+i*500,1.5),annotation_clip=False,color=ltm1_colors,backgroundcolor='#FAF9F6',fontsize=6)
			ax_ltm2CosDistLog.annotate(cue_labels[ltm2_pat],xy=(x[0]+i*500,1.5),annotation_clip=False,color=ltm2_colors,fontsize=6,backgroundcolor='#FAF9F6')


		ax_ltm1CosDistLog.axis('off')
		ax_ltm2CosDistLog.axis('off')
		if plot_mode == 'act':
				vmin = 0
				vmax = 1
		else:
			vmin,vmax = None,None

		colormap1 = ax_ltm1main.imshow(plot_data1,interpolation='none',aspect='auto',cmap = plt.cm.binary,vmin=vmin,vmax=vmax)
		cbar = plt.colorbar(colormap1,ax=ax_ltm1InpDist)
		ax_ltm1main.set_xlabel("Timesteps",fontsize=16)
		ax_ltm1main.set_ylim(N1,-1)
		ax_ltm1main.set_ylabel("LTM1 (lang) \n\n MCs",fontsize=16)
		ax_ltm1main.margins(x=0, y=-0.45)
		for i in range(1,H1):
				ax_ltm1main.axhline(y=i*M1-0.5,color='tab:blue',linewidth=0.5,alpha=0.5)


		colormap2 = ax_ltm2main.imshow(plot_data2,interpolation='none',aspect='auto',cmap = plt.cm.binary,vmin=vmin,vmax=vmax)
		cbar2 = plt.colorbar(colormap2,ax=ax_ltm2InpDist)
		ax_ltm2main.set_xlabel("Timesteps",fontsize=16)
		ax_ltm2main.set_ylabel("LTM2 (od) \n\n MCs",fontsize=16)
		ax_ltm2main.set_ylim(N2,-1)
		#ax_ltm2main.margins(x=0, y=-0.45) 
		for i in range(1,H2):
				ax_ltm2main.axhline(y=i*M2-0.5,color='tab:blue',linewidth=0.5,alpha=0.5)
				
		ax_ltm1main.get_shared_x_axes().join(ax_ltm1main, ax_ltm2main)
		ax_ltm1main.get_shared_y_axes().join(ax_ltm1main, ax_ltm2main)

		ax_ltm1main.axvline(x=plot_recall_start_time,color='orange')
		ax_ltm2main.axvline(x=plot_recall_start_time,color='orange')
		p1 = np.zeros(trpats.shape[0]*H1) 
		p2 = np.zeros(trpats2.shape[0]*H2)

		for i in range(trpats.shape[0]): 
			p1[i*H1:(i+1)*H1] = np.where(trpats[i] > 0)[0] 

		for i in range(trpats2.shape[0]): 
			p2[i*H2:(i+1)*H2] = np.where(trpats2[i] > 0)[0] 


		p1_units, p1_counts = np.unique(p1, return_counts=True)   
		p2_units, p2_counts = np.unique(p2, return_counts=True)   
		ax_ltm1InpDist.barh(p1_units,p1_counts,align='center',height=1)
		ax_ltm2InpDist.barh(p2_units,p2_counts,align='center',height=1)



		for i,gap in enumerate(recall_timelogs):
			if i%2 == 0:
				ax_ltm1main.axvline(x=gap,color='orange')
				ax_ltm2main.axvline(x=gap,color='orange')
			else:
				if i>1:
					ax_ltm1main.axvline(x=gap,color='grey',ls = '--')
					ax_ltm2main.axvline(x=gap,color='grey',ls = '--')

		ax_ltm1InpDist.axis('off')
		ax_ltm2InpDist.axis('off')

		if savef:
			again  = float(Utils.findparamval(parfilename,"again"))
			taua  = float(Utils.findparamval(parfilename,"taua"))
			adgain  = float(Utils.findparamval(parfilename,"adgain"))
			taup  = float(Utils.findparamval(parfilename,"taup"))
			recuwgain  = float(Utils.findparamval(parfilename,"recuwgain"))

			plt.savefig(figPATH+'{}plot_overlapping_npats{}_again{}_taua{}_adgain{}_taup{}_recuwgain{}_inp1e2'.format(plot_mode,trpats.shape[0],again,taua,adgain,taup,recuwgain).replace('.',','),dpi=400)

		plt.show()

def cued_recall(argv,savef=0):

	if len(argv)==2:
		plot_mode = argv[1]  #act,inp,dsup,expdsup
	elif len(argv)==3:
		plot_mode = argv[1]
		sim_no = argv[2]

	else:
		plot_mode = "act"


	os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')



	if  (plot_mode == 'full'):
		data1 = Utils.loadbin("act1.log",N1).T   #(units,timestep)
		data2 = Utils.loadbin("act2.log",N2).T 

		plot_recall_full(data1,data2)
	else:
		trpats = Utils.loadbin("trpats1.bin",N1)
		trpats2 = Utils.loadbin("trpats2.bin",N2)
		data1 = Utils.loadbin(plot_mode+"1.log",N1).T   #(units,timestep)
		data2 = Utils.loadbin(plot_mode+"2.log",N2).T 

		text_file = open("simstages.txt", "r")
		#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
		simstages = [int(line) for line in text_file]
		text_file.close()

		if simstages[0] == -1 or -1 not in simstages:   #preloaded
			encoding_end = 0
		else:
			encoding_steps = simstages[:simstages.index(-1)]
			encoding_end = encoding_steps[-1]




		recall_timelogs = simstages[simstages.index(-2)+1:]
		recall_start_time = recall_timelogs[0]

		recall_cue_steps = [x for i,x in enumerate(recall_timelogs) if i%2==0]

		if cued_net == 'LTM1':
			cue_net = 'LTM2'
			cue_pats = trpats2 #Utils.loadbin("cuepats.bin",N2)
			cued_pats = trpats
			cue_data = data2 #Network that is cueing
			cued_data = data1 #Network being cued
			cue_labels = ODORS_en
			cued_labels = descs_en

		elif cued_net == 'LTM2':
			cue_net = 'LTM1'
			cue_pats = trpats#Utils.loadbin("cuepats.bin",N1)
			cued_pats = trpats2
			cue_data = data1
			cued_data = data2
			cue_labels = descs_en
			cued_labels = ODORS_en

		winners,cued_col_list,cued_recall_score = calc_recallwinner(cued_pats,cue_pats,cued_data,cue_data,recall_start_time,cued_net=cued_net)
		cuenet_winners,cue_col_list,cue_recall_score = check_cuenet(cue_pats,cue_data,recall_start_time,cue_net=cue_net)


		#show main plot from last epoch of assoc training
		# if etrnrep > 1:
		#  assoc_lastepoch = simstages[:simstages.index(etrnrep-1)][-1]
		# else:
		#  assoc_lastepoch = encoding_end

		# plot_data1 = data1 
		# plot_data2 = data2
		# plot_recall_start_time = recall_start_time 
		# plot_start = 0

		# plot_data1 = data1[:,encoding_end:]
		# plot_data2 = data2[:,encoding_end:]
		# plot_recall_start_time = recall_start_time - encoding_end
		# plot_start = encoding_end

		# plot_data1 = data1[:,assoc_lastepoch:]
		# plot_data2 = data2[:,assoc_lastepoch:]
		# plot_recall_start_time = recall_start_time - assoc_lastepoch
		# plot_start = assoc_lastepoch
	
		# assoc_trn_end = simstages[:simstages.index(-2)][-1]

		plot_recall_start_time = 0 #recall_start_time - assoc_trn_end
		plot_start = recall_start_time
		plot_data1 = data1[:,plot_start:]
		plot_data2 = data2[:,plot_start:]


		#print(cued_recall_score[:,0],cued_pats.shape)

		#confusion_mat(trpats,trpats2,winners)

		fig = plt.figure(figsize=(20,12))

		if cued_net == 'LTM1':
			gs_top= gridspec.GridSpec(ncols=1,nrows=3,hspace=0.05,height_ratios = [2,0.25,6],right = 0.9,bottom=0.55,left=0.1)
			gs_bottom = gridspec.GridSpec(ncols=1,nrows=3,wspace=0.05,height_ratios = [2,0.25,6],top=0.43,right=0.9,left=0.1)
			gs_topright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.91,bottom=0.55,top=0.785,right=0.95)
			gs_bottomright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.91,top=0.315,right=0.95)
			#gs_bottom = gridspec.GridSpec(3, 1)
			ax_ltm1main = fig.add_subplot(gs_top[2,0])   #Main LTM1 Act plot
			ax_cuedCosDistLog = fig.add_subplot(gs_top[0,0],sharex=ax_ltm1main)  #Continuous Recall score
			ax_cuedRecallState = fig.add_subplot(gs_top[1,0],sharex=ax_ltm1main) #Cued state
			ax_ltm2main = fig.add_subplot(gs_bottom[2,0])   #LTM2 Act plot
			ax_cueCosDistLog = fig.add_subplot(gs_bottom[0,0],sharex=ax_ltm2main)   #Continuous Recall score
			ax_cueRecallState = fig.add_subplot(gs_bottom[1,0],sharex=ax_ltm2main)  #Cue State
			ax_ltm1InpDist = fig.add_subplot(gs_topright[0,0],sharey=ax_ltm1main)   #LTM1 unit distribution
			ax_ltm2InpDist = fig.add_subplot(gs_bottomright[0,0],sharey=ax_ltm2main) #LTM2 unit distribution

			###### Plot pattern legends
			# l = []
			# l_labels = []
			# for i in range(cued_pats.shape[0]):
			#  l.append(Line2D([0], [0], color=kelly_colors[i], lw=4, ls=linestyles[int(i/cued_pats.shape[0])]))
			#  l_labels.append('{}'.format(ODORS_en[i])) ####NOTE: Need to change this when more patterns are used in lang net

			# ax_ltm1InpDist.legend(l,l_labels,ncol=2,loc='center left',bbox_to_anchor=(1, 0.5),title='LTM1 Pats',title_fontsize=10,fontsize=8)

			# l,l_labels = [],[]
			# for i in range(cue_pats.shape[0]):
			#  l.append(Line2D([0], [0], color=kelly_colors[i], lw=4, ls=linestyles[int(i/cue_pats.shape[0])]))
			#  l_labels.append('{}'.format(ODORS_en[i]))

			# ax_ltm2InpDist.legend(l,l_labels,ncol=2,loc='center left',bbox_to_anchor=(1, 0.5),title ='LTM2 Pats',title_fontsize=10,fontsize=8)
		



		elif cued_net == 'LTM2':
			gs_top= gridspec.GridSpec(ncols=1,nrows=3,hspace=0.05,height_ratios = [1.8,0.25,6],right = 0.9,bottom=0.55,left=0.1)
			gs_bottom = gridspec.GridSpec(ncols=1,nrows=3,wspace=0.05,height_ratios = [2,0.25,6],top=0.43,right=0.9,left=0.1)
			gs_topright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.91,bottom=0.55,top=0.785,right=0.95)
			gs_bottomright = gridspec.GridSpec(ncols=1,nrows=1,wspace=0.05,left=0.91,top=0.315,right=0.95)
			#gs_bottom = gridspec.GridSpec(3, 1)

			ax_ltm1main = fig.add_subplot(gs_top[2,0])   #Main LTM1 Act plot
			ax_cueCosDistLog = fig.add_subplot(gs_top[0,0],sharex=ax_ltm1main)   #Continuous Cos Dist (Cue Net)
			ax_cueRecallState = fig.add_subplot(gs_top[1,0],sharex=ax_ltm1main)  #Cue state (Cue Net)
			ax_ltm2main = fig.add_subplot(gs_bottom[2,0])   #LTM2 Act plot
			ax_cuedCosDistLog = fig.add_subplot(gs_bottom[0,0],sharex=ax_ltm2main)  #Continuous Cos Dist (Cued Net)
			ax_cuedRecallState = fig.add_subplot(gs_bottom[1,0],sharex=ax_ltm2main) #Cued State (Cued Net)
			ax_ltm1InpDist = fig.add_subplot(gs_topright[0,0],sharey=ax_ltm1main)   #LTM1 unit distribution
			ax_ltm2InpDist = fig.add_subplot(gs_bottomright[0,0],sharey=ax_ltm2main) #LTM2 unit distribution


			# x = np.arange(plot_recall_start_time,plot_recall_start_time+len(winners))

			# ax_cuedRecallState.scatter(x=x,y = np.ones(len(x)), c=cue_col_list,s=20,marker='|')
			# ax_cuedRecallState.set_ylim(0.8,2)
			# ax_cuedRecallState.axis('off')

			# ax4.scatter(x=x,y = np.ones(len(x)), c=col_list,s=20,marker='|')
			# ax4.set_ylim(0.8,2)
			# ax4.axis('off') 



		##########################Plot the cue and cued network recall state above the main act plot for each network   
		x = np.arange(plot_recall_start_time,plot_recall_start_time+len(winners))  
		ax_cuedRecallState.scatter(x=x,y = np.ones(len(x)), c= cued_col_list,s=20,marker='|')
		ax_cuedRecallState.set_ylim(0.8,2)
		ax_cuedRecallState.axis('off')

		ax_cueRecallState.scatter(x=x,y = np.ones(len(x)), c=cue_col_list,s=20,marker='|')
		ax_cueRecallState.set_ylim(0.8,2)
		ax_cueRecallState.axis('off') 

		######################Plot continuous recall cosine distances 
		# for i,pat in enumerate(cued_recall_score):
		#  ax_cuedCosDistLog.plot(x,pat,c=kelly_colors[i])

		x_start = plot_recall_start_time
		for i,cue_id in enumerate(cues):
			x_end = x_start+(recall_nstep+recall_ngap)
			x = np.arange(x_start,x_end)

			y = np.arange(x_start-plot_recall_start_time,x_end-plot_recall_start_time)
			pats_cued = cued_recall_score[:,y]
			pats_cue = cue_recall_score[:,y]


				

			min_err_pats_cued = np.argsort(np.nanmin(pats_cued,axis=1))#[:5]  #Get top 5 pats with lowest errors within cue period
			min_err_pats_cue = np.argsort(np.nanmin(pats_cue,axis=1))#[:5]


			# min_err_pats_cued = alternatives[i]  #When having multiple cued alternatives see olflangmain1.recall_extendedcue()
			# min_err_pats_cue = alternatives[i]


			# min_err_pats_cued = np.argsort(np.mean(np.sort(pats_cued)[:,:int((recall_nstep+recall_ngap)/10)],axis=1))[:3]   #Get top 3 pats with lowest average error over 10% recall time duration 
			# min_err_pats_cue = np.argsort(np.mean(np.sort(pats_cue)[:,:int((recall_nstep+recall_ngap)/10)],axis=1))[:3]

			if cued_net == 'LTM1':
				#Get associated odors to the descriptors
				min_err_patstat = patstat_pd[patstat_pd.LTM1.isin(min_err_pats_cued)].reset_index(drop=True) #LTM2.values.tolist()
				associated_ods = min_err_patstat.LTM2.values.tolist()

				#Keep track of od and duplicates to associate linestyle to them
				temp_pair = []
				temp_stack = []
				for s in associated_ods:
					 if s in temp_stack:
						  counter += 1
						  temp_pair.append((s,counter))
					 else:
						  counter = 0
						  temp_pair.append((s,counter))
						  temp_stack.append(s)

				counter_df = pd.DataFrame(temp_pair,columns = ['LTM2','Counts'])


				#merge counter with min_err_patstat for ease of access
				counter_df = min_err_patstat.join(counter_df.Counts)
				
			cued_y_pos_counter = 0 #y position of line label
			cue_y_pos_counter = 0
			for j,(cue_pat_id,cued_pat_id) in enumerate(zip(min_err_pats_cue,min_err_pats_cued)):

				#Match shared descriptor color to odors:

				if cued_net == 'LTM1':
					
					###Match colors to current cue (in case of multiple associations)
					if cue_id in patstat_pd.loc[patstat_pd.LTM1==int(cued_pat_id),'LTM2'].values:
						cued_colors = kelly_colors[cue_id]
					else:
						# print(cued_pat_id,patstat_pd.loc[patstat_pd.LTM1==cued_pat_id,'LTM2'].values)
						# print(cued_pat_id)
						cued_colors = kelly_colors[patstat_pd.loc[patstat_pd.LTM1==cued_pat_id,'LTM2'].values[0]]

					cued_ls = linestyles[counter_df.loc[counter_df.LTM1 == cued_pat_id,'Counts'].values[0]]
					cue_colors = kelly_colors[cue_pat_id]
					cued_labels = descs_en
					cue_labels = ODORS_en
				else:

					cued_colors = kelly_colors[cued_pat_id]
					cue_colors = kelly_colors[patstat_pd.loc[patstat_pd.LTM1==cue_pat_id,'LTM2'].values[0]]
					cued_labels = ODORS_en
					cue_labels = descs_en
					cued_ls = linestyles[counter_df.loc[counter_df.LTM2 == cued_pat_id,'Counts'].values[0]]
				
				#Plot cosine distance lines
				ax_cuedCosDistLog.plot(x,pats_cued[cued_pat_id],c=cued_colors,linewidth=1.5,linestyle=cued_ls)#linestyles[int(cued_pat_id/cued_pats.shape[0])])
				ax_cueCosDistLog.plot(x,pats_cue[cue_pat_id],c=cue_colors,linewidth=1.5,linestyle=linestyles[int(cue_pat_id/cue_pats.shape[0])])

				#label lines
				# if cued_pat_id in winners[x_start:x_end]:
				# 	ax_cuedCosDistLog.annotate(cued_labels[cued_pat_id]+' ('+lines[counter_df.loc[counter_df.LTM1 == cued_pat_id,'Counts'].values[0]]+ ') ',xy=(x[0]+len(x)/10,cued_y_pos_counter/4+1),annotation_clip=False,color=cued_colors,backgroundcolor='#FAF9F6',fontsize=6)
				# 	cued_y_pos_counter+=1
				# if cue_pat_id in cuenet_winners[x_start:x_end]:
				# 	ax_cueCosDistLog.annotate(cue_labels[cue_pat_id],xy=(x[0]+len(x)/10,cue_y_pos_counter/4+1),annotation_clip=False,color=cue_colors,fontsize=6,backgroundcolor='#FAF9F6')
				# 	cue_y_pos_counter+=1

			#ax_cuedCosDistLog.annotate(ODORS_en[i],xy=(x[0]+len(x)/10,-0.25),annotation_clip=False,color=kelly_colors[cued_pat_id],backgroundcolor='#FAF9F6',fontsize=6)
			if cued_net == 'LTM1':
				cue_colors = kelly_colors[cue_id]
				label_annot_yloc = -0.25
			else:
				cue_colors = kelly_colors[patstat_pd.loc[patstat_pd.LTM1==i,'LTM2'].values[0]]
				label_annot_yloc = -0.15
			
			####NOTE: This isnt working well when multiple cues are used in one recall phase. Need to check
			same_item_counter = -1
			for k,item in enumerate(cuenet_winners[x_start:x_end]):
				#if k%100==0: print(k,item)
				if np.isnan(item):
					continue
					same_item_counter = -1 #Check if same item reactivates successively
				elif same_item_counter==item:
					continue
				else:
					same_item_counter = item
					ax_cueCosDistLog.annotate(cue_labels[int(item)],xy=(plot_recall_start_time+ i*(recall_nstep+recall_ngap)+k,label_annot_yloc),annotation_clip=False,color='k',fontsize=10,backgroundcolor='#FAF9F6')

			same_item_counter = -1
			for k,item in enumerate(winners[x_start:x_end]):
				if np.isnan(item):
					continue
					same_item_counter = -1 #Check if same item reactivates successively
				elif same_item_counter==item:
					continue
				else:
					same_item_counter = item
					
					ax_cuedCosDistLog.annotate(cued_labels[int(item)],xy=(plot_recall_start_time+ i*(recall_nstep+recall_ngap)+k,-0.15),annotation_clip=False,color='k',fontsize=10,backgroundcolor='#FAF9F6') #cued_col_list[int(item)]

			#Recall end marker
			ax_cuedCosDistLog.axvline(x=x_end,linestyle=':',c='k',linewidth=0.5)
			ax_cueCosDistLog.axvline(x=x_end,linestyle=':',c='k',linewidth=0.5)
			x_start = x_end
	
		ax_cuedCosDistLog.axhline(y=recall_thresh,linestyle='--',c='k',linewidth=0.7)
		ax_cuedCosDistLog.annotate('cos dist = {}'.format(recall_thresh),xy = (nstep,recall_thresh+0.05),xytext=(nstep,recall_thresh+0.05),fontsize=16)
		#ax_cueCosDistLog.annotate('Cue:',xy = (plot_recall_start_time-(recall_nstep+recall_ngap),label_annot_yloc),xytext=(plot_recall_start_time-(recall_nstep+recall_ngap),label_annot_yloc),fontsize=10,annotation_clip=False)

		#ax_cuedCosDistLog.text(params['initialization_time']+50, 0.9, 'training order ({} epochs)'.format(params['n_train_epochs']), fontsize=10)
		ax_cuedCosDistLog.set_xlim(0,plot_data1.shape[1])
		ax_cuedCosDistLog.axis('off')

		ax_cueCosDistLog.axhline(y=recall_thresh,linestyle='--',c='k',linewidth=0.7)
		ax_cueCosDistLog.annotate('cos dist = {}'.format(recall_thresh),xy = (nstep,recall_thresh+0.05),xytext=(nstep,recall_thresh+0.05),fontsize=16)
		ax_cueCosDistLog.set_xlim(0,plot_data1.shape[1])
		ax_cueCosDistLog.axis('off')


		# add_secondary_cuelabels(cue_mode='Least_lang_overlap',plot_start=plot_start,ax=ax_cuedCosDistLog)

		################# Plot main act/inp/dsup plots
		if plot_mode == 'act':
				vmin = 0
				vmax = 1
		else:
			vmin,vmax = None,None
		colormap1 = ax_ltm1main.imshow(plot_data1,interpolation='none',aspect='auto',cmap = plt.cm.binary,vmin=vmin,vmax=vmax)
		cbar = plt.colorbar(colormap1,ax=ax_ltm1InpDist)
		#ax_ltm1main.set_xlabel("Timesteps",fontsize=16)
		ax_ltm1main.set_ylim(N1,-1)
		ax_ltm1main.set_ylabel("LTM1 (lang) \n\n MCs",fontsize=16)
		# ax_ltm1main.margins(x=0, y=-0.45)
		# for i in range(1,H1):
		# 		ax_ltm1main.axhline(y=i*M1-0.5,color='tab:blue',linewidth=0.5,alpha=0.5)


		colormap2 = ax_ltm2main.imshow(plot_data2,interpolation='none',aspect='auto',cmap = plt.cm.binary,vmin=vmin,vmax=vmax)
		cbar2 = plt.colorbar(colormap2,ax=ax_ltm2InpDist)
		ax_ltm2main.set_xlabel("Timesteps",fontsize=16)
		ax_ltm2main.set_ylabel("LTM2 (od) \n\n MCs",fontsize=16)
		ax_ltm2main.set_ylim(N2,-1)
		#ax_ltm2main.margins(x=0, y=-0.45) 
		# for i in range(1,H2):
		# 		ax_ltm2main.axhline(y=i*M2-0.5,color='tab:blue',linewidth=0.5,alpha=0.5)
				
		ax_ltm1main.get_shared_x_axes().join(ax_ltm1main, ax_ltm2main)
		ax_ltm1main.get_shared_y_axes().join(ax_ltm1main, ax_ltm2main)
		# if cueHCs>0:
		#  ax_ltm1main.axhline(y=cueHCs*M1-1,color='tab:red')
		#  ax_ltm2main.axhline(y=cueHCs*M2-1,color='tab:red')

		##################### INTENSITY Ratings BASED SILENT HC SHADING
		if use_intensity:
			shade_silentHCs(ax_ltm2main,fc='tab:orange',phase='recall',recall_start_time=plot_recall_start_time)

		if cueHCs>0:
			if partial_mode == 'uniform':
				cuedHCs_lists1 = np.tile(np.arange(cueHCs), (trpats.shape[0], 1))  
				cuedHCs_lists2 = np.tile(np.arange(cueHCs), (trpats2.shape[0], 1)) 
			else:
				# cuedHCs_lists1 = get_cuedHCs('LTM1') 
				cuedHCs_lists2 = get_cuedHCs('LTM2') 
				
			# shade_cuedHCs(cuedHCs_lists1,plot_data1,plot_recall_start_time,M1,ax_ltm1main)
			shade_cuedHCs(cuedHCs_lists2,plot_data2,plot_recall_start_time,M2,ax_ltm2main)


		# if distortHCs>0:
		#  if cued_net== 'LTM1':
		#     shade_cuedHCs(distortHCs_ltm2_list,plot_data2,plot_recall_start_time,M2,ax_ltm2main,fc='tab:red')
		#  elif cued_net=='LTM2':
		#     shade_cuedHCs(distortHCs_ltm1_list,plot_data1,plot_recall_start_time,M1,ax_ltm1main,fc='tab:red')


		if distortHCs1>0:
			shade_cuedHCs(distortHCs_ltm1_list,plot_data1,plot_recall_start_time,M1,ax_ltm1main,fc='tab:red')
		if distortHCs2>0:
			shade_cuedHCs(distortHCs_ltm2_list,plot_data2,plot_recall_start_time,M2,ax_ltm2main,fc='tab:red')

		start = plot_recall_start_time

		while start<plot_recall_start_time+len(winners):
			ax_ltm1main.axvline(x=start,color='orange')
			ax_ltm2main.axvline(x=start,color='orange')
			start += recall_nstep+recall_ngap


		plot_recall_vlines(plot_start,recall_timelogs,mode='multi_cue',ax1=ax_ltm1main,ax2=ax_ltm2main)#ax_cueannotate = ax_cuedCosDistLog)


		# if cued_net == 'LTM1':
		#  ax1.set_title("LTM #1 (lang)",fontsize = 14)
		#  ax4.set_title("LTM #2 (od)",fontsize = 14)
		# else:
			# ax1.set_title("LTM #1 (lang)",fontsize = 14)
			# ax4.set_title("LTM #2 (od)",fontsize = 14)


		######################## Plot pattern unit distribution
		p1 = np.zeros(trpats.shape[0]*H1) 
		p2 = np.zeros(trpats2.shape[0]*H2)

		for i in range(trpats.shape[0]): 
			p1[i*H1:(i+1)*H1] = np.where(trpats[i] > 0)[0] 

		for i in range(trpats2.shape[0]): 
			p2[i*H2:(i+1)*H2] = np.where(trpats2[i] > 0)[0] 


		# ax4.hist(p1,bins=H1*M1,range=(0,H1*M1),orientation='horizontal') 
		# ax5.hist(p2,bins=H2*M2,range=(0,H2*M2),orientation='horizontal')


		p1_units, p1_counts = np.unique(p1, return_counts=True)   
		p2_units, p2_counts = np.unique(p2, return_counts=True)   
		ax_ltm1InpDist.barh(p1_units,p1_counts,align='center',height=1)
		ax_ltm2InpDist.barh(p2_units,p2_counts,align='center',height=1)



		# ax4.set_ylim(100,0)
		# ax5.set_ylim(100,0)
		# ax4.margins(y=0)
		# ax5.margins(y=0)

		ax_ltm1InpDist.axis('off')
		ax_ltm2InpDist.axis('off')

	again  = float(Utils.findparamval(parfilename,"again"))
	taua  = float(Utils.findparamval(parfilename,"taua"))
	adgain  = float(Utils.findparamval(parfilename,"adgain"))
	taup  = float(Utils.findparamval(parfilename,"taup"))
	recuwgain  = float(Utils.findparamval(parfilename,"recuwgain"))
	assowgain  = float(Utils.findparamval(parfilename,"assowgain"))
	bgain  = float(Utils.findparamval(parfilename,"bgain"))
	thres  = float(Utils.findparamval(parfilename,"thres"))
	nmean  = float(Utils.findparamval(parfilename,"nmean"))
	namp = float(Utils.findparamval(parfilename,"namp"))

	#plt.savefig(figPATH+'{}plot_overlapping_npats{}_again{}_taua{}_adgain{}_taup{}_recuwgain{}_inp1e2'.format(plot_mode,trpats.shape[0],again,taua,adgain,taup,recuwgain).replace('.',','),dpi=400)


	#fig.tight_layout()
	savef=0
	if savef:
		print('SavingFigure')
		plt.savefig(figPATH+'CueHCs9_Recuwgain1,8_Actplot.png',dpi=300)
		#plt.savefig(figPATH+'Actplot_Overlap_langbgain1,5_adgain{}_recuwgain{}_assowgain{}_nmean{}_namp{}'.format(adgain,recuwgain,assowgain,nmean,namp).replace('.',','),dpi=400)
		#plt.savefig(figPATH+'Actplot_sim{}'.format(sim_no),dpi=400)
		#plt.savefig(figPATH+'Actplot_MaxFlip8HCs_OneShot_recuwgain{}_assowgain{}_maxperturb0,2'.format(recuwgain,assowgain).replace('.',','),dpi=400)
		#plt.savefig(figPATH+'actplot_LeastSimLang_Incong_ltm1cuelength250_ltm2cuelength250_gap375_ltm2distortHCs{}'.format(distortHCs2),dpi=400)
		#plt.savefig(figPATH+'Extended_OdorCue_actplot_iwgain15x_16ods',dpi=400)
		#plt.savefig(figPATH+'24langpats_actplot_HalfNorm_adgain100_taua0,1_tausd0,5',dpi=400)
		#plt.savefig(figPATH+'{}plot_repeatedcue_terpentine_again{}_taua{}_adgain{}_taup{}_recuwgain{}_inp1e2_gapinp5'.format(plot_mode,again,taua,adgain,taup,recuwgain).replace('.',','),dpi=400)

	plt.show()


def add_secondary_cuelabels(cue_mode,plot_start,ax=None,cue_seq = []):
	'''
		Add cue labels for secondary cueing: when a congruent or incongruent cue is presented in the second network 
		after an initial cue is presented in the first

		Valid only for 16 odor 16 labels case
	'''
	if cue_mode == 'Least_lang_overlap':
		cue_seq = [10,15,15,15,0,0,15,4,1,12,0,0,9,0,0,2]
	elif cue_mode == 'Most_lang_overlap':
		cue_seq = [7,6,11,5,10,3,1,6,15,2,12,2,5,4,7,8]
	elif cue_mode == 'Least_od_overlap':
		cue_seq = [4,8,8,8,0,15,8,15,10,0,8,15,8,8,2,5]
	elif cue_mode == 'Most_od_overlap':
		cue_seq = [7,7,10,2,10,10,3,0,15,1,12,1,10,10,15,14]
	elif cue_mode == 'custom':
		if cue_seq == []:
			raise Exception('add_secondary_cuelabels(): No cue_seq provided!')

	else:
		raise Exception('add_secondary_cuelabels(): Invalid cue_mode')


	fname = buildPATH+'cong_incong_recallstages.txt'
	f = open(fname, "r")
	recall_stages = [int(line)-plot_start for line in f]
	f.close()

	for i,step in enumerate(recall_stages[1::3]):
		ax.annotate(ODORS_en[cue_seq[i]],xy=(step,-.15),annotation_clip=False,color=kelly_colors[cue_seq[i]],fontsize=7,backgroundcolor='#FAF9F6')

def plot_recall_vlines(plot_start,recall_timelogs,mode='regular',ax1=None,ax2=None,ax_cueannotate = None):
	'''
	Plot vertical lines to separate input stimulus period and inputless period
	'''

	# print(recall_start_time)

	if mode == 'cong_incong':
		fname = buildPATH+'cong_incong_recallstages.txt'
		f = open(fname, "r")
		recall_stages = [int(line)-plot_start for line in f]
		f.close()

		for i,step in enumerate(recall_stages):
			if i%3==0:
				ax2.axvline(x=step,color='grey',linestyle='-.',lw=0.5)
			else:
				ax1.axvline(x=step,color='grey',linestyle='-.',lw=0.5)

	elif mode == 'extended_cue':
		fname = buildPATH+'extendedcue_recallstages.txt'
		f = open(fname, "r")
		recall_stages = [int(line)-plot_start for line in f]
		f.close()

		cue_alternatives = alternatives.flatten()
		counter = 0
		print(cue_alternatives.shape,cue_alternatives)
		for i,step in enumerate(recall_stages):
			# if i%2==0:
			ax2.axvline(x=step,color='grey',linestyle='-.',lw=0.5)
			ax1.axvline(x=step,color='grey',linestyle='-.',lw=0.5)

			if i%2 == 0:
				start_x = step
				end_x = start_x+recall_ngap/4 
				start_y  = 0
				end_y = N1
				if ax_cueannotate:
					ax_cueannotate.annotate(ODORS_en[cue_alternatives[counter]],xy=(step,-.15),annotation_clip=False,color=kelly_colors[cue_alternatives[counter]],fontsize=7,backgroundcolor=[0.98,0.97,0.96,0.3])
					counter += 1
			rect1 = patches.Rectangle((start_x, start_y-.5), end_x-start_x, end_y-start_y-.5, linewidth=1, edgecolor='none', facecolor='grey',alpha=0.2)
			rect2 = patches.Rectangle((start_x, start_y-.5), end_x-start_x, end_y-start_y-.5, linewidth=1, edgecolor='none', facecolor='grey',alpha=0.2)
			ax1.add_patch(rect1)
			ax2.add_patch(rect2)

			# else:
				# ax2.axvline(x=step,color='grey',linestyle='-.',lw=0.5)
				# ax1.axvline(x=step,color='grey',linestyle='-.',lw=0.5)


	elif mode == 'multi_cue': #A cue is presented multiple times in one cue phase
		for i,step in enumerate(recall_timelogs[1:]):
				ax2.axvline(x=step-plot_start,color='grey',linestyle='-.',lw=0.5)

	elif mode == 'regular':
		for step in recall_timelogs[1::2]:
			if cued_net == 'LTM2':
				ax1.axvline(x=step-plot_start,color='grey',linestyle='-.',lw=0.5)
			else:
				ax2.axvline(x=step-plot_start,color='grey',linestyle='-.',lw=0.5)






def confusion_mat(trpats1,trpats2,winners,savef=0):

	if cued_net=='LTM1':
		cue_net = 'LTM2'
		conf_mat = np.zeros([trpats1.shape[0],trpats2.shape[0]])
		cued_pats = trpats1
	elif cued_net=='LTM2':
		cue_net = 'LTM1'
		conf_mat = np.zeros([trpats2.shape[0],trpats1.shape[0]])
		cued_pats = trpats2

	recall_state = np.empty([(recall_nstep+recall_ngap)])
	print(trpats1.shape)
	for pat_idx in range(cued_pats.shape[0]):
		recall_state = winners[cues.tolist().index(pat_idx)*(recall_nstep+recall_ngap):(cues.tolist().index(pat_idx)+1)*(recall_nstep+recall_ngap)]
		active,counts = np.unique(recall_state,return_counts=True)

		for i,act in enumerate(active):
			if np.isnan(act):
				continue
			else:
				conf_mat[int(act),cues[pat_idx]] = counts[i]/(recall_nstep+recall_ngap)

	fig,ax = plt.subplots(1,1,figsize=(8,8))
	ax.imshow(conf_mat,interpolation='none', aspect='auto',cmap = plt.cm.binary)
	ax.set_xlabel(cue_net+'(Cues)',size=14)
	ax.set_ylabel(cued_net+'(Cued)',size=14)
	ax.set_yticks(np.arange(len(ODORS_en)))
	ax.set_yticklabels(ODORS_en)
	ax.set_xticks(np.arange(len(ODORS_en)))
	ax.set_xticklabels(ODORS_en,rotation=45)
	fig.tight_layout()
	#plt.savefig(figPATH+'ConfMat_cueHC{}'.format(cueHCs),dpi=300)
	if savef:
		plt.savefig(figPATH+'ConfMat_distortHCs{}'.format(distortHCs2),dpi=300)
		


def main(argv):
	cued_recall(argv,savef=1)
	#free_recall(argv)
	#plot_single_net_states()
	#plot_studylag(net='LTM2')
	#analyze_replay(net='LTM1')
if __name__ == "__main__":
	main(sys.argv[:])




#########	Tasks for tomorrow:
#########		Analyze study lag for ortho and overlapping pats, under different noise conditions
#########		Analyze replay activity for overlapping pats wrt pattern representations and time duration a pattern stays active