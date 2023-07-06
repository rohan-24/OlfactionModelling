'''
This code runs multiple simulations to calculate omission (blank) error rates for each odor cue across simulations

'''
import sys, os, select
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr as corr
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from statannotations.Annotator import Annotator
import itertools as it
import pickle
from scipy.spatial import distance
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
import Utils

PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'
buildPATH  = '/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/'
figPATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/DualNet/Network15x15/Semantization/Main/DescriptorAddition&Removal/Gasoline_SingleAssoc/BatchRuns/'

parfilename = PATH+"olflangmain1.par"
od_colors = ['#a40e08',  
		  '#f3d62e', 
		  '#9e642b', 
		  '#514746', 
		  '#623e1b', 
		  '#1d4bbf',
		  '#786e27', 
		  '#606c81', 
		  '#a1e55a', 
		  '#686059', 
		  '#f7b69e',   
		  '#c92464', 
		  '#017058',       
		  '#f99252',
		  '#f48cb6', 
		  '#9b9c82']

ODORS_en = ['Gasoline', 'Leather', 'Cinnamon', 'Pepparmint','Banana', 'Lemon', 'Licorice', 'Terpentine',
			'Garlic', 'Coffee', 'Apple', 'Clove','Pineapple', 'Rose', 'Mushroom', 'Fish']

si = [0.10222183361385999, 0.030231216954294548, 0.07687778801781656, 0.18428997191458849, 0.14396490846060442, 0.07863397456525209, 0.07726495308835027, 0.02292309328794809, 0.0992946323606006, 0.15543696278300378, 0.08877279214067345, 
				0.04825722044158754, 0.054474843302845424, 0.1117470193214893, 0.2125568832697502, 0.06914421001409138]

correct_labels_en= {'Gasoline': ['gasoline'],
 'Leather': ['leather','shoe polish'],
 'Cinnamon': ['cinnamon'],
 'Pepparmint': ['mint'],
 'Banana': ['banana'],
 'Lemon': ['lemon'],
 'Licorice': ['licorice'],
 'Terpentine': ['terpentine'],
 'Garlic': ['garlic'],
 'Coffee': ['coffee'],
 'Apple': ['apple'],
 'Clove': ['clove','dentist'],
 'Pineapple': ['pineapple'],
 'Rose': ['rose'],
 'Mushroom': ['mushroom'],
 'Fish': ['fish','herring','shellfish']}

H1 = int(Utils.findparamval(parfilename,"H"))
M1 = int(Utils.findparamval(parfilename,"M"))
N1 = H1*M1
H2 = int(Utils.findparamval(parfilename,"H2"))
M2 = int(Utils.findparamval(parfilename,"M2"))
N2 = H2*M2
recuwgain = float(Utils.findparamval(parfilename,"recuwgain"))
assowgain = float(Utils.findparamval(parfilename,"assowgain"))
recall_nstep = int(Utils.findparamval(parfilename,"recallnstep"))
recall_ngap = int(Utils.findparamval(parfilename,"recallngap"))
recall_thresh  = float(Utils.findparamval(parfilename,"recall_thresh"))
runflag = Utils.findparamval(parfilename,"runflag")
cueHCs = int(Utils.findparamval(parfilename,"cueHCs"))

cues = Utils.loadbin(buildPATH+"cues.bin").astype(int)

patstat = np.loadtxt('/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_si_nclusters4_topdescs.txt')
if patstat.shape[1]==2:
	patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2'])
else:
	patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2','trn_effort'])


###NOTE: CHANGE DESCS WHEN PATSTAT CHANGES
###Si_4Clusters
descs = ['gasoline', 'turpentine', 'shoe polish', 'leather', 'soap', 'perfume', 'cinnamon', 'spice', 'vanilla', 'mint', 'banana',
	'fruit', 'lemon', 'orange', 'licorice', 'anise', 'pine needles', 'garlic', 'onion', 'disgusting', 'coffee', 'chocolate', 'apple',
	'flower', 'clove', 'dentist', 'pineapple', 'caramel', 'rose', 'mushroom', 'fish', 'herring', 'shellfish']

###Si_4clusters_SNACKData_Filetered
# descs = ['bensin', 'terpentin', 'tjära', 'läder', 'tvål', 'parfym', 'skokräm', 'kanel', 'vanilj', 'blomma', 
# 			'mint', 'banan', 'frukt', 'citron', 'apelsin', 'citrus', 'lakrits', 'anis', 'krydda', 'tallbarr', 
# 			'vitlök', 'lök', 'kaffe', 'choklad', 'äpple', 'nejlika', 'kryddnejlika', 'tandläkare', 'ananas', 
# 			'jordgubbe', 'karamell', 'ros', 'svamp', 'fisk', 'sill', 'skaldjur']
#########PARAMS############
sims = 200 #number of simulations
recall_descriptors = {} #A nested dictionary in the form {OdorCue: {Descriptor:Duration}}

#simulated_omrates= [0.,  0.1, 0.7, 0.7, 0.8, 0.7, 0.6, 0.6, 0.,  0.,  0.3, 0.6, 0.5, 0.,  0.,  0., ]

def analyze_item_replay_transitions(trpats1,trpats2,act1,act2,stabiltiy_thres=200):
	'''
	Analyze recall replay to understand pattern transition sequences
	'''

	if runflag == "full":
		text_file = open(buildPATH+"simstages.txt", "r")
		#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
		simstages = [int(line) for line in text_file]
		text_file.close()

		recall_timelogs = simstages[simstages.index(-2)+1:]
		recall_start_time = recall_timelogs[0]

		print(recall_start_time)
		act1 = act1[:,recall_start_time:]
		act2 = act2[:,recall_start_time:]


	winners_ltm1,_ = calc_recallwinner(trpats1,act1) 
	winners_ltm2,_ = calc_recallwinner(trpats2,act2) 

	winners_ltm1 = [x for x in winners_ltm1 if x != -1]
	winners_ltm2 = [x for x in winners_ltm2 if x != -1]
	ltm1_sequence,ltm2_sequence = [],[]


	### Get sequence of attractors in ltm1 and ltm 2
	prev = None
	count = 1
	for i in winners_ltm1:
		if i != prev:
			if count>stabiltiy_thres:
				ltm1_sequence.append(i)
				count = 0
				prev = i
			else:
				count+=1

	prev = None
	count = 1
	for i in winners_ltm2:
		if i != prev:
			if count>stabiltiy_thres:
				ltm2_sequence.append(i)
				count = 0
				prev = i
			else:
				count+=1

	#print(ltm1_sequence)
	#print(np.array(ODORS_en)[np.array(ltm2_sequence).astype(int)])
	### Get pattern overlaps between successive attractors in both networks
	ltm1_sequence_overlaps,ltm2_sequence_overlaps = [],[]
	for i in range(len(ltm1_sequence)-1):
		cur_item = int(ltm1_sequence[i])
		next_item = int(ltm1_sequence[i+1])

		#check pattern overlap
		o = np.where((trpats1[cur_item]+trpats1[next_item]==2))[0].size
		ltm1_sequence_overlaps.append(o)

	for i in range(len(ltm2_sequence)-1):
		cur_item = int(ltm2_sequence[i])
		next_item = int(ltm2_sequence[i+1])

		#check pattern overlap
		o = np.where((trpats2[cur_item]+trpats2[next_item]==2))[0].size
		ltm2_sequence_overlaps.append(o)        

	# print('Overlap Sequence')
	# print(ltm1_sequence_overlaps)
	# print(ltm2_sequence_overlaps)

	return(ltm1_sequence_overlaps,ltm2_sequence_overlaps)

def plot_item_replay_transitions(ltm1_sequence_overlaps,ltm2_sequence_overlaps):


	# ltm1_sequence_overlaps = np.array(ltm1_sequence_overlaps,dtype=object)
	# ltm2_sequence_overlaps = np.array(ltm2_sequence_overlaps,dtype=object)

	if not isinstance(ltm1_sequence_overlaps[0],list): ###If it is a list and not a list of lists
		ltm1_sequence_overlaps = [ltm1_sequence_overlaps] 

	if not isinstance(ltm2_sequence_overlaps[0],list): ###If it is a list and not a list of lists
		ltm2_sequence_overlaps = [ltm2_sequence_overlaps] 

	#ltm1_sequence_overlaps = [[6,3,1],[2,1,2,1]]
	#### Visualize
	fig,ax = plt.subplots(2,1,figsize=(15,8),sharex=True)
	len1 = [len(a) for a in ltm1_sequence_overlaps]
	len2 = [len(b) for b in ltm2_sequence_overlaps]

	x1 = np.arange(max(len1))
	x2 = np.arange(max(len2))
	for o1 in ltm1_sequence_overlaps:
		x = np.arange(len(o1))
		ax[0].scatter(x+0.5,o1)
		ax[0].plot(x+0.5,o1)

	for o2 in ltm2_sequence_overlaps:
		x = np.arange(len(o2))  
		ax[1].scatter(x+0.5,o2)
		ax[1].plot(x+0.5,o2)
	
	ax[1].set_xlabel('Item Number',size=14)
	if len(x1)>len(x2):
		x = x1
	else:
		x = x2
	ax[0].set_xticks(x)
	ax[0].set_xticklabels(np.array(x)+1)

	ax[0].set_ylabel('Overlapping Units',size=14)
	ax[1].set_ylabel('Overlapping Units',size=14)

	ax[0].set_title('LTM1')
	ax[1].set_title('LTM2')
	ax[0].set_yticks(np.arange(0,H1,2))
	ax[1].set_yticks(np.arange(0,H2,2))
	ax[0].set_yticklabels(np.arange(0,H1,2))
	ax[1].set_yticklabels(np.arange(0,H2,2))
	ax[0].set_ylim([-1,H1])
	ax[1].set_ylim([-1,H2])
	plt.show()

def calc_recallwinner(trpats1, act1):
	'''
	get a list containing winners from lang net at every time step during recall phase 
	based on smallest cosine distance between input patterns and recall activity
	'''


	recall_score = np.zeros((trpats1.shape[0],act1.shape[1]))
	winner = np.zeros(act1.shape[1])

	for i,timestep in enumerate(range(act1.shape[1])):
		cur_act = act1[:,timestep].astype(float)
		

		for j,pat in enumerate(trpats1):


			recall_score[j,i] = 1 - np.dot(cur_act,pat) / (np.linalg.norm(cur_act)*np.linalg.norm(pat))

	
			

		if np.min(recall_score[:,i])<recall_thresh and (np.mean(cur_act.reshape(-1,H1),axis=1).mean()>1e-3):
			winner[i] = np.argmin(recall_score[:,i]) 
			
		else:
			winner[i] = -1

	#print(winner[0:50])
	return(winner,recall_score)

def return_omission_state(winners,stability_thres=200):
	'''
		Takes in the list of winners (in a cue phase) and returns 1 for omission and 0 for non omission
		stability_thres = min number of timesteps for which a pattern should be active to be considered stable
	'''
	counter = 1
	d=[]    ####List of item replay sequences and the time activation for each item at a time
	for i,w in enumerate(winners):
		if i==0:
			prev = w
		else:
			if w == prev: 
				counter+=1 
				prev = w
				if i==len(winners)-1:
					d.append([prev,counter])
			else:
				d.append([prev,counter])
				counter=1
				prev=w

	d = np.array(d)

	print(d)
	###### Completely blank then omission
	if d.shape[0]==1 and d[0,0]==-1:
		return 1

	######Check for stability in each item from replay sequence
	for item,time in d:
		if item==-1:
			continue
		if time>stability_thres:
			return 0

	####This return should be reached if there are short oscillations
	return 1

def return_recall_state(winners,cue,stability_thres=200):
	'''
		Takes in the list of winners (in a cue phase) and returns 1 for correct, 0  for incorrect and 2 for omission
		stability_thres = min number of timesteps for which a pattern should be active to be considered stable
	'''
	counter = 1
	d=[]    ####List of item replay sequences and the time activation for each item at a time
	for i,w in enumerate(winners):
		if i==0:
			prev = w
		else:
			if w == prev: 
				counter+=1 
				prev = w
				if i==len(winners)-1:
					d.append([prev,counter])
			else:
				d.append([prev,counter])
				counter=1
				prev=w

	d = np.array(d)
	# print('\t',ODORS_en[cue])
	# print(d)
	d = d[d[:,1]>=stability_thres]
	# print('########')
	# print(d)



	###### Completely blank or less than stability_thres then return 2
	if np.all(d[:,0]==-1):
		return 2

	###### Check if language network generates correct response or omission
	######Correct response can be any asssociated label or a specified label

	##Any associated label
	# for item,time in d:
	# 	assocs = patstat_pd[patstat_pd.LTM2==cue].LTM1.values
	# 	print(assocs)
	# 	if item in assocs:
	# 		return 1

	##Single association based on correct_descs_en dict
	od = ODORS_en[cue]
	for item,time in d:
		desc = descs[int(item)]
		if desc in correct_labels_en[od]:
			return 1

	# print('************')

	######Check for stability in each item from replay sequence
	# for item,time in d:
	#   if item==-1:
	#       continue
	#   if time>stability_thres:
	#       return 0

	####This return should be reached if response is incorrect
	return 0


def check_lang_recall_durations(act,pats1):
	'''
		Analyze lang net recall patterns durations
	'''
	winners,_ = calc_recallwinner(pats1,act)
	counts = pd.Series(winners).value_counts()

	lang_recall_durations = np.zeros(pats1.shape[0])
	for i in range(pats1.shape[0]):
		if not float(i) in counts.index:
			lang_recall_durations[i] = 0
		else:
			lang_recall_durations[i] = counts[i]

	return lang_recall_durations

def plot_lang_recall_durations(recall_durations,trpats1):

	if sims >1:
		durations_m = recall_durations.mean(axis=0)
		durations_std = recall_durations.std(axis=0)
	else:
		durations_m = recall_durations
		durations_std = np.zeros(len(durations_m))


	p1 = np.zeros(trpats1.shape[0]*H1).astype(int)
	for i in range(trpats1.shape[0]): 
		p1[i*H1:(i+1)*H1] = np.where(trpats1[i] > 0)[0]
	p1=p1.reshape(trpats1.shape[0],H1)

	b21 = Utils.loadbin(buildPATH+"Bj21.bin")
	b11 = Utils.loadbin(buildPATH+"Bj11.bin")

	b1 =  (b11+b21)/2

	desc_biases = [[] for i in range(len(descs_en))]

	for i in range(len(descs_en)):
		#print('Desc: {}'.format(descs_en[i]))
		descpat = p1[i]
		desc_biases[i].extend(b1[descpat])

	desc_biases = np.array(desc_biases)
	biases_mean = desc_biases.mean(axis=1)
	biases_std = desc_biases.std(axis=1)
	# p1_simmat = np.zeros([p1.shape[0],p1.shape[0]])
	# for i in range(p1.shape[0]):
	#   for j in range(p1.shape[0]):
	#       p1_simmat[i,j]=np.count_nonzero(p1[i]==p1[j])

	# p1_simmat_df = pd.DataFrame(p1_simmat,columns=descs_en,index=descs_en)
	# mask = np.eye(len(descs_en),dtype=bool)
	# mean_overlap = p1_simmat_df[mask].mean()


	print('Correlation:' ,corr(durations_m,biases_mean))

	fig,ax = plt.subplots(1,1,figsize=(15,8))
	ax.bar(np.arange(len(durations_m)),durations_m)
	_,caps,_ = ax.errorbar(np.arange(len(durations_m)),durations_m,yerr=durations_std, lolims=True, capsize = 0, ls='None', color='k')
	caps[0].set_marker('_')
	caps[0].set_markersize(10)
	ax.set_xticks(np.arange(len(durations_m)))
	ax.set_xticklabels(descs_en,rotation=45)
	ax.set_xlabel('Descriptors',size=14)
	ax.set_ylabel('Mean activation duration',size=14)

	assocwgain  = float(Utils.findparamval(parfilename,"assowgain"))
	ax.set_title('Pepparmint Cue, Assocwgain: {} , Sims: {}'.format(assocwgain,sims))
	plt.show()

def analyze_omissions(act,pats):
	'''
		For each simulation, get the recall state of the cued network (language net mostly) for each odor cue
		the recall state here is a binary variable: 1 (non-omission, aka not a blank response, could be correct or incorrect)
													0 (omission, blank response in lang net)

		returns a binary vector containing recall state for each odor cue in that simulation

		An omission needs to have no stable attractors in the language network. This could mean:
		- No activations in lang net after cue is lifted
		- Oscillating between multiple descriptors for short durations
		- Short activations of descriptor(s)

		Need to check for long lasting attractors even if there are slight stutters (because of recall threshold)
	'''

	recall_period = recall_ngap+recall_nstep #recall phase for one cue

	recall_start = 0 
	recall_end = recall_start+recall_period #Note: this assumes that your act data begins from recall phase

	omission_states = np.zeros(len(ODORS_en))
	for i,cue in enumerate(cues):
		cur_act = act[:,recall_start+recall_nstep:recall_end]


		winners,_ = calc_recallwinner(pats,cur_act)

		# if i==0:
		#   print(cue)
		#   print(list(winners))
		#####Get omission state
		omission_states[i] = return_omission_state(winners)

		# #check if there are no winners during recall phase of the current cue
		# #1 if there is some winner, 0 if no winner aka blank
		# omission_states[i] = all(v == -1 for v in winners)
		
		#print(list(winners[0:50]))
		#print(cue,omission_states[i])


		
		recall_start = recall_end
		recall_end = recall_start+recall_period

	return(omission_states)


def check_od_recall_durations(act,pats):
	'''
		Look at recall durations for mainly odors during their cue phase. Useful to compare effect of familiarity ratings
	'''
	recall_period = recall_ngap+recall_nstep #recall phase for one cue

	recall_start = 0 
	recall_end = recall_start+recall_period #Note: this assumes that your act data begins from recall phase

	recall_durations = np.zeros(len(ODORS_en))
	for i,cue in enumerate(cues):
		cur_act = act[:,recall_start:recall_end]


		winners,_ = calc_recallwinner(pats,cur_act)

		recall_durations[i] = (winners == cue).sum()/(recall_end-recall_start)

	
		recall_start = recall_end
		recall_end = recall_start+recall_period

	return(recall_durations)

def populate_recall_descriptors_dict(cue,winners,stability_thres=200):
	'''
		Populate the recall descriptor dictionary given the cue and winners
	'''

	###get descriptors and durations from counts
	item,counts = np.unique(winners,return_counts=True) 

	desc_durations = list(zip(list(item),list(counts)))

	###Proceed only if there are active descriptors (not all -1 in winners) 
	###and the descriptors are active for stability_thresh time steps  

	if all(x==-1 for x in winners):
		return

	if all(x[1]<=stability_thres for x in desc_durations):
		return

	###Check if cue is already in the dictionary
	temp= {}
	if cue not in recall_descriptors:
		for desc,duration in desc_durations:
			if desc == -1:
				continue
			temp[desc] = duration
		recall_descriptors[cue] = temp

	###Cue is already stored in dictionary
	else:
		inner_dict = recall_descriptors[cue]
		for desc,duration in desc_durations:
			if desc == -1:
				continue
			###Has this descriptor already been active in the same cue before
			if desc in inner_dict.keys():
				inner_dict[desc] += duration

			###This is descriptor hasnt shown up before
			else:
				inner_dict[desc] = duration		



def analyze_recall_states(act,pats,act_cuenet=None,pats_cuenet=None):

	'''
		For each simulation, get the recall state of the cued network (language net mostly) for each odor cue
		the recall state we quantify correct, incorrect or omission: 
		1 (correct)
		0 (incorrect)
		2 (Omission)

		returns a binary vector containing recall state for each odor cue in that simulation

		An incorrect activation is defined as an attractor activation that is not associated to the active cue odor pattern.

		Need to check for long lasting attractors even if there are slight stutters (because of recall threshold)
	'''

	if runflag == "full":
		text_file = open(buildPATH+"simstages.txt", "r")
		#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
		simstages = [int(line) for line in text_file]
		text_file.close()

		recall_timelogs = simstages[simstages.index(-2)+1:]
		recall_start_time = recall_timelogs[0]

		print(recall_start_time)
		act = act[:,recall_start_time:]
	else:
		recall_start_time = 0

	recall_period = recall_ngap+recall_nstep #recall phase for one cue

	recall_start = 0 
	recall_end = recall_start+recall_period #Note: this assumes that your act data begins from recall phase
	#correct_states = np.zeros(len(ODORS_en))
	recall_states = np.zeros(len(ODORS_en))
	recall_states_cuenet = np.zeros(len(ODORS_en))
	for i,cue in enumerate(cues):
		cur_act = act[:,recall_start+recall_nstep:recall_end]


		winners,_ = calc_recallwinner(pats,cur_act)

		# print(f'{ODORS_en[cue]}: {np.unique(winners)}')
		populate_recall_descriptors_dict(cue,winners)
		recall_states[i] = return_recall_state(winners,cue)


		if act_cuenet is not None and pats_cuenet is not None:
			cur_act_cuenet = act_cuenet[:,recall_start+recall_nstep:recall_end]
			winners_cuenet,_ = calc_recallwinner(pats_cuenet,cur_act_cuenet)
			recall_states_cuenet[i] = return_recall_state(winners_cuenet,cue)
		recall_start = recall_end
		recall_end = recall_start+recall_period

	# print('Recall States: ',recall_states)
	# print('Recall States Cuenet: ',recall_states_cuenet)

	if act_cuenet is None and pats_cuenet is None:
		return(recall_states)
	else:
		return(recall_states,recall_states_cuenet)


def get_omission_rates(omission_states):
	'''
		Calculate omission_rates from stack of omission_states across sim
	'''
	if sims>1:
		omission_rates_m = omission_states.mean(axis=0)
		omission_rates_std = omission_states.std(axis=0)
	else:
		omission_rates_m = omission_states
		omission_rates_std = np.zeros(len(omission_states))

	return(omission_rates_m,omission_rates_std)

def plot_omissionrates_vs_si(omission_rates):
	fig,ax = plt.subplots(1,1,figsize=(15,8),sharex=True,sharey=True)
	for i in range(len(ODORS_en)):
		ax.scatter(si[i],omission_rates[i],s=75,c=od_colors[i])
		ax.annotate(ODORS_en[i], xy=(si[i], omission_rates[i]), xytext=(5, 2), textcoords='offset points', ha='right',va='bottom')

	ax.set_ylabel('Simulated Omission Rate',size=len(omission_rates))
	ax.set_xlabel("Simpson's Diversity Index",size=len(omission_rates))
	ax.set_ylim([0,1])
	#ax.title('Sims = {}'.format(sims))
	plt.show()

def get_recall_state_rates(recall_states,savefname=''):
	'''
		Calculate recall rates from stack of recall states across sims
	'''
	if sims>1:

		recall_state_rates = []
		for i in range(recall_states.shape[1]):
			counts = []
			for j in range(3):
				counts.append(np.where(recall_states[:,i]==j)[0].size/sims)
			recall_state_rates.append(counts)

	else:
		recall_state_rates = []
		for i in range(recall_states.size):
			counts = []
			for j in range(3):
				counts.append(np.where(recall_states[i]==j)[0].size)
			recall_state_rates.append(counts)

	recall_state_rates = np.array(recall_state_rates)

	if savefname:
		print('Saving Recall State Rates')
		np.savetxt(figPATH+'RecallStateRate_'+savefname,recall_state_rates)

	return(recall_state_rates)

def plot_recall_states(recall_rates = 0,savefname=''):
	'''
		Plot correct, incorrect and omission rates for each odor across simulations
		Pass recall_rates or filename storing recall rates as argument
	'''

	if type(recall_rates) == str:
		recall_rates_m,recall_rates_std = np.loadtext(recall_rates)
	elif type(recall_rates) == int:
		raise Exception('plot_recall_states(): Invalid recall_rates!')

	plot_data = pd.DataFrame(recall_rates,columns=['Incorrect','Hits','Omissions'],index=ODORS_en)

	###Reorder columns for visually intuitive ordering
	plot_data = plot_data[['Incorrect','Omissions','Hits']] 

	###Show in Percentage
	plot_data *= 100

	fig,ax = plt.subplots(1,1,figsize=(15,8))
	plot_data.plot.bar(stacked=True,color=['tab:red','tab:orange','tab:green'],rot=45,ax=ax)
	ax.set_title('Recall Statistics {} Simulation(s)'.format(sims))
	ax.legend(bbox_to_anchor=(1.12, 1.0))
	ax.set_ylabel('Percent (%)',size=14)
	if savefname:
		plt.savefig(figPATH+savefname,dpi=300)
	plt.show()




def plot_recall_durations_odors(recall_durations):

	if sims>1:
		recall_durations_m = recall_durations.mean(axis=0)
		recall_durations_std = recall_durations.std(axis=0)/(recall_durations.shape[0]**0.5)
	else:
		recall_durations_m = recall_durations
		recall_durations_std = np.zeros(len(recall_durations))

	fig,ax = plt.subplots(1,1,figsize=(15,8),sharex=True,sharey=True)

	ax.bar(np.arange(len(ODORS_en)),recall_durations_m,yerr=recall_durations_std)

	fam = [4, 1, 3, 4, 4, 3, 1, 2, 3, 3, 2, 3, 2, 3, 3, 2]
	R,pval = corr(fam,recall_durations_m) 
	print(R,pval)

	ax.set_ylabel('Recall Durations (Proportion of total cue time)',size=14)
	ax.set_xticks(np.arange(len(ODORS_en)))
	ax.set_xticklabels(ODORS_en,rotation=45)
	ax.set_xlabel("ODORS",size=14)
	ax.set_title('Without familiarity sims = {}'.format(sims))
	ax.set_ylim([0,1])
	#ax.title('Sims = {}'.format(sims))
	plt.show()


def plot_simulatedomissionrates_vs_trueomissionrates(omission_rates):
	true_omrates = [0.3263546798029557, 0.6267318663406682, 0.5489451476793249, 0.30772341285887583, 0.34061488673139156, 
					0.5198511166253101, 0.49002493765586036, 0.4837133550488599, 0.3730800323362975, 0.4069387755102041, 0.37414141414141416, 
					0.44880615135572643, 0.4087027246848312, 0.38782961460446247, 0.45480340494527766, 0.3202453987730061]



	fig,ax = plt.subplots(1,1,figsize=(15,8),sharex=True,sharey=True)

	print(omission_rates[0].shape)
	ax.errorbar(np.arange(len(ODORS_en)),omission_rates[0],yerr=omission_rates[1],c='tab:blue',marker='o',label='simulated omission rate')
	ax.plot(np.arange(len(ODORS_en)),true_omrates,c='tab:orange',marker='o',label='true omission rate')

	R,pval = corr(omission_rates[0],true_omrates) 
	plt.legend()
	ax.set_ylabel('Omission Rate',size=14)
	ax.set_xticks(np.arange(len(ODORS_en)))
	ax.set_xticklabels(ODORS_en,rotation=45)
	ax.set_xlabel("ODORS",size=14)
	ax.set_title('Correlation: R = {:.3f}, p = {:.3f}'.format(R, pval))
	ax.set_ylim([0,1])
	#ax.title('Sims = {}'.format(sims))
	plt.show()

def get_perceptual_vs_associative_failure_rates(langnet_data,odnet_data):
	'''
		Check how many omissions arise from a failure at a perceptual level (odor network) or
		associative level (odor to lang associations) based on recall state across sims and cues for both patterns
	
		Requires data to be passed as sims x no of cues matrix

		Note this assumes that the odor network is the cueing network
	'''

	percep_assoc_failure_rates = np.zeros([odnet_data.shape[0],2]) #Stores the number of omissions due to percep failure (1st column) and due to assoc failure (2nd column) for each odor (row)
	print(odnet_data.shape)
	for i in range(odnet_data.shape[0]):	#Over each cue
		od_states = odnet_data[i]
		lang_states = langnet_data[i]
		# print('OdNet States: ',od_states)
		# print('LangNet States: ',lang_states)

		omissions = np.where(lang_states==2)[0]
		# print('omissions: ',omissions)
		if omissions.size==0:
			percep_assoc_failure_rates[i] = [0,0]
		else:
			od_states_omissions = od_states[omissions] # Get the state of odor network when lang net has an omission
			percep_failure = np.sum(od_states_omissions==2)
			percep_assoc_failure_rates[i] = [percep_failure,len(omissions)-percep_failure]
		
		# print(' ')

	print(percep_assoc_failure_rates)
	return percep_assoc_failure_rates
def run_sims():

	trpats1 = Utils.loadbin(buildPATH+"trpats1.bin",N1)
	trpats2 = Utils.loadbin(buildPATH+"trpats2.bin",N2)
	
	if sims == 0:
			os.chdir(buildPATH)
			# Wij11 = Utils.loadbin("Wij11.bin",N1,N1) #default size = (simulation steps * N*N, ) #Wij11
			# Wij12 = Utils.loadbin("Wij12.bin",N2,N1)
			# Wij21 = Utils.loadbin("Wij21.bin",N1,N2)
			# Wij22 = Utils.loadbin("Wij22.bin",N2,N2)
			# trpats1 = Utils.loadbin("trpats1.bin",N1)
			# trpats2 = Utils.loadbin("trpats2.bin",N2)

			data1 = Utils.loadbin("act1.log",N1).T  #(units,timestep)
			data2 = Utils.loadbin("act2.log",N2).T 

			os.chdir(PATH)
			########Omission Analysis
			# omission_state = analyze_omissions(data1,trpats1)
			# omission_state = get_omission_rates(omission_state)
			# print(omission_state)
			# plot_simulatedomissionrates_vs_trueomissionrates(omission_state)

			########Cue Recall Durations
			# recall_durations = check_od_recall_durations(data2,trpats2)
			# plot_recall_durations_odors(recall_durations)

			#######Recall attractor sequence analysis
			# ltm1_overlaps,ltm2_overlaps = analyze_item_replay_transitions(trpats1,trpats2,data1,data2)
			# plot_item_replay_transitions(ltm1_overlaps,ltm2_overlaps)

			#######Complete Recall Analysis (omissions,incorrect,correct)
			recall_state,recall_state_cuenet = analyze_recall_states(data1,trpats1,data2,trpats2)
			recall_state = get_recall_state_rates(recall_state,savefname='')
			#plot_recall_states(recall_state)
			
	else:

		for i in range(sims):
			#Run Sim
			os.chdir(buildPATH)
			print('\n\nSimulation No {}\n\n'.format(i+1))
			os.system("./olflangmain1 ")
			################ INIT PARAMS######################################
			trpats1 = Utils.loadbin("trpats1.bin",N1)
			trpats2 = Utils.loadbin("trpats2.bin",N2)
			data1 = Utils.loadbin("act1.log",N1).T  #(units,timestep)
			data2 = Utils.loadbin("act2.log",N2).T 

			###################################################################
			# if i == 0:

			#   Wij11 = Utils.loadbin("Wij11.bin",N1,N1) #default size = (simulation steps * N*N, ) #Wij11
			#   Wij12 = Utils.loadbin("Wij12.bin",N2,N1)
			#   Wij21 = Utils.loadbin("Wij21.bin",N1,N2)
			#   Wij22 = Utils.loadbin("Wij22.bin",N2,N2)

			#   if sims==1:
			#       Wij11 = np.reshape(Wij11,(1,N1,N1))
			#       Wij12 = np.reshape(Wij12,(1,N2,N1))
			#       Wij21 = np.reshape(Wij21,(1,N1,N2))
			#       Wij22 = np.reshape(Wij22,(1,N2,N2))

			# else:
			#   if i==1:
			#       Wij11 = np.stack((Wij11,Utils.loadbin("Wij11.bin",N1,N1)))
			#       Wij12 = np.stack((Wij12,Utils.loadbin("Wij12.bin",N2,N1)))
			#       Wij21 = np.stack((Wij21,Utils.loadbin("Wij21.bin",N1,N2)))
			#       Wij22 = np.stack((Wij22,Utils.loadbin("Wij22.bin",N2,N2)))

			#   else:
			#       Wij11 = np.concatenate((Wij11,Utils.loadbin("Wij11.bin",N1,N1)[None]),axis=0)
			#       Wij12 = np.concatenate((Wij12,Utils.loadbin("Wij12.bin",N2,N1)[None]),axis=0)
			#       Wij21 = np.concatenate((Wij21,Utils.loadbin("Wij21.bin",N1,N2)[None]),axis=0)
			#       Wij22 = np.concatenate((Wij22,Utils.loadbin("Wij22.bin",N2,N2)[None]),axis=0)


			# omission_state = analyze_omissions(data1,trpats1)

			# recall_durations = check_od_recall_durations(data2,trpats2)

			# lang_recall_durations = check_lang_recall_durations(data1,trpats1)

			#ltm1_overlaps,ltm2_overlaps = analyze_item_replay_transitions(trpats1,trpats2,data1,data2)
			#ltm1_overlaps = [ltm1_overlaps]
			#ltm2_overlaps = [ltm2_overlaps]

			#######Complete Recall Analysis (omissions,incorrect,correct)
			recall_state,recall_state_cuenet = analyze_recall_states(data1,trpats1,data2,trpats2)

			if i == 0:
				# omission_state_sims = omission_state
				# recall_durations_sims = recall_durations
				# lang_recall_durations_sims = lang_recall_durations
				# ltm1_overlaps_sims = [ltm1_overlaps]
				# ltm2_overlaps_sims = [ltm2_overlaps]

				recall_state_sims = recall_state
				recall_state_cuenet_sims = recall_state_cuenet
			else:
				# omission_state_sims = np.vstack([omission_state_sims,omission_state])
				# recall_durations_sims = np.vstack([recall_durations_sims,recall_durations])
				# lang_recall_durations_sims = np.vstack([lang_recall_durations_sims,lang_recall_durations])
				# ltm1_overlaps_sims.append(ltm1_overlaps)
				# ltm2_overlaps_sims.append(ltm2_overlaps)

				recall_state_sims = np.vstack([recall_state_sims,recall_state])
				recall_state_cuenet_sims = np.vstack([recall_state_cuenet_sims,recall_state_cuenet])

			# print('Sim: '+str(i))
			print(recall_state_sims)
			print(recall_descriptors)
			# print(omission_state_sims)
			#print(recall_durations_sims)
			#Plot Act Plot
			# os.chdir(PATH)
			# os.system("python3 dualnet_actplot.py act "+str(i+1))

	if sims>0:
		# omission_rates = get_omission_rates(omission_state_sims)
	#plot_omissionrates_vs_si(omission_rates)
		# plot_simulatedomissionrates_vs_trueomissionrates(omission_rates)

		#plot_recall_durations_odors(recall_durations_sims)
		# plot_lang_recall_durations(lang_recall_durations_sims,trpats1)
		# print(ltm1_overlaps_sims,ltm2_overlaps_sims)
		# plot_item_replay_transitions(ltm1_overlaps_sims,ltm2_overlaps_sims)

		###Perceptual vs associative failure
		percep_assoc_omrates = get_perceptual_vs_associative_failure_rates(recall_state_sims.T,recall_state_cuenet_sims.T)

		np.savetxt(figPATH+'Percep_failure_vs_Assoc_failure_Odors_{}sims_{}cueHCs_Recuwgain{}_GasolineSingleAssoc_Batch2.txt'.format(sims,cueHCs,recuwgain).replace('.','.'),percep_assoc_omrates)
		np.savetxt(figPATH+'RecallStateSims_{}Sims_{}cueHCs_LangNetBgain1,2x__Recuwgain{}_GasolineSingleAssoc_Batch2.txt'.format(sims,cueHCs,recuwgain).replace('.','.'),recall_state_sims)
		pickle.dump(recall_descriptors,open(figPATH+'RecallDescriptorsDict_{}Sims_{}cueHCs_LangNetBgain1,2x_Recuwgain{}_GasolineSingleAssoc_Batch2.pkl'.format(sims,cueHCs,recuwgain).replace('.','.'),'wb'))
		recall_state_rates = get_recall_state_rates(recall_state_sims,savefname='{}Sims_{}cueHCs_LangNetBgain1,2x_Recuwgain{}_GasolineSingleAssoc_Batch2.txt'.format(sims,cueHCs,recuwgain).replace('.','.'))
		plot_recall_states(recall_state_rates,savefname='')

		
	#os.system("python3 dualnet_actplot.py")

def analyze_omissions_fromfile(fname):
	'''
		Analyze omissions based on recall state file 

		Can analyze omissions by grouping odors based on:
		- Number of assocs
	'''
	##Data based omission rates for each odor
	actual_Omissions = np.array([35.37716821298911, 66.80112948769666, 59.62081484469544, 32.39209358612344, 
	35.90157321500605, 55.90964098426785, 52.60185558693021, 51.67406212182332, 39.774102460669624, 
	43.646631706333196, 39.81444130697862, 47.35780556676079, 43.32392093586123, 40.822912464703506, 
	48.36627672448568, 34.40903590157321])/100

	##Data based omissions after filtering for age (age<72.3) and MMSE (MMSE>=27)
	# actual_Omissions = np.array([0.25184577522559476, 0.6540983606557377, 0.5656903765690376, 
	# 	0.21579804560260588, 0.2937347436940602, 0.5086848635235732, 0.48635235732009924, 0.45758564437194127, 
	# 	0.35337124289195776, 0.3875715453802126, 0.34796747967479674, 0.3853658536585366, 0.37571312143439284, 
	# 	0.32599837000814996, 0.3854930725346373, 0.29296235679214405])

	data = np.loadtxt(figPATH+fname)
	assocs_dict={}
	for key,value in patstat_pd[['LTM1','LTM2']].to_numpy():
		if value not in assocs_dict:
			assocs_dict[int(value)]=[int(key)]
		else:
			assocs_dict[int(value)].append(int(key))

	assocs_counts = pd.Series(dict((k,len(v)) for k,v in assocs_dict.items()))

	grouped_recall,labels = [[] for i in range(assocs_counts.max())],[[] for i in range(assocs_counts.max())] 

	def custom_stripplot(data,ax,colors,jitter=0):
		x = np.arange(len(data))
		for i,y in enumerate(data):
			for j,pointy in enumerate(y):
				pointx = i+np.random.normal(0,jitter)
				ax.scatter(pointx,pointy,color=colors[i],s=30,alpha=0.8)
				ax.text(pointx,pointy+np.random.normal(0,0.005),labels[i][j],color='k',size=8)
	
	for i,x in enumerate(assocs_counts): 
		grouped_recall[x-1].append(data[i]) 
		labels[x-1].append(ODORS_en[i])

	omission_scores = [[x[2] for x in subl] for subl in grouped_recall]
	
	print(assocs_counts-1)
	actual_omission_scores = [[] for i in range(len(grouped_recall))]
	###Store actual omissions in same format as omission_scores
	for i,count in enumerate(assocs_counts):
		actual_omission_scores[count-1].append(actual_Omissions[i])

	om_mean = [np.mean(x) for x in omission_scores]
	om_std = [np.std(x) for x in omission_scores]

	x = np.arange(assocs_counts.max())

	colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
	fig,(ax,ax2) = plt.subplots(1,2,figsize=(15,8),sharex=True,sharey=True)

	custom_stripplot(omission_scores,ax,colors,0.2)
	custom_stripplot(actual_omission_scores,ax2,colors,0.2)
	# g=sns.stripplot(data=omission_scores,ax=ax,jitter=0.3)
	# g2 = sns.stripplot(data=actual_omission_scores,ax=ax2,jitter=0.3)
	# for i in range(0, len(omission_scores)):
	# 	dots = omission_scores[i]
	# 	for j,dot in enumerate(dots):
	# 		noisex = np.random.uniform(high=0.2)
	# 		noisey = np.random.uniform(high=0.01)
	# 		valign = 'bottom'
	# 		halign='right'
	# 		g.text(i+noisex, dot+noisey, labels[i][j] , horizontalalignment=halign, verticalalignment=valign, size='small', color='black')

	ax.set_title('Simulated')
	ax.bar(x,om_mean,alpha=0.3,color=colors)
	ax.set_xticks(x)
	ax.set_xticklabels(['1 Association','2 Associations','3 Associations','4 Associations'])
	ax.set_ylabel('Omission Rate',size=14)
	ax.set_ylim([0,1])
	# print(assocs_counts.tolist())
	# for i in range(0, len(actual_omission_scores)):
	# 	dots = actual_omission_scores[i]
	# 	for j,dot in enumerate(dots):
	# 		noisex = np.random.uniform(high=0.1)
	# 		noisey = np.random.uniform(high=0.05)
	# 		# if labels[i][j] == 'Mushroom':
	# 		# 	valign = 'top'
	# 		# 	halign='left'
	# 		# else:
	# 		# 	valign = 'bottom'
	# 		# 	halign='right'
	# 		g2.text(i+noisex, dot+noisey, labels[i][j] , horizontalalignment=halign, verticalalignment=valign, size='small', color='black')

	actual_om_mean = [np.mean(x) for x in actual_omission_scores]
	ax2.set_title('SNAC-K')
	ax2.bar(x,actual_om_mean,alpha=0.3,color=colors)
	ax2.set_xticks(x)
	ax2.set_xticklabels(['1 Association','2 Associations','3 Associations','4 Associations'])
	ax2.set_ylabel('Omission Rate',size=14)

	plt.show()

def plot_descriptors_distributions(pkl_fname,secondary_data='biases'):
	'''
		For each odor cue, plot the activation time of descriptors
		one can additionally overlay the subplots with secondary data 
		secondary data can be:
			- biases
			- mean_overlaps (with other descriptors)
			- lang2od_nassocs
	'''
	
	## Read recall_descriptors pickle
	data = pickle.load(open(figPATH+pkl_fname,'rb'))

	if secondary_data == 'mean_overlaps':
		langpats = buildPATH+'trpats1.bin' ##Can specify a different file name if needed
		trpats1 = Utils.loadbin(langpats,N1)

		p1 = np.zeros(trpats1.shape[0]*H1) 
		for i in range(trpats1.shape[0]): 
			p1[i*H1:(i+1)*H1] = np.where(trpats1[i] > 0)[0]
		p1=p1.reshape(trpats1.shape[0],H1)
		p1_simmat = np.zeros([p1.shape[0],p1.shape[0]])
		for i in range(p1.shape[0]):
			for j in range(p1.shape[0]):
				p1_simmat[i,j]=np.count_nonzero(p1[i]==p1[j])

		mean_overlaps = []
		std_overlaps = []
		for i in range(trpats1.shape[0]):
			row = np.delete(p1_simmat[i], i)  # delete the diagonal element
			std_dev = np.std(row)
			mean = np.mean(row)
			mean_overlaps.append(mean)
			std_overlaps.append(std_dev)

		legendpatch = [Patch(facecolor='tab:blue', edgecolor='tab:blue',label='Total duration of activation'),
						Line2D([0], [0],marker='o', color='tab:red', label='Mean overlap with other descriptors',ms=10)]
		

	elif secondary_data == 'lang2od_nassocs':
		ltm1_nassocs = patstat_pd.LTM1.value_counts().sort_index()
		legendpatch = [Patch(facecolor='tab:blue', edgecolor='tab:blue',label='Total duration of activation'),
						Line2D([0], [0],marker='o', color='tab:red', label='Lang->Od No of Assocs', ms=10)]

	elif secondary_data == 'biases':
		b11 = Utils.loadbin(buildPATH+'Bj11.bin')
		b21 = Utils.loadbin(buildPATH+'Bj21.bin')

		b1 = (b21+b11)/2

		langpats = buildPATH+'trpats1.bin' ##Can specify a different file name if needed
		trpats1 = Utils.loadbin(langpats,N1)

		p1 = np.zeros(trpats1.shape[0]*H1) 
		for i in range(trpats1.shape[0]): 
			p1[i*H1:(i+1)*H1] = np.where(trpats1[i] > 0)[0]
		p1=p1.reshape(trpats1.shape[0],H1)

		legendpatch = [Patch(facecolor='tab:blue', edgecolor='tab:blue',label='Total duration of activation'),
						Patch(facecolor='tab:red', edgecolor='tab:red',label='LangNet Biases',alpha=0.9)]

	fig,axs = plt.subplots(4,4,figsize=(20,20))

	for i,od in enumerate(ODORS_en):
		row = i%4
		col = int(i/4)
		ax = axs[row][col]
		###If there were no descriptors for a particular cue
		if i not in data.keys():
			ax.text(0.3,0.5, 'No Descriptiors', style='italic',
							 bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10},transform=ax.transAxes)
			ax.set_title(od)
			ax.set_xticks([])
			ax.set_yticks([])
		else:
			od_data = data[i]
			#sort od_data dictionary in descending order based on descriptor activation durations
			od_data_sorted_keys = sorted(od_data,key=od_data.get,reverse=True)

			desc_labels = [descs[int(j)] for j in od_data_sorted_keys]
			desc_durations = [od_data[int(j)] for j in od_data_sorted_keys]
			x = np.arange(len(desc_labels))

			ax.bar(x=x,height=desc_durations)
			ax.set_xticks(x)
			ax.set_xticklabels(desc_labels,rotation=45)
			ax.set_title(od)
			ax.set_xlim([-1,10])

			if secondary_data == 'mean_overlaps':
				plot_overlaps = [mean_overlaps[int(j)] for j in od_data_sorted_keys]
				plot_yerr = [std_overlaps[int(j)] for j in od_data_sorted_keys]
				ax2 = ax.twinx()
				ax2.set_ylim([0,4])
				ax2.errorbar(x,plot_overlaps,yerr=plot_yerr,c='tab:red',alpha=0.7,marker='o')

			elif secondary_data == 'lang2od_nassocs':
				plot_nassocs = [ltm1_nassocs[int(j)] for j in od_data_sorted_keys]
				ax2 = ax.twinx()
				ax2.set_ylim([0,5])
				ax2.errorbar(x,plot_nassocs,c='tab:red',alpha=0.7,marker='o')

			elif secondary_data == 'biases':
				ids = [int(j) for j in od_data_sorted_keys]
				y = b1[p1[ids].astype(int)]
				y = [list(j) for j in y]
				ax2 = ax.twinx()
				ax2.set_ylim([np.floor(np.min(-5)),np.ceil(np.max(b1))])
				#print([np.min(b1),np.max(b1)])
				# ax2.set_yticks([])
				print(len(y))
				violin = ax2.violinplot(y,positions=x)
				for pc in violin['bodies']:
					pc.set_facecolor('tab:red')
					pc.set_edgecolor('black')
					pc.set_alpha(0.9)

			ax.set_yscale('log')

	if secondary_data:
		plt.legend(handles=legendpatch, bbox_to_anchor=(-1.75, -0.5), loc='upper left')
	fig.text(0.01, 0.5, 'Timesteps', ha='center', va='center', rotation='vertical',size=16)
	plt.subplots_adjust(left = 0.05, bottom = 0.13, right = 0.95, top = 0.95, hspace=0.65,wspace=0.31)
	plt.show()

def plot_percep_vs_associative_omissions(fname,mode='odorwise',savef=0):
	'''
		Plot ratio of omissions due to perceptual vs associative failure
		The left column is perceptual failure and right column is associative failure in data

		mode = odorwise, assocwise, all
	'''	
	data = np.loadtxt(figPATH+fname)

	om_ratios = np.zeros(data.shape)
	for i,(percepf,assocf) in enumerate(data):
		om_ratios[i] = [percepf/(percepf+assocf),assocf/(percepf+assocf)]

	print(om_ratios)

	if mode == 'odorwise':
		width = 0.25
		x = np.arange(16)
		fig,ax=plt.subplots(1,1,figsize=(15,8))
		ax.bar(x-width/2,data[:,0],color='tab:blue',label='Perceptual Failure',width=width)
		ax.bar(x+width/2,data[:,1],color='tab:orange',label='Associative Failure',width=width)

		ax.set_xticks(x)
		ax.set_xticklabels(ODORS_en,rotation=45)
		ax.set_ylabel('Count',size=14)
		ax.set_title('{} simulations'.format(sims),size=16)
		plt.legend()
		plt.show()

	elif mode == 'all':
		om_causes_mean = [data[:,0].mean(),data[:,1].mean()]
		om_causes_std = [data[:,0].std(),data[:,1].std()]

		fig,ax = plt.subplots(1,1,figsize=(10,10))
		ax.bar([0,1],om_causes_mean,yerr = om_causes_std)
		ax.set_xticks([0,1])
		ax.set_xticklabels(['Perceptual \n Failure','Associative \n Failure'],size=14)
		ax.set_ylabel('Omission Rates',size=14)
		# ax.set_ylim([0,1])
		ax.set_title('{} simulations'.format(sims))
		plt.show()

	elif mode == 'assocwise':
		pass ###Need to add


	if savef:
		plt.savefig(figPATH+'Percep_vs_AssocFailure'+mode,format='eps')

def main():

	### Run sims to do analysis
	run_sims()


	### Functions below use saved scripts for analysis

	#Plot Omissions grouped by assocs
	fname = 'RecallStateRate_200Sims_9cueHCs_LangNetBgain1,2x_Recuwgain1.8_GasolineSingleAssoc_Batch2.txt'.format(cueHCs,recuwgain)
	analyze_omissions_fromfile(fname)

	###Plot Descriptor activation durations for each odor
	# fname = 'RecallDescriptorsDict_50Sims_10cueHCs_LangNetBgain1,1x_Recuwgain1,75.pkl'
	# plot_descriptors_distributions(pkl_fname=fname,secondary_data='biases')

	##Plot perceptual failure vs associative failure for omissions
	fname = 'Percep_failure_vs_Assoc_failure_Odors_200sims_9cueHCs_Recuwgain1.8_GasolineSingleAssoc_Batch2.txt'
	plot_percep_vs_associative_omissions(fname)


	
if __name__ == "__main__":
	main()


#plot_simulatedomissionrates_vs_trueomissionrates(simulated_omrates)