'''
This code looks at how sensitive omission rates are to hyperparameters (like recuwgain, assowgain)
The code here borrows some helper functions from analyze_recall

'''
import sys, os
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import re
import pickle
from analyze_recall import analyze_recall_states
sys.path.insert(0, '/home/rohan/Documents/BCPNNSimv2/works/misc/')
import Utils

PATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/'
buildPATH  = '/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/'
figPATH = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/DualNet/Network15x15/Semantization/Main/HyperParameterVsOmissionRates/Bgain/'

parfilename = PATH+"olflangmain1.par"
od_colors = ['#BE0032', #Red
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
recall_nstep = int(Utils.findparamval(parfilename,"recallnstep"))
recall_ngap = int(Utils.findparamval(parfilename,"recallngap"))
recall_thresh  = float(Utils.findparamval(parfilename,"recall_thresh"))
runflag = Utils.findparamval(parfilename,"runflag")

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

#########PARAMS############
sims = 50 #number of simulations per unit of hyperparameter
hyperparam_default = 1. ###Change when using different hyperparam

hyperparam = 'bgain'
lower_bound = hyperparam_default-hyperparam_default*0.2
upper_bound = hyperparam_default+hyperparam_default*0.2
increment = 0.05*hyperparam_default
hyperparam_vals = np.arange(lower_bound,upper_bound+1e-5,increment)
print(hyperparam,hyperparam_vals,hyperparam_vals.shape)
hyperparam2 = 'assowgain'
hyperparam2_vals = np.arange(0.05,0.3,0.05)

omission_rates = [] ###Store omission rate of each odor 
omission_rates_3d = np.zeros([len(cues),len(hyperparam_vals),len(hyperparam2_vals)])


def get_omission_rate(states):
	'''
		Given an mxn matrix of recall states where m is the number of sims, n is the number of odors
		This function returns the proportion of omissions for each odor

		Odors are stored as recall_state = 2 (see analyze_recall_states)
	'''
	global omission_rates
	mask = (states==2)
	if sims > 1:
		omission_rates.append(np.mean(mask,axis=0))
	else:
		omission_rates.append(mask.astype(float))


def populate_3d_omission_rates(omrates,h1,h2):
	'''
		Populates the od x h1 x h2 matrix where od is the index of the odor, h1  is the index of hyperparam1 and h2 is the index of hyperparam2
	'''
	global omission_rates_3d
	print('omrates: ',omrates[0])
	omission_rates_3d[:,h1,h2]=omrates[0]


def run_indiv_hyperparam_simulations(savef=0):
	'''
		Test effect of single hyper param on omission rates
	'''

	global omission_rates
	os.chdir(buildPATH)
	for i,h in enumerate(hyperparam_vals):
		print('\n\n\t Hyperparam: {} = {}\n'.format(hyperparam,h))

		for sim in range(sims):
			print('Simulation No {}\n'.format(sim+1))

			os.system('./olflangmain1 {} {}'.format(hyperparam,h))
			################ INIT States######################################
			trpats1 = Utils.loadbin("trpats1.bin",N1)
			# trpats2 = Utils.loadbin("trpats2.bin",N2)
			data1 = Utils.loadbin("act1.log",N1).T  #(units,timestep)
			# data2 = Utils.loadbin("act2.log",N2).T 


			#######Complete Recall Analysis (omissions,incorrect,correct)
			recall_state = analyze_recall_states(data1,trpats1)

			#######Stack recall states for each odor in each sim 
			if sim == 0:
				recall_state_sims = recall_state
			else:
				recall_state_sims = np.vstack([recall_state_sims,recall_state])

			print('recall_state_sims: ',recall_state_sims)
			# print(recall_descriptors)

		######Get omission rates for all sims using hyperparameter h and store it to omission rates
		get_omission_rate(recall_state_sims)

	omission_rates = np.array(omission_rates)
	print('Omission Rates: \n', omission_rates)

	omission_rates_dict = dict(zip(hyperparam_vals,omission_rates))
	print('Omission Rates Dict: \n',omission_rates_dict)

	if savef:
		pickle.dump(omission_rates_dict,open(figPATH+'Omission_Rates_Dict_Hyperparam-{}_Sims{}.pkl'.format(hyperparam,sims).replace('.','.'),'wb'))


def run_2_hyperparam_simulations(savef=0):
	'''
		Test effect of two hyper param on omission rates
	'''

	global omission_rates
	os.chdir(buildPATH)
	for i,h1 in enumerate(hyperparam_vals):
		for j, h2 in enumerate(hyperparam2_vals):
			for sim in range(sims):
				print('\n\t Hyperparam1: {} = {}	Hyperparam2: {} = {}'.format(hyperparam,h1,hyperparam2,h2))
				print('\t\t\t\tSimulation No {}\n'.format(sim+1))

				os.system('./olflangmain1 {} {} {} {}'.format(hyperparam,h1,hyperparam2,h2))


				################ INIT States######################################
				trpats1 = Utils.loadbin("trpats1.bin",N1)
				# trpats2 = Utils.loadbin("trpats2.bin",N2)
				data1 = Utils.loadbin("act1.log",N1).T  #(units,timestep)
				# data2 = Utils.loadbin("act2.log",N2).T 


				#######Complete Recall Analysis (omissions,incorrect,correct)
				recall_state = analyze_recall_states(data1,trpats1)

				#######Stack recall states for each odor in each sim 
				if sim == 0:
					recall_state_sims = recall_state
				else:
					recall_state_sims = np.vstack([recall_state_sims,recall_state])

				print('recall_state_sims: ',recall_state_sims)


			######Get omission rates for all sims using hyperparameter h and store it to omission rates
			get_omission_rate(recall_state_sims)
			print('Omission Rates: \n',omission_rates)
			populate_3d_omission_rates(omission_rates,i,j)
			print('Omission Rates 3d: \n',omission_rates_3d)
			omission_rates = []

	if savef:
		store_data =  [omission_rates_3d,hyperparam_vals,hyperparam2_vals]
		pickle.dump(store_data,open(figPATH+'Omission_Rates_{}x{}_2Params_Hyperparam1-{}_Hyperparam2-{}_Sims{}.pkl'.format(len(hyperparam_vals),len(hyperparam2_vals),hyperparam,hyperparam2,sims).replace('.','.'),'wb'))

############################################## Plotting Functions ############################################## 

def plot_IndivOdor_OmRates_vs_Hyperparam(data_fname):
	'''
		Plot Omission Rates for each odor across hyperparameter values
		Takes in file name to the saved omission_rates_dict pickle
	'''

	delims =  "_","-",".",","
	regex_delims = '|'.join(map(re.escape,delims))
	fname_split = re.split(regex_delims,data_fname)
	hyperparam_name = fname_split[fname_split.index('Hyperparam')+1]
	print(hyperparam_name)
	data = pickle.load(open(figPATH+data_fname,'rb'))

	data_pd = pd.DataFrame(data,index=ODORS_en).T

	fig,ax = plt.subplots(8,2,figsize=(15,15),sharex=True,sharey=True)
	data_pd.plot.line(ax=ax,style='.-',markersize=15,color=od_colors,subplots=True)
	# ax.set_xlabel(hyperparam_name,size=14)
	# ax.set_ylabel('Omission Rate',size=14)
	fig.text(0.5,0.05,hyperparam_name.capitalize(),size=14)
	fig.text(0.08,0.4,'Omission Rate',size=14,rotation=90)
	plt.subplots_adjust(hspace=0.4)
	# fig.tight_layout()
	plt.show()

def plot_Nassocwise_OmRates_vs_Hyperparam(data_fname,savef=1,format='eps'):
	'''
		Plot Omission Rates grouped by nassocs from odor to language network hyperparameter values
		Takes in file name to the saved omission_rates_dict pickle
		NOTE:: NEED TO WORK ON THIS FUNCTION
	'''

	patstat_fname = PATH+'patstat_si_nclusters4_topdescs.txt'  
	patstat = np.loadtxt(patstat_fname).astype(int)

	if patstat.shape[1]==2:
		patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2'])
	else:
		patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2','trn_effort'])

	assocs_dict={}
	for key,value in patstat_pd[['LTM1','LTM2']].to_numpy():
		if value not in assocs_dict:
			assocs_dict[int(value)]=[int(key)]
		else:
			assocs_dict[int(value)].append(int(key))

	assocs_counts = pd.Series(dict((k,len(v)) for k,v in assocs_dict.items()))

	labels = [[] for i in range(assocs_counts.max())] 
	for i,x in enumerate(assocs_counts): 
		labels[x-1].append(ODORS_en[i])

	print(labels)

	delims =  "_","-",".",","
	regex_delims = '|'.join(map(re.escape,delims))
	fname_split = re.split(regex_delims,data_fname)
	hyperparam_name = fname_split[fname_split.index('Hyperparam')+1]
	print('Plotting Hyperparam: ',hyperparam_name)
	data = pickle.load(open(figPATH+data_fname,'rb'))

	data_pd = pd.DataFrame(data,index=ODORS_en).T
	print(data_pd)
	colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
	fig,ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
	for i,lab in enumerate(labels):
		group_mean = data_pd[lab].mean(axis=1)
		group_std = data_pd[lab].std(axis=1)
		group_mean.plot.line(ax=ax,style='.-',markersize=15,color=colors[i],label='{} Association(s)'.format(i+1))
		ax.errorbar(group_std.index,group_mean,yerr=group_std,capsize=5,ecolor=colors[i])
		#ax.fill_between(group_std.index, group_mean-group_std, group_mean+group_std,color=colors[i],alpha=0.2)

	plt.legend()
	# fig.text(0.5,0.05,hyperparam_name.capitalize(),size=14)
	# fig.text(0.08,0.4,'Omission Rate',size=14,rotation=90)
	x = 1-(1.0-group_std.index.to_numpy())
	x = np.around(x,3)
	ax.set_xticks(group_std.index.to_numpy())
	ax.set_xticklabels(x)
	#ax.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
	ax.set_xlabel(hyperparam_name.capitalize(),size=14)
	ax.set_ylabel('Omission Rate',size=14)
	fig.tight_layout()

	if savef:
		if format=='tiff':
			plt.savefig(figPATH+'{}_{}sims.tiff'.format(hyperparam_name,sims),dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
		if format == 'eps':
			plt.savefig(figPATH+'{}_{}sims.eps'.format(hyperparam_name,sims),format='eps')

	plt.show()
def plot_odorwise_omrate_heatmaps_2hyperparam(data_fname):
	'''
		Plot omission rate heatmaps for two parameter analysis with each odor
	'''
	delims =  "_","-",".",","
	regex_delims = '|'.join(map(re.escape,delims))
	fname_split = re.split(regex_delims,data_fname)
	hyperparam1_name = fname_split[fname_split.index('Hyperparam1')+1]
	hyperparam2_name = fname_split[fname_split.index('Hyperparam2')+1]

	omission_rates_mat, param1_vals,param2_vals = pickle.load(open(figPATH+data_fname,'rb'))
	param1_vals = np.around(param1_vals,2)
	param2_vals = np.around(param2_vals,2
		)
	fig,ax = plt.subplots(4,4,figsize=(15,15),sharex=True,sharey=True)

	for i in range(16):
		row = int(i/4)
		col = i%4
		sns.heatmap(omission_rates_mat[i,:,:],ax=ax[row][col],yticklabels=param1_vals,xticklabels=param2_vals,annot=True,vmin=0,vmax=1)
		ax[row][col].set_yticklabels(param1_vals,rotation=45)
		ax[row][col].set_title(ODORS_en[i],size=14)

	fig.text(0.5,0.05,hyperparam2_name.capitalize(),size=14)
	fig.text(0.08,0.45,hyperparam1_name.capitalize(),size=14,rotation=90)
	plt.show()

def plot_nassocwise_omrate_heatmaps_2hyperparam(data_fname):
	'''
		Plot omission rate heatmaps for two parameter analysis with each odor
	'''
	delims =  "_","-",".",","
	regex_delims = '|'.join(map(re.escape,delims))
	fname_split = re.split(regex_delims,data_fname)
	hyperparam1_name = fname_split[fname_split.index('Hyperparam1')+1]
	hyperparam2_name = fname_split[fname_split.index('Hyperparam2')+1]

	omission_rates_mat, param1_vals,param2_vals = pickle.load(open(figPATH+data_fname,'rb'))
	param1_vals = np.around(param1_vals,2)
	param2_vals = np.around(param2_vals,2)

	patstat_fname = PATH+'patstat_si_nclusters4_topdescs.txt'  
	patstat = np.loadtxt(patstat_fname).astype(int)

	if patstat.shape[1]==2:
		patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2'])
	else:
		patstat_pd = pd.DataFrame(patstat.astype(int),columns=['LTM1','LTM2','trn_effort'])

	assocs_dict={}
	for key,value in patstat_pd[['LTM1','LTM2']].to_numpy():
		if value not in assocs_dict:
			assocs_dict[int(value)]=[int(key)]
		else:
			assocs_dict[int(value)].append(int(key))

	assocs_counts = pd.Series(dict((k,len(v)) for k,v in assocs_dict.items()))

	labels = [[] for i in range(assocs_counts.max())] 
	od_ids = [[] for i in range(assocs_counts.max())]
	for i,x in enumerate(assocs_counts): 
		od_ids[x-1].append(i)
		labels[x-1].append(ODORS_en[i])

	print(labels)

	nassocwise_omrate_mat = [[] for i in range(assocs_counts.max())]
	for i,ods in enumerate(od_ids):
		mat = omission_rates_mat[ods,:,:].mean(axis=0)
		nassocwise_omrate_mat[i] = mat

	fig,ax = plt.subplots(2,2,figsize=(15,15),sharex=True,sharey=True)

	plot_labs = ['{} Association(s)'.format(i+1) for i in range(assocs_counts.max())]
	for i in range(4):
		row = int(i/2)
		col = i%2
		sns.heatmap(nassocwise_omrate_mat[i],ax=ax[row][col],yticklabels=param1_vals,xticklabels=param2_vals,annot=True,vmin=0,vmax=1)
		ax[row][col].set_yticklabels(param1_vals,rotation=45)
		ax[row][col].set_title(plot_labs[i],size=14)

	fig.text(0.45,0.05,hyperparam2_name.capitalize(),size=14)
	fig.text(0.08,0.45,hyperparam1_name.capitalize(),size=14,rotation=90)
	plt.show()

def main():
	# run_indiv_hyperparam_simulations(savef=1)
	# run_2_hyperparam_simulations(savef=1)
	plot_Nassocwise_OmRates_vs_Hyperparam('Omission_Rates_Dict_Hyperparam-bgain_Sims50_cueHCs9.pkl',savef=1,format='tiff')
	# plot_odorwise_omrate_heatmaps_2hyperparam('Omission_Rates_5x5_2Params_Hyperparam1-recuwgain_Hyperparam2-assowgain_Sims20.pkl')
	# plot_nassocwise_omrate_heatmaps_2hyperparam('Omission_Rates_5x5_2Params_Hyperparam1-recuwgain_Hyperparam2-assowgain_Sims20.pkl')

	####TODO: Plotting for 2 hyperparam case
if __name__ == "__main__":
	main()