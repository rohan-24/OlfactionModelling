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


def pos(MCpre,MCpos,N):
	return N*MCpre+MCpos


def run(dorun=False,parfilename=PATH+'olflangmain1.par'):
	if dorun:
		os.chdir('/home/rohan/Documents/BCPNNSimv2/works/build/apps/olflang/')
		os.system("./olflangmain1")

	
	fig, (ax2,ax3, ax4) = plt.subplots(3, 1, figsize=(14, 10),constrained_layout=True,sharex=True)


	H1 = int(Utils.findparamval(parfilename,"H"))
	M1 = int(Utils.findparamval(parfilename,"M"))

	H2 = int(Utils.findparamval(parfilename,"H2"))
	M2 = int(Utils.findparamval(parfilename,"M2"))
	N1 = H1 * M1
	N2 = H2 * M2


	MC_pre1 = 18
	MC_post1 = 1

	MC_pre2 = 18
	MC_post2 = 2

	pos1 = pos(MC_pre1,MC_post1,N1)
	pos2 = pos(MC_pre2,MC_post2,N2)
	#pos3 = pos(MC_pre1,MC_post2,N2)
	
	assoc_cspeed = float(Utils.findparamval(parfilename,"assoc_cspeed"))
	assoc_dist  = float(Utils.findparamval(parfilename,"assoc_dist"))
	delay = int(assoc_dist/assoc_cspeed/1e-3)

	text_file = open(buildPATH+"simstages.txt", "r")
	#Contains each pattern's encoding start timestep, end of training timestep, start of recall timestep
	simstages = [int(line) for line in text_file]
	text_file.close()


	recall_start_time = simstages[simstages.index(-2)+1:][0]
	#pos2 = pos(MC_pre2,MC_post2)
	# pos3 = pos(MC_pre1,MC_post2)
	# pos4 = pos(MC_pre1,MC_post3)

	#  # P traces

	# pi = Utils.loadbin("Pi11.log",H*N) #shape of pi is (timesteps,H*M)
	# pi = np.reshape(pi,(pi.shape[0],H,N)) #Pi is stored as (timestep, target hypercolumn, source unit) -> (timestep,H,N)
	# pj = Utils.loadbin("Pj11.log",N)
	# pij = Utils.loadbin("Pij11.log",N*N)


	# pj = Utils.loadbin(buildPATH+"Pj21.log",N1)
	# pij = Utils.loadbin(buildPATH+"Pij11.log",N1*N1)
	# p = Utils.loadbin(buildPATH+"P11.log",H1)



	# duration = np.arange(0,pj.shape[0],1)
	# linestyles = ["-","-","-","--","--","--","-.","-.","-.",":"] # - for Pi, -- for Pj, -. for Pij
	# datapoints = [
	# 	#pi[:,1,MC_pre1],
	# 	pi[:6500,0,0],
	# 	pi[:6500,0,1],
	# 	pi[:6500,0,2],

	# 	pj[:6500,MC_post1],
	# 	pj[:6500,MC_post2],
	# 	pj[:6500,MC_post3],
	# 	pij[:6500,pos1],
	# 	pij[:6500,pos(1,1)],
	# 	pij[:6500,pos(2,2)],
	# 	p[:6500,0]
	# 	]
	# legend = [
	# 	#'Pi LTM2 HC'+str(MC_pre1/M)+' MC'+str(MC_pre1%M),
	# 	'Pi LTM1 MC'+str(0),	
	# 	'Pi LTM1 MC'+str(1),
	# 	'Pi LTM1 MC'+str(2),
	# 	'Pj LTM2 HC MC'+str(MC_post1%M+1),
	# 	'Pj LTM2 HC MC'+str(MC_post2%M+1),
	# 	'Pj LTM2 HC MC'+str(MC_post3%M+1),
	# 	'Pij LTM2 HC'+str(MC_pre1/M)+' MC'+str(MC_pre1%M)+'-'+'LTM1 HC'+str(MC_post1/M)+'  MC'+str(MC_post1%M),
	# 	'Pij LTM2 HC'+str(MC_pre1/M)+' MC'+str(MC_pre1%M)+'-'+'LTM1 HC'+str(MC_post2/M)+'  MC'+str(MC_post2%M),
	# 	'Pij LTM2 HC'+str(MC_pre1/M)+' MC'+str(MC_pre1%M)+'-'+'LTM1 HC'+str(MC_post3/M)+' MC'+str(MC_post3%M),
	# 	'P LTM1 HC 0'
	# 		 ]
	# for i in range(len(datapoints)):
	#     #ax1.plot(duration,pi[:,0,MC_pre1],linestyles[0], duration,pj[:,pos1],linestyles[1], duration,pj[:,pos2],linestyles[1], duration,pij[:,pos1],linestyles[2] duration,pij[:,pos2],linestyles[2])
	#    	if linestyles[i] == "-.":
	#    		linewidth = 2
	#    	else:
	#    		linewidth = 1
	#    	ax1.plot(duration,datapoints[i],linestyles[i],label=legend[i],linewidth=linewidth)
	#    	ax1.margins(x=0)

	# ax1.tick_params(axis='x', labelsize=12)
	# ax1.tick_params(axis='y', labelsize=12)
	# ax1.title.set_text("Pj-traces")
	# ax1.legend(loc=4)


	###########WEIGHTS###########

	wij21 = Utils.loadbin(buildPATH+"Wij21.log",N2*N1)
	duration = np.arange(0,wij21.shape[0],1)
	#wij21 = Utils.loadbin("Wij21.log",N*N)


	ax2.plot(duration,wij21[:,pos1],
			 duration,wij21[:,pos2],
			 #duration,wij21[:,pos3],
			 #duration,calc_w[:6500]
			 #np.arange(0,len(calc_w),1),calc_w
			 )#,np.arange(0,len(wij21),1),wij21[:,pos4])
	
	ax2.margins(x=0)



	ax2.title.set_text("Associative Weights (LTM Od -> LTM Lang) ")
	ax2.set_ylabel("Wij21",fontsize=16)
	# ax2.legend(['LTM2 HC'+str(MC_pre1/M)+' MC'+str(MC_pre1%M)+'-'+'LTM1 HC'+str(MC_post1/M)+' MC'+str(MC_post1%M),
	# 			'LTM2 HC'+str(MC_pre1/M)+' MC'+str(MC_pre1%M)+'-'+'LTM1 HC'+str(MC_post2/M)+' MC'+str(MC_post2%M),
	# 			'LTM2 HC'+str(MC_pre1/M)+' MC'+str(MC_pre1%M)+'-'+'LTM1 HC'+str(MC_post3/M)+' MC'+str(MC_post3%M),
	# 			'calc_w LTM2 HC0 MC0 - LTM1 HC1 MC0'
	# 			#'LTM2 HC'+str(pre1/M)+' MC'+str(MC_pre1%M)+'-'+'LTM1 HC'+str(MC_post3/M)+' MC'+str(MC_post3%M)
	# 			])
	ax2.legend(['LTM Od MC18-> LTM Lang MC1','LTM Od MC18 -> LTM Lang MC 2'])

	
	###########BWSUP###########
	# data1 = Utils.loadbin(buildPATH+"bwsup1.log",N1).T 	#(units,timestep)
	# data2 = Utils.loadbin(buildPATH+"bwsup2.log",N2).T 
	# colormap_bwsup1 = ax1.imshow(data1[0:40,:],interpolation='none',aspect='auto',cmap = plt.cm.binary)
	# ax1.set_ylabel("MCs",fontsize=16)
	# ax1.set_title('LTM Lang BWSUP')

	# colormap_bwsup2 = ax2.imshow(data2[0:40,:],interpolation='none',aspect='auto',cmap = plt.cm.binary)
	# ax2.set_ylabel("MCs",fontsize=16)
	# ax2.set_title('LTM Od BWSUP')

	# ax1.axvline(x=recall_start_time,color='orange')
	# ax2.axvline(x=recall_start_time,color='orange')
	###########ACT###########

	data1 = Utils.loadbin(buildPATH+"act1.log",N1).T 	#(units,timestep)
	data2 = Utils.loadbin(buildPATH+"act2.log",N2).T 
	colormap1 = ax3.imshow(data1[0:H1,:],interpolation='none',aspect='auto',cmap = plt.cm.binary)
	#cbar = plt.colorbar(colormap1)
	#ax2.set_xlabel("Timesteps",fontsize=16)
	
	ax3.set_ylabel("MCs",fontsize=16)
	ax3.set_title('LTM Lang ACT')
	ax3.axvline(x=recall_start_time,color='orange')
	# for i in range(1,H1):
	# 		ax3.axhline(y=i*M1-1,color='tab:blue')


	colormap2 = ax4.imshow(data2[0:H2,:],interpolation='none',aspect='auto',cmap = plt.cm.binary)
	#cbar = plt.colorbar(colormap1)
	ax4.set_xlabel("Timesteps",fontsize=16)
	ax4.set_ylabel("MCs",fontsize=16)
	ax4.set_title('LTM Od ACT')
	ax4.axvline(x=recall_start_time,color='orange')
	#plt.suptitle('Delay: {} ms'.format(delay),weight='heavy',size=16)

	# for i in range(1,H2):
	# 		ax4.axhline(y=i*M2-1,color='tab:blue')

	# HCshow = 1
	# data1 = Utils.loadbin("act1.log",N)[:,HCshow*M:HCshow*M+M]
	# data1 = data1[6500:12500,0:M]
	# r1 = 12500
	# data1 = data1.T
	# colormap1 = ax3.imshow(data1[:, 0:],interpolation='none',aspect='auto',cmap = 'jet')
	# ax3.title.set_text("LTM #1 Activity")
	# ax3.yaxis.set_major_formatter(tck.FuncFormatter(mjrFormatter))
	# ax3.set_ylabel('Object',fontsize=16)
	# ax3.tick_params(axis='x', labelsize=12)
	# ax3.tick_params(axis='y', labelsize=12)
	# plt.colorbar(colormap1,ax=ax3)


	# data2 = Utils.loadbin("act2.log",N)[:,HCshow*M:HCshow*M+M]
	# data2 = data2[6500:12500,0:M]
	# r2 = 12500
	# data2 = data2.T
	# colormap1 = ax4.imshow(data2[0:, 0:],interpolation='none',aspect='auto',cmap = 'jet')
	# ax4.title.set_text("LTM #2 Activity")
	# ax4.yaxis.set_major_formatter(tck.FuncFormatter(mjrFormatter))
	# ax4.set_ylabel('Context',fontsize=16)
	# plt.colorbar(colormap1,ax=ax4)
	# ax4.tick_params(axis='x', labelsize=12)
	# ax4.tick_params(axis='y', labelsize=12)
	# plt.gcf().set_size_inches(18, 13.5)


	plt.show()

def main(dorun = False):
	#run1(dorun)
	run()

if __name__ == "__main__":
    main()