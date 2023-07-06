Anders Lansner, 2020-10-14, latest updated 2020-04-03

README file for the BCPNNSim Threaded API(Version 0.9.6)
========================================================

Aims of this code
-----------------

This code is intended to allow realization and simulation of all
different aspects of the BCPNN model. Thus, it allows to build
abstract computational networks as well as biologically detailed
networks of cortex with leaky integrate-and-fire model neurons and
conductance based synapses. It was initially intended as a template
for development of FPGA and ASIC. The plan is to develop it for a
wider use, e.g. for parallel execution on clusters. Currently, the
code is threaded but a cluster parallel version is planned.

Neural units can be spiking or non-spiking and one can mix both types
in the same network. Some things remain to be done in order to achieve
this though.


Disclaimer
----------

The code is new, poorly documented likely to have several bugs. No
responsibility is accepted for correcting bugs or for any damage
occuring when used.


Contributions
-------------

This first restrictively distributed version was developed by Anders
Lansner. Additions and improvements have been done by Naresh
Ravishandran and the cmake setup was provided by Artur Podobas.


Components of code and distribution
-----------------------------------

This API allows a user to spefify a network built of Population and
Projection components. A population is an array of neural units and a
projection is a (possibly sparse) matrix of connections between a
source and a target population.

The basic classes are Pop/PopH and Prj/PrjH with derived classes like
HCU/BCU and BCC. The helper class Logger is used for logging and
printing state variables. The helper class Parseparam is used to
interpret the parameter files (named *.par) and makes easy to explore
different parameter settings. The helper class Timer is used for easy
timing of code sections. Utils.py and Figs1.py can be used for initial
data visualization and analysis.

A few examples of test and application codes are included. The
intention is to illustrate some basic features and to show examples of
use of the API. Simulation output is binary and stored in files with
name *.log and *.bin.

List of applications (*main*.cpp)

ammain1        A recurrent BCPNN as associative memory
hcumain1       A hypercolumn unit (HCU) with non-spiking or spiking neural units
lifmain1       Testing input-output relations of i-f spiking neurons
lifburst1      A bursting half-center network built from AdEx neurons
plasticmain1   Two HCU:s connected via a plastic BCPNN projection
facdepmain1    Two HCU:s connected via a projection with synaptic depression
	       and facilitation
prjhmain1      A source and target PopH:s testing built-in W configurations
bcumain1       A source and target BCU:s of non-spiking or spiking neurons connected
	       via a plastic BCC
shclassmain2   A three layer BCPNN set up for MNIST classification. Dataset need to be downloaded from e.g.
	       KTH-box:
	       https://kth.box.com/s/mrpqpgi66n2dnzi4nuu7czfkx43vxtnb
	       Place it in the same directory as your BCPNNSim code.
emmain1        Two recurrent BCPNN:s recurrently connected to do episodic memory


Quick start
-----------

To start using the code, download the BCPNNSim_0.9.6.zip file, unpack
it and follow the instructions in Readme_cmake.


Saltsjöbaden Oct 14 2020

/Anders Lansner


