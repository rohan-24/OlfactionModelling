# OlfactionModelling
The files here contain the most recent and necessary folders needed to run the olfaction model. It also contains the most recent Figures generated from the model

To get started with running simulations: 

1. Follow the instructions in _Readme_cmake.txt_ in the works folder to set up the bcpnn simulator.
2. You should have a "build" folder within /works now. Importantly, the build folder should contain apps/olflang/. This folder contains the executable for the model cpp and any weights,biases or network state logging files get stored here at the end of the simulation.
3. The major files to be concerned with for running the olfaction model are located in works/apps/olflang (not to be confused by works/build/apps/olflang/)
   - olflangmain.cpp (contains the cpp code for the model)
   - olflangmain1.par (parameter file for the model)
   - dualnet_actplot.py (plots the raster for the simulations)
   - preload.py (If you want to create new preloaded biases and weights)   
4. Make sure the filepaths in the code files mentioned above match your local directory.
5. To run the model go to works/build/olflang and type "make olflangmain1 && ./olflangmain1"

## Important Pointers

If you want to run the simulations with preloaded weights and biases, you must copy the .bin files for the weights and biases to works/build/apps/olflang. Use the weights and biases in ModellingFigures/Main/PreloadedBW/ to run simulations with the most recent preloads. The other subfolders in ModellingFigures contain the different scenarios simulated and contain the Preloaded Bias and Weights for those scenarios as well. 

Before you run simulations with a set of preloaded weights and biases, make sure to check that the parameter _runflag_ in olflangmain1.par is set to "preload_localasso" . Also make sure that the preloadBW() function in olflangmain.cpp is referring to the correct file names for the preloaded weights and biases. 



