Simple Installation procedures using CMake:

1) Ensure that you have CMake version 3.10+ installed.
2) Create a build directory, for example inside the BCPNNSim/works directory by typing: mkdir build
3) Enter that directory: cd build
4) Invokve CMAKE from that build directory onto the top-level directory that contains the CMakelists.txt (in this case, "BCPNNSim/works").
   Also, you may provide an installation path.
   For example, type: cmake ../. -DCMAKE_INSTALL_PREFIX:PATH=../.
   (Note that above example will install everything into /usr and requires root)
5) type: make && make install
6) add bin folder to your path: export PATH="<your home catalog>/MyPrograms/BCPNNSim/works/bcpnn/bin:$PATH"

Artur Podobas, June, 2020
extended by Anders Lansner, Sept 2020
