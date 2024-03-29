
# LOGKPREDICT

LOGKPREDICT is a code that links [HostDesigner](https://sourceforge.net/projects/hostdesigner/)
with [Chemprop](https://github.com/chemprop/chemprop). LOGKPREDICT predicts stability constants, log K values, 
using [model](https://github.com/Critical-Materials-Institute/LOGKPREDICT/blob/main/model.pt) 
trained with a modified chemprop and a stability constant database that is yet-to-be released. 
The primary purpose for this code is to provide an additional 
metric to rank molecules created using HostDesigner. 

Please check the documentation
of HostDesigner for how to use LOGPREDICT once it is installed.

Installation of LOGKPREDICT together with the corresponding modified version 
of CHEMPROP:

0) Make sure that you have the Conda environment installed on your system (Linux, Mac OS, 
or Windows). For this purpose, use either the full [Conda](https://anaconda.org) environment
or the minimal version of Conda, which is called [Miniconda](https://conda.io/miniconda.html).


1) Download chemprop-1.5.2 from [chemprop repository](https://github.com/chemprop/chemprop).


2) Go into the main chemprop-1.5.2 directory and issue the following commands:

`conda env create -f environment.yml`

`conda activate chemprop`

`pip install -e .`


3) Download LOGKPREDICT from this repository, put it into a directory from the PATH 
on your system, and change the permissions of LOGKPREDICT to make it executable, for
example by issuing the command:

`chmod 755 LOGKPREDICT`


4) Download model.pt from this repository and set the environmental variable LOGKPREDICT_DIR
to point to the directory where model.pt resides. 

Example: 
If using bash shell, add the line to the .bashrc file (or file containing environment variables):

`export LOGKPREDICT_DIR='/path/to/directory/'`


5) You can test the installation by downloading logk_input and logk_output from this repository 
(and renaming logk_output to a different name, for example logk_predict0) and issuing 
the command "LOGKPREDICT" in the same directory where logk_input resides. The logk_output
should coincide with logk_output0. 
