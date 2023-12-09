IMPORTANT
___________
***************************************************************************
- you need to comment out the lines with os.chdir()
or add your own directory where the data is, so you don't get an error. 
***************************************************************************

To run .ipynb files:
________________________
1- create a new enviornment and install the packages in the textfile packages using the command line. Check the versions needed (especially for numpy,  and tensorflow). The rest of the packages should follow the instructions in lab#2. Also check the file Packages that include all the modules and their versions that were installed in the environment used to run my codes. 

conda create --name dana-hw2 
conda activate dana-hw2
example on installing a module of a specific version:
conda install module-name=desired-version



2- The packages used in this homework include:
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import  sklearn
from sklearn.model_selection import train_test_split
import os
import math
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers.schedules as schedules
import pickle
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import logging



3- Next, while the environment is activated, run the following command line to create a kernel:
python -m ipykernel install --user --name iems469 --display-name "iems469" 

(in this case iems469 is the kernel name, but one can customize it)

4- In new launcher, create a new notebook from the new kernel that was created. 
Next , click on kernel in the tool bar and choose restart kernel. Every time a new package is installed, 
one needs to run the command line in step 3, and restart the kernel again, to update the kernel. Unless the installations took place consecutively, then run step 3 only once after all the installations and restart the kernel. 

5- Next upload the code, and run it using jupyter hub on deepdish. 
6- Note that to be able to run the code on GPU, install the following, by running this command:
(not applicable for this hw as PyTorch was not used)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


Alternatively, use google colab to convert the .ipynb files to .py files. ( by downloading as .py)

To run .py files:
__________________


type the following commands for a code in a file named "name.py", in the powershell in windows:
1- ssh netid@mlds-deepdish3.ads.northwestern.edu
2- enter password 
3- conda activate environment-name
4- cd directory 
5- python name.py
now the code will run and the output of the code will be displayed in the same terminal. 
The saved files of the code will be saved in the directory. 


If for some reason, you need to load the trained model.h5 file or the .pickle or .npy files :
__________________________________________________________________________________________________

- make sure they're in the same directory as the code. 
- if you get an error, make sure there are no preset directories, 
such as os.chdir()


