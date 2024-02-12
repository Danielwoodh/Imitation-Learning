   README.md 

Imitation Learning - README
===========================

This file outlines the use of the code that was constructed as part of the MSc Robotics thesis at Cranfield University.

The jupyter-notebook can be run from either a Windows or Linux based operating system.

The Simulation requires a Linux-based operating system, with Ubuntu 21.04 being used. If a different operating system is being used, it cannot be guaranteed that the simulation will be able to run.

Library Requirements
--------------------
Before any code can be run, please ensure all the following library requirements are installed.

1. Most requirements can be installed through the requirements.txt file, using the following command: `pip install -r requirements.txt`

_The following libraries and software will not be installed through the `pip install -r requirements.txt` command and must be installed manually as follows:_

*   **RLLab** - This can be installed following the instructions here: [https://rllab.readthedocs.io/en/latest/user/installation.html](https://rllab.readthedocs.io/en/latest/user/installation.html) and must be contained in the same folder as the rest of the code.
*   _Preferably_ **CUDA** & **cuDNN** - To allow for faster computation during training.  
    CUDA can be installed following the instruction here: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)  
    cuDNN can be installed following the instructions here: [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

Usage
=====

Real-world
----------
**Note**: This repository does not contain the data necessary to replicate the steps in the Jupyter Notebook. The data used was proprietary and cannot be uploaded. Examples of the data and processing are shown within the notebooks.

1.  Navigate to the “notebooks” folder in the command prompt.
2.  Run the command `jupyter-notebook`. If this fails, attempt the following commands: `jupyter notebook`, `jupyter lab`
3.  Open the _Pushing.ipynb_ file within jupyter.
4.  Run the jupyter-notebook, paying attention the markdown cells for further instruction on operation.

Simulation
----------

For the simulation, there are two methods to load the data, either using the _gen\_videos.py_ file to generate the dataset from within RLlab, explaing below. Alternatively, the _expert\_push.pkl_ can be used to automatically generate the dataset from within the jupyter-notebook.

1.  Navigate to the “notebooks” folder in the command prompt.
2.  Run the command `jupyter-notebook`. If this fails, attempt the following commands: `jupyter notebook`, `jupyter lab`
3.  Open the _Pushing-sim.ipynb_ file within jupyter.
4.  Run the jupyter-notebook, paying attention the markdown cells for further instruction on operation.

For the simulation, RLlab is required which can be installed by following the instructions here: [https://rllab.readthedocs.io/en/latest/user/installation.html](https://rllab.readthedocs.io/en/latest/user/installation.html) 

MuJoCo physics-based simulator is also required, which requires a license (For this project, a temporary student-license was used).

6. Once RLlab is installed, launch RLlab and run the _gen\_videos.py_ file to generate the dataset within the simulation.
   
7. Use the dataset generated from _gen\_videos.py_ in the _Pushing-sim.ipynb_ to train the model.
   
8. Save the model using the code within the jupyter-notebook and link the directory of the model and the generated dataset to the _run\_ddpg\_push.py_ file in the following line of code `return dict(nvp=1, vp=vp, object=object_, goal=goal, imsize=(64, 64), geoms=geoms, name="push", modelname='Models/ctxskiprealtransform127_11751', meanfile=None, modeldata='Models/greenctxstartgoalvpdistractvalid.npy')`, replacing the “modelname” and “modeldata” parameters with the saved model from training and the dataset respectively; with the dataset being stored as .npy file.
    
9. Run the _run\_ddpg\_push.py_ file to perform the reinforcement learning component in simulation.
