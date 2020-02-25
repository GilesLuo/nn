# Neural Network Solution of HW2
Author and coder : Giles Luo

This repository is a suppliant resource of HW2 for NUS courses ME5404 Neural Network. It contains the complete version of source code with #some typical GIFs result.

Training process is visualized by GIFs. An is example is shown here:

![image](https://github.com/GilesLuo/nn/blob/master/Q2/sequential_training/plot_no_noise/hidden%3D10_epoch%3D7999.gif)

The prediction output only fits x=[-1,1], because the network is trained on [-1,1], which intuitively verifies that NN only learn what is taught.

# Environment and modules
My souce code runs on:
    python 3.7.6
    
    pytorch 1.3.1
    
    torchvision 0.4.2
    
    tqdm 4.42.1 (this is a unnecessary terminal toolbar module. If you don't like it,  you can simply delete tqdm() outside the iterator)
    
    pyopengl 3.1.1a1
    
# Description    
The folder 'homework requirements' contains homework questions for hw2 and data set for hw2 Q3.

In folder 'source code and plots', Q1, Q2, Q3 are respectively the folder to solve corresponding questions in hw requirements.    

Q1 is solved using MATLAB, and Q2 Q3 are solved using Python.

In Q2, in addition to solve given question, I assume that the training data are disturbed by unknown Gaussian noise. The training result is shown in 'plot_with_noise' folder. 


