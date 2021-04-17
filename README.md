# GAN art generator
## About and Technologies
This is an Implmentation of a Generative adversial network. The aim of this projet was to train the GAN such that it produces images that are indstiguishable from exisiting art.  
The GAN neural network was built using Python and utilised the PyTorch library for consturcting the discriminator and generator neural networks. Tkinter was used for the GUI aspects of the project.
## Enviroment Setup
Before you are able to run this project you need to first setup the proper enviroment on your computer. As this project was only run on Windows with Pytorch in Cuda mode this guide is only going to describe how to setup for that enviroment
#### Installing Python 3 and Tkinter
The first thing you need to do is to install python3 on your Machine. if you already have python installed you can skip this step.:  
- Go to Python’s [download page](https://www.python.org/downloads/release/python-394/):
- Download the appropriate installer for your machine
- Make sure to check the box that asks you to add python to your path when prompted
- You should now have python installed and added to your path 
- Tkinter comes prepackeged with Python3 so you now should also have Tkinter installed
#### Installing PyTorch in Cuda Mode
- After Installing Python3 and adding it to your path
- Open Command Prompt
- run the "pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html" without the qoutes to install Pytorch
- You have now succesfully installed PyTorch for Cuda 11
- Note that if your GPU device does not support Cuda 11 , you can try other installations of PyTorch here https://pytorch.org/ however this may require minor changes to the code     in order to get it to work
## Code Documentation
The codebase is small and consists of only 5 classes. Here is a quick run down of these classes and how to use them
#### Viewer.pyw
This is file responsible for providing the GUI and running the GAN in inference mode. The viewer class allows the user to ask the GAN to generate new Fake images. The user can change which dataset they want the GAN to emulate and which epoch of training the GAN stopped at. The Viewer is a simple class extends a TK window. On initialization it creates 3 radio buttons that allows the user to select which GAN model they want to be loaded. The GAN models differ on which training set they were trained on. It also providers a slider that allows the user to specify which epoch of training to load. It also intialises 2 buttons "Generate New random vector" and "Generate New". "Generate New" instructs the GAN to generate new images using the internal intialization vector and then displays these images on the screen  ."Generate New random vector" generates a new random vector to be used by the GAN when genrating images. To see the changes the new vector to produces you need to always click on the "Generate New" button after clicking on the "Generate New random vector" button. You can run the GUI by double clicking on the Viewer.pyw file after installing python. Note that the GUI will not generate new images if trained models do not exist in the specified diretory stucture (decribed later) and it also wont generate images if the model with the specified Model Number doesnt Exists. Currently the trainer saves every 10 iterations so if you try to load iteration that is not a multiple of 10 the gui will not be able to load it. reading https://docs.python.org/3/library/tk.html May be useful to get a basic understanding on How Tkinter works  

![image](https://user-images.githubusercontent.com/69083495/115099624-6adfc680-9f37-11eb-9113-29b83e0e27a6.png)  
#### Discriminator.py and Genrator.py
these file contain the implmentation of the Generator and Discriminator. They are built using PyTorchs sequentail class. The generator takes in a random 1d array vector and through 5 convoltional layers transposes that random vector onto the sample space of the training images of size 64x64. On the other hand the discrimnator also uses 5 convoltional layers but instead it transposes the input image into a binary decision of wether or not an image was produced by the generator.  
The following resources may be useful when using these classes:
- https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
- https://pytorch.org/docs/stable/nn.html
#### Configuaration.py and Trainer.py
the Trainer.py file is responsible for training the GAN network on a specified dataset and storing and updated model after a specified number of epochs so that it can be used by GUI. The configuration.py file contains the configuration settings of the Trainer.py class. Here you can change which data set the Trainer.py file will use, how often and where the models will be saved (Other settings should not be changed by the user) and which epoch the trainer should resume training from. When you run the Trainer.py file it will use the settings from configuration.py and start training your model. 
## Folder Sturcture 
As the models and Training datasets are pretty large they were not uploaded to Github. As such you will need to download the training sets and train your models for a bit before your are able to use GUI class. Here are the links to the datasets that were used when training this model:  
- https://www.kaggle.com/ipythonx/van-gogh-paintings}
- https://www.kaggle.com/arnaud58/landscape-pictures
- http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
Note that your datasets need to be inserted in a folder structure that looks like this relative to trainer.py "data/datasetname/1/". You would put your training images inside the 1 folder. Similarly when saving your models they will be save in a folder structure that looks like this "models/datasetname/"



 

