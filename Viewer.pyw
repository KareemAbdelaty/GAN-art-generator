from tkinter import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from PIL import Image, ImageTk
from Configuration import *
from Generator import *
from Discriminator import *
class Viewer(Tk):
    def __init__(self):
        
        Tk.__init__(self)
        self.init_interface()
        
    def show_frame(self, cont):
        frame  = self.currentFrame
        frame.pack_forget()
        frame = self.frames[cont]
        frame.pack(fill ='both',expand = True)
        self.currentFrame = frame
    def init_interface(self):
        container = Frame(self)
        self.title("GAN Generator Demo")
        self.geometry("1060x1060")
        container.pack(side="top", fill="both", expand = True)
        self.frames = {}
        for F in [MainMenu]:

            frame = F(container, self)

            self.frames[F] = frame

        self.currentFrame = self.frames[MainMenu]
        self.show_frame(MainMenu)
        
class MainMenu(Frame):

    def __init__(self,master,controller):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)
        self.controller = controller
        #reference to the master widget, which is the tk window
        self.master = master
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        self.noise = torch.randn(64, nz, 1, 1, device=self.device)
        self.model = Generator(ngpu).to(self.device)
        self.dataset = IntVar()
        self.num = IntVar()
        self.init_window()
    def init_window(self):
        self.R1 = Radiobutton(self, text="CelebFaces", variable= self.dataset, value=1,
                )
        self.R1.select()
        self.R2 = Radiobutton(self, text="Landscapes", variable=self.dataset, value=2,
                )
        self.R3 = Radiobutton(self, text="Van Goch", variable=self.dataset, value=3,
                )
        self.epoch = Scale(self,length=1060,bg ="cyan",orient=HORIZONTAL,from_ = 0, to = 200,variable = self.num)
        self.vectorz = Button(self,width = 1060, text = "New Random Vector" ,bg = "cyan",command =lambda: self.update_vector())
        self.generateNew = Button(self,width = 1060 ,text = "Generate new" ,bg = "cyan",command =lambda: self.generateNewImage())
        self.imageMatrix = Label(self)
        self.R1.pack()
        self.R2.pack()
        self.R3.pack()
        self.epoch.pack()
        self.vectorz.pack()
        self.generateNew.pack()
        self.imageMatrix.pack()
    def update_vector(self):
        self.noise = torch.randn(64, nz, 1, 1, device=self.device)
    def generateNewImage(self):
        if(self.dataset.get() == 1):
            path = "models/celeba/celeba"+str(self.num.get())+".model"
            firstModel = torch.load(path)
            netG = Generator(ngpu).to(self.device)
            netG.load_state_dict(firstModel["generator_state_dict"])
            netG.eval()
            fake1 = netG(self.noise).detach().cpu()
            array = np.transpose(vutils.make_grid(fake1, padding=2, normalize=True),(1,2,0)).numpy()
            im = np.array(Image.fromarray((array * 255).astype(np.uint8)).convert('RGB'))
            self.img =  ImageTk.PhotoImage(image=Image.fromarray(im))
            self.imageMatrix.configure(image = self.img)
            self.imageMatrix.image = self.img
            self.imageMatrix.pack()
        elif(self.dataset.get() == 2):
            path = "models/landscapes/scenery"+str(self.num.get())+".model"
            firstModel = torch.load(path)
            netG = Generator(ngpu).to(self.device)
            netG.load_state_dict(firstModel["generator_state_dict"])
            netG.eval()
            fake1 = netG(self.noise).detach().cpu()
            array = np.transpose(vutils.make_grid(fake1, padding=2, normalize=True),(1,2,0)).numpy()
            im = np.array(Image.fromarray((array * 255).astype(np.uint8)).convert('RGB'))
            self.img =  ImageTk.PhotoImage(image=Image.fromarray(im))
            self.imageMatrix.configure(image = self.img)
            self.imageMatrix.image = self.img
            self.imageMatrix.pack()
        elif(self.dataset.get()==3):
            path = "models/art/art"+str(self.num.get())+".model"
            firstModel = torch.load(path)
            netG = Generator(ngpu).to(self.device)
            netG.load_state_dict(firstModel["generator_state_dict"])
            netG.eval()
            fake1 = netG(self.noise).detach().cpu()
            array = np.transpose(vutils.make_grid(fake1, padding=2, normalize=True),(1,2,0)).numpy()
            im = np.array(Image.fromarray((array * 255).astype(np.uint8)).convert('RGB'))
            self.img =  ImageTk.PhotoImage(image=Image.fromarray(im))
            self.imageMatrix.configure(image = self.img)
            self.imageMatrix.image = self.img
            self.imageMatrix.pack()        
            

        
        
        
if __name__ == "__main__":
    app = Viewer()
    app.mainloop()
