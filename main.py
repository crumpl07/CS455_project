import PySimpleGUI as sg
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os


sg.theme('SystemDefault')

layout = [  [sg.Image()],
            [sg.Button('Generate'), sg.Button('Cancel')] ]


window = sg.Window('GANS for Animal Faces', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break  
    if event == 'Generate': 
        print("Run GANS")

window.close()

def danny(): 

    project_name = 'CS455_PROJECT'
    DATA_DIR = 'afhq/'

    image_size = 64
    batch_size = 128
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # I lov emy girlfiredn vivian
    def denorm(img_tensors):
        return img_tensors * stats[1][0] + stats[0][0]

    def show_images(images, nmax=64):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
        plt.savefig(os.join(os.getcwd())

    def show_batch(dl, nmax=64):
        for images, _ in dl:
            show_images(images, nmax)
            break

    print(os.listdir(DATA_DIR)[:10])

    # print(DATA_DIR)

    train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats)]))

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)

    show_batch(train_dl)

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl: 
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device) 

