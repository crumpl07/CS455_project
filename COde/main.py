# I lov emy girlfiredn vivian


from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

project_name = '06b-anime-dcgan'
DATA_DIR = '../input/animefacedataset/'

image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)]))

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

