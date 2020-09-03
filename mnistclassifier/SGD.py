from fastai.vision.all import *
from fastbook import *

## MNIST Sample of just 3s and 7s
path = untar_data(URLs.MNIST_SAMPLE, dest = './')
Path.BASE_PATH = path
Path.BASE_PATH, path

path.ls() ## labels.csv, valid/, train/
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()

## Open Images
im3_path = threes[0]
im3 = Image.open(im3_path)
im3.show()

## Preparing training data
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)

## Preparing the validation data
valid_3_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens.shape,valid_7_tens.shape

## Dataset needs to return (x,y) tuple
dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))

## Initialize Parameters
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()

weights = init_params((28*28,1))
weights.shape

