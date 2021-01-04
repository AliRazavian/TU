import torch
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='/tmp', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(
    root='/tmp', train=False, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=False, num_workers=2)
valloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

filenames = []
object_type = []

storagePath = '../../../data/cifar10/'
for batch_idx, (inputs, targets) in enumerate(trainloader):
    filename = '%strain/%s/%05d.png' % (
        storagePath, classes[targets], batch_idx)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.utils.save_image(inputs, filename)
    filenames.append('train/%s/%05d.png' % (classes[targets], batch_idx))
    object_type.append(classes[targets])

pd.DataFrame({'Filename': filenames,
              'object_type': object_type}).to_csv('%strain_set.csv' % storagePath, index=False)


filenames = []
object_type = []

for batch_idx, (inputs, targets) in enumerate(valloader):
    filename = '%sval/%s/%05d.png' % (
        storagePath, classes[targets], batch_idx)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.utils.save_image(inputs, filename)
    filenames.append('val/%s/%05d.png' % (classes[targets], batch_idx))
    object_type.append(classes[targets])

pd.DataFrame({'Filename': filenames,
              'object_type': object_type}).to_csv('%sval_set.csv' % storagePath, index=False)
