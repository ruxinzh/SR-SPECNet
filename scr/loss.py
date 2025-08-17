import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from scr.helpers import *

class SNRLoss(nn.Module):
    def __init__(self):
        super(SNRLoss, self).__init__()

    def forward(self, perdict, target, weight):
        """
        Calcula
        weighted_squared_errors = squared_errors * weight
        
        # Calculate the mean of the weighted squared errors
        loss = torch.mean(weighted_squtes the weighted mean squared error loss between the perdict and the target.
        
        Parameters:
        - perdict: Predicted values (tensor).
        - target: Ground truth values (tensor).
        - weight: Weight for each element in the batch (tensor).

        Returns:
        - loss: Weighted mean squared error loss.
        """
        # Calculate squared errors
        squared_errors = (perdict - target) ** 2
        
        # Apply weights
        weighted_squared_errors = squared_errors *( weight**2)
        # Calculate the mean of the weighted squared errors
        loss = torch.mean(weighted_squared_errors)
        return loss
    
class fftSNRLoss(nn.Module):
    def __init__(self):
        super(fftSNRLoss, self).__init__()

    def forward(self, perdict, target, weight):
        """
        Calcula
        weighted_squared_errors = squared_errors * weight
        
        # Calculate the mean of the weighted squared errors
        loss = torch.mean(weighted_squtes the weighted mean squared error loss between the perdict and the target.
        
        Parameters:
        - perdict: Predicted values (tensor).
        - target: Ground truth values (tensor).
        - weight: Weight for each element in the batch (tensor).

        Returns:
        - loss: Weighted mean squared error loss.
        """
        # Calculate squared errors
        perdict = perdict[:,:,:,0] + 1j*perdict[:,:,:,1]
        target = target[:,:,:,0] + 1j*target[:,:,:,1]
        nBatch,nSig,nEle = perdict.shape
        perdict = perdict.view(nBatch*nSig,-1)
        target = target.view(nBatch*nSig,-1)
        linear_space = torch.linspace(-1, 1, 512)
        angle_axis = torch.asin(linear_space) * (180 / torch.pi)
        a_theta = steering_vector(nEle, angle_axis).conj()
        AH = torch.transpose(a_theta, 0, 1).unsqueeze(0)
        AH = AH.repeat(nBatch*nSig, 1, 1).to(perdict.device)        
        perdict = torch.abs(torch.matmul(AH,perdict.unsqueeze(-1))).squeeze(-1)
        target = torch.abs(torch.matmul(AH,target.unsqueeze(-1))).squeeze(-1)
        
        squared_errors = (torch.log10(perdict) - torch.log10(target)) ** 2 
        
        # Apply weights
        weight = weight.view(nBatch*nSig,-1)
        weighted_squared_errors = squared_errors * weight
        # Calculate the mean of the weighted squared errors
        loss = torch.mean(weighted_squared_errors)
        return loss