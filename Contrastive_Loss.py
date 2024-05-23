import numpy
import torch
import torch.nn as nn

class Contrastive_Loss(nn.Module):
    def __init__(self, margin, p = 2):
        super(Contrastive_Loss, self).__init__()
        '''
        margin 
        
        distance > margin ) predict : False
        distance <= margin ) predict : True
        
        p : p of Lp-norm
        p = 1, Manhattan Norm, p = 2, Euclidean Norm
        '''
        
        self.margin = margin 
        self.p = p
        
    def forward(self, X_1, X_2, equlity):
        '''
        X_1, X_2 : feature vector
        equlity : y_1 == y_2
        '''
        
        equlity = equlity.int()
        
        distance = torch.norm(X_1-X_2, p = self.p) 
        
        loss = ((1-equlity)*(distance**2)+equlity*(torch.clamp(self.margin-distance, min=0)**2))/2
        
        predict = True if distance > self.margin else False
        
        return loss, predict
    