import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attacker.linf_sgd import Linf_SGD
from torch.optim import SGD, Adam




import torch
from torch import nn
import torch.nn.functional as F


class AttackPGD(nn.Module):
    def __init__(self, model):
        super(AttackPGD, self).__init__()
        config = {'attack': True,
                'epsilon': 8 / 255.,
                'num_steps': 1,
                'step_size': 2 / 255.,
                'random_start': True}
        
        
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.attack = config['attack']

    def forward(self, inputs, targets):
        if not self.attack:
            return self.model(inputs), inputs

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
############################################################################################################                
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
############################################################################################################                
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            delta = torch.clamp(x - inputs, min=-self.epsilon, max=self.epsilon)
            x = torch.clamp(inputs + delta, min=0, max=1).detach()
    
#             x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
#             x = torch.clamp(x, 0, 1)

        return self.model(x), delta, x




# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def Linf_PGD_alpha(model, X, y, epsilon, steps=1, random_start=True):
    training = model.training
    if training:
        model.eval()
    saved_params = [p.clone() for p in model.arch_parameters()]
    optimizer = Linf_SGD(model.arch_parameters(), lr=2*epsilon/steps)
    with torch.no_grad():
        loss_before = model._loss(X, y, updateType='weight')
    if random_start:
        for p in model.arch_parameters():
            p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
        model.clip()
        
    for _ in range(steps):
        optimizer.zero_grad()
        model.zero_grad()
        loss = -model._loss(X, y, updateType='weight')
        loss.backward()
        optimizer.step()
        diff = [(model.arch_parameters()[i] - saved_params[i]).clamp_(-epsilon, epsilon)
                for i in range(len(saved_params))]
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(diff[i] + saved_params[i])
        model.clip()
        
    optimizer.zero_grad()
    model.zero_grad()
    with torch.no_grad():
        loss_after = model._loss(X, y, updateType='weight')
    if loss_before > loss_after:
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(saved_params[i])
    if training:
        model.train()
        
    return diff


def Random_alpha(model, X, y, epsilon):
    for p in model.arch_parameters():
        p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
    model.clip()


    
    
    
    
    
    
    
    
    
def Linf_PGD_alpha_RNN(model, X, y, hidden, epsilon, steps=7, random_start=True):
    training = model.training
    if training:
        model.eval()
    saved_params = [p.clone() for p in model.arch_parameters()]
    optimizer = Linf_SGD(model.arch_parameters(), lr=2*epsilon/steps)
    with torch.no_grad():
        loss_before, _ = model._loss(hidden, X, y, updateType='weight')
    if random_start:
        for p in model.arch_parameters():
            p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
        model.clip()

    for _ in range(steps):
        optimizer.zero_grad()
        model.zero_grad()
        loss, _ = model._loss(hidden, X, y, updateType='weight')
        loss = -loss
        loss.backward()
        optimizer.step()
        diff = [(model.arch_parameters()[i] - saved_params[i]).clamp_(-epsilon, epsilon)
                for i in range(len(saved_params))]
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(diff[i] + saved_params[i])
        model.clip()
    
    optimizer.zero_grad()
    model.zero_grad()
    with torch.no_grad():
        loss_after, _ = model._loss(hidden, X, y, updateType='weight')
    if loss_before > loss_after:
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(saved_params[i])
    if training:
        model.train()
    
        
def Random_alpha_RNN(model, X, y, hidden, epsilon):
    for p in model.arch_parameters():
        p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
    model.clip()





