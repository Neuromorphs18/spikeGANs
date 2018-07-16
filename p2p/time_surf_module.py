import pdb
import torch
import numpy as np

class time_surf_module(torch.nn.Module):
    def __init__(self, D_in, H, decay_constant):
        super(time_surf_module, self).__init__()
        self.input_size = D_in
        self.surface_state = torch.FloatTensor(np.zeros([D_in,H]))
        self.dumb = torch.nn.Linear(D_in,1)
        
    def forward(self, input):
        self.decay_constant = self.dumb(input).clamp(min=0)
        self.surface_state = np.multiply(self.surface_state, np.exp(-self.decay_constant.detach().numpy()))
        self.surface_state = self.surface_state + input
        return self.surface_state

    #This don't work
    def backward(self, grad_output):
        something = np.multiply((-np.exp(-self.decay_constant)),grad_output.clone())
        return grad_output