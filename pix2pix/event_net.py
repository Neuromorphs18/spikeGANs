import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
class AutoGRU(nn.Module):
    def __init__(self, x, y, seq_depth):
        super(Sequence,self).__init__()
        self.x, self.y = x, y
        self.seq_depth = seq_depth
        self.gru1 = nn.GRU(c + 1, 500) # x * y + 1 (==)
        self.gru2 = nn.GRU(self.x * self.y, 500) # x, y, pol, don't send
 
    def forward(self,seq, hc = None):
        if hc == None:
            hc1, hc2 = None, None
        else:
            hc1, hc2 = hc
        gru1_out, hc1 = self.gru1(seq,hc1)
        gru2_out, hc2 = self.gru2(seq,hc2)
        out = torch.stack(out).squeeze(1)

        # wrap into 2d
        out = out.reshape(x,y)

        self.sent_packets = (out * out).sum() # OR (out * out) > 0 

        #GAN IT
        return gan(out)

    def compute_loss(self):
        gan_loss + self.sent_packets // (self.seq_depth)
        pass

def prep_data(filename):
    data = np.loadtxt(filename, skiprows=1, delimiter=",")
    
    # frame_now, event_id, prev_frame, x, y, polarity, timestamp
    x = data[:,np.array([3,4,5,6])]

    # take time difference
    times = x[:,3]
    time_d = times[1:] - times[:-1]
    x[:,3] = np.concatenate([np.zeros(1), time_d])

    # map 0,1 to -1,1
    x[2] = (x[2]*2) - 1


prep_data("Tobi1.csv")

print(np.unique(prep_data[2]))




