from os.path import join, expanduser
import numpy as np
import pandas as pd
from scipy.stats import mode

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pretrained_gan import TrainedGAN
from time_surf_module import time_surf_module


class AutoGRU(nn.Module):
    def __init__(self, x, y, seq_depth, p2pmodel, batch_size=10):
        super(AutoGRU,self).__init__()
        self.x, self.y = x, y
        self.decay_constant = 0.1
        self.seq_depth = seq_depth
        self.n_layers = 2
        self.batch_size = batch_size
        self.hidden_size = 700
        self.hid2 = 200
        self.gru1 = nn.GRU(input_size=self.x + self.y + 1,  hidden_size=self.hidden_size, batch_first=False) # x * y + 1 (==)
        self.gru2 = nn.GRU(input_size=self.hidden_size, hidden_size=self.hid2, batch_first=False)
        self.gru3 = nn.GRU(input_size=200, hidden_size=self.x + self.y, batch_first=False)
        self.gather_x = torch.LongTensor(list(range(self.x)))
        self.gather_y = torch.LongTensor(list(range(self.x, self.x + self.y)))
        self.pass_on = nn.Linear(self.x + self.y, 3)

        decay_constant = 0.1
        #TODO timesurf
        #self.time_surf = time_surf_module(606, (692, 260), decay_constant)
        self.GAN = p2pmodel
 
    def forward(self, spikes):
        #h = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        out, h = self.gru1(spikes) #TODO link hidden layers
        out, h = self.gru2(out)
        out, h = self.gru3(out)

        final_layer = out.index_select(1, torch.LongTensor([self.batch_size - 1])).squeeze()

        out = F.softmax(self.pass_on(final_layer), dim=1) #comment out for batch-wise spike output

        """out = F.sigmoid(out)
        # wrap into 2d
        x = torch.index_select(out, 1, self.gather_x).unsqueeze(1)
        y = torch.index_select(out, 1, self.gather_y).unsqueeze(1)
        #TODO this outputs for last 64 steps, do we want just the last time-step
        print(out.size())
        mask = torch.bmm(x.round().permute(0,2,1), (y).round()).abs().clamp(min=-1, max=1)
        print(x.size(), y.size(), mask.size())
        x = x.squeeze().unsqueeze(-1).expand(self.batch_size, self.x, self.y)
        y = y.squeeze().unsqueeze(-1).expand(self.batch_size, self.x, self.y)
        out = torch.bmm(x.round().permute(0,2,1), (y).round()).clamp(min=-1, max=1)
        """

        self.sent_packets = out.max(1)[1].abs().sum()

        return out, out.max(1)[1]


    def compute_loss(self, spikes, x, y, target_img):
        gan_input = np.zeros((self.batch_size, 3, self.x, self.y))
        blanks = np.zeros((self.batch_size, 3, self.x, self.y))
        gan_input[np.arange(self.batch_size), spikes.detach().numpy(), x, y] = 255
        gan_input = torch.cat([torch.Tensor(gan_input), torch.Tensor(blanks)], dim=1)

        #TODO randomly remove spkes from image and see if salient ones are picked out

        g_loss = self.GAN.generate_img(gan_input)

        loss = self.sent_packets / (self.seq_depth * self.batch_size)
        loss += g_loss
        return loss

def row_to_vec(row):
    x, y, polarity, timed = row
    r = np.zeros([TOTX, TOTY])
    r[int(x), int(y)] = polarity
    r = np.concatenate([r.flatten(), np.array([timed])])
    return r

def prep_data(filename):
    data = np.loadtxt(filename, skiprows=1, delimiter=",")

    # frame_now, event_id, prev_frame, x, y, polarity, timestamp
    x = data[:,np.array([3,4,5,6])] # x, y, pol, time

    # take time difference
    xypol = x[:,np.array([0,1,2])].astype(np.int16)
    xypol[:, 2] = (xypol[:, 2] * 2) - 1

    times = x[:,3]
    time_d = times[1:] - times[:-1]
    time_d = np.concatenate([np.zeros(1), time_d])

    return xypol, time_d, data[:0]

def get_batch(index, data, time_d, targets, seq_len=64, batch_size=10):
    total_size = seq_len + batch_size

    data = data[index:index + total_size]
    time_d = time_d[index:index + total_size]
    targets = targets[index:index + total_size]

    newd = np.zeros([total_size, TOTX + TOTY + 1])

    r = range(len(data[:, 0]))
    newd[r, data[:, 0]] = data[r, 2]
    newd[r, TOTX + data[:, 1]] = data[r, 2]
    """out = np.zeros([total_size, TOTX*TOTY + 1])
    r = range(len(data[:, 0]))
    out[r, TOTX * data[:, 0] + data[:, 1]] = data[r, 2]
    """
    newd[:, -1] = time_d # TODO scale time_d to 0,1 or scale spikes by this - what about when the camera stays still for a long time? - will go to 0 - does this matter/
    in_data = []

    for i in range(batch_size):
        in_data.append(newd[i:i+seq_len])

    targets = targets[seq_len:]

    x = data[:,0][seq_len:]
    y = data[:,1][seq_len:]
    in_data = np.array(in_data)
    in_data.reshape((seq_len, batch_size,  TOTX + TOTY + 1))

    return torch.Tensor(in_data), targets, x, y

if __name__ == "__main__":
    # python3 event_net.py --dataroot ~/TelGanData/Tobi1 --name spikes2tobi --model pix2pix --which_direction AtoB --gpu_ids -1
    TOTX = 346
    TOTY = 260
    df = pd.read_csv("Tobi1_small.csv")
    #df["time_d"] = 0
    #df["time_d"] = np.concatenate([df["timestamp"][1:] - df["timestamp"][:-1]]).squeeze()

    xypol, time_d, img_ids = prep_data("Tobi1_small.csv")
    seq_len = 32
    batch_size=10

    print(df.columns.values)

    #gan = TrainedGAN("/Users/Tilda/TelGanData/Tobi1/", "tobi_model1", df[df["frame_now"] == df["frame_now"].min()][["x", "y", "polarity", 'timestamp']].as_matrix())

    gan = TrainedGAN("/Users/Tilda/TelGanData/Tobi1/", "tobi_model1")

    gru = AutoGRU(TOTX, TOTY, seq_len, gan, batch_size=batch_size)

    optimi = optim.Adam(gru.parameters())

    max_epochs = 100

    for epoch in range(max_epochs):
        print("epoch {}".format(epoch))

        for i in range(len(time_d) - seq_len*batch_size):
            print("batch", i, end=" ")
            optimi.zero_grad()
            data, targets, x, y = get_batch(i, xypol, time_d, img_ids, seq_len=seq_len, batch_size=batch_size) # DOES NOT SLIDE #TODO move outside loop
            output, hardmax_spikes = gru(data)
            loss = gru.compute_loss(hardmax_spikes, x, y, targets)
            loss.backward()
            optimi.step()
