from os.path import join, expanduser
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
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
        # self.conv1 = nn.Conv2d(3, 3, (3, 3))
        # self.conv2 = nn.Conv2d(3, 3, (3, 3))
        # self.conv3 = nn.Conv2d(3, 3, (3, 3))
        self.gru1 = nn.GRU(input_size=self.x + self.y + 1,  hidden_size=self.hidden_size, batch_first=False) # x * y + 1 (==)
        self.gru2 = nn.GRU(input_size=self.hidden_size, hidden_size=self.hid2, batch_first=False)
        self.gru3 = nn.GRU(input_size=self.hid2, hidden_size=1, batch_first=False)
        self.pass_on = nn.Linear(self.seq_depth, self.x*self.y)
        self.time_surf_module = time_surf_module(TOTX, TOTY, 1e-4)
        # decay_constant = 0.1
        #TODO timesurf
        #self.time_surf = time_surf_module(606, (692, 260), decay_constant)
        self.GAN = p2pmodel

    # def spikes_to_img(self, spikes):
    #     spikes_arr = spikes.numpy()
    #     spikes_arr_x = np.where(spikes_arr[:, :, :TOTX] != 0)[2].astype(np.int16)
    #     spikes_arr_y = np.where(spikes_arr[:, :, TOTX:-1] != 0)[2].astype(np.int16)
    #     # spike_arr_pol = spike_arr_x[np.where(spike_arr_x != 0)].astype(np.int8)
    #     tot_events = batch_size * seq_len
    #     # spikes_arr_pol = spikes_arr[:, :, :-1].reshape((tot_events, 606))[range(tot_events), spikes_arr_x]
    #     times = spikes_arr[:, :, -1].reshape((tot_events,))
    #     img = np.zeros((TOTX, TOTY))
    #     img[spikes_arr_x, spikes_arr_y] = times
    #     return torch.Tensor(img)

    def forward(self, spikes):
        #h = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        out, h = self.gru1(spikes) #TODO link hidden layers
        out, h = self.gru2(out)
        out, h = self.gru3(out)
        out.squeeze_()
        time_surf = self.pass_on(out)
        time_surf = F.tanh(time_surf.view([self.batch_size, self.x, self.y]))

        # out, h = self.conv1(spikes) #TODO link hidden layers
        # out, h = self.conv2(out)
        # out, h = self.conv3(out)

        # out = F.sigmoid(out).round() #, dim=2) #comment out for batch-wise spike output
        #
        # time_surf = self.time_surf_module(spikes, out)
        #
        # self.sent_packets = out.sum() / (self.batch_size*self.seq_depth)

        return time_surf

    def compute_loss(self, time_surf, target_imgs):
        if time_surf is not None:
            gan_input = time_surf

        folder = "less_spikes"

        try:
            os.mkdir(folder)
        except FileExistsError:
            pass



        for i in range(self.batch_size):
            #plt.imshow(gan_input.numpy())
            #plt.show()
            R = np.zeros((self.x, self.y))
            ids = np.where(gan_input[i] > 0)
            R[ids] = gan_input.detach().numpy()[i][ids]
            B = np.zeros((self.x, self.y))
            ids = np.where(gan_input[i] < 0)
            B[ids] = gan_input.detach().numpy()[i][ids] * -1
            G = np.zeros((self.x, self.y))
            img_a = np.zeros((self.x, self.y, 3))
            img_a[:, :, 0] = R
            img_a[:, :, 1] = G
            img_a[:, :, 2] = B
            img_a = Image.fromarray((255*img_a).astype(np.uint8).reshape(self.y, self.x, 3)) #TODO why is this differnt to matplotlib?
            img_b = Image.open(os.path.join(images_folder, "targets", "{}.png".format(target_imgs[i])))
            aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
            aligned_image.paste(img_a, (0, 0))
            aligned_image.paste(img_b, (img_a.size[0], 0))
            aligned_image.save(os.path.join(folder, "train", "{}.png".format(i)))

        #self.GAN.opt.dataroot = folder
        g_loss, d_loss = self.GAN.generate_img(folder)
        g_loss, d_loss = torch.stack(g_loss), torch.stack(d_loss)

        # loss = torch.stack(self.sent_packets)
        loss = time_surf.norm()
        print("l", loss.mean(), "g", g_loss.mean(), "d", d_loss.mean())
        loss = g_loss
        return loss

def row_to_vec(row):
    x, y, polarity, timed = row
    r = np.zeros([TOTX, TOTY])
    r[int(x), int(y)] = polarity
    r = np.concatenate([r.flatten(), np.array([timed])])
    return r

def prep_data(filename):
    data = np.loadtxt(filename, skiprows=1, delimiter=",")

    # remove first image as this will be the baseline
    data = data[np.where(data[:,2] != 1)]

    # frame_now, event_id, prev_frame, x, y, polarity, timestamp
    x = data[:,np.array([3,4,5,6])] # x, y, pol, time

    # take time difference
    xypol = x[:,np.array([0,1,2])].astype(np.int16)
    xypol[:, 2] = (xypol[:, 2] * 2) - 1

    times = x[:,3]
    time_d = times[1:] - times[:-1]
    time_d = np.concatenate([np.zeros(1), time_d])

    return xypol, time_d, data[:,2]

def get_batch(index, data, time_d, targets, seq_len=64, batch_size=10):
    total_size = seq_len * batch_size

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
    tgts = []

    for i in range(batch_size):
        step = i*seq_len
        in_data.append(newd[step:step+seq_len])
        tgts.append(int(mode(targets[step:step+seq_len]).mode))


    in_data = np.array(in_data)
    in_data.reshape((seq_len, batch_size, TOTX + TOTY + 1))

    return torch.Tensor(in_data), tgts

if __name__ == "__main__":
    # python3 event_net.py --dataroot ~/TelGanData/Tobi1 --name spikes2tobi --model pix2pix --which_direction AtoB --gpu_ids -1
    TOTX = 346
    TOTY = 260
    file = "Tobi1_small.csv"

    print('Loading...')
    xypol, time_d, img_ids = prep_data(file)
    print('Done!')

    seq_len = 128
    batch_size = 32

    images_folder = "../Tobi1/"
    gan = TrainedGAN(images_folder, "tobi_model1", batch_size=batch_size)

    gru = AutoGRU(TOTX, TOTY, seq_len, gan, batch_size=batch_size)

    optimi = optim.Adam(gru.parameters())

    max_epochs = 100

    for epoch in range(max_epochs):
        print("epoch {}".format(epoch))

        for i in range(len(time_d) - seq_len*batch_size):
            print("batch", i, end=" ")
            optimi.zero_grad()
            data, targets = get_batch(i, xypol, time_d, img_ids, seq_len=seq_len, batch_size=batch_size) # DOES NOT SLIDE #TODO move outside loop
            output = gru(data)
            loss = gru.compute_loss(output, targets)
            loss.sum().backward()
            optimi.step()
