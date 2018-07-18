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
        self.conv1 = nn.Conv2d(3, 6, (3, 3))
        self.conv2 = nn.Conv2d(6, 2, (2, 2))
        self.conv3 = nn.Conv2d(2, 3, (3, 3))
        self.GAN = p2pmodel

    def forward(self, spikes):
        print(spikes.shape)
        out = self.conv1(spikes)
        out = self.conv2(out)
        out = self.conv3(out)

        print(out.shape)

        return out

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
    xypol = x[:,np.array([0,1,2,3])].astype(np.int16)
    xypol[:, 2] = (xypol[:, 2] * 2) - 1

    times = x[:,3]
    time_d = times[1:] - times[:-1]
    time_d = np.concatenate([np.zeros(1), time_d])
    xypol[:,3] = time_d

    return xypol, data[:,2]


def ConvertTDeventsToAccumulatedFrameConvert(td_events, width=346, height=260, tau=1e6):
    """ Converts the specified TD data to a frame by accumulating events """
    state = {}
    state['last_spike_time'] = np.zeros([width, height])
    state['last_spike_polarity'] = np.zeros([width, height])
    td_image = np.zeros([width, height])

    last_t = 0
    for row in td_events:
        [x, y, p, t] = row

        state['last_spike_time'][x, y] = t
        state['last_spike_polarity'][x, y] = 1 if p == True else -1
        last_t = t

        td_image[x, y] = p

    surface_image = state['last_spike_polarity'] * np.exp((state['last_spike_time'] - last_t) / tau)

    return state, surface_image.transpose()

def get_batch(index, data, targets, seq_len=64, batch_size=10, x=346, y=260):
    total_size = seq_len * batch_size
    cmap = plt.cm.jet

    data = data[index:index + total_size]
    tgts = []
    in_data = []

    for i in range(batch_size):
        step = i*seq_len
        state, img = ConvertTDeventsToAccumulatedFrameConvert(data[step:step+seq_len])
        img = np.fliplr(np.flipud(img))

        normed_data = (img - np.min(img)) / (np.max(img) - np.min(img))
        mapped_data = cmap(normed_data)

        mapped_data = mapped_data[:,:,:3]

        in_data.append(mapped_data)
        tgts.append(int(mode(targets[step:step+seq_len]).mode))


    in_data = np.array(in_data).reshape(batch_size, 3, x, y)

    return in_data, tgts

if __name__ == "__main__":
    # python3 event_net.py --dataroot ~/TelGanData/Tobi1 --name spikes2tobi --model pix2pix --which_direction AtoB --gpu_ids -1
    TOTX = 346
    TOTY = 260
    file = "Tobi1_small.csv"

    device = "cpu" if not(torch.cuda.is_available()) else "cuda"

    print('Loading...')
    xypol, img_ids = prep_data(file)
    print('Done!')

    seq_len = 1000
    batch_size = 64

    images_folder = "../Tobi1/"
    gan = TrainedGAN(images_folder, "tobi_model1", batch_size=batch_size)

    gru = AutoGRU(TOTX, TOTY, seq_len, gan, batch_size=batch_size)

    optimi = optim.Adam(gru.parameters())

    max_epochs = 100

    for epoch in range(max_epochs):
        print("epoch {}".format(epoch))

        for i in range(len(xypol) - seq_len*batch_size):
            print("batch", i, end=" ")
            optimi.zero_grad()
            data, targets = get_batch(i, xypol, img_ids, seq_len=seq_len, batch_size=batch_size) # DOES NOT SLIDE #TODO move outside loop
            output = gru(torch.Tensor(data, device=device))
            loss = gru.compute_loss(output, targets)
            loss.sum().backward()
            optimi.step()
