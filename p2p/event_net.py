import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc

from models.pix2pix_edited import Pix2PixModel
 
class AutoGRU(nn.Module):
    def __init__(self, x, y, seq_depth, p2pmodel, batch_size=10):
        super(AutoGRU,self).__init__()
        self.x, self.y = x, y
        self.seq_depth = seq_depth
        self.n_layers = 2
        self.batch_size = batch_size
        self.hidden_size = 700
        self.hid2 = 200
        self.gru1 = nn.GRU(input_size=self.x + self.y + 1,  hidden_size=self.hidden_size, batch_first=True) # x * y + 1 (==)
        self.gru2 = nn.GRU(input_size=self.hidden_size, hidden_size=self.hid2, batch_first=True)
        self.gru3 = nn.GRU(input_size=200, hidden_size=606, batch_first=True)
        self.GAN = p2pmodel
 
    def forward(self, seq):
        #h = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        out, h = self.gru1(seq) #TODO link hidden layers
        out, h = self.gru2(out)
        out, h = self.gru3(out)

        out = F.sigmoid(out)

        # wrap into 2d
        out = out.sum(dim=1)

        self.sent_packets = out.abs().sum()

        return out

        return self.GAN.forward(out)

    def compute_loss(self):
        loss = self.sent_packets / (self.seq_depth * self.batch_size * (self.x + self.y))
        #loss = self.GAN.get_loss_G() + self.sent_packets // (self.seq_depth)
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

    return xypol, time_d

def get_batch(index, data, time_d, seq_len=64, batch_size=10):
    total_size = seq_len*batch_size

    data = data[index:index + total_size]
    time_d = time_d[index:index + total_size]

    out = np.zeros([total_size, TOTX + TOTY + 1])

    out[:,xypol[:,0]] = xypol[:,2]
    out[:,TOTX + xypol[:,1]] = xypol[:,2]
    out[:,-1] = time_d

    out = torch.Tensor(out.reshape([batch_size, seq_len, TOTX + TOTY + 1]))

    # OLD VERSION - massive vector
    #out = np.zeros([len(x), TOTX, TOTY + 1])
    #out[:,xypol[0],xypol[1]] = xypol[2]
    #out = out.reshape(len(out), out.shape[1]*out.shape[2])
    #out[:,TOTY*TOTX] = time_d
    #out = np.concatenate([out, time_d.reshape(len(time_d), 1)], axis=1)

    return out

if __name__ == "__main__":
    # python3 event_net.py --dataroot ~/TelGanData/Tobi1 --name spikes2tobi --model pix2pix --which_direction AtoB --gpu_ids -1
    TOTX = 346
    TOTY = 260
    xypol, time_d = prep_data("Tobi1.csv")
    seq_len = 32
    batch_size=10

    # GRU

    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    #visualizer = Visualizer(opt)


    gru = AutoGRU(TOTX, TOTY, seq_len, model, batch_size=batch_size)


    """"opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    """


    optimi = optim.Adam(gru.parameters())

    max_epochs = 100

    for epoch in range(max_epochs):
        print("epoch {}".format(epoch))

        for i in range(len(time_d) - seq_len*batch_size):

            print("batch", i, end=" ")

            optimi.zero_grad()
            data = get_batch(i, xypol, time_d, seq_len=seq_len, batch_size=batch_size) # DOES NOT SLIDE #TODO move outside loop
            output = gru(data)
            loss = gru.compute_loss()
            loss.backward()
            optimi.step()

            print(loss)

    """
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    """
