import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt


class time_surf_module(torch.nn.Module):
    def __init__(self, input_width, input_height, decay_constant):
        super(time_surf_module, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_size = input_width
        # self.decay_constant = torch.Tensor(np.full((input_width, input_height), decay_constant))
        self.decay_constant = decay_constant
        self.input_width = input_width
        self.input_height = input_height
        self.time_map_pos = torch.Tensor(np.full((self.input_width, self.input_height), np.iinfo(np.int32).max, dtype=np.int32), device=self.device)
        self.time_map_neg = torch.Tensor(np.full((self.input_width, self.input_height), np.iinfo(np.int32).max, dtype=np.int32), device=self.device)
        self.time_surface = np.zeros((input_width, input_height, 3), dtype=np.float)
        # self.time_surface [:, :, 0] = 109
        # self.time_surface [:, :, 1] = 255
        # self.time_surface [:, :, 2] = 106
        # self.time_surface = torch.Tensor(self.time_surface)
        # self.dumb = torch.nn.Linear(1, D_in)

    def forward(self, spikes, yes_no):
        # self.decay_constant = self.dumb(input).clamp(min=0)
        yn = yes_no.cpu().detach().numpy().astype(np.bool).reshape((-1,))
        times = np.cumsum(spikes.cpu().numpy()[:, :, -1])
        cur_time = times[-1]
        spikes_arr_x = np.where(spikes[:, :, :self.input_width] != 0)[2].astype(np.int16)
        spikes_arr_y = np.where(spikes[:, :, self.input_width:-1] != 0)[2].astype(np.int16)
        spikes_arr_pol = spikes.cpu().numpy()[:, :, :-1].reshape((len(spikes_arr_x), 606))[range(len(spikes_arr_x)), spikes_arr_x]
        spikes_arr_y = spikes_arr_y[yn]
        spikes_arr_x = spikes_arr_x[yn]
        spikes_arr_pol = spikes_arr_pol[yn]
        times = times[yn]
        spikes_arr_y_pos = spikes_arr_y[spikes_arr_pol == 1]
        spikes_arr_x_pos = spikes_arr_x[spikes_arr_pol == 1]
        spikes_arr_y_neg = spikes_arr_y[spikes_arr_pol == -1]
        spikes_arr_x_neg = spikes_arr_x[spikes_arr_pol == -1]
        times_pos = times[spikes_arr_pol == 1]
        times_neg = times[spikes_arr_pol == -1]
        ids_neg = np.where(self.time_map_neg.numpy() >= 0)
        self.time_map_neg.cpu().numpy()[ids_neg] += cur_time
        self.time_map_neg.cpu().numpy()[spikes_arr_x_neg, spikes_arr_y_neg] = times_neg
        channel_neg = np.exp(-self.decay_constant * self.time_map_neg.numpy())
        ids_pos = np.where(self.time_map_pos.numpy() >= 0)
        self.time_map_pos.cpu().numpy()[ids_pos] += cur_time
        self.time_map_pos.cpu().numpy()[spikes_arr_x_pos, spikes_arr_y_pos] = times_pos
        channel_pos = np.exp(-self.decay_constant * self.time_map_pos.numpy())
        self.time_surface[:, :, 1] = 1
        self.time_surface[:, :, 1][(channel_pos != 0) | (channel_neg != 0)] = 0
        channel_pos[channel_pos == 0] = 109/255
        channel_neg[channel_neg == 0] = 106/255
        self.time_surface[:, :, 0] = channel_pos
        self.time_surface[:, :, 2] = channel_neg
        return torch.Tensor(self.time_surface, device=self.device)

    # #This don't work
    # def backward(self, grad_output):
    #     something = np.multiply((-np.exp(-self.decay_constant)),grad_output.clone())
    #     return grad_output