import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from PIL import Image
import os

class Batch_loader:
    def __init__(self, filename):
        print('Loading...')
        self.xypol, self.img_ids = self.prep_data(filename)
        print('Done!')
        self.unique_ids = np.unique(self.img_ids)
        self.position = self.unique_ids[0]

    def prep_data(self, filename):
        data = np.loadtxt(filename, skiprows=1, delimiter=",")

        # remove first image as this will be the baseline
        data = data[np.where(data[:, 2] != 1)]

        # frame_now, event_id, prev_frame, x, y, polarity, timestamp
        x = data[:, np.array([3, 4, 5, 6])]  # x, y, pol, time

        # take time difference
        xypol = x[:, np.array([0, 1, 2, 3])].astype(np.int16)
        xypol[:, 2] = (xypol[:, 2] * 2) - 1

        times = x[:, 3]
        time_d = times[1:] - times[:-1]
        time_d = np.concatenate([np.zeros(1), time_d])
        xypol[:, 3] = time_d

        return xypol, data[:, 2].astype(np.int)

    def ConvertTDeventsToAccumulatedFrameConvert(self, td_events, width=346, height=260, tau=1e6):
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

    def get_batch(self, batch_size=10):
        cmap = plt.cm.jet
        tgts = []
        in_data = []
        for i in self.unique_ids[self.position:self.position + batch_size]:
            data = self.xypol[np.where(self.img_ids == i)]
            state, img = self.ConvertTDeventsToAccumulatedFrameConvert(data)
            img = np.fliplr(np.flipud(img))
            img = np.array(Image.fromarray((255 * img)).resize((256, 256))) / 255

            normed_data = (img - np.min(img)) / (np.max(img) - np.min(img))

            mapped_data = cmap(normed_data)
            mapped_data = mapped_data[:, :, :3]
            mapped_data = np.moveaxis(mapped_data, -1, 0)

            in_data.append(mapped_data)
            target_img = Image.open(os.path.join(opt.dataroot, "targets", "{}.png".format(i))).resize((256,256))
            target_img = np.moveaxis((np.array(target_img) /255)[:, :, :3], -1, 0)
            tgts.append(target_img)

        self.position += batch_size
        if self.position >= len(self):
            self.position = 0

        in_data = np.array(in_data)

        return in_data, tgts

    def __len__(self):
        return len(self.unique_ids)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    # data_loader = CreateDataLoader(opt)
    # dataset = data_loader.load_data()
    # dataset_size = len(data_loader)
    # print('#training images = %d' % dataset_size)
    data_loader = Batch_loader(opt.spikes_file)
    dataset_size = len(data_loader)
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i in range(len(data_loader)//opt.batchSize + 1):
            print('Proccessing batch {} of epoch {}: '.format(i, epoch))
            spikes, targets = data_loader.get_batch(batch_size=opt.batchSize)
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(spikes, targets)
            model.optimize_parameters()
            # model.no_optimisation_run_through()

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
              (epoch, opt.niter, time.time() - epoch_start_time))
        model.update_learning_rate()
