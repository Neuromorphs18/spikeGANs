import os
from PIL import Image

import torch

from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import time_surface_rgb, rgb_to_tensor


class TrainedGAN():
    def __init__(self, dataroot, weight_file, batch_size):
        opt = TrainOptions(dataroot, weight_file, batch_size)
        opt = opt.parse()
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.display_id = -1  # no visdom display
        opt.dataroot = weight_file
        opt.model = "pix2pix"
        model = create_model(opt)
        print(torch.__version__)
        model.setup(opt)
        self.model = model
        self.opt = opt
        # create website
        img_root = os.path.join(dataroot, "generator")
        imgs = os.listdir(img_root)
        imgs.sort(key=lambda f: int(f.split(".")[0]))

        self.current_img = rgb_to_tensor(
                Image.open(os.path.join(img_root, imgs[0])).convert('RGB')
            )


        self.web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
        self.webpage = html.HTML(self.web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


    def generate_img(self, data_folder):
        # test
        self.opt.dataroot = data_folder
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()

        d_loss = []
        g_loss = []

        for i, data in enumerate(dataset):
            #if i >= self.opt.how_many:
            #    break
            self.model.set_input(data)
            self.model.no_optimisation_run_through()
            d = self.model.get_D_loss()
            g = self.model.get_G_loss()

            print(d, g)

            d_loss.append(d)
            g_loss.append(g)

            #TODO get loss and return it
            visuals = self.model.get_current_visuals()
            img_path = self.model.get_image_paths()
            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(self.webpage, visuals, img_path, aspect_ratio=self.opt.aspect_ratio, width=self.opt.display_winsize)

        self.webpage.save()

        return g_loss, d_loss


