import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html



class TrainedGAN():
    def __init__(self, dataroot, weight_file):
        opt = TestOptions(dataroot, weight_file)
        opt = opt.parse()
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.display_id = -1  # no visdom display
        opt.dataroot = weight_file

        self.opt = opt
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        # create website
        self.web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
        self.webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
        # test


    def generate(self, data):
        opt.dataset = data
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()

        for i, data in enumerate(dataset):
            if i >= self.opt.how_many:
                break
            model.set_input(data)
            model.test()
            #TODO get loss and return it
            visuals = self.model.get_current_visuals()
            img_path = self.model.get_image_paths()
            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(self.webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        self.webpage.save()


