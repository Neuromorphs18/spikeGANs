from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self, dataroot, weight_file):
        super(TestOptions, self).__init__()
        self.weight_file = weight_file
        self.dataroot = dataroot

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        #MR edit
        parser.add_argument('--dataroot', type=str, default=self.dataroot, help='loc of weight files')
        parser.add_argument('--name', type=str, default=self.weight_file, help='loc of weight files')

        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))

        self.isTrain = False
        return parser
