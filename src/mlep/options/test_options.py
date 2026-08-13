from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument('--dataroot')
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        # Test-time corruption sweep (mirrors mlep.harness.data.EVAL_SCENARIOS):
        # every listed scenario is evaluated on the SAME images so you can read
        # off how much blur / JPEG compression hurts the pretrained model.
        parser.add_argument('--corruptions', default='clean,blur,jpeg',
                            help="comma list of test-time corruption scenarios to "
                                 "run: any of clean, blur, jpeg")
        parser.add_argument('--out', default='test_results.txt',
                            help='text file to save the per-model results table to')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')

        self.isTrain = False
        return parser
