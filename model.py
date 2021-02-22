import os
from copy import deepcopy

import torch
import numpy as np
import torch.nn as nn
from PIL import Image

# dirty hack so imports work within the TransGAN repo
import sys
sys.path.insert(0, './TransGAN')

from functions import load_params, copy_params, cur_stages
from utils.utils import set_log_dir, create_logger
from models.Celeba64_TransGAN import Generator


torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True


class TransGAN:
    def __init__(self, args=None):
        self.args = args
        self.epoch = 300
        self.gen_net = self.setup_generator(args)

    def setup_generator(self, args):
        """
        Code setting up the network.
        """

        # import the network
        gen_net = Generator(args=args).cuda()
        gen_net.set_arch(args.arch, cur_stage=2)

        # initialize the weights
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                if args.init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif args.init_type == 'orth':
                    nn.init.orthogonal_(m.weight.data)
                elif args.init_type == 'xavier_uniform':
                    nn.init.xavier_uniform(m.weight.data, 1.)
                else:
                    raise NotImplementedError('{} unknown inital type'.format(args.init_type))
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        gen_net.apply(weights_init)

        # configure model to run on availible GPUs
        gpu_ids = [i for i in range(int(torch.cuda.device_count()))]
        gen_net = torch.nn.DataParallel(gen_net.to("cuda:0"), device_ids=gpu_ids)

        gen_net.module.cur_stage = 0
        gen_net.module.alpha = 1.

        # initial params
        gen_avg_param = copy_params(gen_net)

        # load checkpoint
        if args.load_path:
            assert os.path.exists(args.load_path)
            checkpoint_file = os.path.join(args.load_path)
            assert os.path.exists(checkpoint_file)
            checkpoint = torch.load(checkpoint_file)
            start_epoch = checkpoint['epoch']
            gen_net.load_state_dict(checkpoint['gen_state_dict'])
            avg_gen_net = deepcopy(gen_net)
            avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
            gen_avg_param = copy_params(avg_gen_net)
            del avg_gen_net
            cur_stage = cur_stages(start_epoch, args)
            gen_net.module.cur_stage = cur_stage
            gen_net.module.alpha = 1.
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

        logger.info(args)

        load_params(gen_net, gen_avg_param)
        return gen_net

    @staticmethod
    def postprocess(t):
        """
        Postprocessing that transforms model weights into an image.
        """
        t = t.cpu().detach().numpy()[0]
        t = np.maximum(np.minimum(t, 1.0), -1.0)
        t = (((t + 1) / 2) * 255).astype(np.uint8)
        t = np.transpose(t, (1, 2, 0))
        return Image.fromarray(t)

    def __call__(self):
        """
        Actual method for generating images based on a random seed.
        """
        with torch.no_grad():
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, self.args.latent_dim)))
            tensor = self.gen_net(z, self.epoch)
            return self.postprocess(tensor)