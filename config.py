from dataclasses import dataclass


@dataclass
class ModelParams:
    D_downsample = 'avg'
    arch = None
    baseline_decay = 0.9
    beta1 = 0.0
    beta2 = 0.99
    bottom_width = 8
    channels = 3
    controller = 'controller'
    ctrl_lr = 0.00035
    ctrl_sample_batch = 1
    ctrl_step = 30
    d_depth = 7
    d_lr = 0.0001
    d_spectral_norm = True
    data_path = './data'
    dataset = 'stl10'
    df_dim = 384
    diff_aug = 'translation,cutout,color'
    dis_batch_size = 16
    dis_model = 'ViT_8_8'
    dynamic_reset_threshold = 0.001
    dynamic_reset_window = 500
    entropy_coeff = 0.001
    eval_batch_size = 25
    exp_name = 'celeba64_test'
    fade_in = 0.0
    fid_stat = None
    g_depth = 5
    g_lr = 0.0001
    g_spectral_norm = False
    gen_batch_size = 32
    gen_model = 'Celeba64_TransGAN'
    gf_dim = 1024
    grow_step1 = 25
    grow_step2 = 55
    grow_steps = [0, 0]
    hid_size = 100
    img_size = 64
    init_type = 'xavier_uniform'
    latent_dim = 1024
    load_path = './pretrained_weight/celeba64_checkpoint.pth'
    loss = 'wgangp-eps'
    lr_decay = False
    max_epoch = 300
    max_iter = 500000
    max_search_iter = 90
    n_classes = 0
    n_critic = 5
    noise_injection = False
    num_candidate = 10
    num_eval_imgs = 50000
    num_workers = 36
    optimizer = 'adam'
    patch_size = 4
    path_helper = {'prefix': 'logs/celeba64_test_2021_02_23_13_42_44',
                   'ckpt_path': 'logs/celeba64_test_2021_02_23_13_42_44/Model',
                   'log_path': 'logs/celeba64_test_2021_02_23_13_42_44/Log',
                   'sample_path': 'logs/celeba64_test_2021_02_23_13_42_44/Samples'}
    phi = 1.0
    print_freq = 50
    random_seed = 12345
    rl_num_eval_img = 5000
    shared_epoch = 15
    topk = 5
    val_freq = 1
    wd = 0.001
