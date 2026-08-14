from fvcore.common.config import CfgNode

# Define config
cfg = CfgNode()

# Data
cfg.data = CfgNode()
cfg.data.data_root = "data"
cfg.data.patch_size = 256
cfg.data.use_augmentation = True
cfg.data.num_workers = 4
cfg.data.pin_memory = True
cfg.data.lq_dir_name = "LQ"
cfg.data.gt_dir_name = "GT"
cfg.data.val_ratio = 0.1
cfg.data.val_split_seed = 42

# Model
cfg.model = CfgNode()
cfg.model.arch_name = "NAFNet"
# NAFNet
cfg.model.nafnet = CfgNode()
cfg.model.nafnet.width = 32
cfg.model.nafnet.num_enc_blks = (1, 1, 1, 28)
cfg.model.nafnet.middle_blk_num = 1
cfg.model.nafnet.num_dec_blks = (1, 1, 1, 1)
cfg.model.nafnet.dropout_rate = 0.0
# Restormer
cfg.model.restormer = CfgNode()
cfg.model.restormer.inp_channels = 3
cfg.model.restormer.out_channels = 3
cfg.model.restormer.embed_dim = 48


# Loss
cfg.loss = CfgNode()
cfg.loss.losses = [("mse", 1.0)]

# Training
cfg.train = CfgNode()
cfg.train.output_dir = "results/default"
cfg.train.batch_size = 16
cfg.train.epochs = 100
cfg.train.lr = 1e-3
cfg.train.lr_min = 1e-6
cfg.train.weight_decay = 1e-3
cfg.train.seed = 42
cfg.train.amp = True

# Logging
cfg.logging = CfgNode()
cfg.logging.log_dir = "results/mlruns"
cfg.logging.experiment_name = "image_restoration_allinone"
cfg.logging.log_img_limit = 4


def get_default_cfg() -> CfgNode:
    """Return a copy of the default config."""
    return cfg
