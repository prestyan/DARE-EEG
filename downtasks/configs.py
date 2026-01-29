import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--mode", type=str, default="single")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--save_ckpt', default=True)
    parser.add_argument('--update_freq', default=1, type=int)


    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
            weight decay. We use a cosine schedule for WD and using a larger decay by
            the end of training improves performance for ViTs.""")
    parser.add_argument("--distributed", default=True)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    # Data paths
    parser.add_argument('--dataset', default='TUAB', type=str)
    parser.add_argument("--data_path_1", type=str, default="E:/data/EEG/Sub_S1_single/Sub_S1_")
    parser.add_argument("--data_path_2", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default='./logs/')

    # Model save name
    parser.add_argument("--output_dir", type=str, default='./logs/checkpoints')
    parser.add_argument("--model_save_name", type=str, default="checkpoint-ours_TUEV_")

    # ConvHead parameters
    parser.add_argument("--in_channels", type=int, default=23) # recommand to use in command line
    parser.add_argument("--conv_out_channels", type=int, default=58) # recommand to use in command line
    parser.add_argument("--in_time_length", type=int, default=2000) # recommand to use in command line
    parser.add_argument("--out_time_length", type=int, default=2000)
    parser.add_argument("--conv_layers", type=int, default=2)

    # Encoder parameters
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--freeze_encoder", action="store_true") 

    # MLP parameters
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--embed_num", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=1) # recommand to use in command line
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128]) 
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pooling_method", type=str, default="mean")
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    known_args, _ = parser.parse_known_args()
    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            import traceback
            traceback.print_exc()
            print("DeepSpeed import failed. Either disable --enable_deepspeed or install a compatible deepspeed version.")
            exit(1)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

tag = "deep"
variant = "D"

BASE_MODELS_CONFIGS = {
    "tiny1": {
        "embed_dim":64, "embed_num":1, "depth":[2,2,4], "num_heads":4},
    "tiny2": {
        "embed_dim":64, "embed_num":4, "depth":[2,2,4], "num_heads":4},
    "tiny3": {
        "embed_dim":64, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "little": {
        "embed_dim":128, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "base1": {
        "embed_dim":256, "embed_num":1, "depth":[6,6,6], "num_heads":4},
    "base2": {
        "embed_dim":256, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "base3": {
        "embed_dim":512, "embed_num":1, "depth":[6,6,6], "num_heads":8},
    "deep": {
        "embed_dim":512, "embed_num":4, "depth":[8,8,8], "num_heads":8},
}

def get_base_config(embed_dim=512, embed_num=4, depth=[8,8,8], num_heads=4):
    
    models_configs = {
            'encoder': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'depth': depth[0],
                    'num_heads': num_heads,
                },
            'predictor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'predictor_embed_dim': embed_dim,
                    'depth': depth[1],
                    'num_heads': num_heads,
                },
            'reconstructor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'reconstructor_embed_dim': embed_dim,
                    'depth': depth[2],
                    'num_heads': num_heads,
                },
    }
    return models_configs