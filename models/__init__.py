from models.vae import VAE

def args_for_model(parser, model):
    parser.add_argument('--act', type=str, default="ELU", help="activate function in VAE")
