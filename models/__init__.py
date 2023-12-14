from models.vae import VAE

def args_for_model(parser, model):
    parser.add_argument('--something', type=str, default="?", help="?")
