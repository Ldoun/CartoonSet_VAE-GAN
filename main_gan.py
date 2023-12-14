import os
import sys
import logging
from functools import partial
from sklearn.model_selection import train_test_split

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset

from models import Discriminator, Generator
from data import DataSet
from trainer_gan import Trainer
from config import get_args
from lr_scheduler import get_sch
from utils import seed_everything, handle_unhandled_exception, save_to_json

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    result_path = os.path.join(args.result_path, 'GAN' +'_'+str(len(os.listdir(args.result_path))))
    os.makedirs(result_path)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    save_to_json(vars(args), os.path.join(result_path, 'config.json'))
    sys.excepthook = partial(handle_unhandled_exception,logger=logger)
    
    fold=0
    dataset = DataSet(base_path=args.path, is_for_gan=True)
    train_index, valid_index = train_test_split(range(len(dataset)), test_size=0.2)

    logger.info(f'start training of {fold+1}-fold')

    train_dataset = Subset(dataset, train_index)
    valid_dataset = Subset(dataset, valid_index)

    G_model = Generator(args.act).to(device)
    D_model = Discriminator(args.act).to(device)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer_G = torch.optim.Adam(G_model.parameters(), lr=args.lr)
    optimizer_D = torch.optim.SGD(D_model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, #pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, #pin_memory=True
    )
    
    trainer = Trainer(
        train_loader, valid_loader, [G_model, D_model], loss_fn, [optimizer_G, optimizer_D], device, args.patience, args.epochs, result_path, logger, len(train_dataset), len(valid_dataset))
    trainer.train() #start training