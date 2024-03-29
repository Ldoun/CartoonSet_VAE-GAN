import os
import sys
import torch
import numpy as np
from tqdm import tqdm

from torchvision.utils import save_image

class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, scheduler, device, patience, epochs, result_path, fold_logger, len_train, len_valid):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.epochs = epochs
        self.logger = fold_logger
        self.path = result_path
        self.best_model_path = os.path.join(result_path, 'best_model.pt')
        self.len_train = len_train
        self.len_valid = len_valid
    
    def train(self):
        best = np.inf
        for epoch in range(1,self.epochs+1):
            self.cur_epoch = epoch
            loss_train = self.train_step()
            loss_val = self.valid_step()
            self.scheduler.step()

            self.logger.info(f'Epoch {str(epoch).zfill(5)}: t_loss:{loss_train:.3f} v_loss:{loss_val:.3f}')

            if loss_val < best:
                best = loss_val
                torch.save(self.model.state_dict(), self.best_model_path)
                bad_counter = 0

            else:
                bad_counter += 1

            if bad_counter == self.patience:
                break

    def train_step(self):
        self.model.train()

        total_loss = 0
        for batch in tqdm(self.train_loader, file=sys.stdout): #tqdm output will not be written to logger file(will only written to stdout)
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            output, mean, logvar = self.model(batch)            
            loss = self.loss_fn(output, batch, mean, logvar)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        return total_loss/self.len_train
    
    def valid_step(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in self.valid_loader:
                batch = batch.to(self.device)
                output, mean, logvar = self.model(batch)            
                loss = self.loss_fn(output, batch, mean, logvar)

                total_loss += loss.item()
        with torch.no_grad():
            gen_imgs = self.model.generate_sample(torch.randn(25, 512, device='cuda'))
            save_image(gen_imgs.data, os.path.join(self.path, f"{self.cur_epoch}epoch.png"), nrow=5, normalize=True)
                
        return total_loss/self.len_valid

    # def test(self, test_loader):
    #     self.model.load_state_dict(torch.load(self.best_model_path))
    #     self.model.eval()
    #     with torch.no_grad():
    #         result = []
    #         for batch in test_loader:
    #             x = batch['data'].to(self.device)
    #             output, _, _ = self.model(x).detach().cpu().numpy()
    #             result.append(output)

    #     return np.concatenate(result,axis=0)