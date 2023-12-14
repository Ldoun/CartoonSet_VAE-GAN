import os
import sys
import torch
import numpy as np
from tqdm import tqdm

from torchvision.utils import save_image

class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, device, patience, epochs, result_path, fold_logger, len_train, len_valid):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.generator, self.discriminator = model
        self.loss_fn = loss_fn
        self.optimizer_G, self.optimizer_D = optimizer
        self.device = device
        self.patience = patience
        self.epochs = epochs
        self.logger = fold_logger
        self.path = result_path
        self.len_train = len_train
        self.len_valid = len_valid
    
    def train(self):
        #best = np.inf
        for epoch in range(1,self.epochs+1):
            loss_G_train, loss_D_train = self.train_step()
            loss_G_val, loss_D_val = self.valid_step()

            self.logger.info(f'Epoch {str(epoch).zfill(5)}: GT_loss:{loss_G_train:.3f} DT_loss:{loss_D_train:.3f} GV_loss:{loss_G_val:.3f} DV_loss:{loss_D_val:.3f}')
            torch.save(self.generator.state_dict(), os.path.join(self.path, f'{epoch}.pt'))

            # Disable Early Stop
            # if loss_G_val < best:
            #     best = loss_G_val
            #     torch.save(self.generator.state_dict(), self.best_model_path)
            #     bad_counter = 0

            # else:
            #     bad_counter += 1

            # if bad_counter == self.patience:
            #     break

    def train_step(self):
        self.generator.train()
        self.discriminator.train()

        total_G_loss, total_D_loss = 0, 0
        i = 0
        for batch in tqdm(self.train_loader, file=sys.stdout): #tqdm output will not be written to logger file(will only written to stdout)
            i+=1
            batch = batch.to(self.device)
            valid = torch.ones((batch.shape[0], 1), requires_grad=False, dtype=torch.float, device=self.device)
            fake = torch.zeros((batch.shape[0], 1), requires_grad=False, dtype=torch.float, device=self.device)

            self.optimizer_G.zero_grad()
            z = torch.randn((batch.shape[0], 512), device=self.device)
            gen_imgs = self.generator(z)
            g_loss = self.loss_fn(self.discriminator(gen_imgs), valid)
            g_loss.backward()
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            real_loss = self.loss_fn(self.discriminator(batch), valid)
            fake_loss = self.loss_fn(self.discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            self.optimizer_D.step()

            total_G_loss += g_loss.item() * batch.shape[0]
            total_D_loss += d_loss.item() * batch.shape[0]

            if i % 100 == 0:
                save_image(gen_imgs.data[:25], os.path.join(self.path, f"{i}.png"), nrow=5, normalize=True)
        
        return total_G_loss/self.len_train, total_D_loss/self.len_train
    
    def valid_step(self):
        self.generator.eval()
        self.discriminator.eval()
        with torch.no_grad():
            total_G_loss, total_D_loss = 0, 0
            for batch in self.valid_loader:
                batch = batch.to(self.device)
                valid = torch.ones((batch.shape[0], 1), requires_grad=False, dtype=torch.float, device=self.device)
                fake = torch.zeros((batch.shape[0], 1), requires_grad=False, dtype=torch.float, device=self.device)
                z = torch.randn((batch.shape[0], 512), device=self.device)
                gen_imgs = self.generator(z)
                g_loss = self.loss_fn(self.discriminator(gen_imgs), valid)

                real_loss = self.loss_fn(self.discriminator(batch), valid)
                fake_loss = self.loss_fn(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                total_G_loss += g_loss.item() * batch.shape[0]
                total_D_loss += d_loss.item() * batch.shape[0]
                
        return total_G_loss/self.len_valid, total_D_loss/self.len_valid

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