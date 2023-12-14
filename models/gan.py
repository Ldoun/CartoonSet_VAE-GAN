import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, act):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), stride=1), getattr(nn, act)(),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, (3, 3), stride=2), getattr(nn, act)(),
            nn.Conv2d(64, 64, (3, 3), stride=1), getattr(nn, act)(),
            nn.MaxPool2d((3, 3), stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=1), getattr(nn, act)(),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, (3, 3), stride=2), getattr(nn, act)(),
            nn.Conv2d(256, 256, (3, 3), stride=1), getattr(nn, act)(),
            nn.MaxPool2d((3, 3), stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, (3, 3), stride=1), getattr(nn, act)(),
            nn.Dropout(0.25),
            nn.Conv2d(256, 512, (3, 3), stride=1), getattr(nn, act)(),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 1)
        )
        
    def forward(self, x,):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 512)
        return self.fc(x)
    
class Generator(nn.Module):
    def __init__(self, act):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            getattr(nn, act)(),
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, (3, 3), stride=2), getattr(nn, act)(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(512, 256, (3, 3), stride=1), getattr(nn, act)(),
            nn.ConvTranspose2d(256, 256, (3, 3), stride=2, padding=1), getattr(nn, act)(),
            nn.Dropout(0.25),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3), stride=2, padding=1), getattr(nn, act)(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(128, 64, (3, 3), stride=1), getattr(nn, act)(),
            nn.ConvTranspose2d(64, 64, (3, 3), stride=1), getattr(nn, act)(),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2, output_padding=1), getattr(nn, act)(),
            nn.ConvTranspose2d(32, 3, (3, 3), stride=1), nn.Tanh(),
        )
        
    def forward(self, x,):
        x = self.fc(x)
        x = x.view(-1, 512, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x