import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import mlflow
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from datasets import load_dataset
from PIL import Image
from itertools import islice

# MLflow setup
EXPERIMENT_NAME = "Colorizer_Experiment"

def setup_mlflow():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id

# Data ingestion
class ColorizeIterableDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __iter__(self):
        for item in self.dataset:
            try:
                img = item['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = self.transform(img)
                
                lab = rgb2lab(img.permute(1, 2, 0).numpy())
                l_chan = lab[:, :, 0]
                l_chan = (l_chan - 50) / 50
                ab_chan = lab[:, :, 1:]
                ab_chan = ab_chan / 128
                
                yield torch.Tensor(l_chan).unsqueeze(0), torch.Tensor(ab_chan).permute(2, 0, 1)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                continue

def create_dataloaders(batch_size=32):
    try:
        print("Loading ImageNet dataset in streaming mode...")
        dataset = load_dataset("imagenet-1k", split="train", streaming=True)
        print("Dataset loaded in streaming mode.")
        
        print("Creating custom dataset...")
        train_dataset = ColorizeIterableDataset(dataset)
        print("Custom dataset created.")
        
        print("Creating dataloader...")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
        print("Dataloader created.")
        
        return train_dataloader
    except Exception as e:
        print(f"Error in create_dataloaders: {str(e)}")
        return None

# Model definition
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, bn=True, dropout=False):
        super(UNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False) if down \
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.dropout = nn.Dropout(0.5) if dropout else None
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return nn.ReLU()(x) if self.down else nn.ReLU(inplace=True)(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = UNetBlock(1, 64, bn=False)
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512)
        self.down5 = UNetBlock(512, 512)
        self.down6 = UNetBlock(512, 512)
        self.down7 = UNetBlock(512, 512)
        self.down8 = UNetBlock(512, 512, bn=False)
        
        self.up1 = UNetBlock(512, 512, down=False, dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, dropout=True)
        self.up3 = UNetBlock(1024, 512, down=False, dropout=True)
        self.up4 = UNetBlock(1024, 512, down=False)
        self.up5 = UNetBlock(1024, 256, down=False)
        self.up6 = UNetBlock(512, 128, down=False)
        self.up7 = UNetBlock(256, 64, down=False)
        self.up8 = nn.ConvTranspose2d(128, 2, 4, 2, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return torch.tanh(self.up8(torch.cat([u7, d1], 1)))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

def init_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# Training utilities
def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 128.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def visualize_results(epoch, generator, train_loader, device):
    generator.eval()
    with torch.no_grad():
        for inputs, real_AB in train_loader:
            inputs, real_AB = inputs.to(device), real_AB.to(device)
            fake_AB = generator(inputs)
            
            fake_rgb = lab_to_rgb(inputs.cpu(), fake_AB.cpu())
            real_rgb = lab_to_rgb(inputs.cpu(), real_AB.cpu())
            
            img_grid = make_grid(torch.from_numpy(np.concatenate([real_rgb, fake_rgb], axis=3)).permute(0, 3, 1, 2), normalize=True, nrow=4)
            
            plt.figure(figsize=(15, 15))
            plt.imshow(img_grid.permute(1, 2, 0).cpu())
            plt.axis('off')
            plt.title(f'Epoch {epoch}')
            plt.savefig(f'results/epoch_{epoch}.png')
            mlflow.log_artifact(f'results/epoch_{epoch}.png')
            plt.close()
            break
    generator.train()

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    mlflow.log_artifact(filename)

def load_checkpoint(filename, generator, discriminator, optimizerG, optimizerD):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0

# Training function
def train(generator, discriminator, train_loader, num_epochs, device, lr=0.0002, beta1=0.5):
    criterion = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth.tar")
    start_epoch = load_checkpoint(checkpoint_path, generator, discriminator, optimizerG, optimizerD)

    experiment_id = setup_mlflow()
    with mlflow.start_run(experiment_id=experiment_id, run_name="training_run") as run:
        try:
            for epoch in range(start_epoch, num_epochs):
                generator.train()
                discriminator.train()

                num_iterations = 1000
                pbar = tqdm(enumerate(islice(train_loader, num_iterations)), total=num_iterations, desc=f"Epoch {epoch+1}/{num_epochs}")
                
                for i, (real_L, real_AB) in pbar:
                    real_L, real_AB = real_L.to(device), real_AB.to(device)
                    batch_size = real_L.size(0)

                    # Train Discriminator
                    optimizerD.zero_grad()

                    fake_AB = generator(real_L)
                    fake_LAB = torch.cat([real_L, fake_AB], dim=1)
                    real_LAB = torch.cat([real_L, real_AB], dim=1)

                    pred_fake = discriminator(fake_LAB.detach())
                    loss_D_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

                    pred_real = discriminator(real_LAB)
                    loss_D_real = criterion(pred_real, torch.ones_like(pred_real))

                    loss_D = (loss_D_fake + loss_D_real) * 0.5
                    loss_D.backward()
                    optimizerD.step()

                    # Train Generator
                    optimizerG.zero_grad()

                    fake_AB = generator(real_L)
                    fake_LAB = torch.cat([real_L, fake_AB], dim=1)
                    pred_fake = discriminator(fake_LAB)

                    loss_G_GAN = criterion(pred_fake, torch.ones_like(pred_fake))
                    loss_G_L1 = l1_loss(fake_AB, real_AB) * 100  # L1 loss weight

                    loss_G = loss_G_GAN + loss_G_L1
                    loss_G.backward()
                    optimizerG.step()

                    pbar.set_postfix({
                        'D_loss': loss_D.item(),
                        'G_loss': loss_G.item(),
                        'G_L1': loss_G_L1.item()
                    })

                    mlflow.log_metrics({
                        "D_loss": loss_D.item(),
                        "G_loss": loss_G.item(),
                        "G_L1_loss": loss_G_L1.item()
                    }, step=epoch * num_iterations + i)

                visualize_results(epoch, generator, train_loader, device)

                checkpoint = {
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                }
                save_checkpoint(checkpoint, filename=checkpoint_path)

            print("Training completed successfully.")
            
            mlflow.pytorch.log_model(generator, "generator_model")
            model_uri = f"runs:/{run.info.run_id}/generator_model"
            mlflow.register_model(model_uri, "colorizer_generator")
            
            return run.info.run_id
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            mlflow.log_param("error", str(e))
            return None

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        batch_size = 32
        num_epochs = 50

        train_loader = create_dataloaders(batch_size=batch_size)
        if train_loader is None:
            raise Exception("Failed to create dataloader")

        generator = Generator().to(device)
        discriminator = Discriminator().to(device)

        generator.apply(init_weights)
        discriminator.apply(init_weights)

        run_id = train(generator, discriminator, train_loader, num_epochs=num_epochs, device=device)
        if run_id:
            print(f"Training completed successfully. Run ID: {run_id}")
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)
        else:
            print("Training failed.")
    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")