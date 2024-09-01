import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import mlflow
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from skimage.color import lab2rgb
import argparse
from itertools import islice

from data_ingestion import ColorizeIterableDataset, create_dataloaders
from model import Generator, Discriminator, init_weights

EXPERIMENT_NAME = "Colorizer_Experiment"

def setup_mlflow():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id

def lab_to_rgb(L, ab):
    """Convert L and ab channels to RGB image"""
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
            break  # Only visualize one batch
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

                # Use a fixed number of iterations per epoch
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
            
            # Log the generator model
            mlflow.pytorch.log_model(generator, "generator_model")
            
            # Register the model
            model_uri = f"runs:/{run.info.run_id}/generator_model"
            mlflow.register_model(model_uri, "colorizer_generator")
            
            return run.info.run_id
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            mlflow.log_param("error", str(e))
            return None

def test_training(generator, discriminator, train_loader, device):
    print("Testing training process...")
    try:
        train(generator, discriminator, train_loader, num_epochs=1, device=device)
        print("Training process test passed.")
        return True
    except Exception as e:
        print(f"Training process test failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Colorizer model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    try:
        train_loader = create_dataloaders(batch_size=args.batch_size)

        generator = Generator().to(device)
        discriminator = Discriminator().to(device)

        generator.apply(init_weights)
        discriminator.apply(init_weights)

        if args.test:
            if test_training(generator, discriminator, train_loader, device):
                print("All tests passed.")
            else:
                print("Tests failed.")
        else:
            run_id = train(generator, discriminator, train_loader, num_epochs=args.num_epochs, device=device)
            if run_id:
                print(f"Training completed. Run ID: {run_id}")
                # Save the run ID to a file for easy access by the inference script
                with open("latest_run_id.txt", "w") as f:
                    f.write(run_id)
            else:
                print("Training failed.")

    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")