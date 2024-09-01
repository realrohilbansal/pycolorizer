import os
import mlflow
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from skimage.color import rgb2lab
from PIL import Image
import numpy as np


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
                
                # Convert to LAB color space
                lab = rgb2lab(img.permute(1, 2, 0).numpy())
                
                # Normalize L channel to range [-1, 1]
                l_chan = lab[:, :, 0]
                l_chan = (l_chan - 50) / 50
                
                # Normalize AB channels to range [-1, 1]
                ab_chan = lab[:, :, 1:]
                ab_chan = ab_chan / 128
                
                yield torch.Tensor(l_chan).unsqueeze(0), torch.Tensor(ab_chan).permute(2, 0, 1)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                continue

def create_dataloaders(batch_size=32):
    try:
        print("Loading ImageNet dataset in streaming mode...")
        # Load ImageNet dataset from Hugging Face in streaming mode
        dataset = load_dataset("imagenet-1k", split="train", streaming=True)
        print("Dataset loaded in streaming mode.")
        
        print("Creating custom dataset...")
        # Create custom dataset
        train_dataset = ColorizeIterableDataset(dataset)
        print("Custom dataset created.")
        
        print("Creating dataloader...")
        # Create dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
        print("Dataloader created.")
        
        return train_dataloader
    except Exception as e:
        print(f"Error in create_dataloaders: {str(e)}")
        return None

def test_data_ingestion():
    print("Testing data ingestion...")
    try:
        dataloader = create_dataloaders(batch_size=4)
        if dataloader is None:
            raise Exception("Dataloader creation failed")
        
        # Get the first batch
        for sample_batch in dataloader:
            if len(sample_batch) != 2:
                raise Exception(f"Unexpected batch format: {len(sample_batch)} elements instead of 2")
            
            l_chan, ab_chan = sample_batch
            if l_chan.shape != torch.Size([4, 1, 256, 256]) or ab_chan.shape != torch.Size([4, 2, 256, 256]):
                raise Exception(f"Unexpected tensor shapes: L={l_chan.shape}, AB={ab_chan.shape}")
            
            print("Data ingestion test passed.")
            return True
    except Exception as e:
        print(f"Data ingestion test failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        print("Starting data ingestion pipeline...")
        mlflow.start_run(run_name="data_ingestion")
        
        try:
            # Log parameters
            print("Logging parameters...")
            mlflow.log_param("batch_size", 32)
            mlflow.log_param("dataset", "imagenet-1k")
            print("Parameters logged.")
            
            # Create dataloaders
            print("Creating dataloaders...")
            train_dataloader = create_dataloaders(batch_size=32)
            if train_dataloader is None:
                raise Exception("Failed to create dataloader")
            print("Dataloaders created successfully.")
            
            # Log a sample batch
            print("Logging sample batch...")
            for sample_batch in train_dataloader:
                l_chan, ab_chan = sample_batch
                
                # Log sample input (L channel)
                sample_input = l_chan[0].numpy()
                mlflow.log_image(sample_input, "sample_input_l_channel.png")
                
                # Log sample target (AB channels)
                sample_target = ab_chan[0].permute(1, 2, 0).numpy()
                mlflow.log_image(sample_target, "sample_target_ab_channels.png")
                
                print("Sample batch logged.")
                break  # We only need one batch for logging
            
            print("Data ingestion pipeline completed successfully.")
        
        except Exception as e:
            print(f"Error in data ingestion pipeline: {str(e)}")
            mlflow.log_param("error", str(e))
        
        finally:
            mlflow.end_run()
    
    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")

    if test_data_ingestion():
        print("All tests passed.")
    else:
        print("Tests failed.")