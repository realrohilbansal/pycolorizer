import argparse
import os
import torch
import mlflow
from data_ingestion import create_dataloaders, test_data_ingestion
from model import Generator, Discriminator, init_weights, test_models
from train import train, test_training
from app import setup_gradio_app

EXPERIMENT_NAME = "Colorizer_Experiment"

def setup_mlflow():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id

def run_pipeline(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    experiment_id = setup_mlflow()

    if args.ingest_data or args.run_all:
        print("Starting data ingestion...")
        train_loader = create_dataloaders(batch_size=args.batch_size)
        if train_loader is None:
            print("Data ingestion failed.")
            return
    else:
        train_loader = None

    if args.create_model or args.train or args.run_all:
        print("Creating and testing models...")
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
        generator.apply(init_weights)
        discriminator.apply(init_weights)
        if not test_models():
            print("Model creation or testing failed.")
            return
    else:
        generator = None
        discriminator = None

    if args.train or args.run_all:
        print("Starting model training...")
        if train_loader is None:
            print("Creating dataloader for training...")
            train_loader = create_dataloaders(batch_size=args.batch_size)
            if train_loader is None:
                print("Failed to create dataloader for training.")
                return
        if generator is None or discriminator is None:
            print("Creating models for training...")
            generator = Generator().to(device)
            discriminator = Discriminator().to(device)
            generator.apply(init_weights)
            discriminator.apply(init_weights)
        run_id = train(generator, discriminator, train_loader, num_epochs=args.num_epochs, device=device)
        if run_id:
            print(f"Training completed. Run ID: {run_id}")
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)
        else:
            print("Training failed.")
            return

    if args.test_training:
        print("Testing training process...")
        if train_loader is None:
            print("Creating dataloader for testing...")
            train_loader = create_dataloaders(batch_size=args.batch_size)
            if train_loader is None:
                print("Failed to create dataloader for testing.")
                return
        if generator is None or discriminator is None:
            print("Creating models for testing...")
            generator = Generator().to(device)
            discriminator = Discriminator().to(device)
            generator.apply(init_weights)
            discriminator.apply(init_weights)
        if test_training(generator, discriminator, train_loader, device):
            print("Training process test passed.")
        else:
            print("Training process test failed.")

    if args.serve or args.run_all:
        print("Setting up Gradio app for serving...")
        if not args.run_id:
            try:
                with open("latest_run_id.txt", "r") as f:
                    args.run_id = f.read().strip()
            except FileNotFoundError:
                print("No run ID provided and couldn't find latest_run_id.txt")
                return
        iface = setup_gradio_app(args.run_id, device)
        iface.launch(share=args.share)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Colorizer Pipeline")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--run_id", type=str, help="MLflow run ID of the trained model for inference")
    parser.add_argument("--ingest_data", action="store_true", help="Run data ingestion")
    parser.add_argument("--create_model", action="store_true", help="Create and test the model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test_training", action="store_true", help="Test the training process")
    parser.add_argument("--serve", action="store_true", help="Serve the model using Gradio")
    parser.add_argument("--run_all", action="store_true", help="Run all steps")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly")
    args = parser.parse_args()

    run_pipeline(args)