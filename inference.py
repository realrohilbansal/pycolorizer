import os
import torch
import mlflow
import mlflow.pytorch
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
import argparse

from model import Generator

EXPERIMENT_NAME = "Colorizer_Experiment"

def setup_mlflow():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id

def load_model(run_id, device):
    print(f"Loading model from run: {run_id}")
    model_uri = f"runs:/{run_id}/generator_model"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    return model

# Configuration variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_ID = "your_run_id_here"  # Replace with the actual run ID
IMAGE_PATH = "path/to/your/image.jpg"  # Replace with the path to your input image
SAVE_MODEL = False
SERVE_MODEL = False
SERVE_PORT = 5000

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)
    lab_img = rgb2lab(img_tensor.permute(1, 2, 0).numpy())
    L = lab_img[:,:,0]
    L = (L - 50) / 50
    L = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float()
    return L

def postprocess_output(L, ab):
    L = L.squeeze().cpu().numpy()
    ab = ab.squeeze().cpu().numpy()
    L = (L + 1.) * 50.
    ab = ab * 128.
    Lab = np.concatenate([L[..., np.newaxis], ab], axis=2)
    rgb_img = lab2rgb(Lab)
    return (rgb_img * 255).astype(np.uint8)

def colorize_image(model, image_path, device):
    L = preprocess_image(image_path).to(device)
    with torch.no_grad():
        ab = model(L)
    colorized = postprocess_output(L, ab)
    return colorized

def save_model(model, run_id):
    with mlflow.start_run(run_id=run_id):
        # Log the model
        mlflow.pytorch.log_model(model, "model")
        
        # Register the model
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "colorizer_model")
        
        print(f"Model saved and registered with run_id: {run_id}")

def serve_model(run_id, port=5000):
    model_uri = f"runs:/{run_id}/model"
    mlflow.pytorch.serve(model_uri, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorizer Inference")
    parser.add_argument("--run_id", type=str, help="MLflow run ID of the trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input grayscale image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # If run_id is not provided, try to load it from the file
    if not args.run_id:
        try:
            with open("latest_run_id.txt", "r") as f:
                args.run_id = f.read().strip()
        except FileNotFoundError:
            print("No run ID provided and couldn't find latest_run_id.txt")
            exit(1)

    experiment_id = setup_mlflow()
    with mlflow.start_run(experiment_id=experiment_id, run_name="inference_run"):
        try:
            model = load_model(args.run_id, device)
            colorized = colorize_image(model, args.image_path, device)
            output_path = f"colorized_{os.path.basename(args.image_path)}"
            Image.fromarray(colorized).save(output_path)
            print(f"Colorized image saved as: {output_path}")
            
            mlflow.log_artifact(output_path)
            mlflow.log_param("input_image", args.image_path)
            mlflow.log_param("model_run_id", args.run_id)
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            mlflow.log_param("error", str(e))
        finally:
            mlflow.end_run()