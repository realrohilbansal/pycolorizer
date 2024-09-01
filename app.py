import gradio as gr
import torch
import mlflow
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms

from model import Generator

EXPERIMENT_NAME = "Colorizer_Experiment"
RUN_ID = "your_run_id_here"  # Replace with your actual run ID

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

def preprocess_image(image):
    img = Image.fromarray(image).convert("RGB")
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

def colorize_image(image, model, device):
    L = preprocess_image(image).to(device)
    with torch.no_grad():
        ab = model(L)
    colorized = postprocess_output(L, ab)
    return colorized

def setup_gradio_app(run_id, device):
    model = load_model(run_id, device)

    def gradio_colorize(input_image):
        colorized = colorize_image(input_image, model, device)
        return Image.fromarray(colorized)

    iface = gr.Interface(
        fn=gradio_colorize,
        inputs=gr.Image(label="Upload a grayscale image"),
        outputs=gr.Image(label="Colorized Image"),
        title="Image Colorizer",
        description="Upload a grayscale image and get a colorized version!",
    )

    return iface