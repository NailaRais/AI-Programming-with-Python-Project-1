import os
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
import ast

# Load models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# Obtain ImageNet labels
with open('data/imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def classifier(img_path, model_name):
    # Check if the image file exists
    if not os.path.exists(img_path):
        print(f"Warning: Image file not found: {img_path}")
        return None  # Handle the case where the image is not found
    
    # Load the image
    img_pil = Image.open(img_path)

    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image
    img_tensor = preprocess(img_pil)
    
    # Resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    
    # Check PyTorch version for Variable handling
    pytorch_ver = __version__.split('.')
    
    # Handle tensor gradients for PyTorch v0.4 and above
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)
    
    # Apply model to input
    model = models[model_name]
    model = model.eval()  # Put model in evaluation mode
    
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        output = model(img_tensor)
    else:
        data = Variable(img_tensor, volatile=True)
        output = model(data)

    # Return the index corresponding to the predicted class
    pred_idx = output.data.numpy().argmax()
    return imagenet_classes_dict[pred_idx]

def classify_images(images_dir, results, model_name):
    for key in results:
        img_path = os.path.join(images_dir, key)  # Dynamically build the image path
        try:
            model_label = classifier(img_path, model_name)
            results[key] = model_label
        except FileNotFoundError as e:
            print(f"Error: {e}")


