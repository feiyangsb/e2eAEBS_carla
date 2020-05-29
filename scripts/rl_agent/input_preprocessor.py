import numpy as np
from PIL import Image
from torchvision import transforms

class InputPreprocessor():
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __call__(self, state):
        image = state[0]
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))[:,:,:3][:,:,::-1]
        img = Image.fromarray(img)
        img = self.transforms(img).numpy()
        
        return [img, state[1]/40.0]