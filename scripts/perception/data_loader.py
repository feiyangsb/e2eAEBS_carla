import os
import numpy as np
import csv
import collections
from torch.utils import data
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms

class CarlaDataLoader(data.DataLoader):
    def __init__(self, data_path, split='training'):
        self.split = split
        self.image_path = collections.defaultdict(list)
        self.gt_distance = collections.defaultdict(list)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image_path_list = []
        gt_distance_list = []
        for folder in os.listdir(data_path):
            with open(os.path.join(data_path, folder, "label.csv")) as csv_file:
                readCSV = csv.reader(csv_file, delimiter=',')
                skip_title_flag = False
                for row in readCSV:
                    if skip_title_flag == False:
                        skip_title_flag = True
                        continue
                    image_path_list.append(row[0])
                    gt_distance_list.append(row[1])
        image_path_train, image_path_calibration, gt_distance_train, gt_distance_calibration = train_test_split(
                                                image_path_list, gt_distance_list, test_size=0.2, random_state=42)
        self.image_path["training"] = image_path_train     
        self.image_path["calibration"] = image_path_calibration
        self.gt_distance["training"] = gt_distance_train
        self.gt_distance["calibration"] = gt_distance_calibration

        print("totally found {} images: {} for training, {} for calibrating".format(len(image_path_list), len(image_path_train), len(image_path_calibration)))
    
    def __len__(self):
        return len(self.gt_distance[self.split])
    
    def __getitem__(self, index):
        img_path = self.image_path[self.split][index]
        img = Image.open(img_path+".png").convert("RGB")
        img = self.transform(img)
        gt_distance = self.gt_distance[self.split][index]
        gt_distance = np.array([gt_distance]).astype('float')/120.0
        return img, gt_distance