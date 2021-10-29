#import config
from .model import Net

import os

import numpy as np
import cv2
import time
import torch
from torch import nn
from torchvision import transforms

class VehicleEmbedder:
    def __init__(self, pretrained_weight_path):
        #torch.backends.cudnn.enabled = True
        self.im_size = (64, 128)
        device = torch.device("cuda")
        self.model = Net(n_classes=-1).to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        pretrained_dict = torch.load(pretrained_weight_path)
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.eval()
        
    def infer(self, vehicle_images):
        vehicle_image_array = [cv2.resize(img, self.im_size) for img in vehicle_images]

        with torch.no_grad():
            vehicle_image_tensor = torch.from_numpy(np.stack(vehicle_image_array)).cuda()
            vehicle_image_tensor = vehicle_image_tensor/255.0
            vehicle_image_tensor = vehicle_image_tensor.transpose(1, 2).transpose(1, 3).contiguous()
            features = []

            batch_size = 64
            n_img = len(vehicle_image_array)
            n_left = n_img

            while n_left > 0:
                n_samples = min(batch_size, n_left)
                start_index = n_img-n_left
                end_index = start_index + n_samples
                start = time.time()
                vehicle_image_batch = torch.autograd.Variable(vehicle_image_tensor[start_index:end_index])
                feature_batch = self.model.extract_features(vehicle_image_batch)

                execution_time = time.time() - start
                #print(execution_time)
                for i in range(n_samples):
                    features.append(feature_batch[i])
                n_left -= n_samples
            features = torch.stack(features, dim=0)
            return features
