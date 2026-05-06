import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from datasets.ISIC_2018_Task_3.augmentation import train_transform, val_transform, center_crop

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


base_dir = r'/public/ywq/Project/datasets/ISIC_2018_Task_3'


def read_imgs_path_list(img_dir, label_csv_path):
    reindex_columns_list = ['image', 'NV', 'DF', 'BKL', 'VASC', 'AKIEC','BCC', 'MEL']

    label_csv_pd = pd.read_csv(label_csv_path, header=0)
    label_csv_pd = label_csv_pd.reindex(columns=reindex_columns_list)
    label_argmax = np.argmax(label_csv_pd[['NV', 'DF', 'BKL', 'VASC', 'AKIEC','BCC', 'MEL']].values, axis=1)
    label_csv_pd['argmax'] = label_argmax

    imgs_path_list = []
    for idx, row in label_csv_pd.iterrows():
        img_path = os.path.join(img_dir, row['image'] + '.jpg')
        if not os.path.exists(img_path):
            raise ValueError(f'image does not exist: {img_path}')
        imgs_path_list.append([img_path, row['argmax']])

    return imgs_path_list


class ISIC2018Task3_DataSets(Dataset):
    def __init__(
            self,
            base_dir=base_dir,
            split='train',
            num_patch=5,
            img_size=(224, 224)
    ):
        super(ISIC2018Task3_DataSets, self).__init__()

        self.base_dir = base_dir
        self.split = split
        self.numpatch = num_patch
        if self.split == 'train':
            self.img_dir = os.path.join(self.base_dir, 'ISIC2018_Task3_Training_Input')
            self.label_csv_path = os.path.join(self.base_dir, 'ISIC2018_Task3_Training_GroundTruth',
                                               'ISIC2018_Task3_Training_GroundTruth.csv')
            self.transform = train_transform(img_size=img_size)
        elif self.split == 'val':
            self.img_dir = os.path.join(self.base_dir, 'ISIC2018_Task3_Validation_Input')
            self.label_csv_path = os.path.join(self.base_dir, 'ISIC2018_Task3_Validation_GroundTruth',
                                               'ISIC2018_Task3_Validation_GroundTruth.csv')
            self.transform = val_transform(img_size=img_size)
        elif self.split == 'test':
            self.img_dir = os.path.join(self.base_dir, 'ISIC2018_Task3_Test_Input')
            self.label_csv_path = os.path.join(self.base_dir, 'ISIC2018_Task3_Test_GroundTruth',
                                               'ISIC2018_Task3_Test_GroundTruth.csv')
            self.transform = val_transform(img_size=img_size)
        else:
            raise ValueError(f"The split ({split}) must be between 'train', 'val' or 'test'")
        
        self.images_path_list = read_imgs_path_list(self.img_dir, self.label_csv_path)
        
        crop_size = img_size[0]
        self.five_crop = T.Compose([
            T.Resize((img_size[0]*2, img_size[1]*2)),     
            T.FiveCrop(crop_size)
        ])   

        print(f'split: {self.split}, total {len(self.images_path_list)} samples')

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):
        image_path = self.images_path_list[idx][0]
        cls_label = self.images_path_list[idx][1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transform_res = self.transform(image=image)
        image_full = transform_res['image']

        image_pil = Image.fromarray(image)
        crops_pil = self.five_crop(image_pil) 
        
        crops_tensor_list = []
        for crop in crops_pil:
            crop_np = np.array(crop)
            
            aug_crop = self.transform(image=crop_np)['image']
   
            crops_tensor_list.append(aug_crop)
        
        selected_patches = torch.stack(crops_tensor_list)


        return {'image': image_full.float(),
                'patch': selected_patches.float(),
                'cls_label': cls_label,
                'image_name': os.path.basename(image_path)}
