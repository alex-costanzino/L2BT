import os
import torch
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.general_utils import SquarePad

def visa_classes():
    return [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum"
    ]

class BaseAnomalyDetectionDataset(Dataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]

        self.size = img_size
        self.img_path = os.path.join(dataset_path, class_name, split)
                
        self.rgb_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((self.size, self.size), interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.IMAGENET_MEAN, std = self.IMAGENET_STD)
            ])

class TrainValDataset(BaseAnomalyDetectionDataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        super().__init__(split = split, class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.img_paths = self.load_dataset()

    def load_dataset(self):
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.JPG")
        rgb_paths.sort()
        return rgb_paths

    def __len__(self):
        return len(self.img_paths)
    
    def get_size(self):
        return max(Image.open(self.img_paths[0]).convert('RGB').size)

    def __getitem__(self, idx):
        rgb_path = self.img_paths[idx]
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)

        return img

class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split = "test", class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.img_paths, self.gt_paths, self.labels = self.load_dataset()

        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)
        defect_types.remove('ground_truth')

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                rgb_paths.sort()
                img_tot_paths.extend(rgb_paths)
                gt_tot_paths.extend([0] * len(rgb_paths))
                tot_labels.extend([0] * len(rgb_paths))
            elif defect_type == 'bad':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.img_path, 'ground_truth', defect_type) + "/*.png")
                rgb_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(rgb_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(rgb_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def get_size(self):
        return max(Image.open(self.img_paths[0]).convert('RGB').size)
    
    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if gt == 0:
            gt = torch.zeros(
                [1, img.size[1], img.size[0]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        img = self.rgb_transform(img)

        return img, gt, label, img_path


def get_data_loader(split, class_name, dataset_path, img_size, batch_size=None):
    if split in ['train']:
        dataset = TrainValDataset(split = "train", class_name = class_name, img_size = img_size, dataset_path = dataset_path)
        data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 8, drop_last = False, pin_memory = False)
    elif split in ['test']:
        dataset = TestDataset(class_name = class_name, img_size = img_size, dataset_path = dataset_path)
        data_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = False, num_workers = 8, drop_last = False, pin_memory = False)
    
    return data_loader, dataset.get_size()