import os
import csv
from PIL import Image
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

class SuperResDataset(Dataset):
    def __init__(self, low_res_paths, high_res_paths, transform=None):
        self.low_res_paths = low_res_paths
        self.high_res_paths = high_res_paths
        self.transform = transform

    def __len__(self):
        return len(self.low_res_paths)

    def __getitem__(self, idx):
        low_res_img = Image.open(self.low_res_paths[idx]).convert("RGB")
        high_res_img = Image.open(self.high_res_paths[idx]).convert("RGB")

        if self.transform:
            low_res_img = self.transform(low_res_img)
            high_res_img = self.transform(high_res_img)

        return low_res_img, high_res_img

class DataManager():
    def __init__(self, SEED: int = 23) -> None:
        self.SEED = SEED

    def createCSV(self, root_dirs, csv_file = 'imagePaths.csv', augmented=True) -> None:
        '''Creates a CSV file containing relations between low resolution input and high resolution target.
        Provide all directories with subdirectories "low_res" and "high_res" containing low resolution and high resolution images respectively
        which you want to use in the experiment. If augmented is set to True the CSV will contain paths to augmented images as well.'''
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['lowResPath', 'highResPath'])

            for root_dir in root_dirs:
                low_res_dir = os.path.join(root_dir, 'low_res')
                high_res_dir = os.path.join(root_dir, 'high_res')
                low_res_augmented_dir = os.path.join(root_dir, 'low_res_augmented')
                augmented_dir = os.path.join(root_dir, 'augmented')

                for filename in os.listdir(low_res_dir):
                    low_res_path = os.path.join(low_res_dir, filename)
                    high_res_path = os.path.join(high_res_dir, filename)
                    writer.writerow([low_res_path, high_res_path])

                if augmented:
                    for filename in os.listdir(low_res_augmented_dir):
                        low_res_augmented_path = os.path.join(low_res_augmented_dir, filename)
                        augmented_path = os.path.join(augmented_dir, filename)
                        writer.writerow([low_res_augmented_path, augmented_path])

    def prepareData(self, root_dir: str, data_directories: List[str]) -> None:
        '''Expects paths to "high_res" directories containing high resolution images.
        Prepares data for the experiment. Goes over the images in the given directories.
        Resizes them to smaller resoltion and back to original resolution to create low resolution images.
        Saves the low resolution images to "low_res" directories. 
        Also creates "augmented" directories that contain the augmented images.'''
        for data_dir in data_directories:
            high_res_dir = os.path.join(root_dir, data_dir, 'high_res')
            low_res_dir = os.path.join(root_dir, data_dir, 'low_res')
            augmented_dir = os.path.join(root_dir, data_dir, 'augmented')
            low_res_augmented_dir = os.path.join(root_dir, data_dir, 'low_res_augmented')

            os.makedirs(low_res_dir, exist_ok=True)

            for filename in os.listdir(high_res_dir):
                high_res_path = os.path.join(high_res_dir, filename)
                low_res_path = os.path.join(low_res_dir, filename)

                if not os.path.exists(low_res_path):
                    with Image.open(high_res_path) as img:
                        low_res_img = img.resize((128, 128))
                        low_res_img = low_res_img.resize((256, 256))
                        low_res_img.save(low_res_path)

            os.makedirs(low_res_augmented_dir, exist_ok=True)
            
            for filename in os.listdir(augmented_dir):
                augmented_path = os.path.join(augmented_dir, filename)
                low_res_augmented_path = os.path.join(low_res_augmented_dir, filename)

                if not os.path.exists(low_res_augmented_path):
                    with Image.open(augmented_path) as img:
                        low_res_img = img.resize((128, 128))
                        low_res_img = low_res_img.resize((256, 256))
                        low_res_img.save(low_res_augmented_path)

    def processImages(self, dataPaths: List[str]) -> None:
        '''Not yet implemented correctly. Do not use'''
        experiment_folder = "experiment_data"

        for data_path in dataPaths:
            if not os.path.exists(data_path):
                print(f"Directory '{data_path}' does not exist.")
                continue

            experiment_path = os.path.join(data_path, experiment_folder)
            os.makedirs(experiment_path, exist_ok=True)

            high_res_folder = os.path.join(experiment_path, "high_res")
            low_res_folder = os.path.join(experiment_path, "low_res")

            os.makedirs(high_res_folder, exist_ok=True)
            os.makedirs(low_res_folder, exist_ok=True)

            for filename in os.listdir(data_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(data_path, filename)

                    high_res_image = Image.open(img_path)
                    low_res_image = high_res_image.resize((128, 128), Image.ANTIALIAS)
                    high_res_image_resized = low_res_image.resize((256, 256), Image.ANTIALIAS)

                    high_res_image_resized_path = os.path.join(high_res_folder, f"{data_path}_{filename}")
                    high_res_image_resized.save(high_res_image_resized_path)

                    low_res_image_path = os.path.join(low_res_folder, f"{data_path}_{filename}")
                    low_res_image.save(low_res_image_path)

            print(f"Processed images in '{data_path}'")

    def augmentImages(self, directories: List[str]):
        '''Expects paths to directories containing "high_res" directories with high resolution images.
        Creates "augmented" directories containing augmented images. With the augmentations defined in performAugmentations method.'''
        for directory in directories:
            high_res_folder = os.path.join(directory, "high_res")
            augmented_folder = os.path.join(directory, "augmented")

            if not os.path.exists(augmented_folder):
                os.makedirs(augmented_folder)

            if not os.path.exists(high_res_folder):
                print(f"High-resolution folder not found in {directory}")
                continue

            for filename in tqdm(os.listdir(high_res_folder)):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(high_res_folder, filename)
                    img = Image.open(image_path)

                    augmented_images = self.performAugmentations(img)

                    for augmentation_suffix, augmented_img in augmented_images.items():
                        augmented_filename = f"{os.path.splitext(filename)[0]}_{augmentation_suffix}{os.path.splitext(filename)[1]}"
                        augmented_path = os.path.join(augmented_folder, augmented_filename)
                        
                        if not os.path.exists(augmented_path):
                            augmented_img.save(augmented_path)


    def performAugmentations(self, image: Image.Image) -> Dict[str, Image.Image]:
        '''Expects a PIL Image object.
        Returns a dictionary of augmented images.'''
        augmented_images = {
            "rotation": image.rotate(45), 
            "transpose": image.transpose(Image.FLIP_LEFT_RIGHT)
        }

        return augmented_images
    
    def build(self, dataDirs: List[str], csvFile: str = 'image_paths.csv', augmented: bool = True) -> List[Tuple[DataLoader, DataLoader]]:
        '''Builds the experiment data. Calls the methods in the desired order.
        Returns a list of tuples (Dataset, DataLoader) for train, validation and test sets.'''

        self.augmentImages(dataDirs)
        self.prepareData('', dataDirs)
        self.createCSV(dataDirs, csvFile, augmented)

        imagePaths = pd.read_csv(csvFile)
        lowResPaths = imagePaths['lowResPath'].tolist()
        highResPaths = imagePaths['highResPath'].tolist()

        trainLowResPaths, testLowResPaths, trainHighResPaths, testHighResPaths = train_test_split(lowResPaths, highResPaths, test_size=0.2, random_state=self.SEED)

        trainLowResPaths, valLowResPaths, trainHighResPaths, valHighResPaths = train_test_split(trainLowResPaths, trainHighResPaths, test_size=0.1, random_state=self.SEED)

        print("Train set shape (High-Res):", len(trainHighResPaths))
        print("Train set shape (Low-Res):", len(trainLowResPaths))
        print("Validation set shape (High-Res):", len(valHighResPaths))
        print("Validation set shape (Low-Res):", len(valLowResPaths))
        print("Test set shape (High-Res):", len(testHighResPaths))
        print("Test set shape (Low-Res):", len(testLowResPaths))

        transform = transforms.Compose([transforms.ToTensor()])

        trainDataset = SuperResDataset(trainLowResPaths, trainHighResPaths, transform=transform)
        trainLoader = DataLoader(trainDataset, batch_size=1, shuffle=True)

        valDataset = SuperResDataset(valLowResPaths, valHighResPaths, transform=transform)
        valLoader = DataLoader(valDataset, batch_size=1, shuffle=True)

        testDataset = SuperResDataset(testLowResPaths, testHighResPaths, transform=transform)
        testLoader = DataLoader(testDataset, batch_size=1, shuffle=True)

        return [(trainDataset, trainLoader), (valDataset, valLoader), (testDataset, testLoader)]
