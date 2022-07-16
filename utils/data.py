from pathlib import Path
from typing import Union
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class KittiSemanticsDataset(Dataset):
    def __init__(self, X: Union[list, np.ndarray], Y: Union[list, np.ndarray]) -> None:
        super().__init__()
        self.X = torch.stack(X)
        self.Y = torch.stack(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class KittiDepthDataset(KittiSemanticsDataset):
    def __init__(self, X_color: Union[list, np.ndarray], X_sensor: Union[list, np.ndarray], X_mask: Union[list, np.ndarray], Y: Union[list, np.ndarray]) -> None:
        super().__init__(X_color, Y)

        self.X_sensor = torch.stack(X_sensor)
        self.X_mask = torch.stack(X_mask)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return self.X[index], self.X_sensor[index], self.X_mask[index], self.Y[index]

def load_image(filename: Union[Path, str]) -> Image:
    image = Image.open(filename)
    return image

def generate_mask(image: Image.Image):
    mask = np.array(image)
    mask[mask > 0.0] = 1.0

    return Image.fromarray(mask)

def load_kitti_semantics_dataset(load_dir: Union[Path, str], val_ratio=0.2, shuffle=True, random_state=None):
    if isinstance(load_dir, (Path, str)):
        load_dir = Path(load_dir)
    else:
        raise TypeError(f"load_dir must be Path or str")

    preprocess = transforms.Compose([
        transforms.Resize((352, 1216)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # load train & validation dataset
    X, Y = [], []
    for filename in tqdm(load_dir.glob("training/image_2/*.png"), desc="loading train & validation data"):
        x = load_image(filename)
        x = preprocess(x)
        X.append(x)
        
        y = load_image(load_dir / f"training/semantic_rgb/{filename.name}")
        y = preprocess(y)
        Y.append(y)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=val_ratio,
        shuffle=shuffle, random_state=random_state
    )  

    X_test, Y_test = [], []
    for filename in tqdm(load_dir.glob("testing/image_2/*.png"), desc="loading test data"):
        x = load_image(filename)
        x = preprocess(x)
        X_test.append(x)
        Y_test.append(torch.zeros_like(x))

    train_dataset = KittiSemanticsDataset(X_train, Y_train)
    val_dataset = KittiSemanticsDataset(X_val, Y_val)
    test_dataset = KittiSemanticsDataset(X_test, Y_test)

    return train_dataset, val_dataset, test_dataset

def load_kitti_depth_dataset(load_dir: Union[Path, str], val_ratio=0.2, shuffle=True, random_state=None, train=True):
    if isinstance(load_dir, (Path, str)):
        load_dir = Path(load_dir)
    else:
        raise TypeError(f"load_dir must be Path or str")

    preprocess_c = transforms.Compose([
        transforms.Resize((352, 1216)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    preprocess_d = transforms.Compose([
        transforms.Resize((352, 1216)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.0],
            std=[1.0]
        )
    ])

    if train:
        # load train & validation dataset
        X_color, X_sensor, X_mask, Y = [], [], [], []
        for filename in tqdm(load_dir.glob("val_selection_cropped/image/*.png"), desc="loading train & validation data"):
            x_c = load_image(filename)
            x_c = preprocess_c(x_c)
            X_color.append(x_c)

            filename = filename.name.split("_image_")
            file_prefix, file_suffix = filename[0], f"{filename[1]}_image_{filename[2]}"

            x_s = load_image(load_dir / f"val_selection_cropped/velodyne_raw/{file_prefix}_velodyne_raw_{file_suffix}")
            x_m = generate_mask(x_s)
            x_s = preprocess_d(x_s.convert("F"))
            X_sensor.append(x_s)

            x_m = preprocess_d(x_m.convert("F"))
            X_mask.append(x_m)
            
            y = load_image(load_dir / f"val_selection_cropped/groundtruth_depth/{file_prefix}_groundtruth_depth_{file_suffix}")
            y = preprocess_d(y.convert("F"))
            Y.append(y)

        X_color_train, X_color_val, X_sensor_train, X_sensor_val, X_mask_train, X_mask_val, Y_train, Y_val = train_test_split(
            X_color, X_sensor, X_mask, Y, test_size=val_ratio,
            shuffle=shuffle, random_state=random_state
        )

        train_dataset = KittiDepthDataset(X_color_train, X_sensor_train, X_mask_train, Y_train)
        val_dataset = KittiDepthDataset(X_color_val, X_sensor_val, X_mask_val, Y_val)

        return train_dataset, val_dataset

    else:
        X_color_test, X_sensor_test, X_mask_test, Y_test = [], [], [], []
        for filename in tqdm(load_dir.glob("test_depth_completion_anonymous/image/*.png"), desc="loading test data"):
            x_c = load_image(filename)
            x_c = preprocess_c(x_c)
            X_color_test.append(x_c)

            x_s = load_image(load_dir / f"test_depth_completion_anonymous/velodyne_raw/{filename.name}")
            x_m = generate_mask(x_s)
            x_s = preprocess_d(x_s.convert("F"))
            X_sensor_test.append(x_s)

            x_m = preprocess_d(x_m.convert("F"))
            X_mask_test.append(x_m)

            Y_test.append(torch.zeros_like(x_c))

    
        test_dataset = KittiDepthDataset(X_color_test, X_sensor_test, X_mask_test, Y_test)

        return test_dataset

def show_image(image: Union[torch.Tensor, Image.Image, np.ndarray], show=True):
    if isinstance(image, torch.Tensor):
        image = image.detach().numpy()
    elif isinstance(image, (Image.Image, np.ndarray)):
        image = np.array(image)
    else:
        raise ValueError(f"image must be Tensor, Image or ndarray")

    a1, a2, a3 = image.shape
    if 1 <= a1 and a1 <= 3:
        image = np.transpose(image, (1, 2, 0))
    
    ax = plt.imshow(image)
    if show:
        plt.show()
        plt.clf()
    
    return ax

if __name__ == "__main__":
    load_dir = Path("/home/matsuura/Development/waseda/m1/lecture_spring_2022/cgo/group_work/dataset/data_semantics")

    train_dataset, val_dataset, test_dataset = load_kitti_semantics_dataset(load_dir)

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))