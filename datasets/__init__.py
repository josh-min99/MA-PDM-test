import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from datasets.addata import ADLoader


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    imsize = config.data.image_size
    patch_size = config.data.patch_size
    dataset_type = config.data.dataset
    time_step = config.data.time_step
    # 데이터 루트 경로를 config.data.data_dir에서 읽는다 (원본은 "/home/data/datasets/"로 하드코딩되어 있었음)
    video_folder = config.data.data_dir
    TrainD = ADLoader(video_folder=video_folder,dataset_type=dataset_type,phase="train",time_step=time_step,patch_size = patch_size,transform=transforms.Compose([
             transforms.ToTensor(),]),resize_height=imsize,resize_width=imsize)
    TestD = ADLoader(video_folder=video_folder,dataset_type=dataset_type,phase="test",time_step=time_step,patch_size = patch_size,transform=transforms.Compose([
             transforms.ToTensor(),]),resize_height=imsize,resize_width=imsize,parse_patches=False)

    return TrainD, TestD


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
