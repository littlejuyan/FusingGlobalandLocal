import cv2
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import copy
import os
import imageio

def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}

def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)



class read_data():
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        real_img_list = [os.path.join(self.root, '0_real', train_file) for train_file in 
                        os.listdir(os.path.join(self.root, '0_real'))]
                       
        real_label_list = [0 for _ in range(len(real_img_list))]

        fake_img_list = [os.path.join(self.root, '1_fake', train_file) for train_file in
                        os.listdir(os.path.join(self.root, '1_fake'))]

        fake_label_list = [1 for _ in range(len(fake_img_list))]


        self.img = real_img_list+fake_img_list
        self.label = real_label_list+fake_label_list
        
        print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))


    def __getitem__(self, index):
        img, target = imageio.imread(self.img[index]), self.label[index]
        #print("img file: ", self.img[index])
        imgname = self.img[index]
        #print(imgname)
        if len(img.shape) < 3:
            img=np.asarray(img)[..., np.newaxis]
        if len(img.shape) == 3 and img.shape[-1]==1:
            img=np.tile(np.asarray(img), (1,1,3))
        img = Image.fromarray(img, mode='RGB')

        # compute scaling

        height, width = img.height, img.width
        img = data_augment(img, self.opt)

        if self.opt.isTrain and not self.opt.no_flip:
            img = transforms.RandomHorizontalFlip()(img)
        
        input_img = copy.deepcopy(img)
        input_img = transforms.ToTensor()(input_img)
        input_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_img)

        img = transforms.Resize(self.opt.cropSize)(img)
        img = transforms.CenterCrop(self.opt.cropSize)(img)
        cropped_img = transforms.ToTensor()(img)
        cropped_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(cropped_img)


        scale = torch.tensor([height, width])

        return input_img, cropped_img, target, scale, imgname

    def __len__(self):
        return len(self.label) 


