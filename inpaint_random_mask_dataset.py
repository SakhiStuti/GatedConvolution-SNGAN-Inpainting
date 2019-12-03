from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
import numpy as np
import cv2
import torch


class Places2_rmask(Dataset):
    """
    Dataset for Inpainting task
    Params:
        img_root
        resize_shape(tuple): The shape of the final image (default:(256,256))
        transforms_oprs(list) : Determine which transformation used on the imgae (default:['random_crop', 'to_tensor'])
    Return:
        img, mask
    """
    def __init__(self, img_root, resize_shape=(256, 256), transforms_oprs=['random_crop', 'to_tensor'],
                random_ff_setting={'img_shape':[256,256],'mv':5, 'ma':4.0, 'ml':40, 'mbw':10}):
        self.img_paths = glob('{:s}/**/**/*.jpg'.format(img_root), recursive=True) #gt img paths
        self.resize_shape = resize_shape
        self.random_ff_setting = random_ff_setting
        self.transform = self.transform_initialize(resize_shape, transforms_oprs)


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, index):
        
        
        img_path = self.img_paths[index]
        img = self.transform(self.read_img(img_path)) #0-1, 3x256x256
        img = 2*img - 1 #-1 to 1, 3x256x256
        mask = self.transform(self.read_mask())#0-1, 3x256x256
        
        return img, mask[:1,:,:]
    
    
    def read_img(self, path):
        """
        Returns PIL Image
        """
        return  Image.open(path).convert('RGB')#0-255
    

    def read_mask(self):
        """
        Returns PIL Mask
        """
        mask_np = 255*self.random_ff_mask() #0-255
        return Image.fromarray(mask_np.astype(np.uint8))
    
    

    def transform_initialize(self, crop_size, config=['random_crop', 'to_tensor']):
        """
        Initialize the transformation oprs and create transform function for img
        """
        self.transforms_oprs = {}
        self.transforms_oprs["hflip"]= transforms.RandomHorizontalFlip(0.5)
        self.transforms_oprs["vflip"] = transforms.RandomVerticalFlip(0.5)
        self.transforms_oprs["random_crop"] = transforms.RandomCrop(crop_size)
        self.transforms_oprs["to_tensor"] = transforms.ToTensor()
        self.transforms_oprs["norm"] = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transforms_oprs["resize"] = transforms.Resize(crop_size)
        self.transforms_oprs["center_crop"] = transforms.CenterCrop(crop_size)
        self.transforms_oprs["rdresizecrop"] = transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0), ratio=(1,1), interpolation=2)
        return transforms.Compose([self.transforms_oprs[name] for name in config])
    
    
    def random_ff_mask(self):
        """Generate a random free form mask with configuration.

        self.random_ff_setting should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        Returns:
            np.array: (h,w,3) of type float32 which is 1 at maksed regions and zero otherwise
        """

        h,w = self.random_ff_setting['img_shape']
        mask = np.zeros((h,w))
        num_v = 12+np.random.randint(self.random_ff_setting['mv'])#tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1+np.random.randint(5)):
                angle = 0.01+np.random.randint(self.random_ff_setting['ma'])
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10+np.random.randint(self.random_ff_setting['ml'])
                brush_w = 10+np.random.randint(self.random_ff_setting['mbw'])
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
                
        mask = mask.reshape(mask.shape+(1,)).astype(np.float32)#reshape to (h,w,1)
        mask = np.tile(mask,(1,1,3))#add channels (h,w,3)
        return mask