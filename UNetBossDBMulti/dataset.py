# import the necessary packages
from torch.utils.data import Dataset
import cv2
# for bossdb
from typing import Tuple, List
from intern import array
from torchvision.transforms import ToTensor
import numpy as np
import torch
import config


# from https://github.com/aplbrain/bossdb_cookbook/blob/main/notebooks/BossDB-Dataset-Classes-for-Pytorch-DataLoaders.ipynb
# under heading "Dataset class for when you need single image slices and corresponding segmentation masks from the data"
class BossDBSliceDataset(Dataset):
    
    def __init__(
        self, 
        image_boss_uri: str, 
        mask_boss_uri: str, 
        boss_config: dict, 
        centroid_list_zyx: List[Tuple[int, int, int]],
        px_radius_yx: Tuple[int, int],
        image_transform=ToTensor(),
        mask_transform=None,
        verbose=False, 
        zshift = 1, # change to 2 for celegans data I guess?
        binary_classification = True # assume just 2 levels
       # xmin = None, ymin=None, zmin=None
    ):
        self.config = boss_config
        self.image_array = array(image_boss_uri, boss_config=boss_config)
        self.mask_array = array(mask_boss_uri, boss_config=boss_config)
        if verbose:
            print('image,mask uri=', image_boss_uri,mask_boss_uri)
        self.centroid_list = centroid_list_zyx
        rad_y, rad_x = px_radius_yx
        self.px_radius_y = rad_y
        self.px_radius_x = rad_x
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.verbose = verbose
        self.zshift = zshift
        #self.binary_classification = False
        # self.xmin,self.ymin,self.zmin = 0,0,0
        # if xmin is not None: # have xmin,ymin
        #     self.xmin = xmin
        # if ymin is not None:
        #     self.ymin = ymin
        # if zmin is not None:
        #     self.zmin = zmin
    
    def __getitem__(self, key):
        z, y, x = self.centroid_list[key]
        # if self.verbose:
        #     print('z,y,x=', z,y,x)
            #print('subtracts z,y,x=', z-self.zmin, y-self.ymin,x-self.xmin)
        # z -= self.zmin
        # y -= self.ymin
        # x -= self.xmin
        # if self.verbose:
        #     print('subtracts:', z,y,x)
        #print('**** Image shape:',z,y,x)
        #print('**** Image shape (array)=', self.image_array.shape)
        # JPN -- add try/except
        try:
            image_array =  self.image_array[
				z : z + self.zshift,
				y - self.px_radius_y : y + self.px_radius_y,
				x - self.px_radius_x : x + self.px_radius_x,
			]
            mask_array =  self.mask_array[
				z : z + self.zshift,
				y - self.px_radius_y : y + self.px_radius_y,
				x - self.px_radius_x : x + self.px_radius_x,
			]
        except Exception as e:
            #return None
            if self.verbose:
                print('could not find a slice!')
                print('[FULL ERROR]:', str(e))
            # empty array
            #print('***HERE 2!')
            image_array = np.zeros([self.zshift, self.px_radius_y*2, self.px_radius_x*2])
            mask_array = np.zeros([self.zshift, self.px_radius_y*2, self.px_radius_x*2])
            
		# take last?  I guess?
        #print('*** HERE 1:', image_array.shape)
        if self.zshift > 1:
            i = np.random.randint(2)
            image_array = image_array[i,:,:]
            mask_array = mask_array[i,:,:]
            mask_array = np.expand_dims(mask_array,0)
            
		# put into the right formatting
        mask_array = mask_array[0,:,:]
        mask_array = mask_array.astype(np.float32)#/255.0
        image_array = image_array.astype(np.float32)#/255.0

        # max out
        mask_array[mask_array > config.NUM_CLASSES] = config.NUM_CLASSES-1
        
		# # # binary classification?
        # if self.binary_classification:
        #     mask_array[mask_array > 0] = 1.0
        
        # if self.verbose:
        #     print('after read in shapes are: (image)', image_array.shape, 
        #           '(mask)', mask_array.shape)
        if self.image_transform:
            image_array= self.image_transform(image_array)
            
        if self.mask_transform:
            mask_array = self.mask_transform(mask_array)
        
        #mask_array = torch.from_numpy(mask_array)#.astype('int64')).long()
        #image_array = torch.from_numpy(image_array)

        
		# also copy image array into expected shape
        image_array = np.tile(image_array, (3,1,1))
        #print('**** array shape:', image_array.shape, mask_array.shape)
            
        #return image_array, mask_array
        # JPN with () added
        return image_array, mask_array
        # image, y, self.XImg_list[index]

    def __len__(self):
        return len(self.centroid_list)