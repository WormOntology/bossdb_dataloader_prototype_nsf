import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from utils import *
from model import UNET
from torch.utils.data import DataLoader


verbose = True
LOAD_MODEL = False # start with a built model?

import config


# if torch.cuda.is_available():
#     DEVICE = 'cuda:0'
#     #print('Running on the GPU')
# else:
#     DEVICE = "cpu"
#     #print('Running on the CPU')
DEVICE = config.DEVICE

#ROOT_DIR = '../datasets/cityscapes'
# IMG_HEIGHT = 110  
# IMG_WIDTH = 220  
import config
MODEL_PATH = config.MODEL_PATH
IMG_HEIGHT = config.INPUT_IMAGE_HEIGHT  
IMG_WIDTH = config.INPUT_IMAGE_WIDTH
BATCH_SIZE = 16 
LEARNING_RATE = 0.0005
EPOCHS = 10
num_workers = 0

def train_function(data, model, optimizer, loss_fn, device):
    print('Entering into train function')
    loss_values = []
    #data = tqdm(data)
    if verbose:
        print('after tqdm')
    #for index, batch in enumerate(data): 
    for (i, (X, y)) in enumerate(data):
        #X, y = batch
        if verbose:
            print('after batch')
        #X, y = X.to(device), y.to(device)
        (X, y) = (X.to(config.DEVICE), y.to(config.DEVICE))
        preds = model(X)

        if verbose:
            print('after preds')
    
        # [batch_size, nb_classes=3, height, width]
        # RuntimeError: only batches of spatial targets supported (3D tensors) but got targets of dimension: 4
        #loss = loss_fn(preds, y)
        # check 
        if len(y.shape) > 3: # JPN
            loss = loss_fn(preds, y[:,0,:,:].long()) # JPN
        else:
            loss = loss_fn(preds, y.long()) # JPN
        if verbose:
            print('after loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
        

def main():
    global epoch
    epoch = 0 # epoch is initially assigned to 0. If LOAD_MODEL is true then
              # epoch is set to the last value + 1. 
    LOSS_VALS = [] # Defining a list to store loss values after every epoch
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.NEAREST),
    ]) 

    from dataset import BossDBSliceDataset
    # for bossdb
    from intern import array
    import config
    from numpy import random, column_stack

    # get dataset URI
    boss_uri_images = config.IMAGE_DATASET_URI
    boss_uri_masks = config.SEGMASK_DATASET_URI
    # JPN -- have to get dataset size and centroids
    channel = array(boss_uri_images)
    # get shape
    shape = channel.shape
    print('dataset shape=', shape)

    # select a set of centroids randomly
    ncentroid_train = config.NCENTROID_TRAIN
    ncentroid_test = config.NCENTROID_TEST
    zshift = config.ZSHIFT

    # sadly, we have to ping the dataset to find example indicies... sigh
    z = random.randint(config.ZMIN,config.ZMAX-zshift,size=ncentroid_train) # +/- 1
    y = random.randint(config.YMIN+config.INPUT_IMAGE_WIDTH//2,config.YMAX-config.INPUT_IMAGE_WIDTH//2,size=ncentroid_train)
    x = random.randint(config.XMIN+config.INPUT_IMAGE_WIDTH//2,config.XMAX-config.INPUT_IMAGE_WIDTH//2,size=ncentroid_train)
    centroids_train = column_stack([z,y,x])

    z = random.randint(config.ZMIN,config.ZMAX-1,size=ncentroid_test)
    y = random.randint(config.YMIN+config.INPUT_IMAGE_WIDTH//2,config.YMAX-config.INPUT_IMAGE_WIDTH//2,size=ncentroid_test)
    x = random.randint(config.XMIN+config.INPUT_IMAGE_WIDTH//2,config.XMAX-config.INPUT_IMAGE_WIDTH//2,size=ncentroid_test)
    centroids_test = column_stack([z,y,x])

    # save test centroids
    print("[INFO] saving test centroids to "+config.TEST_PATHS+" ...")
    f = open(config.TEST_PATHS, "w")
    #print(centroids_test.astype('str').tolist())
    for t in centroids_test:
        f.write(str(t)+'\n')
    #f.write("\n".join(centroids_test.astype('str').tolist()))
    f.close()

    # create the train and test datasets
    trainDS = BossDBSliceDataset(image_boss_uri=boss_uri_images, mask_boss_uri=boss_uri_masks, 
                                 centroid_list_zyx = centroids_train, 
                                 boss_config = None, # None for public dataset
                                 px_radius_yx = [config.INPUT_IMAGE_WIDTH//2,config.INPUT_IMAGE_WIDTH//2],
                                 zshift=zshift, verbose=verbose)
    train_set = DataLoader(trainDS, shuffle=True,
            batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
            num_workers=num_workers) ##os.cpu_count())

    print('Data Loaded Successfully!')

    # Defining the model, optimizer and loss function
#    unet = UNET(in_channels=3, classes=19).to(DEVICE).train()
    unet = UNET(in_channels=3, classes=config.NUM_CLASSES).to(DEVICE).train()
    optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=255) 

    # Loading a previous stored model from MODEL_PATH variable
    if LOAD_MODEL == True:
        checkpoint = torch.load(MODEL_PATH)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']+1
        LOSS_VALS = checkpoint['loss_values']
        print("Model successfully loaded!")    

    #Training the model for every epoch. 
    for e in tqdm(range(epoch, EPOCHS)):
        print(f'Epoch: {e}')
        loss_val = train_function(train_set, unet, optimizer, loss_function, DEVICE)
        if verbose:
            print('after loss function')
        print('Val loss:', loss_val)
        LOSS_VALS.append(loss_val) 
        torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e,
            'loss_values': LOSS_VALS
        }, MODEL_PATH)
        print("Epoch completed and model successfully saved!")


if __name__ == '__main__':
    main()