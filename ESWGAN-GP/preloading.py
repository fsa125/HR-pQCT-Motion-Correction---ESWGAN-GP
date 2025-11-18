#################### Modification ###########################  
from utils import *


#pre-loading the data to Memory

# makes a stack of images, where motion-corrupoted image and ground truth are paired, then run the model
# take pairs, put into the RAM, stack them, take random sample from the stack
class Store_train_data():

    def __init__(self, train_dataloader, directory_name):
            


        print('pre-loading the data to Memory')
        stacked_all = []

        for imgs in tqdm(train_dataloader):
            #print(data)

            #X_s, Y_s = data['lr'], data['hr']                                  ##The efficient training idea is taken from one of my old projects
            imgs_lr = imgs['lr']
            #print(imgs_lr.shape)
            imgs_hr = imgs['hr']                                                                              ##https://github.com/fsa125/Two-Stage-Network-Super-Resolution
            #X_s = data['img_Corr']
            stacked_dataset = TensorDataset(imgs_lr, imgs_hr)

            stacked_all.append(stacked_dataset)

        torch.save(stacked_all, directory_name)

        #return stacked_all
        return None