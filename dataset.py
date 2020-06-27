import torch
import torchvision
import numpy as np
from PIL import Image

class CUBContrast(torch.utils.data.Dataset):
# """
#     A custom class used to sample positive and negative pairs of images with equal probability, the outputs of which are
#     fed to a contrastive loss function.
# """
    def __init__(self,cub_dataset):
        ## cub_dataset IS AN ImageFolder DATASET,
        ## THIS FUNCTION MERELY COPIES ITS RELEVANT ATTRIBUTES
        self.classes = cub_dataset.classes
        self.imgs = cub_dataset.imgs
        self.transform = cub_dataset.transform

    def __getitem__(self,index):
        ## CHOOSE EITHER POSITIVE PAIR (0) OR NEGATIVE PAIR (1)
        self.target = np.random.randint(0,2)
        ## HERE THE FIRST IMAGE IS CHOSEN BY VIRTUE OF INDEX ITSELF
        img1,label1 = self.imgs[index]
        ## CREATE NEW LIST OF IMAGES TO AVOID RE-SELECTING ORIGINAL IMAGE
        new_imgs = list(set(self.imgs) - set(self.imgs[index]))
        length = len(new_imgs)
        # print(length)
        random = np.random.RandomState(42)
        if self.target == 1:
            ## GET NEGATIVE COUNTERPART
            label2 = label1
            while label2 == label1:
                choice = random.choice(length)
                img2,label2 = new_imgs[choice]
        else:
            ## GET POSITIVE COUNTERPART
            label2 = label1 + 1
            while label2 != label1:
                choice = random.choice(length)
                img2,label2 = new_imgs[choice]

        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1,img2,self.target)

    def __len__(self):
        return(len(self.imgs))
