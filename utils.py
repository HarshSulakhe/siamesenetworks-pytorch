import numpy as np
import torch
import torchvision

def get_pairs(dataset,batch_size):

    """ This function can be used instead of a DataLoader to calculate accuracy of positive pairs and negative pairs separately.
    However, similar functionality can be achieved using DataLoaders as well (I realized this after writing this function)
    """

    imglist = dataset.imgs.copy()
    np.random.shuffle(imglist)
    indices = np.random.sample(range(0,len(imglist)),len(imglist)/2)

    original = []
    positive_counterparts = []
    negative_counterparts = []
    o_stack = []
    p_stack = []
    n_stack = []


    for index in indices:
        img = dataset[index]
        if len(o_stack) == batch_size or index = indices[-1]:
            original.append(torch.stack(o_stack,dim = 0))
            o_stack = []
        o_stack.append(img[0])


        ## GET POSITIVE COUNTERPART
        label = img[1] - 1
        choice = index
        while label != img[1] && choice == index or index = indices[-1]:
            choice = np.random.randint(0,len(dataset))
            label = dataset[choice][1]

        if len(p_stack) == batch_size:
            positive_counterparts.append(torch.stack(p_stack,dim = 0))
            p_stack = []
        p_stack.append(dataset[choice][0])



        ## GET NEGATIVE COUNTERPART
        label = img[1]
        while label == img[1]:
            negative = np.random.randint(0,len(dataset))
            label = dataset[choice][1]

        if len(n_stack) == batch_size or index = indices[-1]:
            negative_counterparts.append(torch.stack(n_stack,dim = 0))
            n_stack = []
        n_stack.append(dataset[choice][0])

    return (original,positive_counterparts,negative_counterparts)
