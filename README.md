# Siamese Networks
An implementation of Contrastive Loss in PyTorch using Siamese Networks

## Dataset Used
[Link to Dataset](https://www.kaggle.com/kasikrit/att-database-of-faces)<br>
Out of the 40 faces available within the dataset, 32 were used for training the model, 4 for validation, and the remaining 4 were treated as "unseen data" for one shot classification.

## File descriptions

- ```split.py``` - Script to segregate classes into train-val-test
- ```loss.py``` - Class definition of Contrastive Loss
- ```dataset.py``` - Class definition of dataset for Contrastive Loss.
- ```model.py``` - Class definition of SiameseNetwork Model
- ```utils.py``` - Helper functions
- ```main.ipynb``` - Main script to proceed with training.


## Dependencies

- PyTorch 1.4.0
- Python 3.7.6
