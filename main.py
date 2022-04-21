import cv2
import numpy as np
import tqdm as t
import os
from matplotlib import pyplot as plt

def getImagesFromSpecies(path, _class):
    "Function that Return a list of images from a species"
    images = []
    for _img in t.tqdm(os.listdir(path + _class)):
        img = cv2.imread(path + _class + '/' + _img)
        images.append(img)
    return images

def getImages(path, species):
    "Function that Return a list of images and a list of labels"
    images = []
    labels = []
    for _class in species:
        print('Getting images from class: ' + _class)
        images.append(getImagesFromSpecies(path, _class))
        labels.append(_class)
    return images, labels

def printImages(images, labels):
    "Function that print images"
    for i in range(len(images)):
        for j in range(len(images[i])):
            plt.imshow(images[i][j])
            plt.title(labels[i])
            plt.show()

def main():
    "Main function"
    path = './archive/train/'
    species = os.listdir(path)
    images, labels = getImages(path, species)
    printImages(images, labels)

main()
