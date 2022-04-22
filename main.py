import cv2
from matplotlib.image import imread
import numpy as np
import tqdm as t
import os
from matplotlib import pyplot as plt
import operator

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

def bird_classification(imagesTrain, labelsTrain, imagesTest, labelsTest, imagesValid, labelsValid):
    "Function that train a classifier and test it"
    # Train
    print('Training...')
    classifier = cv2.ml.SVM_create()
    classifier.setKernel(cv2.ml.SVM_LINEAR)
    classifier.setType(cv2.ml.SVM_C_SVC)
    classifier.setC(2.67)
    classifier.setGamma(5.383)
    classifier.train(np.array(imagesTrain), cv2.ml.ROW_SAMPLE, np.array(labelsTrain))
    # Test
    print('Testing...')
    correct = 0
    for i in range(len(imagesTest)):
        for j in range(len(imagesTest[i])):
            _, result = classifier.predict(np.array(imagesTest[i][j]))
            if result[0][0] == labelsTest[i]:
                correct += 1
    print('Accuracy: ' + str(correct / (len(imagesTest) * len(imagesTest[0]))))
    # Valid
    print('Validating...')
    correct = 0
    for i in range(len(imagesValid)):
        for j in range(len(imagesValid[i])):
            _, result = classifier.predict(np.array(imagesValid[i][j]))
            if result[0][0] == labelsValid[i]:
                correct += 1
    print('Accuracy: ' + str(correct / (len(imagesValid) * len(imagesValid[0]))))

def grabCutImages(images, labels):
    "Function that apply grabcut to images"
    imagesGrabcut = []
    for i in range(len(images)):
        for j in range(len(images[i])):
            img = images[i][j]
            mask = np.zeros(img.shape[:2], np.uint8)
            bgdModel = np.zeros((1,65), np.float64)
            fgdModel = np.zeros((1,65), np.float64)
            rect = (50,50,450,290)
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            img = img*mask2[:,:,np.newaxis]
            imagesGrabcut.append(img)
    return imagesGrabcut

def main():
    "Bird Classification"
    pathTrain = './archive/train/'
    pathTest = './archive/test/'
    pathValid = './archive/valid/'
    train = os.listdir(pathTrain)
    test = os.listdir(pathTest)
    imagesTrain, labelsTrain = getImages(pathTrain, train)
    # imagesTest, labelsTest = getImages(pathTest, test)
    # imagesValid, labelsValid = getImages(pathValid, test)
    imagesGrabcut = grabCutImages(imagesTrain, labelsTrain)
    printImages(imagesGrabcut, labelsTrain)

if __name__ == '__main__':
    main()