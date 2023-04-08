import os
import sys
import random
import numpy as np

def density(input_file, numformats):
    data = np.load(input_file)
    images = data['images']
    features = data['features']
    labels = data['labels']
    print('images', images.shape)
    print('features', features.shape)
    print('labels', labels.shape)

    images_size = labels.shape[0]
    numft = features.shape[1]
    dimensions = images.shape[1]
    resolution = images.shape[2]

    tnslist = list(range(0, images_size))

    testlist = random.sample(tnslist, int(images_size / 5))
    trainlist = tnslist
    for i in range(len(testlist)):
        trainlist.remove(testlist[i])

    print('trainlist', len(trainlist))
    print('testlist', len(testlist))

    train_images = np.zeros((len(trainlist), resolution, resolution), dtype='float32')
    test_images  = np.zeros((len(testlist), resolution, resolution), dtype='float32')
    train_features = np.zeros((len(trainlist), numft), dtype='float32')
    test_features = np.zeros((len(testlist), numft), dtype='float32')
    train_labels = np.zeros((len(trainlist), numformats), dtype='int32')
    test_labels  = np.zeros((len(testlist), numformats), dtype='int32')

    for i in range(len(trainlist)):
        train_images[i, :, :] = images[trainlist[i], :, :]
        # print('train_images[i]\n', train_images[i, :, :])
    for i in range(len(testlist)):
        test_images[i, :, :] = images[testlist[i], :, :]
            
    for i in range(len(trainlist)):
        train_labels[i][labels[trainlist[i]]] = 1
        for j in range(numft):
            train_features[i][j] = features[trainlist[i]][j] / np.absolute(features[:, j]).max()
    # print('train_features', train_features)
    for i in range(len(testlist)):
        test_labels[i][labels[testlist[i]]] = 1
        for j in range(numft):
            test_features[i][j] = features[testlist[i]][j] / np.absolute(features[:, j]).max()

    np.savez('data/npz/train_density.npz', images=train_images, features=train_features, labels=train_labels)
    np.savez('data/npz/test_density.npz', images=test_images, features=test_features, labels=test_labels)
    print('density finished!!!')

def histogram(input_file, numformats):
    data = np.load(input_file)
    images_dim0 = data['images_dim0']
    images_dim1 = data['images_dim1']
    features = data['features']
    labels = data['labels']
    print('numformats', numformats)
    print('images_dim0', images_dim0.shape)
    print('images_dim1', images_dim1.shape)
    print('features', features.shape)
    print('labels', labels.shape)
    
    print('labels 0', labels[0])
    print('labels 1', labels[1])

    images_size = labels[0].shape[0]
    numft = features.shape[1]
    dimensions = images_dim1.shape[1]
    resolution = images_dim1.shape[2]

    tnslist = list(range(0, images_size))

    testlist = random.sample(tnslist, int(images_size / 5))
    trainlist = tnslist
    for i in range(len(testlist)):
        trainlist.remove(testlist[i])

    print('trainlist', len(trainlist))
    print('testlist', len(testlist))

    train_images_dim0 = np.zeros((len(trainlist), resolution, resolution), dtype='float32')
    test_images_dim0  = np.zeros((len(testlist), resolution, resolution), dtype='float32')
    train_images_dim1 = np.zeros((len(trainlist), resolution, resolution), dtype='float32')
    test_images_dim1  = np.zeros((len(testlist), resolution, resolution), dtype='float32')
    train_features = np.zeros((len(trainlist), numft), dtype='float32')
    test_features = np.zeros((len(testlist), numft), dtype='float32')
    train_labels = np.zeros((2, len(trainlist), numformats), dtype='int32')
    test_labels  = np.zeros((2, len(testlist), numformats), dtype='int32')

    for i in range(len(trainlist)):
        if np.absolute(images_dim0[trainlist[i], :, :]).max() > 0:
            train_images_dim0[i, :, :] = images_dim0[trainlist[i], :, :] / np.absolute(images_dim0[trainlist[i], :, :]).max()
        if np.absolute(images_dim1[trainlist[i], :, :]).max() > 0:
            train_images_dim1[i, :, :] = images_dim1[trainlist[i], :, :] / np.absolute(images_dim1[trainlist[i], :, :]).max()
        
    for i in range(len(testlist)):
        if np.absolute(images_dim0[testlist[i], :, :]).max() > 0:
            test_images_dim0[i, :, :] = images_dim0[testlist[i], :, :] / np.absolute(images_dim0[testlist[i], :, :]).max()
        if np.absolute(images_dim1[testlist[i], :, :]).max() > 0:
            test_images_dim1[i, :, :] = images_dim1[testlist[i], :, :] / np.absolute(images_dim1[testlist[i], :, :]).max()
            
    for i in range(len(trainlist)):
        train_labels[0][i][int(labels[0][trainlist[i]])] = 1
        train_labels[1][i][int(labels[1][trainlist[i]])] = 1
        for j in range(numft):
            if np.absolute(features[:, j]).max() > 0:
                train_features[i][j] = features[trainlist[i]][j] / np.absolute(features[:, j]).max()
    for i in range(len(testlist)):
        test_labels[0][i][int(labels[0][testlist[i]])] = 1
        test_labels[1][i][int(labels[1][testlist[i]])] = 1
        for j in range(numft):
            if np.absolute(features[:, j]).max() > 0:
                test_features[i][j] = features[testlist[i]][j] / np.absolute(features[:, j]).max()

    np.savez('data/npz/train_histogram.npz', images_dim1=train_images_dim1, images_dim0=train_images_dim0, features=train_features, labels=train_labels)
    np.savez('data/npz/test_histogram.npz', images_dim1=test_images_dim1, images_dim0=test_images_dim0, features=test_features, labels=test_labels)
    print('finished!!!')

def sptfs(input_file, images_name, numformats):
    data = np.load(input_file)
    images_dim0 = data[images_name][:, 0, :, :]
    images_dim1 = data[images_name][:, 1, :, :]
    features = data['features']
    labels = data['labels']
    print('images_dim0', images_dim0.shape)
    print('images_dim1', images_dim1.shape)
    print('features', features.shape)
    print('labels', labels.shape)

    images_size = labels[0].shape[0]
    numft = features.shape[1]
    dimensions = images_dim1.shape[1]
    resolution = images_dim1.shape[2]

    tnslist = list(range(0, images_size))

    testlist = random.sample(tnslist, int(images_size / 5))
    trainlist = tnslist
    for i in range(len(testlist)):
        trainlist.remove(testlist[i])

    print('trainlist', len(trainlist))
    print('testlist', len(testlist))

    train_images_dim0 = np.zeros((len(trainlist), resolution, resolution), dtype='float32')
    test_images_dim0  = np.zeros((len(testlist), resolution, resolution), dtype='float32')
    train_images_dim1 = np.zeros((len(trainlist), resolution, resolution), dtype='float32')
    test_images_dim1  = np.zeros((len(testlist), resolution, resolution), dtype='float32')
    train_features = np.zeros((len(trainlist), numft), dtype='float32')
    test_features = np.zeros((len(testlist), numft), dtype='float32')
    train_labels = np.zeros((len(trainlist), numformats), dtype='int32')
    test_labels  = np.zeros((len(testlist), numformats), dtype='int32')

    for i in range(len(trainlist)):
        if np.absolute(images_dim0[trainlist[i], :, :]).max() > 0:
            train_images_dim0[i, :, :] = images_dim0[trainlist[i], :, :] / np.absolute(images_dim0[trainlist[i], :, :]).max()
        if np.absolute(images_dim1[trainlist[i], :, :]).max() > 0:
            train_images_dim1[i, :, :] = images_dim1[trainlist[i], :, :] / np.absolute(images_dim1[trainlist[i], :, :]).max()
        
    for i in range(len(testlist)):
        if np.absolute(images_dim0[testlist[i], :, :]).max() > 0:
            test_images_dim0[i, :, :] = images_dim0[testlist[i], :, :] / np.absolute(images_dim0[testlist[i], :, :]).max()
        if np.absolute(images_dim1[testlist[i], :, :]).max() > 0:
            test_images_dim1[i, :, :] = images_dim1[testlist[i], :, :] / np.absolute(images_dim1[testlist[i], :, :]).max()
            
    for i in range(len(trainlist)):
        train_labels[i][labels[trainlist[i]]] = 1
        for j in range(numft):
            if np.absolute(features[:, j]).max() > 0:
                train_features[i][j] = features[trainlist[i]][j] / np.absolute(features[:, j]).max()
    for i in range(len(testlist)):
        test_labels[i][labels[testlist[i]]] = 1
        for j in range(numft):
            if np.absolute(features[:, j]).max() > 0:
                test_features[i][j] = features[testlist[i]][j] / np.absolute(features[:, j]).max()

    np.savez('data/npz/train_' + images_name + '.npz', images_dim1=train_images_dim1, images_dim0=train_images_dim0, features=train_features, labels=train_labels)
    np.savez('data/npz/test_' + images_name + '.npz', images_dim1=test_images_dim1, images_dim0=test_images_dim0, features=test_features, labels=test_labels)
    print('finished!!!')


if __name__=='__main__':
    if sys.argv[1] == 'density':
        density(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'histogram':
        histogram(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'flat':
        sptfs(sys.argv[2], 'flatten_imgs', int(sys.argv[3]))
    elif sys.argv[1] == 'map':
        sptfs(sys.argv[2], 'map_imgs', int(sys.argv[3]))
