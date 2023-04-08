import collections
import numpy as np

DataSets = collections.namedtuple('DataSets', ['train', 'test'])

class DataSet(object):
    def __init__(self, images, features, labels, sample='histogram'):
        self.sample = sample
        if sample == 'histogram':
            print('images 0 shape', images[0].shape)
            print('labels 0 shape', labels[0].shape)
            print('labels 0', labels[0])
            print('labels 1', labels[1])
            assert images[0].shape[0] == labels[0].shape[0], (
                'images.shape: %s labels[0].shape %s' % (images[0].shape, labels[0].shape)
            )
            assert images[1].shape[0] == labels[0].shape[0], (
                'images.shape: %s labels[0].shape %s' % (images[1].shape, labels[0].shape)
            )
        else:
            assert images.shape[0] == labels[0].shape[0], (
                'images.shape: %s labels[0].shape %s' % (images.shape, labels[0].shape)
            )
        assert features.shape[0] == labels[0].shape[0], (
            'images.shape: %s labels[0].shape %s' % (features.shape, labels[0].shape)
        )
        assert labels[0].shape[0] == labels[1].shape[0], (
            'labels[0]: %s labels[1].shape %s' % (labels[0].shape, labels[1].shape)
        )
        self._num_examples = labels[0].shape[0]
        if sample == 'histogram':
            self._images_dim0 = images[0]
            self._images_dim1 = images[1]
        else:
            self._images = images
        self._features = features
        self._labels0 = labels[0]
        self._labels1 = labels[1]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images_dim0(self):
        return self._images_dim0
    
    @property
    def images_dim1(self):
        return self._images_dim1
    
    @property
    def images(self):
        return self._images

    @property
    def features(self):
        return self._features

    @property
    def labels0(self):
        return self._labels0

    @property
    def labels1(self):
        return self._labels1

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            if self.sample == 'histogram':
                self._images_dim0 = self.images_dim0[perm0]
                self._images_dim1 = self.images_dim1[perm0]
            else:
                self._images = self.images[perm0]
            self._features = self.features[perm0]
            self._labels0 = self.labels0[perm0]
            self._labels1 = self.labels1[perm0]
        # next epoch
        if start + batch_size > self._num_examples:
            # finished epoch
            self._epochs_completed += 1
            # rest examples in this epoch
            rest_num_examples = self._num_examples - start
            if self.sample == 'histogram':
                images_dim0_rest_part = self._images_dim0[start:self._num_examples]
                images_dim1_rest_part = self._images_dim1[start:self._num_examples]
            else:
                images_rest_part = self._images[start:self._num_examples]
            features_rest_part = self._features[start:self._num_examples]
            labels0_rest_part = self._labels0[start:self._num_examples]
            labels1_rest_part = self._labels1[start:self._num_examples]
            # shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                if self.sample == 'histogram':
                    self._images_dim0 = self.images_dim0[perm]
                    self._images_dim1 = self.images_dim1[perm]
                else:
                    self._images = self.images[perm]
                self._features = self.features[perm]
                self._labels0 = self.labels0[perm]
                self._labels1 = self.labels1[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            if self.sample == 'histogram':
                images_dim0_new_part = self._images_dim0[start:end]
                images_dim1_new_part = self._images_dim1[start:end]
            else:
                images_new_part = self._images[start:end]
            features_new_part = self._features[start:end]
            labels0_new_part = self._labels0[start:end]
            labels1_new_part = self._labels1[start:end]
            if self.sample == 'histogram':
                return (
                    np.concatenate((images_dim0_rest_part, images_dim0_new_part), axis=0), 
                    np.concatenate((images_dim1_rest_part, images_dim1_new_part), axis=0), 
                    np.concatenate((features_rest_part, features_new_part), axis=0), 
                    np.concatenate((labels0_rest_part, labels0_new_part), axis=0), 
                    np.concatenate((labels1_rest_part, labels1_new_part), axis=0)
                )
            else:
                return (
                    np.concatenate((images_rest_part, images_new_part), axis=0),
                    np.concatenate((features_rest_part, features_new_part), axis=0),
                    np.concatenate((labels0_rest_part, labels0_new_part), axis=0), 
                    np.concatenate((labels1_rest_part, labels1_new_part), axis=0)
                )
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            if self.sample == 'histogram':
                return (
                    self._images_dim0[start:end], 
                    self._images_dim1[start:end], 
                    self._features[start:end], 
                    self._labels0[start:end], 
                    self._labels1[start:end]
                )
            else:
                return (
                    self._images[start:end],
                    self._features[start:end], 
                    self._labels0[start:end], 
                    self._labels1[start:end]
                )
