"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import, print_function

import numpy as np
from keras.preprocessing import sequence


class DataSet(object):
    _MAX_FAKE_SENTENCE_LEN = 50

    def __init__(self, instances, labels, sequences, fake_data=False):
        if fake_data:
            self._num_examples = 1000
        else:
            assert len(labels) == 0 or instances.shape[0] == labels.shape[0], (
                    "images.shape: %s labels.shape: %s" % (instances.shape,
                                                           labels.shape))
            self._num_examples = instances.shape[0]

        self._instances = np.array(instances, dtype=np.int32)
        self._labels = np.array(labels, dtype=np.int32)
        self._sequences = np.array(sequences, dtype=np.int32)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def instances(self):
        return self._instances

    @property
    def labels(self):
        return self._labels

    @property
    def labels(self):
        return self._sequences
 
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_instance = [0 for _ in range(self._MAX_FAKE_SENTENCE_LEN)]
            fake_label = 0
            return ([fake_instance for _ in range(batch_size)],
                    [fake_label for _ in range(batch_size)])
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._instances = self._instances[perm]
            if len(self._labels) > 0:
                self._labels = self._labels[perm]
                self._sequences = self._sequences[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples, "%d - %d" % (batch_size, self._num_examples)
        end = self._index_in_epoch

        if len(self._labels) > 0:
            return self._instances[start:end], self._labels[start:end], self._sequences[start:end]
        else:
            return self._instances[start:end], None


class SemiDataSet(object):
    def __init__(self, instances, labels, sequences, n_labeled, n_classes):
        self.n_labeled = n_labeled

        # Unlabled DataSet
        self.unlabeled_ds = DataSet(instances, [], [])
        self.num_examples = self.unlabeled_ds.num_examples

        # Labeled DataSet
        indices = np.arange(self.num_examples)
        shuffled_indices = np.random.permutation(indices)
        instances = instances[shuffled_indices]
        labels = labels[shuffled_indices]
        sequences = sequences[shuffled_indices]
        y = np.array([lbl[-1] for lbl in labels])
        n_from_each_class = n_labeled // n_classes
        i_labeled = []
        for c in range(n_classes):
            i = indices[y == c][:n_from_each_class]
            i_labeled += list(i)
        l_instances = instances[i_labeled]
        l_labels = labels[i_labeled]
        l_sequences = sequences[i_labeled]
        self.labeled_ds = DataSet(l_instances, l_labels, l_sequences)

    def next_batch(self, batch_size):
        unlabeled_instances, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_instances, labels, sequences = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_instances, labels, sequences = self.labeled_ds.next_batch(batch_size)
        return labeled_instances, labels, sequences, unlabeled_instances


def read_data_sets(data_path, n_classes, n_labeled=100, fake_data=False):
    class DataSets(object):
        pass

    data_sets = DataSets()

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets

    train_data = np.load("%s/train.pkl" % data_path)
    train_instances = train_data["x"]
    train_labels = train_data["y"]
    train_sequences = np.array(train_data["s"], dtype=np.int32)
    data_sets.train = SemiDataSet(train_instances, train_labels, train_sequences, n_labeled, n_classes)

    validation_data = np.load("%s/validation.pkl" % data_path)
    validation_instances = validation_data["x"]
    validation_labels = validation_data["y"]
    validation_sequences = np.array(validation_data["s"], dtype=np.int32)
    data_sets.validation = DataSet(validation_instances, validation_labels, validation_sequences)

    test_data = np.load("%s/test.pkl" % data_path)
    test_instances = test_data["x"]
    test_labels = test_data["y"]
    test_sequences = np.array(test_data["s"], dtype=np.int32)
    data_sets.test = DataSet(test_instances, test_labels, test_sequences)

    return data_sets
