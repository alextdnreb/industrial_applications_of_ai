import tensorflow as tf
import matplotlib.pyplot as plt


def plot_feature_maps_for_convs(num_convs, num_filters_first, sample, model):
    for i in range(num_convs):
        visualization_model = tf.keras.models.Model(
            inputs=model.input, outputs=model.layers[i * 2].output
        )

        feature_maps = visualization_model.predict(sample, verbose=0)

        plt.figure(figsize=(10, 10))
        num_filters = num_filters_first * (i + 1)
        for j in range(num_filters):
            plt.subplot(int(num_filters / 4), int(num_filters / 2), j + 1)
            plt.imshow(feature_maps[0, :, :, j], cmap="gray")
            plt.suptitle(f"Feature maps for layer {i+1}")
            plt.axis("off")
            plt.tight_layout()
        plt.show()


def encode_labels(encoder, y_train, y_validation, y_test):
    y_train_encoded = encoder.fit_transform(y_train)
    y_validation_encoded = encoder.transform(y_validation)
    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_validation_encoded, y_test_encoded


# based on https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
import tensorflow as tf
import numpy as np
import math


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]

        return np.array(
            [
                tf.keras.utils.img_to_array(tf.keras.utils.load_img(path)) / 255
                for path in batch_x
            ]
        ), np.array(batch_y)
