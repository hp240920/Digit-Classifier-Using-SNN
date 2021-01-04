from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing

from lif_model import lif
from lif_model import count_spikes


def load_digit():
    digit = load_digits()
    print(digit.data.shape)
    return digit


if __name__ == '__main__':
    # plt.gray()
    digits = load_digit()
    print(len(digits.data[0]))

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    X_train, X_temp, y_train, y_temp = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False)

    X_validate, X_test, y_validate, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    # fit_value = np.mean(X_train)

    max_spike_rate = 25
    weights = np.zeros([10, 64])
    count = 0
    neuron = lif()
    for pixel_arr in X_train:
        print(count)
        pixel_count = 0
        for pixel in pixel_arr:
            pre_syn_current = pixel
            neuron.I = pre_syn_current
            pre_rate = count_spikes(neuron)/max_spike_rate  # between 0 and 1

            for i in range(10):
                current = weights[i].dot(np.transpose(pixel_arr))
                neuron.I = current
                post_rate = count_spikes(neuron)/max_spike_rate  # between 0 and 1

                if i == y_train[count]:
                    weights[i][pixel_count] += 0.05 * pre_rate
                else:
                    weights[i][pixel_count] -= 0.05 * pre_rate
            pixel_count += 1
        count += 1

        # Validating
        v_count = 0
        correct = 0
        incorrect = 0

        for x in X_validate:
            ans = x.dot(np.transpose(weights))
            output = np.argmax(ans)
            if output == y_validate[v_count]:
                correct += 1
            else:
                incorrect += 1
            v_count += 1
        print("Correct :", correct)
        print("Incorrect :", incorrect)
        print("Accuracy (Validate):", 100 * (correct / (correct + incorrect)))


    count = 0
    correct = 0
    incorrect = 0
    for x in X_train:
        ans = x.dot(np.transpose(weights))
        # temp = ans.argsort()[-2]
        # temp1 = ans.argsort()[-3]
        output = np.argmax(ans)
        if output == y_train[count]:
            correct += 1
        else:
            incorrect += 1
        count += 1
    print("Correct :", correct)
    print("Incorrect :", incorrect)
    print("Accuracy (Training):", 100 * (correct / (correct + incorrect)))

