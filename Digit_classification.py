from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


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

    weights = np.zeros([10, 65])
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    fit_value = np.mean(X_train)

    X_validate = min_max_scaler.fit_transform(X_validate)
    ones = np.ones(len(X_validate))
    X_validate = np.insert(X_validate, 0, ones, axis=1)

    X_test = min_max_scaler.fit_transform(X_test)
    ones = np.ones(len(X_test))
    X_test = np.insert(X_test, 0, ones, axis=1)

    # print(fit_value)
    count = 0
    ones = np.ones(len(X_train))
    X_train = np.insert(X_train, 0, ones, axis=1)

    for k in range(15):
        count = 0
        for x in X_train:
            expected_y = y_train[count]
            count += 1
            x = x - fit_value

            ans = x.dot(np.transpose(weights))
            output = np.argmax(ans)
            if output == expected_y:
                continue
            for i in range(10):
                a = 0
                if expected_y == i:
                    a = 1
                weights[i][1:] = weights[i][1:] + x[1:]*(a-0.1)
                weights[i][0] = weights[i][0] + a

        # print("Weights: ", weights)

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

    exit(0)
        # Train testing
    count = 0
    correct = 0
    incorrect = 0
    for x in X_train:
        ans = x.dot(np.transpose(weights))
        temp = ans.argsort()[-2]
        temp1 = ans.argsort()[-3]
        output = np.argmax(ans)
        if output == y_train[count] or temp == y_train[count] or temp1 == y_train[count]:
            correct += 1
        else:
            incorrect += 1
        count += 1
    print("Correct :", correct)
    print("Incorrect :", incorrect)
    print("Accuracy (Training):", 100 * (correct / (correct + incorrect)))
    # print(output , y_train[125])

    # Validate testing
    count = 0
    correct = 0
    incorrect = 0

    for x in X_validate:
        ans = x.dot(np.transpose(weights))
        output = np.argmax(ans)
        if output == y_validate[count]:
            correct += 1
        else:
            incorrect += 1
        count += 1
    print("Correct :", correct)
    print("Incorrect :", incorrect)
    print("Accuracy (Validate):", 100 * (correct / (correct + incorrect)))

    # Final testing
    count = 0
    correct = 0
    incorrect = 0

    for x in X_test:
        ans = x.dot(np.transpose(weights))
        output = np.argmax(ans)
        if output == y_test[count]:
            correct += 1
        else:
            incorrect += 1
        count += 1
    print("Correct :", correct)
    print("Incorrect :", incorrect)
    print("Accuracy (Test):", 100 * (correct / (correct + incorrect)))

    # plt.matshow(digits.images[100])
    print(len(digits.target))
    # plt.show()
