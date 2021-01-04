import numpy as np
import lif_model

if __name__ == '__main__':

    # TRAINING
    # bias, x1, x2

    input_1 = lif_model.lif()
    input_1.threshold = 0.5

    input_2 = lif_model.lif()
    input_2.threshold = 0.5

    inputs = np.array([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    y_expected = np.array([-1, -1, -1, 1])
    weights = np.array([0, 0, 0])

    count = 0
    for i in inputs:
        # print(type(i))
        # print(i)
        input_1.I = i[1]
        input_2.I = i[2]
        a = -1
        b = -1

        # lif_model.plot_potential_decay(input_1)
        # lif_model.plot_potential_decay(input_2)

        if lif_model.count_spikes(input_1) > 0:
            a = 1
        if lif_model.count_spikes(input_2) > 0:
            b = 1

        i = np.array([i[0], a, b])

        weights = weights + i.dot(y_expected[count])
        # print(weights)
        count += 1

    print("Weights: ", weights)
    # TESTING
    x = 1
    y = 1

    bias = 1
    output = 0

    input_1.I = x
    input_2.I = y

    #print(lif_model.count_spikes(input_1))

    if lif_model.count_spikes(input_1) > 0:
        a = 1
    else:
        a = -1

    if lif_model.count_spikes(input_2) > 0:
        b = 1
    else:
        b = -1

    test_input = np.transpose(np.array([bias, a, b]))
    test_output = test_input.dot(weights)
    if test_output < 0:
        output = -1
    else:
        output = 1
    print("Output : ", output)
