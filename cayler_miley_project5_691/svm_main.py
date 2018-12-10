from svm import *
from helpers import *

if __name__ == '__main__':
    plt.style.use('ggplot')
    # for i in range(4):
    #     data = generate_training_data_binary(i+1)
    #     w, b, S = svm_train_brute(data)
    #     plot_decision_boundary_binary(data, w, b, save=False, title=str(i+1))
    #     print(svm_test_brute(w, b, [2, 4]))

    for i in range(2):
        [data, Y] = generate_training_data_multi(i+1)
        # plot_training_data_multi(data)
        W, B = svm_train_multiclass(data)
        plot_decision_boundary_multi(data, W, B, save=True, title=i+1)
