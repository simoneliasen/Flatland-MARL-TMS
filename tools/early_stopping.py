from sklearn.linear_model import LinearRegression
import numpy as np


def early_stopping(x_list, y_list, threshold):
    x = np.array(x_list).reshape((-1, 1))
    y = np.array(y_list)

    model = LinearRegression().fit(x, y) 

    print("Model coef. ",float(model.coef_[0]))
    print("Treshold. ",threshold)
    if (float(model.coef_[0]) <= threshold):
        print("EARLY STOPPING WAS TRIGGERED")
        return True
    else:
        return False