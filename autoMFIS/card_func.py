import numpy as np


def card_func(values,min_act):
    mean_activation = np.mean(values[values>0],axis=0)

    if mean_activation > min_act:
        activation = True
    else:
        activation = False

    return activation, mean_activation



def activation_func(values, min_freq):
    freq = np.sum(values>0)/len(values)

    if freq > min_freq:
        activation = True
    else:
        activation = False

    return activation, freq

def card_hybrid_func(values, min_val):
    mean_activation = np.mean(values[values>0],axis=0)

    freq_rel = np.sum(values>0)/len(values)

    hybrid_card = mean_activation * freq_rel

    if hybrid_card > min_val:
        activation = True
    else:
        activation = False

    return activation, hybrid_card
