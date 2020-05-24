__author__ = 'ERRISON'

import numpy as np
import itertools


def div_indices(c_values, index):
    if index == 0:  # Disagreement Measure
        diversity_value = (c_values[1] + c_values[2]) / np.sum(c_values)
    elif index == 1:  # Alternate Double-Fault Measure (1-df)
        diversity_value = 1 - (c_values[3] / np.sum(c_values))
    return diversity_value


def wad_calc(labels, decisions, acc_list, params):  # deprecated
    decisions_qtd = len(decisions)
    decisions_len = decisions[0].shape
    mat_div = np.zeros((decisions_qtd, decisions_qtd))

    for i in range(decisions_qtd):
        vec_qtd_clf = list(range(decisions_qtd))
        vec_qtd_clf.remove(i)
        n11, n10, n01, n00 = 0, 0, 0, 0

        for (j, k) in itertools.product(vec_qtd_clf, range(decisions_len[0])):

            if np.array_equal(decisions[i][k], decisions[j][k]) is True:
                if np.array_equal(decisions[i][k], labels[k]) is True:
                    n11 += 1
                elif np.array_equal(decisions[i][k], labels[k]) is False:
                    n00 += 1

            elif np.array_equal(decisions[i][k], decisions[j][k]) is False:
                if np.array_equal(decisions[i][k], labels[k]) is True:
                    n10 += 1
                elif np.array_equal(decisions[j][k], labels[k]) is True:
                    n01 += 1

            n_values = [n11, n10, n01, n00]
            mat_div[i, j] = div_indices(n_values, params[2])

    div_clf = mat_div.sum(axis=1) / (decisions_qtd - 1)
    acc_vec = np.array(acc_list)/100
    wad_vec = np.divide(np.multiply(acc_vec, div_clf), (params[0] * acc_vec) + (params[1] * div_clf))
    return wad_vec


def wad_calc_v2(labels, decisions, acc_list, params):
    decisions_qtd = len(decisions)
    mat_div = np.zeros((decisions_qtd, decisions_qtd))

    for i in range(decisions_qtd):
        vec_qtd_clf = list(range(decisions_qtd))
        vec_qtd_clf.remove(i)
        n11, n10, n01, n00 = 0, 0, 0, 0

        for j in vec_qtd_clf:
            equ_dec = np.where(np.all(np.equal(decisions[i], decisions[j]), axis=1) == 1)
            dif_dec = np.where(np.all(np.equal(decisions[i], decisions[j]), axis=1) == 0)

            n11 = np.sum(np.all(np.equal(decisions[i][equ_dec], labels[equ_dec]), axis=1))
            n00 = np.sum(np.all(np.not_equal(decisions[i][equ_dec], labels[equ_dec]), axis=1))
            n10 = np.sum(np.all(np.equal(decisions[i][dif_dec], labels[dif_dec]), axis=1))
            n01 = np.sum(np.all(np.equal(decisions[j][dif_dec], labels[dif_dec]), axis=1))

            n_values = [n11, n10, n01, n00]
            mat_div[i, j] = div_indices(n_values, params[1])

    div_clf = mat_div.sum(axis=1) / (decisions_qtd - 1)
    acc_vec = np.array(acc_list)/100
    wad_vec = np.divide(np.multiply(acc_vec, div_clf), ((1-params[0]) * acc_vec) + (params[0] * div_clf))
    return wad_vec


def wad_calc_v2_inc(labels, decisions, acc_list, params, i_model1):
    decisions_qtd = len(decisions)
    mat_div = [0.01] * decisions_qtd

    i = i_model1
    vec_qtd_clf = list(range(decisions_qtd))
    vec_qtd_clf.remove(i)

    for j in vec_qtd_clf:
        if isinstance(decisions[j], (list, tuple, np.ndarray)):
            equ_dec = np.where(np.all(np.equal(decisions[i], decisions[j]), axis=1) == 1)
            dif_dec = np.where(np.all(np.equal(decisions[i], decisions[j]), axis=1) == 0)

            n11 = np.sum(np.all(np.equal(decisions[i][equ_dec], labels[equ_dec]), axis=1))
            n00 = np.sum(np.all(np.not_equal(decisions[i][equ_dec], labels[equ_dec]), axis=1))
            n10 = np.sum(np.all(np.equal(decisions[i][dif_dec], labels[dif_dec]), axis=1))
            n01 = np.sum(np.all(np.equal(decisions[j][dif_dec], labels[dif_dec]), axis=1))

            mat_div[j] = div_indices([n11, n10, n01, n00], params[1])

    div_clf = np.array(mat_div)
    acc_vec = np.array(acc_list)/100
    wad_vec = np.divide(np.multiply(acc_vec, div_clf), ((1-params[0]) * acc_vec) + (params[0] * div_clf))
    return wad_vec


def reg_calc(labels, decisions, acc_list, params):
    decisions_qtd = len(decisions)
    mat_div = np.zeros((decisions_qtd, decisions_qtd))

    for i in range(decisions_qtd):
        vec_qtd_clf = list(range(decisions_qtd))
        vec_qtd_clf.remove(i)
        n11, n10, n01, n00 = 0, 0, 0, 0

        for j in vec_qtd_clf:
            equ_dec = np.where(np.all(np.equal(decisions[i], decisions[j]), axis=1) == 1)
            dif_dec = np.where(np.all(np.equal(decisions[i], decisions[j]), axis=1) == 0)

            n11 = np.sum(np.all(np.equal(decisions[i][equ_dec], labels[equ_dec]), axis=1))
            n00 = np.sum(np.all(np.not_equal(decisions[i][equ_dec], labels[equ_dec]), axis=1))
            n10 = np.sum(np.all(np.equal(decisions[i][dif_dec], labels[dif_dec]), axis=1))
            n01 = np.sum(np.all(np.equal(decisions[j][dif_dec], labels[dif_dec]), axis=1))

            n_values = [n11, n10, n01, n00]
            mat_div[i, j] = div_indices(n_values, params[1])

    div_clf = mat_div.sum(axis=1) / (decisions_qtd - 1)
    acc_vec = np.array(acc_list)/100
    reg_vec = acc_vec + (params[0] * div_clf)
    return reg_vec







