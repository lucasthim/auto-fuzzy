__author__ = 'jparedes'

# -*- coding: utf-8 -*-
from numpy import array, tile, any
from itertools import compress


class Decision:
    def __init__(self, estimate_values, classes_info):
        self.aggregation_values = estimate_values.copy()
        self.decision_parameters = classes_info

    def dec_max_pert(self):
        """
        This function decide to which class belong each instance
        :return: Binary classification
        Example:
           self.aggregation_values [0.80 0.15 0.05
                                    0.03 0.90 0.07
                                    0.20 0.40 0.40
                                    0.20 0.10 0.70]
           self.decision_parameters:
                            C2: [0.33 0.33 0.33] frecuencia de cada clase:
           Output:
                                    [1, 0, 0]
                                    [0, 1, 0]
                                    [0, 1, 0]
                                    [0, 0, 1]
        """
        m = self.aggregation_values
        freq_classes = self.decision_parameters

        repeated_max_values_rows = tile(m.max(1), (m.shape[1], 1))  # m.max(1) maximo de cada fila
        out = 1 * (m == repeated_max_values_rows.T)  # First binary decision (possibles ties)
        d_row = out.sum(axis=1)  # Instances with compatibility in more than 1 class (tie)
        d_row_max = any([(d_row > 1), (d_row == 0)], axis=0)
        index_ties = list(compress(range(out.shape[0]), d_row_max.tolist()))  # index de las instancias com empate

        classes_number = len(freq_classes)
        for j in index_ties:
            tiebreaker = classes_number * [0]
            freq_classes_in_tie = [freq_classes[i] * out[j, i] for i in range(classes_number)]
            # Assigning to the class with more patterns (instances)
            tiebreaker[freq_classes_in_tie.index(max(freq_classes_in_tie))] = 1
            out[j, :] = tiebreaker

        return out


def main():
    '''
    P1 = d0.Lecture()
    #file_path = "/media/jorg/8E468C6E468C593B/Cursos/Proposta Disertacao/autoFIS_python/iris/iris.dat"
    file_path = 'E:/autoFIS_python/data' + '/' + 'iris.dat'
    dataset = P1.read_1file(file_path) # lista de 2 array numpy
    y = dataset[1]
    print type(y), y.shape
    print "===============================\n"
    '''
    M1 = array([[0.8, 0.15, 0.05],
                   [0.03, 0.9, 0.07],
                   [0.2, 0.4,  0.4],
                   [0.2, 0.1, 0.7]])
    # param = [[1, 0, 0], np.array([0.25, 0.4, 0.35])]
    param = [[0.25, 0.4, 0.35], [1, 0, 0]]
    obj = Decision(M1, param)
    print (obj.dec_max_pert())

if __name__ == '__main__':
    main()