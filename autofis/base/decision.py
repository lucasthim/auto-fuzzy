import numpy as np
from itertools import compress

class Decision:
    def __init__(self, aggregation_rules, percentage_of_classes):
        self.aggregation_rules = aggregation_rules.copy()
        self.percentage_of_classes = percentage_of_classes

    def predict(self,uX,t_norm):

        num_classes = len(self.aggregation_rules)
        y_estimation = np.zeros((uX.shape[0], num_classes))
        for i in range(num_classes):
            y_estimation[:, [i]] = self.predict_class(self.aggregation_rules[i], uX, t_norm)
        y_one_hot_estimation = self.decide_class(y_estimation)
        return y_one_hot_estimation


    def decide_class(self,y_estimation):

        """
        This function decide to which class belong each instance
        :return: One hot classification vector
        Example
           y_estimation : [0.80 0.15 0.05
                                    0.03 0.90 0.07
                                    0.20 0.40 0.40
                                    0.20 0.10 0.70]
           self.percentage_of_classes: [0.33 0.33 0.33]

           Output:
                    [1, 0, 0]
                    [0, 1, 0]
                    [0, 1, 0]
                    [0, 0, 1]
        """
        m = y_estimation

        repeated_max_values_rows = np.tile(m.max(1), (m.shape[1], 1))  # m.max(1) maximo de cada fila
        out = 1 * (m == repeated_max_values_rows.T)  # First binary decision (possibles ties)
        d_row = out.sum(axis=1)  # Instances with compatibility in more than 1 class (tie)
        d_row_max = np.any([(d_row > 1), (d_row == 0)], axis=0)
        index_ties = list(compress(range(out.shape[0]), d_row_max.tolist()))  # index de las instancias com empate

        classes_number = len(self.percentage_of_classes)
        for j in index_ties:
            tiebreaker = classes_number * [0]
            freq_classes_in_tie = [self.percentage_of_classes[i] * out[j, i] for i in range(classes_number)]
            # Assigning to the class with more patterns (instances)
            tiebreaker[freq_classes_in_tie.index(max(freq_classes_in_tie))] = 1
            out[j, :] = tiebreaker

        return out


    def predict_class(self,class_aggregation_rules, uX, tnorm): 
        premises = class_aggregation_rules[0]
        aggregation_method = class_aggregation_rules[2]

        num_col = len(premises)
        new_u = np.zeros((uX.shape[0], num_col))
        for i in range(num_col):
            temp = uX[:, premises[i]]
            new_u[:, i] = np.prod(temp, axis=1) if tnorm == 'prod' else np.min(temp, axis=1);

        if aggregation_method == 'MQR' or aggregation_method == 'intMQR' or aggregation_method == 'lin_regre':
            weights = class_aggregation_rules[1]
            return np.dot(new_u, weights)

        elif aggregation_method == 'max':
            aux = new_u.max(axis=1)
            return aux.reshape(len(aux), 1)

        else:  # 'logistic regression'
            log_regression = class_aggregation_rules[1]
            aux = log_regression.predict(new_u)
            return aux.reshape(len(aux), 1)



def main():
        print ('Module 5 <<Decision>>')


if __name__ == '__main__':
    main()