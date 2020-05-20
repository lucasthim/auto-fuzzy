from autoFIS.lecture import Lecture
from autoFIS.fuzzification import Fuzzification
from autoFIS.formulation.formulation import Formulation
from autoFIS.association import Association
from autoFIS.aggregation import Aggregation
from autoFIS.decisions import Decisions
from autoFIS.evaluation import Evaluation


def main():
    print ("Testing autoFIS modules")
    filezip_name = "D:\\Jorg\Projects\\autoFIS\\test\\datas" + '\\' + 'saheart-10-fold_csv.zip'
    train_file = 'saheart-10-7tra.csv'
    test_file = 'saheart-10-7tst.csv'

    # Lecture
    L = Lecture()
    L.read_1cv(filezip_name, train_file, test_file)
    info = L.info_data()

    # Fuzzification
    X = info[0].copy()
    cat_bool = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    F1 = Fuzzification(X, cat_bool)
    F1.build_uX('tukey', 5)
    # F1.add_negation()

    # Split to train and test
    index_train = info[-1]
    ux_train = F1.uX[:index_train, :]
    ux_test = F1.uX[index_train:, :]
    cbin_train = info[1][:index_train, :]
    cbin_test = info[1][index_train:, :]

    # Formulation
    np_by_attr = F1.size_attributes
    p_by_attr = F1.premises_attributes
    F2 = Formulation(ux_train, cbin_train, np_by_attr, p_by_attr)
    # Parameters:
    ordem = 2
    t_norm = 'min'
    par_area = ['cardinalidade_relativa', 0.1]  # 'cardinalidade_relativa', 'frequencia_relativa'
    par_over = [1, 0.95]
    par_pcd = [1]

    arbol = F2.gen_ARB(ordem, t_norm, par_area, par_over, par_pcd)

    for i in arbol:
        print (len(i[0]))

    # Division
    F3 = Association(arbol, cbin_train)
    premises_by_class = F3.division("freq_max")
    # OK con: MQR (11, 17), PMQR(11, 17), CD(295, 49), PCD(303, 57), freq_max(313, 27)
    # falta parar el proceso ya que en ciertos casos no se generan premisas para una clase

    for i in premises_by_class:
        print (len(i[0]))
        print (i[0])

    # Aggregation:
    F4 = Aggregation(premises_by_class, cbin_train)
    x = F4.aggregation("MQR")

    classes_premises = []
    print ("Premises:\n=========")
    for i in range(len(x)):
        print ('Premises of Class ', i, x[0][i][0])
        classes_premises.append(x[0][i][0])
        print ('weights', i, x[0][i][1].T)

    # Decision
    F5 = Decisions(x[1], info[3])
    train_bin_prediction = F5.dec_max_pert()

    # Evaluation
    models_classes = x[0]

    F6 = Evaluation(models_classes, classes_premises, info[3])
    metrics_train = F6.eval_train(cbin_train, train_bin_prediction)
    print (metrics_train)
    metrics_test = F6.eval_test(cbin_test, ux_test, t_norm)
    print (metrics_test)


if __name__ == '__main__':
    main()