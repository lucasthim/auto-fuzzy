from .fuzzification import Fuzzification
from .formulation import Formulation
from .association import Association
from .aggregation import Aggregation
from .decision import Decision
# from .evaluation import Evaluation


class AutoFISClassifier():

    def __init__(self,categorical_attributes):


        # -------------------------
        # Fuzzyfication parameters
        # -------------------------
        self.categorical_attributes = [False, False, False, False]
        self.triangle_format = 'normal'  # "tukey", "normal"
        self.n_fuzzy_sets = 5  # 3, 5, 7
        self.enable_negation = False

        # -------------------------
        # Formulation parameters
        # -------------------------
        self.ordem_max_premises = 3
        # Area filter parameters:
        self.criteria_area = "frequencia_relativa"  # "cardinalidade_relativa", "frequencia_relativa"
        self.area_threshold = 0.075
        # PCD filter parameter:
        self.is_enable_pcd = [0, 0]
        # Overlapping filter parameters:
        self.is_enable_overlapping = [1, 1]
        self.overlapping_threshold = 0.95

        # -------------------------
        # Association:
        # -------------------------
        self.method_association = "CD"  # "MQR", "PMQR", "CD", "PCD", "freq"

        # -------------------------
        # Aggregation
        # -------------------------
        self.method_aggregation = "MQR"  # "MQR", "PMQR", "intMQR", "CD", "PCD", "max"

        self.fuzzyfication = None
        self.formulation = None
        self.association = None
        self.aggregation = None
        self.decision = None


    def set_parameters(self,categorical_attributes_mask):
        
        self.parameters = [categorical_attributes_mask, self.triangular_fuzzification_type, 
                    self.num_partitions_by_attribute, self.t_norm, self.is_enable_negation,
                    self.ordem_max_premises, self.criteria_area, self.area_threshold, self.is_enable_pcd,
                    self.is_enable_overlapping, self.overlapping_threshold,
                    self.method_association, self.method_aggregation]

    def load_data(self,data,categorical_attributes_mask = None):

        self.reading_module = Lecture()
        self.reading_module.read_data(data)
        self.info_data = self.reading_module.info_data()
        self.categorical_attributes_mask = categorical_attributes_mask

    def fuzzify(self):
        # Fuzzification of inputs
        X = self.info_data[0].copy()
        self.fuzzyfication = Fuzzification(X, self.categorical_attributes_mask)
        self.fuzzyfication.build_uX('tukey', 3)
        self.fuzzyfication.add_negation()

    def formulate(self):
        pass

    def fit(self):

        
        # Split to train and test
        index_train = int(X.shape[0] * 0.70) # Consertar isso depois para um Stratified KFold ou algo do tipo
        ux_train = self.fuzzyfication.uX[:index_train, :]
        ux_test = self.fuzzyfication.uX[index_train:, :]
        y_train_binary = self.info_data[1][:index_train, :]
        y_test_binary = self.info_data[1][index_train:, :]

        # Formulation
        self.formulation = Formulation(
            ux = ux_train, 
            c_bin = y_train_binary, 
            np_by_attribute = self.fuzzyfication.num_of_premises_by_attribute, 
            p_by_attribute = self.fuzzyfication.attribute_premises,
            ref_attributes = self.fuzzyfication.ref_attributes,
            attributes_contain_negation = [])

        # Parameters:
        max_number_of_premises = 2
        t_norm = 'min'
        par_area = ['cardinalidade_relativa', 0.1]  # 'cardinalidade_relativa', 'frequencia_relativa'
        par_over = [1, 0.95]
        par_pcd = [1]

        arbol = self.formulation.gen_ARB(max_number_of_premises, t_norm, par_area, par_over, par_pcd)

        for i in arbol:
            print (len(i[0]))

        # Division
        self.association = Association(arbol, y_train_binary)
        premises_by_class = self.association.division("freq_max")
        # OK con: MQR (11, 17), PMQR(11, 17), CD(295, 49), PCD(303, 57), freq_max(313, 27)
        # falta parar el proceso ya que en ciertos casos no se generan premisas para una clase

        for i in premises_by_class:
            print (len(i[0]))
            print (i[0])

        # Aggregation:
        self.aggregation = Aggregation(premises_by_class, y_train_binary)
        x = self.aggregation.aggregation("MQR")

        classes_premises = []
        print ("Premises:\n=========")
        for i in range(len(x)):
            print ('Premises of Class ', i, x[0][i][0])
            classes_premises.append(x[0][i][0])
            print ('weights', i, x[0][i][1].T)

        # Decision
        self.decision = Decisions(x[1], self.info_data[3])
        train_bin_prediction = self.decision.dec_max_pert()

        # Evaluation
        models_classes = x[0]

        self.evaluation = Evaluation(models_classes, classes_premises, self.info_data[3])
        self.metrics_train = self.evaluation.eval_train(y_train_binary, train_bin_prediction)
        print ('metrics_train: ')
        print (metrics_train)
        self.metrics_test = self.evaluation.eval_test(y_test_binary, ux_test, t_norm)
        print ('')
        print ('metrics_test: ')
        print (metrics_test)
