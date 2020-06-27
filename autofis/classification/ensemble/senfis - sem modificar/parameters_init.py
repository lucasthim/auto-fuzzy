class GlobalParameter:
    def __init__(self):
        self.t_data = 0  # 1 = binario, 0 = multipla
        self.n_classifiers = [5]
        self.percent_resample = 0.9
        self.blb = 10
        self.method_aggregation = 4  # 4 RandomFIS_Expert, 1 RandomFIS_E
        self.size_ensemble = [2, 3]
        self.sel_param = [1]
        self.sel_method = [0]  # 0 WAD, 1 ACC+reg
