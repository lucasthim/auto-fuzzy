import timeit


# TODO: break this into atomic parameters
parameters_database = [[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ['tukey', 3, 'prod', 1, 2, 'frequencia_relativa', 0.075, [0, 0], [1, 1], 0.95, 'MQR', 'MQR', 5, 4, 0.9, 10]]
data = 'automobile-10-fold_csv.zip'
report_folder = '/home/lucas-thimoteo/Projects/Personal/auto-fuzzy/2020-08-22_09-42-30/5'
clf_n = 5
folder__tree_outpus = '/home/lucas-thimoteo/Projects/Personal/auto-fuzzy/2020-08-22_09-42-30/5/tree_outputs'
data_name_key = 'automobile'
root = '/home/lucas-thimoteo/Projects/Personal/auto-fuzzy/autofis/classification/ensemble/senfis'

t0 = timeit.default_timer()







