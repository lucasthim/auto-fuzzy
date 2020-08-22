def create_dictionary():
    iris = 4 * [0]
    glass = 9 * [0]
    wine = 13 * [0]
    ecoli = 7 * [0]
    balance = 4 * [0]  # ['0/1']
    cleveland = [0] + 2 * [1] + 2 * [0] + 2 * [1] + [0] + [1] + [0] + [1] + [0] + [1]
    pima = 8 * [0]
    titanic = 3 * [1]
    banana = 2 * [0]
    winequality_red = 11 * [0]
    phoneme = 5 * [0]
    segment = 2 * [0] + [1] + 2 * [0] + 14 * [0]
    ring = 20 * [0]
    twonorm = 20 * [0]
    magic = 10 * [0]
    texture = 40 * [0]
    spambase = 57 * [0]

    # Group 2
    hayes_roth = 4 * [1]
    tae = 4 * [1] + [0]
    haberman = 3 * [0]
    new_thyroid = 5 * [0]
    hepatitis = [0] + 12 * [1] + 5 * [0] + [1]
    bupa = 6 * [0]
    heart = [0] + 2 * [1] + 2 * [0] + 2 * [1] + [0] + [1] + [0] + 3 * [0]
    wisconsin = 9 * [0]  # [1]
    ionosphere = [0] + 32 * [0]
    spectfheart = 44 * [0]
    dermatology = 33 * [1] + [0]
    contraceptive = [0] + 2 * [1] + [0] + 5 * [1]
    vehicle = 18 * [0]
    page_blocks = 10 * [0]
    thyroid = [0] + 15 * [1] + 5 * [0]
    penbased = 16 * [0]
    satimage = 36 * [0]
    optdigits = [1] + [0] + 6 * [0] + [0] + 7 * [0] + [0] + 7 * [0] + [0] + 6 * [0] + 2 * [0] + \
                7 * [0] + [0] + 6 * [0] + 2 * [0] + 7 * [0] + [0] + 7 * [0]
    coil2000 = 85 * [0]

    # Group 3
    saheart = 4 * [0] + [1] + 4 * [0]
    automobile = [0] + 7 * [1] + 5 * [0] + 2 * [1] + [0] + [1] + 8 * [0]
    australian = [1] + 2 * [0] + 3 * [1] + [0] + 2 * [1] + [0] + 2 * [1] + 2 * [0]
    crx = [1] + 2 * [0] + 2 * [1] + [0] + [1] + [0] + 2 * [1] + [0] + 2 * [1] + 2 * [0]
    german = [1] + [0] + 2 * [1] + [0] + 5 * [1] + [0] + [1] + [0] + 2 * [1] + [0] + [1] + [0] + 2 * [1]

    monk = 6 * [1]
    vowel = 3 * [1] + 10 * [0]
    census = [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
              1, 0, 1, 0, 0, 0]
    connect = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1]
    fars = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    fars_bin = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    fars_no_injury_vs_injury = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    fars_fatal_inj_vs_no_inj = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    kddcup = [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0]
    poker = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    poker_1_vs_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    poker_0_vs_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    covtype = 10 * [0] + 4 * [1] + 40 * [1]
    cover_1_vs_2 = 10 * [0] + 4 * [1] + 40 * [1]
    # ==========================================================================================================

    databases = {}

    databases["iris"] = iris, 3
    databases["glass"] = glass, 6
    databases["wine"] = wine, 3
    databases["ecoli"] = ecoli, 8
    databases["balance"] = balance, 3
    databases["cleveland"] = cleveland, 5
    databases["pima"] = pima, 2
    databases["titanic"] = titanic, 2
    databases["banana"] = banana, 2
    databases["winequality-red"] = winequality_red, 11
    databases["phoneme"] = phoneme, 2
    databases["segment"] = segment, 7
    databases["ring"] = ring, 2
    databases["twonorm"] = twonorm, 2
    databases["magic"] = magic, 2
    databases["texture"] = texture, 11
    databases["spambase"] = spambase, 2
    databases["hayes-roth"] = hayes_roth, 3
    databases["tae"] = tae, 3
    databases["haberman"] = haberman, 2
    databases["newthyroid"] = new_thyroid, 3
    databases["hepatitis"] = hepatitis, 2
    databases["bupa"] = bupa, 2
    databases["heart"] = heart, 2
    databases["wisconsin"] = wisconsin, 2
    databases["ionosphere"] = ionosphere, 2
    databases["spectfheart"] = spectfheart, 2
    databases["dermatology"] = dermatology, 6
    databases["contraceptive"] = contraceptive, 3
    databases["vehicle"] = vehicle, 4
    databases["page-blocks"] = page_blocks, 5
    databases["thyroid"] = thyroid, 3
    databases["penbased"] = penbased, 10
    databases["satimage"] = satimage, 6
    databases["optdigits"] = optdigits, 10
    databases["coil2000"] = coil2000, 2
    databases["saheart"] = saheart, 2
    databases["automobile"] = automobile, 6
    databases["australian"] = australian, 2
    databases["crx"] = crx, 2
    databases["german"] = german, 2

    databases["movement_libras"] = 90 * [0], 15
    databases["yeast"] = 4 * [0] + [1] + 3 * [0], 10
    databases["appendicitis"] = 7 * [0], 2
    databases["monk-2"] = monk, 2
    databases["sonar"] = 60 * [0], 2
    databases["vowel"] = vowel, 2
    databases["wdbc"] = 30 * [0], 2
    databases['census'] = census, 2
    databases['connect-4'] = connect, 3
    databases['fars'] = fars, 8
    databases['fars_bin'] = fars_bin, 2
    databases['fars_no_injury_vs_injury'] = fars_no_injury_vs_injury, 2
    databases['fars_fatal_inj_vs_no_inj'] = fars_fatal_inj_vs_no_inj, 2
    databases['kddcup'] = kddcup, 23
    databases['poker'] = poker, 10
    databases['poker_1_vs_0'] = poker_1_vs_0, 2
    databases['poker_0_vs_1'] = poker_0_vs_1, 2
    databases['covtype'] = covtype, 2
    databases['cover_1_vs_2'] = cover_1_vs_2, 2

    return databases


# Dictionary
def dictionary_data(database_name):
    databases = create_dictionary()
    return databases[database_name]


def main():
    databases = create_dictionary()
    print (databases["iris"][0])
    print (databases["australian"][0])

    try:
        print (databases["win"][0])
    except KeyError:
        print ([0])

    # for i in databases:
    #     print i  # Se impremen los keys
    #     print "LogicCategorical:", databases[i][0]
    #     print "Number of attributes", len(databases[i][0])
    #     print "Number of classes:", databases[i][1]
    #     print "\n"


if __name__ == '__main__':
    main()

    # all_numeric = ['iris', 'glass', 'wine', 'ecoli', 'pima', 'banana', 'winequality_red', 'phoneme', 'ring',
    # 'twonorm', 'magic', 'texture', 'spambase', 'satimage', 'penbased', 'coil2000',
    #                'page_blocks', 'vehicle', 'spectfheart', 'new-thyroid', 'haberman']
    # all_numeric_allpossible_categoric = ['balance', 'wisconsin']
    # all_numeric_1possible_categoric = ['ionosphere']
    #
    # delete_1attr = ['segment', 'automobile', ]
    # # segment tiene 2 candidatos a categorico
    # # automobile: atributos mixtos
    #
    # delete_2attr = ['optdigits']
    # # optdigits tiene algunos candidatos a categorico
    #
    #
    # mix = ['cleveland', 'saheart', 'crx', 'australian', 'thyroid', 'contraceptive', 'Dermatology', 'hepatitis', 'Tae']
    # # dermatology solo tiene 1 numerica
    # # tae solo tiene 1 numerica
    #
    # mix_possible_categoric = ['german', 'heart']
    #
    # all_categoric = ['titanic', 'hayes-roth']
