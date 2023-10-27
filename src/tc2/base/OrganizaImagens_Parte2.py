import cv2
import numpy as np


def getDadosImagens(red):
    folderRoot = "data/tc2/faces/"  # MODIFIQUE para o caminho do conjunto de dados no seu computador.
    individual = [
        "an2i",
        "at33",
        "boland",
        "bpm",
        "ch4f",
        "cheyer",
        "choon",
        "danieln",
        "glickman",
        "karyadi",
        "kawamura",
        "kk49",
        "megak",
        "mitchell",
        "night",
        "phoebe",
        "saavik",
        "steffi",
        "sz24",
        "tammo",
    ]  # os 20 sujeitos no conjunto de dados.
    expressoes = [
        "_left_angry_open",
        "_left_angry_sunglasses",
        "_left_happy_open",
        "_left_happy_sunglasses",
        "_left_neutral_open",
        "_left_neutral_sunglasses",
        "_left_sad_open",
        "_left_sad_sunglasses",
        "_right_angry_open",
        "_right_angry_sunglasses",
        "_right_happy_open",
        "_right_happy_sunglasses",
        "_right_neutral_open",
        "_right_neutral_sunglasses",
        "_right_sad_open",
        "_right_sad_sunglasses",
        "_straight_angry_open",
        "_straight_angry_sunglasses",
        "_straight_happy_open",
        "_straight_happy_sunglasses",
        "_straight_neutral_open",
        "_straight_neutral_sunglasses",
        "_straight_sad_open",
        "_straight_sad_sunglasses",
        "_up_angry_open",
        "_up_angry_sunglasses",
        "_up_happy_open",
        "_up_happy_sunglasses",
        "_up_neutral_open",
        "_up_neutral_sunglasses",
        "_up_sad_open",
        "_up_sad_sunglasses",
    ]
    QtdIndividuos = len(individual)
    QtdExpressoes = len(expressoes)
    X = np.empty((red * red, 0))
    Y = np.empty((QtdIndividuos, 0))

    for i in range(QtdIndividuos):
        for j in range(QtdExpressoes):
            path = (
                folderRoot
                + individual[i]
                + "/"
                + individual[i]
                + expressoes[j]
                + ".pgm"
            )
            PgmImg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if PgmImg is None:
                continue

            ResizedImg = cv2.resize(PgmImg, (red, red))

            VectorNormalized = ResizedImg.flatten("F")
            ROT = -np.ones((QtdIndividuos, 1))
            ROT[i, 0] = 1

            # cv2.imshow("Foto", PgmImg)
            # cv2.waitKey(0)

            VectorNormalized.shape = (len(VectorNormalized), 1)
            X = np.append(X, VectorNormalized, axis=1)
            Y = np.append(Y, ROT, axis=1)

    print("-------------------------------------------------------------")
    print()
    print(f"Quantidade de amostras do conjunto de dados: {X.shape[1]}")
    print("A quantidade de preditores esta relacionada ao redimensionamento!")
    print(f"Para esta rodada escolheu-se um redimensionamento de {red}")
    print(f"Portanto, a quantidade de preditores desse conjunto de dados: {X.shape[0]}")
    print(f"Este conjunto de dados possui {Y.shape[0]} classes")
    print(f"X tem ordem {X.shape[0]}x{X.shape[1]}")
    print(f"Y tem ordem {Y.shape[0]}x{Y.shape[1]}")
    print()
    return X, Y
