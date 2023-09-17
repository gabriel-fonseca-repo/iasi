nome_arquivo_csv_in = "data/EMG.csv"
nome_arquivo_csv_out = "data/EMG_Classes.csv"


classes = ["Neutro", "Sorrindo", "Aberto", "Surpreso", "Rabugento"]
novas_linhas = []
with open(nome_arquivo_csv_in, "r") as dados_emg:
    for i in range(10):
        for classe_id, classe in enumerate(classes):
            for linha in range(1000):
                novas_linhas.append(
                    dados_emg.readline().strip() + "," + str(classe_id) + "\n"
                )

with open(nome_arquivo_csv_out, "w") as arquivo_escrita:
    arquivo_escrita.writelines(novas_linhas)
