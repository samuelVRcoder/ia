from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import pandas as pd

import numpy

tabela = pd.read_csv("./clientes.csv")

codificador = LabelEncoder()

for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        tabela[coluna] = codificador.fit_transform(tabela[coluna])


for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        print(coluna)

x = tabela.drop(["id_cliente", "score_credito"], axis=1)

y = tabela["score_credito"]

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=1)

modelo_arvore = RandomForestClassifier()

modelo_vizinho = KNeighborsClassifier()

modelo_arvore.fit(x_treino, y_treino)

modelo_vizinho.fit(x_treino, y_treino)

previsao_arvore = modelo_arvore.predict(x_teste)

previsao_vizinhos = modelo_vizinho.predict(x_teste.to_numpy())

print(accuracy_score(previsao_arvore, y_teste))

print(accuracy_score(previsao_vizinhos, y_teste))

input()


