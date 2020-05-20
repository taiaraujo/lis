# importação da pasta do sistema operacional
import os
# importação da biblioteca para a escrita do arquivo
import xlwt
# importação da biblioteca para encontrar os maiores valores
import heapq
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from pandas import crosstab
import numpy as np

# lista que vai armazenar os arquivos
dados = []
usuarios = []
# abrindo arquivo .csv com o nome 'arq'
with open('dataset_pontuacao_livros.csv', encoding='ISO-8859-1') as arq:
    for linha in arq.readlines():
        atrib = linha.split(";")
        dados.append(list(atrib))

    arq.close()

for i in range(len(dados)):
    for j in range(len(dados[i])):
        try:
            dados[i][j] = eval(dados[i][j])
        except:
            dados[i][j] = dados[i][j]
        if dados[i][j] == '' or dados[i][j] == '\n':
            del(dados[i-1][j-1])

print(dados)

with open('usuarios.csv', encoding='ISO-8859-1') as arq:
    for linha in arq.readlines():
        atrib = linha.split(";")
        usuarios.append(list(atrib))

    arq.close()

for i in range(len(usuarios)):
    for j in range(len(usuarios[i])):
        try:
            usuarios[i][j] = eval(usuarios[i][j])
        except:
            usuarios[i][j] = usuarios[i][j]
        if usuarios[i][j] == '' or usuarios[i][j] == '\n':
            del(usuarios[i-1][j-1])

print(usuarios)

x, y = [], []

for i in range(1, 19):
    x.append([usuarios[i][1], usuarios[i][2], usuarios[i][3], usuarios[i][4], usuarios[i][5], usuarios[i][6],
              usuarios[i][7], usuarios[i][8], usuarios[i][9], usuarios[i][11], usuarios[i][12], usuarios[i][13],
              usuarios[i][14], usuarios[i][15], usuarios[i][16], usuarios[i][17], usuarios[i][18], usuarios[i][19]])

    y.append(usuarios[i][20])
print('Livros mais bem pontuados pelos usuarios')
ind = []
for i in range(5):
    c = np.argmax(y)
    y[c] = 0
    ind.append(c)
for i in range(5):
    print(i, dados[ind[i]][1])

x = np.array(x)
y = np.array(y)
u = 1

model = GaussianNB()

model.fit(x, y)
temp = []
pontos = []
for j in range(19, len(dados)):
    predicted = model.predict([[usuarios[u][1], usuarios[u][2], usuarios[u][3], usuarios[u][4], usuarios[u][5], usuarios[u][6],
                                usuarios[u][7], usuarios[u][8], usuarios[u][9], dados[j][3], dados[j][4], dados[j][5], dados[j][6], dados[j][7], dados[j][8], dados[j][9], dados[j][10], dados[j][11]]])
    # print(predicted)
    temp.append([usuarios[u][1], usuarios[u][2], usuarios[u][3], usuarios[u][4], usuarios[u][5], usuarios[u][6],
                 usuarios[u][7], usuarios[u][8], usuarios[u][9], dados[j][3], dados[j][4], dados[j][5], dados[j][6], dados[j][7], dados[j][8], dados[j][9], dados[j][10], dados[j][11]])
    pontos.append(predicted)
print('Livros que indicamos')
best = heapq.nlargest(5, pontos)
ind = []
for i in range(5):
    c = np.argmax(pontos)
    pontos[c] = 0
    ind.append(c+19)
for i in range(5):
    print(i, dados[ind[i]][1])
print('Matriz de Confusão')
# print(confusion_matrix(y, model.predict(x)))
print(crosstab(y, model.predict(x), rownames=['Real'], colnames=['Predito'], margins=True))
# medias = cross_val_score(predicted, x, y)
# media = sum(medias) / len(medias)
# print(medias,'||', media)
print('Acurácia')
print(model.score(x, y, sample_weight=None))

