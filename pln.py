import pandas as pd
resenha = pd.read_csv("imdb.csv")
resenha.head()
resenha


from sklearn.model_selection import train_test_split

treino, teste, classe_treino, classe_teste = train_test_split(resenha.text_pt,
                                                              resenha.sentiment,
                                                              random_state = 42)

from sklearn.linear_model import LogisticRegression

#regressao_logistica = LogisticRegression()
#regressao_logistica.fit(treino, classe_treino)
#acuracia = regressao_logistica.score(teste, classe_teste)
#print(acuracia)


print(resenha.sentiment.value_counts())


resenha.head()


classificacao = resenha["sentiment"].replace(["neg", "pos"], [0,1])


# In[64]:


resenha.head()


resenha["classificacao"] = classificacao


resenha.tail()

from sklearn.feature_extraction.text import CountVectorizer

texto = ["Assisti um filme ótimo","Assisti um filme ruim"]

vetorizar = CountVectorizer(lowercase=False)
bag_of_words = vetorizar.fit_transform(texto)


# In[59]:


bag_of_words


# In[60]:


matriz_sparsa = pd.SparseDataFrame(bag_of_words,
                     columns=vetorizar.get_feature_names())


matriz_sparsa


vetorizar = CountVectorizer(lowercase=False,max_features=50)
bag_of_words = vetorizar.fit_transform(resenha.text_pt)
print(bag_of_words.shape)


def classificar_texto(texto,coluna_texto,coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                                  texto[coluna_classificacao],
                                                                  random_state = 42)

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)
    return regressao_logistica.score(teste, classe_teste)

print(classificar_texto(resenha,"text_pt","classificacao"))

get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud

todas_palavras = ' '.join([texto for texto in resenha.text_pt])
nuvem_palavras = WordCloud(width=800,height=500,
                           max_font_size=110,
                          collocations = False).generate(todas_palavras)

import matplotlib.pyplot as plt
plt.figure (figsize=(10,7)
plt.imshow(nuvem_palavras,interpolation='bilinear')
plt.show("off")

resenha.query("sentiment == 'pos' ")

def nuvem_palavras_neg(texto,coluna_texto):
    texto_negativo = texto.query("sentiment == 'neg' ")
    todas_palavras = ' '.join([texto for texto in texto_negativo[coluna_texto]])
    nuvem_palavras = WordCloud(width=800,height=500,
                               max_font_size=110,
                              collocations = False).generate(todas_palavras)

    plt.figure (figsize=(10,7))
    plt.imshow(nuvem_palavras,interpolation='bilinear')
    plt.axis("off")
    plt.show()

def nuvem_palavras_pos(texto,coluna_texto):
    texto_negativo = texto.query("sentiment == 'pos' ")
    todas_palavras = ' '.join([texto for texto in texto_negativo[coluna_texto]])
    nuvem_palavras = WordCloud(width=800,height=500,
                               max_font_size=110,
                              collocations = False).generate(todas_palavras)

    plt.figure (figsize=(10,7))
    plt.imshow(nuvem_palavras,interpolation='bilinear')
    plt.axis("off")
    plt.show()

nuvem_palavras_neg(resenha,"text_pt")
nuvem_palavras_pos(resenha,"text_pt")
            
import nltk
nltk.download("all")

frase = ["um filme ruim","um filme bom"]
frequencia = nltk.FreqDist(frase)
frequencia


from nltk import tokenize

frase = "Bem vindo ao mundo do PLN!"

token_espaco = tokenize.WhitespaceTokenizer()
token_frase = token_espaco.tokenize(frase)
print(frase)

token_frase = token_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_frase)
df_frequencia = pd.DataFrame({"Palavra":list(frequencia.keys()),
                             "Frequência":list(frequencia.values())})

df_frequencia.nlargest(columns = "Frequência",n=10)

import seaborn as sns
import matplotlib.pyplot as plt
def pareto(texto,coluna_texto,quantidade):
    todas_palavras = ' '.join([texto for texto in texto[coluna_texto]])
    token_frase = token_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)
    df_frequencia = pd.DataFrame({"Palavra":list(frequencia.keys()),
                                 "Frequência":list(frequencia.values())})
    df_frequencia = df_frequencia.nlargest(columns = "Frequência",n = quantidade)
    plt.figure(figure=(12,8))
    ax = sns.barplot(data = df_frequencia,x = "Palavra", y = "Frequência", color = "gray")
    ax.set(ylabel = "Contagem")
    plt.show()

pareto(resenha,"text_pt",10)


palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
frase_processada = list()
for opniao in resenha.text_pt:
    nova_frase = list()
    palavras_texto = token_espaco.tokenize(opniao)
    for palavra in palavras_texto:
        if palavra not in palavras_irrelevantes:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))
    
resenha["tratamento_1"] = frase_processada


print(palavras_irrelevantes)


resenha.head()

classificar_texto(resenha, "tratamento_1","classificacao")

# pareto(resenha,"tratamento_1",10)




