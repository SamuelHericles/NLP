#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Aula 01 - Dados Textuais: classificação e exploração.


# In[12]:


import pandas as pd

resenha = pd.read_csv("imdb.csv")
resenha.head()


# In[ ]:





# In[11]:


from sklearn.model_selection import train_test_split

treino, teste, classe_treino, classe_teste = train_test_split(resenha.text_pt,
                                                              resenha.sentiment,
                                                              random_state = 42)


# In[12]:


from sklearn.linear_model import LogisticRegression

regressao_logistica = LogisticRegression()
regressao_logistica.fit(treino, classe_treino)
acuracia = regressao_logistica.score(teste, classe_teste)
print(acuracia)


# In[ ]:


print("Negativa \n")
print(resenha.text_pt[189])


# In[ ]:


print("Positivo \n")
print(resenha.text_pt[49002])


# In[26]:


print(resenha.sentiment.value_counts())


# In[25]:


resenha.head()


# In[21]:


classificacao = resenha["sentiment"].replace(["neg", "pos"], [0,1])


# In[22]:


resenha["classificacao"] = classificacao


# In[23]:


resenha.head()


# In[24]:


resenha.tail()


# ## Aula 2 - Bag of Words: criando representações da linguagem humana.

# In[27]:


from sklearn.feature_extraction.text import CountVectorizer

texto = ["Assisti um filme ótimo", "Assisti um filme ruim"]

vetorizar = CountVectorizer(lowercase=False)
bag_of_words = vetorizar.fit_transform(texto)


# In[28]:


vetorizar.get_feature_names()


# In[29]:


bag_of_words


# In[30]:


matriz_esparsa = pd.SparseDataFrame(bag_of_words,
                      columns=vetorizar.get_feature_names())


# In[31]:


matriz_esparsa


# In[32]:


vetorizar = CountVectorizer(lowercase=False, max_features=50)
bag_of_words = vetorizar.fit_transform(resenha.text_pt)
print(bag_of_words.shape)


# In[33]:


def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                              texto[coluna_classificacao],
                                                              random_state = 42)
    regressao_logistica = LogisticRegression(solver = "lbfgs")
    regressao_logistica.fit(treino, classe_treino)
    return regressao_logistica.score(teste, classe_teste)
print(classificar_texto(resenha, "text_pt", "classificacao"))









# ## Aula 3 - Visualizando os dados com WordCloud.

# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud

todas_palavras = ' '.join([texto for texto in resenha.text_pt])

nuvem_palvras = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(todas_palavras)


# In[35]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
plt.imshow(nuvem_palvras, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[44]:


def nuvem_palavras_neg(texto, coluna_texto):
    texto_negativo = texto.query("sentiment == 'neg'")
    todas_palavras = ' '.join([texto for texto in texto_negativo[coluna_texto]])

    nuvem_palvras = WordCloud(width= 800, height= 500,
                              max_font_size = 110,
                              collocations = False).generate(todas_palavras)
    plt.figure(figsize=(10,7))
    plt.imshow(nuvem_palvras, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[45]:


def nuvem_palavras_pos(texto, coluna_texto):
    texto_positivo = texto.query("sentiment == 'pos'")
    todas_palavras = ' '.join([texto for texto in texto_positivo[coluna_texto]])

    nuvem_palvras = WordCloud(width= 800, height= 500,
                              max_font_size = 110,
                              collocations = False).generate(todas_palavras)
    plt.figure(figsize=(10,7))
    plt.imshow(nuvem_palvras, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[38]:


nuvem_palavras_neg(resenha, "text_pt")


# In[39]:


nuvem_palavras_pos(resenha, "text_pt")


# In[7]:


import nltk 
frase = ["um filme bom", "um filme ruim"]
frequencia = nltk.FreqDist(frase)
frequencia


# In[8]:


from nltk import tokenize

frase = "Bem vindo ao mundo do PLN!"

token_espaco = tokenize.WhitespaceTokenizer()
token_frase = token_espaco.tokenize(frase)
print(token_frase)


# In[9]:


token_frase = token_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_frase)
df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),
                                   "Frequência": list(frequencia.values())})


# In[47]:


df_frequencia.nlargest(columns = "Frequência", n = 10)


# ## Aula 5 - Uma nova visualização e os stop words.

# In[ ]:


import seaborn as sns
    
def pareto(texto, coluna_texto, quantidade):
    todas_palavras = ' '.join([texto for texto in texto[coluna_texto]])
    token_frase = token_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)
    df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),
                                   "Frequência": list(frequencia.values())})
    df_frequencia = df_frequencia.nlargest(columns = "Frequência", n = quantidade)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequencia, x = "Palavra", y = "Frequência", color = 'gray')
    ax.set(ylabel = "Contagem")
    plt.show()



pareto(resenha, "text_pt", 10)


# In[ ]:





# In[15]:


palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")

frase_processada = list()
for opiniao in resenha["text_pt"]:
    nova_frase = list()
    palavras_texto = token_espaco.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in palavras_irrelevantes:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))
    
resenha["tratamento_1"] = frase_processada


# In[16]:


resenha.head()


# In[17]:


print("Frase text_pr\n")
print(resenha["text_pt"][0])
print("\n Frase tratamento_1\n")
print(resenha["tratamento_1"][0])


# In[18]:


classificar_texto(resenha, "tratamento_1", "classificacao")


# In[ ]:


pareto(resenha,"tratamento_1", 10)


# In[ ]:


#StopWords: palavras para tirar algumas que não sao relevantes
#Tokenização: forma de pegar as palavras de uma frase,separação de espaço ou pontos
from nltk import tokenize
frase = "Olá mundo"
token_pontuacao = tokenize.WordPunctTokenizer()
token_frase = token_pontuacao.tokenize(frase)
print(token_frase)


# In[ ]:


from string import punctuation
pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)
pontuacao_stopwords = pontuacao + palavras_irrelevantes
frase_processada = list()
for opiniao in resenha["tratamento_1"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))
resenha["tratamento_2"] = frase_processada


# In[ ]:


resenha["tratamento_1"][0]


# In[ ]:


resenha["tratamento_2"][0]


# In[ ]:


#STEMMER: REDUZIR A UM RADICAL COMUM
#RSLP: REMOVEDOR DE SUFIXOS
stemmer = nltk.RSLPStemmer()
stemmer.stem("corre")


# In[ ]:


resenha.head()


# In[ ]:


import unidecode

acentos = "ótimo péssimo não é tão"

teste = unidecode.unidecode(acentos)
print(teste)
stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]


# In[ ]:





# In[ ]:





# In[ ]:


frase_processada = list()
for opiniao in resenha["tratamento_2"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento_3"] = frase_processada


# In[ ]:


resenha.head()


# In[ ]:


acuracia_tratamento3 = classificar_texto(resenha,"tratamento_3","classificacao")
print(acuracia_tratamento3)


# In[ ]:





# nuvem_palavras_neg(resenha, "tratamento_3")
# 

# In[ ]:


nuvem_palavras_pos(resenha, "tratamento_3")


# In[ ]:


frase = "O Thiago é o novo instrutor da Alura"
print(frase.lower())


# In[ ]:


frase_processada = list()
for opiniao in resenha["tratamento_3"]:
    nova_frase = list()
    opiniao = opiniao.lower()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))
    
resenha["tratamento_4"] = frase_processada


# In[ ]:


resenha["text_pt"][0]


# In[ ]:


resenha["tratamento_4"][0]


# In[ ]:


acuracia_tratamento4 = classificar_texto(resenha,"tratamento_4","classificacao")
print(acuracia_tratamento4)
print(acuracia_tratamento3)


# In[ ]:


nuvem_palavras_neg(resenha,"tratamento_4")


# In[ ]:


nuvem_palavras_pos(resenha,"tratamento_4")


# In[72]:


frase_processada = list()
for opiniao in resenha["tratamento_4"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(stemmer.stem(palavra))
    frase_processada.append(' '.join(nova_frase))
resenha["tratamento_5"] = frase_processada


# In[73]:


acuracia_tratamento5 = classificar_texto(resenha,"tratamento_5","classificacao")
print(acuracia_tratamento5)
print(acuracia_tratamento4)


# In[74]:


#stemizador diferentes para cada lingua


# In[75]:


nuvem_palavras_neg(resenha,"tratamento_4")


# In[76]:


nuvem_palavras_pos(resenha,"tratamento_4")


# In[77]:


#TF-IDF: FREQUÊNCIA DOS TERMOS - INVERSOS DE FREQUÊNCIA DO DOCUMENTO
# analisar a frequencias das palavras normalizando o valor dela
# algumas palavras devem ser normalizadas para todas terem pesos próximos


# In[78]:


from sklearn.feature_extraction.text import TfidfVectorizer
frases = ["Assisti um filme ótimo","Assisti um filme pésssimo"]
tfidf = TfidfVectorizer(lowercase = False, max_features=50)
caracteristicas = tfidf.fit_transform(frases)
pd.DataFrame(caracteristicas.todense(),
            columns=tfidf.get_feature_names()
)


# In[79]:


tfidf_bruto = tfidf.fit_transform(resenha["text_pt"])
treino, teste, classe_treino, classe_teste = train_test_split(tfidf_bruto,
                                                              resenha["classificacao"],
                                                              random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_bruto = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_bruto)


# In[80]:


tfidf_tratados = tfidf.fit_transform(resenha["tratamento_5"])
treino, teste, classe_treino, classe_teste = train_test_split(tfidf_tratados,
                                                              resenha["classificacao"],
                                                              random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_tratados = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_tratados)


# In[81]:


#Ngrams:o tf-idf não gera um fator para manter uma sequência de
#palavras.O Ngrams, faz isso, manter uma certa sequencia no texto
from nltk import ngrams
frase = "Assiti um ótimo filme."
frase_separada = token_espaco.tokenize(frase)
pares = ngrams(frase_separada, 2)
list(pares)


# In[10]:


#evitando flexões e derivações nas palavas
stemmer = nltk.RSLPStemmer()
stemmer.stem("correria")


# In[13]:


frase_processada = list()
for opiniao in resenha["tratamento_4"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavra_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(stemmer.stem(palavra))
        frase_processada.append(' '.join(nova_frase))
resenha["tratamento_5"] = frase_processada


# In[ ]:


acuracia_tratamento5 = classificar_texto(resenha, "tratamento_5","classificacao")
print(acuracia_tratamento5)
print(acuracia_tratamento4)


# In[ ]:


nuvem_palavras_neg(resenha,"tratamento_5")


# In[ ]:


nuvem_palavras_pos(resenha,"tratamento_5")


# In[ ]:


pareto(resenha,"tratamento_5",10)

