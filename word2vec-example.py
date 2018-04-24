
from gensim.models.word2vec import Word2Vec, LineSentence
import gensim.downloader as api
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Entrenamiento
corpus = api.load('text8')
model = Word2Vec(sentences=corpus, size=100, sg=1, hs=1)
model.save('./model-text8')

# Evaluación del modelo
import os
import gensim
from pprint import pprint
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
pprint( model.accuracy(test_data_dir + 'questions-words.txt', case_insensitive=False) )
print("...............................")
pprint( model.evaluate_word_pairs(test_data_dir + 'wordsim353.tsv', case_insensitive=False) )



# Ejemplos de similaridad..
print("v('king') - v('man') + v('woman') = ???")

print("Método 1. Suma directa")
king, man, woman = model['king'], model['man'], model['woman']
result = king - man + woman
print(model.most_similar(positive=[result], topn=6))
print("...............................")

print("Método 2. most_similar()")
print(model.most_similar(positive=['king', 'woman'], negative=['man'], topn=5))

print("¿Qué es más coseno-parecido a 'perro'? ¿'tanque' o 'gato'?")
dogcat = model.similarity("dog", "cat")
dogtank = model.similarity("dog", "tank")
if dogcat > dogtank:
	print("'gato'!")
else:
	print("....'tanque' (¿no quieres revisar algunos parámetros del entrenamiento?)")

# Ahora con WMD..
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
stop_words = stopwords.words('english')

sentence_obama = 'Obama speaks to the media in Illinois'
sentence_obama = sentence_obama.lower().split()
sentence_obama = [w for w in sentence_obama if w not in stop_words]
sentence_president = 'The president greets the press in Chicago'
sentence_president = sentence_president.lower().split()
sentence_president = [w for w in sentence_president if w not in stop_words]
sentence_orange = 'Oranges are my favorite fruit'
sentence_orange = sentence_orange.lower().split()
sentence_orange = [w for w in sentence_orange if w not in stop_words]

print(" D1 = 'Obama speaks to the media in Illinois'")
print(" D2 = 'The president greets the press in Chicago'")
print(" D3 = 'Oranges are my favorite fruit'")
print("...............................")
print(" WMD(D1, D2) = ")
print( model.wmdistance(sentence_obama, sentence_president) )
print(" WMD(D1, D3) = ")
print( model.wmdistance(sentence_obama, sentence_orange) )
print(" WMD(D2, D3) = ")
print( model.wmdistance(sentence_president, sentence_orange) )



# Visualización de embeddings de alguans palabras según nuestro modelo
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

some_words = ["dog",  "animal", "cat",  "mice", "vegetables", "king", "queen", "man", "woman"]

X = model[model.wv.vocab] # convertimos palabras del vocabulario
pca = PCA(n_components=2) # proyección bidimensional
result = pca.fit(X)
print("Ploteando proyección PCA en 2-dims..")
some_X = model[some_words]
some_transforms = pca.transform(some_X)
pyplot.scatter(some_transforms[:, 0], some_transforms[:, 1]) # scatter plot
# Etiquetamos puntos con las palabras correspondientes
for i, word in enumerate(some_words):
    pyplot.annotate(word, xy=(some_transforms[i, 0], some_transforms[i, 1]))
pyplot.savefig("PCA-w2v.png")