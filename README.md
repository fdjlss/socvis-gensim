# Ejemplos para SocVis con word2vec de gensim
0. [Instalación](#instalación)
1. [Preparación de datos](#preparación-de-datos)
2. [Entrenamiento](#entrenamiento)
3. [Persistencia](#persistencia)
4. [Evaluación](#evaluación)
5. [Experimentos](#experimentos)
6. [Enlaces relevantes](#enlaces-relevantes)

## Instalación
### Requerimientos
Python 2.x (x >= 6), 3.x (x >= 3) <br>
NumPy >= 1.3 <br>
SciPy >= 0.7
### gensim
```
easy_install -U gensim
```
o bien
```
pip install --upgrade gensim
```
Más info: [Installation](https://radimrehurek.com/gensim/install.html)

## Preparación de datos
```python
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```
### Alt 1. Forma básica
Cargar todo en memoria
```python
sentences = [['arroz', 'con', 'pollo'], ['pastel', 'de', 'papas']]
model = gensim.models.Word2Vec(sentences)
```

### Alt 2. ¿Y si tengo (muchos) más datos?
Le pasamos un iterador memory-friendly..
```python
import os
from smart_open import smart_open

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for filename in os.listdir(self.dirname):
            for line in smart_open(os.path.join(self.dirname, filename), 'rb'):
                yield line.split()
```
Si `os.listdir('./data/') = ['f1.txt', 'f2.txt', ..., 'fn.txt']` ..
```python
sentences = MySentences('./data/')
model = gensim.models.Word2Vec(sentences)
```
Todo el pre-procesamiento (lowercase, remover números, ...) conviene hacerla en la definición `MySentences`. *Todo lo que se necesita es que el input "yields" una frase (lista de palabras utf8) una tras otra.*
### Alt 3. Usando gensim-data
Podemos usar corpus y/o modelos preentrenados de [gensim-data](https://github.com/RaRe-Technologies/gensim-data)
```python
import gensim.downloader as api
```
#### Cargando corpus
```python
corpus = api.load('text8')  # descarga de corpus y devuelto como un iterable
model = gensim.models.Word2Vec(corpus)  # train a model from the corpus
```
#### Cargando modelo
```python
model = api.load('word2vec-google-news-300') # listo para usar
```
Estas descargas (de modelo o corpus) van a una carpeta temporal.
Para descargar permanentemente y devolver el path, cambiamos el valor por defecto del argumento `return_path`:
```python
print(api.load("20-newsgroups", return_path=True))
```




## Entrenamiento
Podemos hacer los explícitos los pasos en la creación del modelo
```python
model = gensim.models.Word2Vec()    # modelo vacío
model.build_vocab(sentences)        # puede ser un iterable
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter) 
```
### (Algunos) Parámetros del entrenamiento
```python
class gensim.models.word2vec.Word2Vec(
    sentences=None, 
    size=100,           # n-dims de los vectores
    alpha=0.025,        # learning rate inicial
    window=5,           # distancia máxima entre palabra y su contexto
    min_count=5,        # se descartan palabras con frecuencia total < min_count
    max_vocab_size=None,# si hay más palabras, saca las con menor frecuencia
    sg=0,               # 0: CBoW, 1: Skip-gram
    hs=0,               # 0: HSMax, 1: NS
    negative=5,         # limita "palabras ruidosas" tener para el NS (sólo si > 0)
    cbow_mean=1,        # 0: suma el contexto, 1: promedio (sólo si sg=0)
    iter=5,             # iter+1 épocas de entrenamiento (la primera para construir vocabulario)
    callbacks=())       # callback para ejecutar en etapas específicas del entrenamiento
```
...y más en la [API Reference de gensim](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec)
Modelos word2vec de gensim se guardan como matrices NumPy de floats de tamaño `len(w2v_model.wv.vocab)`\*`model.size`


## Persistencia
```python
model.save('./mi-modelo')
del model
model = gensim.models.Word2Vec.load('./mi-modelo')
```

### Online training
Podemos parcialmente entrenar modelo con algunos documentos, guardarlo, y continuar luego el entrenamiento con más documentos con palabras posiblemente nuevas para el vocabulario:
```python
model = gensim.models.Word2Vec.load('./mi-modelo')
more_sentences = [['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']]
model.build_vocab(more_sentences, update=True) # importante update=True
model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)
```

## Evaluación
Gensim incluye algunos datasets contra los que hacer evaluaciones del modelo.
En `questions-words.txt`, dataset de Google, se encuentran analogías sintácticas y semánticas de la forma a:a' :: b:b'. Con `accuracy()` hacemos que el modelo responda a:a' :: b:__ y se obtienen puntajes de aciertos.
```python
import os
import gensim
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
```
```python
>> model.accuracy(test_data_dir + 'questions-words.txt', case_insensitive=False)

2018-04-24 14:54:13,412 : INFO : precomputing L2-norms of word weight vectors
2018-04-24 14:54:14,321 : INFO : family: 58.2% (178/306)
2018-04-24 14:54:16,506 : INFO : gram1-adjective-to-adverb: 17.6% (133/756)
2018-04-24 14:54:17,393 : INFO : gram2-opposite: 21.2% (65/306)
2018-04-24 14:54:21,031 : INFO : gram3-comparative: 56.1% (707/1260)
2018-04-24 14:54:22,494 : INFO : gram4-superlative: 35.2% (178/506)
2018-04-24 14:54:25,358 : INFO : gram5-present-participle: 31.1% (309/992)
2018-04-24 14:54:29,224 : INFO : gram7-past-tense: 31.9% (425/1332)
2018-04-24 14:54:32,103 : INFO : gram8-plural: 45.4% (450/992)
2018-04-24 14:54:33,981 : INFO : gram9-plural-verbs: 29.2% (190/650)
2018-04-24 14:54:33,982 : INFO : total: 37.1% (2635/7100)
```
Con `evaluate_word_pairs()` se evalúan las correlaciones que hace el modelo contra juicios de similitud humana con el coeficiente de correlación de Pearson y el coeficiente de correlación de Spearman, como realizado en el [primero de los papers de Mikolov et al. del 2013](https://arxiv.org/abs/1301.3781)
```python
>> model.evaluate_word_pairs(test_data_dir + 'wordsim353.tsv', case_insensitive=False)

2018-04-24 14:58:26,198 : INFO : Pearson correlation coefficient against /home/jschellman/twitsrc/lib/python3.5/site-packages/gensim/test/test_data/wordsim353.tsv: 0.6565
2018-04-24 14:58:26,198 : INFO : Spearman rank-order correlation coefficient against /home/jschellman/twitsrc/lib/python3.5/site-packages/gensim/test/test_data/wordsim353.tsv: 0.6651
2018-04-24 14:58:26,198 : INFO : Pairs with unknown words ratio: 5.7%
((0.6564798796510927, 1.986859872406572e-42),
 SpearmanrResult(correlation=0.6650673182540592, pvalue=6.992561183956873e-44),
 5.6657223796034)
```
Nota (de [este notebook](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb) ): 
*Once again, good performance on Google's or WS-353 test set doesn’t mean word2vec will work well in your application, or vice versa. It’s always best to evaluate directly on your intended task. For an example of how to use word2vec in a classifier pipeline, see this [tutorial](https://github.com/RaRe-Technologies/movie-plots-by-genre).*

## Experimentos
Podemos obtener el embedding de una palabra con..
```python
>> model['raccoon']
array([ 1.0514954 , -1.0882462 ,  0.5977194 , -1.8239858 , -1.1860061 ,
        0.33046788, -1.0863892 , -2.48743   ,  1.2902331 , -1.0205759 ,
       -1.9141145 ,  1.3184149 ,  0.17544365, -1.5896797 ,  0.19770409,
        1.6801156 ,  0.9820397 ,  0.7584974 ,  2.3135023 , -0.30250603,
        0.3365632 , -1.5647495 ,  0.04142677,  0.03303481, -1.0773582 ,
        1.0583849 , -0.49168533,  2.0129025 , -0.61064714,  0.37005958,
        0.43913102, -0.10467256,  2.1104326 ,  0.15914543,  0.47913054,
       -0.57707447,  0.4913059 , -0.8651208 , -0.18684553,  1.8520341 ,
        1.235165  , -1.2043599 , -1.131661  , -2.406969  , -1.5776978 ,
       -0.40217164,  0.5498088 , -0.26626858, -0.71042967,  0.32916796,
       -0.05505305,  0.90935963, -1.3081107 , -0.7167593 ,  1.9541544 ,
       -1.9741548 , -1.1004103 ,  1.5270965 , -1.5337243 ,  0.9146289 ,
        0.11212104,  1.0708483 ,  0.16307715,  0.717044  ,  1.0354604 ,
        0.59897286,  0.9068039 ,  1.2016143 ,  0.47849917, -0.17389452,
        0.82398045,  0.5355287 , -0.15972371, -0.39870992,  0.21480806,
       -0.2749907 ,  0.46969727, -0.96976584, -0.3214242 , -0.10745487,
       -1.1139208 , -0.64185214,  2.0910223 ,  0.6251665 , -0.3245018 ,
       -0.19661197,  1.4374366 ,  1.2764899 , -0.20185828,  1.5685318 ,
        1.1295198 ,  0.09176906,  1.2449435 , -0.4895615 , -1.6255064 ,
        0.18820573, -0.45227936,  0.09359731,  0.9148596 ,  0.50943875],
      dtype=float32)
```
Cálculo de la log-probabilidad de una frase (por ahora sólo implementado con hs=1 y sg=1):
```python
>> model.score(["video game".split()])
array([-10.407113], dtype=float32)
```

### Interfaz de similaridad
#### Algunos ejemplos:
El famoso v('king') - v('man') + v('woman') ...
```python
>> model.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)
[('queen', 0.7044106721878052), ('princess', 0.6393158435821533), ('empress', 0.6258441209793091), ('prince', 0.624630331993103), ('elizabeth', 0.6178595423698425)]
```
Obtenemos similares resultados si hacemos directamente la aritmética:
```python
>> king, man, woman = model['king'], model['man'], model['woman']
>> result = king - man + woman
>> model.most_similar(positive=[result], topn=6)
[('king', 0.861211359500885), ('queen', 0.7236999273300171), ('prince', 0.6600159406661987), ('princess', 0.6493911743164062), ('empress', 0.6436548233032227), ('elizabeth', 0.6303344964981079)]
```
La diferencia es que `most_similar()` primero precalcula la L2-normalización de los vectores, hace la suma con los vectores normalizados, luego revisa los `topn` vecinos cercanos del vector resultante y remueve el input del output. Todos estos cálculos de distancia se hacen con similaridad coseno.

```python
>> model.wv.doesnt_match("dog horse cat mouse human".split())
'human'
```
```python
>> model.similarity("dog", "cat")
0.8375475584168148

>> model.similarity("dog", "tank")
0.15581708106840225
```
#### WMD
```python
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
```
```python
>> model.wmdistance(sentence_obama, sentence_president)
2.596937661854488

>> model.wmdistance(sentence_obama, sentence_orange)
3.238054760814676

>> model.wmdistance(sentence_president, sentence_orange)
3.6691700002904417
```
Como WMD es prácticamente distancia euclidiana, convendría precalcular previamente las L2-normalizaciones de los vectores con ```model.init_sims()``` (para quedarse sólo con las normalizaciones ```init_sims(replace=True)``` antes de computar las wmd-distancias. Para eliminar todos los vectores normalizados: ```model.clear_sims()```

### Visualizando los embeddings
#### Proyección PCA
```python
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

some_words = ["dog",  "animal", "cat",  "mice", "vegetables", "king", "queen", "man", "woman"]

X = model[model.wv.vocab] # convertimos palabras del vocabulario
pca = PCA(n_components=2) # proyección bidimensional
result = pca.fit(X)

some_X = model[some_words]
some_transforms = pca.transform(some_X)
pyplot.scatter(some_transforms[:, 0], some_transforms[:, 1]) # scatter plot
# Etiquetamos puntos con las palabras correspondientes
for i, word in enumerate(some_words):
    pyplot.annotate(word, xy=(some_transforms[i, 0], some_transforms[i, 1]))
pyplot.savefig("PCA-w2v-3.png")
```

## Enlaces relevantes
* [Sitio oficial gensim](https://radimrehurek.com/gensim/index.html)
* [Gensim-data: corpus y modelos preentrenados para uso en gensim con gloVe o w2v](https://github.com/RaRe-Technologies/gensim-data)
* [El tutorial de w2v más simple del mundo](https://rare-technologies.com/word2vec-tutorial/)
* [Notebooks con temas específicos](https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks)
* [Word2Vec tutorial con TensorFlow y Keras](http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/)
* \( [Formulario de Reclamos y Quejas de gensim](https://twitter.com/radimrehurek) \)
