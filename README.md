# Ejemplos para SocVis con word2vec de gensim
0. [Instalación](#instalación)
1. [Preparación de datos](#preparación-de-datos)
2. [Entrenamiento](#entrenamiento)
3. [Persistencia](#persistencia)
4. [Experimentos](#experimentos)
5. [Enlaces relevantes](#enlaces-relevantes)

## 0. Instalación
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

## 1. Preparación de datos
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




## 2. Entrenamiento
Podemos hacer los pasos explícitos
```python
model = gensim.models.Word2Vec()  # modelo vacío
model.build_vocab(sentences)                 # puede ser un iterable
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

### Online training




## 3. Persistencia



## 4. Experimentos
### Interfaz de similaridad

### WMD

### Visualizando los embeddings




## 5. Enlaces relevantes
* [Sitio oficial gensim](https://radimrehurek.com/gensim/index.html)
* [C̶o̶r̶p̶i̶i̶ C̶o̶r̶p̶u̶s̶e̶s̶  Corpus (plural) y modelos preentrenados para uso en gensim con gloVe o w2v](https://github.com/RaRe-Technologies/gensim-data)
* [El tutorial de w2v más simple del mundo](https://rare-technologies.com/word2vec-tutorial/)
* [Notebooks con temas específicos](https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks)
* [Word2Vec tutorial con TensorFlow y Keras](http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/)
* \( [Formulario de Reclamos y Quejas de gensim](https://twitter.com/radimrehurek) \)
