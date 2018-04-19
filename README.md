# Ejemplos para SocVis con word2vec de gensim
0. [Instalación](#instalación)
1. [Preparación de datos](#preparación-de-datos)
2. [Entrenamiento](#entrenamiento)
3. [Persistencia](#persistencia)
4. [Experimentos](#experimentos)
5. [Enlaces relevantes](#enlaces-relevantes)

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
sentences = [['arroz', 'con', 'pollo'], ['pastel', ,'de', 'papas']]
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

### Alt 3. Usando gensim-data
Podemos usar corpus y/o modelos preentrenados de [gensim-data](https://github.com/RaRe-Technologies/gensim-data)
```python
```


## Entrenamiento
Podemos hacer los pasos explícitos
```python
model = gensim.models.Word2Vec(min_count=1)  # modelo vacío
model.build_vocab(sentences)                 # puede ser un iterable
model.train(sentences, total_examples=model.corpus_count, epochs=new_model.iter) 
```


## Persistencia



## Experimentos
### Interfaz de similaridad

### WMD

### Visualizando los embeddings




## Enlaces relevantes
* [Sitio oficial gensim](https://radimrehurek.com/gensim/index.html)
* [C̶o̶r̶p̶i̶i̶ C̶o̶r̶p̶u̶s̶e̶s̶  Corpus (plural) y modelos preentrenados para uso en gensim con gloVe o w2v](https://github.com/RaRe-Technologies/gensim-data)
* [El tutorial de w2v más simple del mundo](https://rare-technologies.com/word2vec-tutorial/)
* [Notebooks con temas específicos](https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks)
* [Word2Vec tutorial con TensorFlow y Keras](http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/)
* \( [Formulario de Reclamos y Quejas de gensim](https://twitter.com/radimrehurek) \)
