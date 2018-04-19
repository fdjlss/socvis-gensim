import gensim
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec, LineSentence
import gensim.downloader as api

from pprint import pprint
from copy import deepcopy
from multiprocessing import cpu_count
from smart_open import smart_open

# def write_wiki(wiki, name, titles = []):
#   with smart_open('{}.wiki'.format(name), 'wb') as f:
#     wiki.metadata = True
#     for text, (page_id, title) in wiki.get_texts():
#       if title not in titles:
#         f.write(b' '.join(text)+b'\n')
#         titles.append(title)
#   return titles

# new = WikiCorpus('enwiki-20180201-pages-articles.xml.bz2')
# newwiki = LineSentence('new.wiki')
# model = Word2Vec(newwiki, size=100, min_count=10, workers=4)
# model.save('newwikimodel')

# class MyText(object):
#   def __iter__(self):
#     for line in open(lee_train_file):
#       # assume there's one document per line, tokens separated by whitespace
#       yield line.lower().split()

# test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
# lee_train_file = test_data_dir + 'lee_background.cor'
# sentences = MyText()
# print(sentences)
# model = gensim.models.Word2Vec(sentences, min_count=5, size=200)
# model.accuracy('./questions-words.txt')

corpus_patent = api.load('patent-2017')
model_patent = Word2Vec(corpus_patent)
model_patent.save('./model_patent')

# corpus_wiki = api.load('wiki-english-20171001') # api.load('text8')
# model_wiki = Word2Vec(corpus_wiki)
model.most_similar("car")
