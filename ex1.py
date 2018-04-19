import gensim
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec, LineSentence
from pprint import pprint
from copy import deepcopy
from multiprocessing import cpu_count
from smart_open import smart_open

def write_wiki(wiki, name, titles = []):
  with smart_open('{}.wiki'.format(name), 'wb') as f:
    wiki.metadata = True
    for text, (page_id, title) in wiki.get_texts():
      if title not in titles:
        f.write(b' '.join(text)+b'\n')
        titles.append(title)
  return titles

# old, new = [WikiCorpus('enwiki-{}-pages-articles.xml.bz2'.format(ymd)) for ymd in ['20101011', '20180201']]
# old_titles = write_wiki(old, 'old')
# all_titles = write_wiki(new, 'new', old_titles)
# oldwiki, newwiki = [LineSentence(f+'.wiki') for f in ['old', 'new']]

new = WikiCorpus('enwiki-20180201-pages-articles.xml.bz2')
newwiki = LineSentence('new.wiki')
model = Word2Vec(oldwiki, size=100, min_count=10, workers=4)
model.save('newwikimodel')

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