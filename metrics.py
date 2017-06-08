#####################################################
# Class containing all the metrics methods          #
# mentioned in the memory of the project.           #
#####################################################

from __future__ import print_function
import os
import random
import argparse
import time
import tensorflow as tf
import numpy as np
from scipy import spatial
import codecs
from annoy import AnnoyIndex
import translationMatrixLS.py as tmls

def distances(embeddings, tokens, en, es, algorithm):
    d_en = dict((x,y) for x,y in zip(en[0],en[1]))
    d_es = dict((x,y) for x,y in zip(es[0],es[1]))
    d_en_es = dict((x,y) for x,y in zip(en[0],es[0]))
    print('Computing distance between top5 and the result of the algorithm.')
    top = 0
    medias = {}
    medias_top = []
    losses = {}
    losses_top = []
    for i in range(0,6):
        medias.setdefault(i,[])
        losses.setdefault(i,[])
    with open(os.getcwd()+'/d-500-'+algorithm+'.tsv','w') as tsv:
        with open(os.getcwd()+'/d-media.tsv','a') as m:
            for token,embedding in zip(tokens,embeddings):
                n = tmls.getSimilars(embedding, 5, es)
                top = tmls.getTop(d_en_es[token],n,5)
                #distances = [spatial.distance.cosine(embedding,d_es[d]) for d in n]
		distances = map(lambda x: spatial.distance.cosine(embedding,d_es[x]), n)
                media = np.mean(np.asarray(distances))
                loss = spatial.distance.cosine(embedding,d_es[d_en_es[token]])
                medias[top].append(media)
                losses[top].append(loss)
                tsv.write(token+'\t'+str(n)+'\t'+str(loss)+'\t'+str(media)+'\n')
		print(media)
            m.write(algorithm+'\n')
            for top in medias:
                if medias[top]:
                    media_top = np.mean(np.asarray(medias[top]))
                    loss_top = np.mean(np.asarray(losses[top]))
                    medias_top.append(media_top)
                    losses_top.append(loss_top)
                    m.write(str(top)+'\t'+str(loss_top)+'\t'+str(media_top)+'\t'+str(len(medias[top]))+'\n')
            m.write('Total:\t'+str(np.mean(np.asarray(losses_top)))+'\t'+str(np.mean(np.asarray(medias_top)))+'\n')

def top5(t, embeddings, tokens, en, es, algorithm):
  d_en_es = {}
  d_es_en = {}
  with codecs.open(os.getcwd()+'/../word2vec/en2es-lemma-dict.txt','r',encoding='utf-8') as lemmas:
    for line in lemmas:
      (key,val) = line.split(':') #EN -> SP
      d_en_es[key] = val.strip('\n')
      (val,key) = line.split(':') #SP -> EN
      d_es_en[key.strip('\n')] = val
  with open(os.getcwd()+'/../word2vec/en_es.txt','r') as syncons:
    for line in syncons:
      (key,val) = line.split()
      d_en_es[key] = val
      (val,key) = line.split()
      d_es_en[key] = val
  print('Computing top5 of translations')
  top = 0
  exists = 0
  i = 0
  with codecs.open(os.getcwd()+'/t5-'+str(len(tokens))+'-'+algorithm+'.tsv','w',encoding='utf-8') as tsv:
    #with open(os.getcwd()+'/t5-media.tsv','a') as m:
      for token,embedding in zip(tokens,embeddings):
        fin = 0
        start = 0
        if token in d_en_es:
	  start = time.time()
          items = t.get_nns_by_vector(embedding, 5, search_k=100000)
          fin = time.time()
          #n = tmls.getSimilars(embedding, 5, es)
          n = es[0][items]
          top = tmls.getTop(d_en_es[token],n,5)
          if d_en_es[token] in n:
            exists = 1 #+= 1
          tsv.write(token+'\t'+str(top)+'\t'+str(exists)+'\t'+str(1)+'\n')
          exists = 0
	else:
	  tsv.write(token+'\t'+str(0)+'\t'+str(0)+'\t'+str(0)+'\n')
          #m.write(algorithm+'_annoy\t'+str(top/float(len(tokens)))+'\t'+str(exists/float(len(tokens)))+'\n')
	i+=1
        print(i, fin-start)
          


def top5_2(t, embeddings, tokens, en, es, algorithm):
  d_en_es = {}
  d_es_en = {}
  with open(os.getcwd()+'/../word2vec/en2es-lemma-dict.txt','r') as lemmas:
    for line in lemmas:
      (key,val) = line.split(':') #EN -> SP
      d_en_es[key] = val.strip('\n')
      (val,key) = line.split(':') #SP -> EN
      d_es_en[key.strip('\n')] = val
  with open(os.getcwd()+'/../word2vec/en_es.txt','r') as syncons:
    for line in syncons:
      (key,val) = line.split()
      d_en_es[key] = val
      (val,key) = line.split()
      d_es_en[key] = val
    d_es = dict((x,y) for x,y in zip(es[0],es[1]))
    print('Computing top5 of translations')

    with codecs.open(os.getcwd()+'/t5-100-'+algorithm+'_annoy.tsv','w',encoding='utf-8') as tsv:
        for token,embedding in zip(tokens,embeddings):
            tsv.write(token)
            items, distances = t.get_nns_by_vector(embedding, 5, include_distances=True)
            #n = tmls.getSimilars(embedding, 5, es)
            n = es[0][items]
            for translation in n:
		tsv.write('\t'+translation.strip('\n')+'\t'+str(spatial.distance.cosine(embedding,d_es[translation])))
            try:
                tsv.write('\t'+str(1)+'\t'+d_en_es[token]+'\t'+str(spatial.distance.cosine(embedding,d_es[d_en_es[token]]))+'\n')
            except KeyError:
                tsv.write('\t'+str(0)+'\n')
            print(token)

def distance_from_algorithm(tokens, en, es, algorithm):
    d_en_es = dict((x,y) for x,y in zip(en[0],es[0]))
    d_es = dict((x,y) for x,y in zip(es[0],es[1]))
    top = 0
    media = 0
    with open(os.getcwd()+'/distances_from_algorithm.tsv','a') as m:
        for token in tokens:
            n = tmls.getSimilars(d_es[d_en_es[token]], 5, es)
	    distances = map(lambda x: spatial.distance.cosine(d_es[d_en_es[token]],d_es[x]), n)
            media += np.mean(np.asarray(distances))
            print(media/float(500))
        m.write(algorithm+'\t'+str(media/float(500))+'\n')

