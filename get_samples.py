#####################################################
# Obtains random samples for testing the algorithms #
#####################################################


import os
import random
import codecs
import translationMatrixLS.py as tmls

sbin = os.getcwd()+'/vecs_en_full.bin'
svocab = os.getcwd()+'/vocab_en_full.txt'

en = tmls.generateVectors(sbin, svocab, 300, 'en')

indexes = random.sample(range(len(en[0])),100)
samples = dict((x,y) for x,y in zip(en[0][indexes],en[1][indexes]))
with open(os.getcwd()+'/samples_en_full.txt', 'w') as txt:
    for s in samples:
        txt.write(s+'\n')

'''
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

i = 0
with codecs.open(os.getcwd()+'/samples_en_full.txt', 'r', encoding = 'utf-8') as txt:
    for l in txt:
        if '#' in l:
            try:
                d_en_es[l.strip('\n')]
	    except KeyError:
                i += 1
		print(l.strip('\n'))
        else:
            try:
                d_en_es[l.strip('\n')]
	    except KeyError:
                i += 1
		print(l.strip('\n'))
print(i)
'''
