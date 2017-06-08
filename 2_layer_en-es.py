from __future__ import print_function
import os
import random
import argparse
import tensorflow as tf
import numpy as np
import time
from scipy import spatial
import metrics
import codecs
from collections import OrderedDict
from annoy import AnnoyIndex
import translationMatrixLS.py as tmls

class Sgd:
    
    global num_steps, batch_size, emb_size, h1_size
    
    num_steps = 1501
    batch_size = 500
    emb_size = 300
    h1_size = 10000
    
    def __init__(self, sv, tv, svlang):
        s = sv[1]
        t = tv[1]
        '''
        indexes = random.sample(range(len(s)),len(s))
        p80 = len(s)*80/100
        p10 = len(s)*10/100
        train = indexes[:p80]
        valid = indexes[p80:p80+p10]
        test = indexes[p80+p10:]
	
        self.train_dataset = s[train]
        self.train_labels = t[train]
        self.valid_dataset = s[valid]
        self.valid_labels = t[valid]
        self.test_dataset = s[test]
        self.test_labels = t[test]
        '''
        self.sv = sv
        self.tv = tv

        self.svlang = svlang

    def train_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            # Input data. For the training data, we use a placeholder that
            # will be fed at tun time with a training minibatch.
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,emb_size))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,emb_size))
            tf_valid_dataset = tf.constant(self.valid_dataset, tf.float32)
            tf_valid_labels = tf.constant(self.valid_labels, tf.float32)
            tf_test_dataset = tf.constant(self.test_dataset, tf.float32)
            tf_test_labels = tf.constant(self.test_labels, tf.float32)

            # To do individual translations
            tf_input = tf.placeholder(tf.float32, shape=(1,emb_size))

            global_step = tf.Variable(0)
            
            # First layer
            w_shape1 = [emb_size,h1_size]
            weights1 = tf.get_variable("weights1", w_shape1, initializer=tf.contrib.layers.xavier_initializer())
            biases1 = tf.Variable(tf.zeros([h1_size]))

            # Forth layer
            w_shape2 = [h1_size,emb_size]
            weights2 = tf.get_variable("weights3", w_shape2, initializer=tf.contrib.layers.xavier_initializer())
            biases2 = tf.Variable(tf.zeros([emb_size]))

            def calculate_embeddings(tf_in):
                h1 = tf.nn.relu(tf.matmul(tf_in, weights1) + biases1)
                return tf.matmul(h1, weights2) + biases2

            embeddings = calculate_embeddings(tf_train_dataset)
            
            beta = 0.0001
            batch_loss = self.real_loss(embeddings,tf_train_labels)
            loss = batch_loss + beta*tf.nn.l2_loss(weights1) + beta*tf.nn.l2_loss(weights2)

            # Optimizer
            alpha = 2.5
	    learning_decay = 1
            learning_rate = tf.train.exponential_decay(alpha, global_step, 500, learning_decay)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

            # Predictions for the training, validation and test data
            train_prediction = embeddings

            v3 = calculate_embeddings(tf_valid_dataset)
            valid_loss = self.real_loss(v3, tf_valid_labels)
            
            t3 = calculate_embeddings(tf_test_dataset)
            test_loss = self.real_loss(t3,tf_test_labels)
            
            # For results
            output3 = calculate_embeddings(tf_input)
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
	    saver = tf.train.Saver()
            for step in range(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (self.train_labels.shape[0] - batch_size)
                # Generate a minibatch.
                batch_data = self.train_dataset[offset:(offset + batch_size), :]
                batch_labels = self.train_labels[offset:(offset + batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                _, l, rl, predictions = session.run(
                    [optimizer, loss, batch_loss, train_prediction], feed_dict=feed_dict)
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Validation loss: %.5f" % valid_loss.eval())
                    print("Real loss:", rl)
            print("Test accuracy: %.5f" % test_loss.eval())
	    tf.add_to_collection('vars', output3)
            tf.add_to_collection('vars', tf_input)
            save_path = saver.save(session, os.getcwd()+"/ckpt/translation_matrix.ckpt",
                                   meta_graph_suffix='meta', write_meta_graph=True)
            
    def test_graph(self, ttype):

        with tf.Session() as session:
            # Restore previous saved graph
            saver = tf.train.import_meta_graph(os.getcwd()+"/ckpt/translation_matrix.ckpt.meta")
            saver.restore(session,os.getcwd()+"/ckpt/translation_matrix.ckpt")
            output2 = tf.get_collection('vars')[0]
            tf_input2 = tf.get_collection('vars')[1]

            
            # Get values for translation
            print('Getting samples')
            start = time.time()
            samples = OrderedDict()
            aux = dict((x,y) for x,y in zip(self.sv[0], self.sv[1]))
            #with codecs.open(os.getcwd()+'/vocab_'+self.svlang+'.txt','r',encoding='utf-8') as s:
            with codecs.open(os.getcwd()+'/samples_'+self.svlang+'_full.txt','r',encoding='utf-8') as s:
                for line in s:
                    	samples[line.strip('\n')] = aux[line.strip('\n')]
            print('Fin:', time.time()-start)

            print('Getting translations')
            start = time.time()
            embeddings = []
            for sample in samples:
      	        feed_dict = {tf_input2: samples[sample].reshape(1,emb_size)}
      	        translation = session.run(output2,feed_dict=feed_dict)
		translation = translation.reshape(emb_size,)
                embeddings.append(translation)
            print('Fin:', time.time()-start)
            
            '''
            # Indexing target vectors for Annoy
            print('Indexing Annoy vectors')
            start = time.time()
            norms = np.linalg.norm(self.tv[1],axis=1).reshape(-1,1)
            vectors = self.tv[1]/norms
            t = AnnoyIndex(300)
            for i,v in zip(xrange(len(vectors)),vectors):
                t.add_item(i,v)
            t.build(1000)
            t.save('index1000.ann')
            print('Fin:',time.time()-start)
            '''
	    
            print('Loading index file')
            start = time.time()
            t = AnnoyIndex(emb_size)
            t.load('index1000.ann')
            print('Fin:', time.time()-start)
	               
	    if ttype == 'distance':
                metrics.distances(embeddings, samples, self.sv, self.tv, 'NN2')
		#metrics.distance_from_algorithm(samples, self.sv, self.tv, 'NN2')
            else:
		start = time.time()
                metrics.top5_2(t,embeddings, samples, self.sv, self.tv, 'NN2')
                #metrics.top5(t, embeddings, samples, self.sv, self.tv, 'NN2')
		print('Fin:',time.time()-start)
            
	       
    def real_loss(self,tf1,tf2):
        return tf.losses.cosine_distance(tf.nn.l2_normalize(tf1,dim=1),tf.nn.l2_normalize(tf2,dim=1),dim=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--top5', action='store_true')
    parser.add_argument('--distance', action='store_true')
    args = vars(parser.parse_args())
    sbin = os.getcwd()+'/vecs_'+args['source']+'.bin'
    svocab = os.getcwd()+'/vocab_'+args['source']+'.txt'
    tbin = os.getcwd()+'/vecs_'+args['target']+'.bin'
    tvocab = os.getcwd()+'/vocab_'+args['target']+'.txt'
    sbin_full = os.getcwd()+'/vecs_'+args['source']+'_full.bin'
    svocab_full = os.getcwd()+'/vocab_'+args['source']+'_full.txt'
    tbin_full = os.getcwd()+'/vecs_'+args['target']+'_full.bin'
    tvocab_full = os.getcwd()+'/vocab_'+args['target']+'_full.txt'

    en = tmls.generateVectors(sbin_full, svocab_full, 300, args['source'])
    # Descomentar estas dos lineas para entrenar y comentar la de test
    #es = tmls.generateVectors(tbin, tvocab, 300, args['target'])
    #es = tmls.paralellizeVectors(en,es,'en')
    # Descomentar esta linea para test y comentar las de entrenamiento
    es = tmls.generateVectors(tbin_full, tvocab_full, 300, args['target'])

    s = Sgd(en,es,'en')

    if args['train']:
        s.train_graph()
    if args['distance']:
        s.test_graph('distance')
    if args['top5']:
        s.test_graph('top5')
    if not args['train'] and not args['distance'] and not args['top5']:
            print('Introduce the option for training (\'--train\') or a type of test (\'--distance\' or \'--top5\' )')

    
if __name__ == "__main__":
    main()
