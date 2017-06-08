from __future__ import print_function
import os
import random
import argparse
import time
import tensorflow as tf
import numpy as np
from scipy import spatial
import metrics
import translationMatrixLS.py as tmls

class Sgd:
    
    global num_steps, batch_size, emb_size
    
    num_steps = 801
    batch_size = 500
    emb_size = 300
    
    def __init__(self, sv, tv, svlang):
        s = sv[1]
        t = tv[1]
        indexes = random.sample(range(len(s)),len(s))
        p80 = len(s)*80/100
        p10 = len(s)*10/100
        train = indexes[:p80]#range(0,5000)
        valid = indexes[p80:p80+p10]
        test = indexes[p80+p10:]

        self.train_dataset = s[train]
        self.train_labels = t[train]
        self.valid_dataset = s[valid]
        self.valid_labels = t[valid]
        self.test_dataset = s[test]
        self.test_labels = t[test]

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

            # Variables
            w_shape = [emb_size,emb_size]
            weights = tf.get_variable("weights", w_shape, initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.Variable(tf.zeros([emb_size]))

            # Training computation
            embeddings = tf.matmul(tf_train_dataset, weights) + biases

            beta = 0.001
            batch_loss = self.real_loss(embeddings,tf_train_labels)
            loss = batch_loss + beta*tf.nn.l2_loss(weights)

            # Optimizer
            alpha = 5
            optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

            # Predictions for the training, validation and test data
            train_prediction = embeddings
            valid_prediction = tf.matmul(tf_valid_dataset, weights) + biases
            valid2 = self.real_loss(valid_prediction, tf_valid_labels)
            test_prediction = tf.matmul(tf_test_dataset, weights) + biases
            test2 = self.real_loss(test_prediction,tf_test_labels)
            
            # For results
            output = tf.matmul(tf_input, weights) + biases
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
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
                    print("Validation loss: %.5f" % valid2.eval())
                    print("Real loss:", rl)
            print("Test accuracy: %.5f" % test2.eval())
            tf.add_to_collection('vars', output)
            tf.add_to_collection('vars', tf_input)
            saver = tf.train.Saver()
            save_path = saver.save(session, os.getcwd()+"/ckpt/translation_matrix.ckpt",
                                   meta_graph_suffix='meta', write_meta_graph=True)

    def test_graph(self, ttype):
        with tf.Session() as session:
            # Restore previous saved graph
            saver = tf.train.import_meta_graph(os.getcwd()+"/ckpt/translation_matrix.ckpt.meta")
            saver.restore(session,os.getcwd()+"/ckpt/translation_matrix.ckpt")
            output2 = tf.get_collection('vars')[0]
            tf_input2 = tf.get_collection('vars')[1]
            
            # Get samples
            samples = {}
            aux = dict((x,y) for x,y in zip(self.sv[0], self.sv[1]))
            with open(os.getcwd()+'/samples_'+self.svlang+'.txt','r') as s:
                for line in s:
                    samples[line.strip('\n')] = aux[line.strip('\n')] 

            #Dictionaries for translations and vectors itself
            d_en_es = dict((x,y) for x,y in zip(self.sv[0], self.tv[0]))
            d_es_en = dict((x,y) for x,y in zip(self.tv[0], self.sv[0]))
            
            d_en = dict((x,y) for x,y in zip(self.sv[0], self.sv[1]))
            d_es = dict((x,y) for x,y in zip(self.tv[0], self.tv[1]))

            embeddings = []
            for sample in samples:
      	        feed_dict = {tf_input2: samples[sample].reshape(1,emb_size)}
      	        translation = session.run(output2,feed_dict=feed_dict)
                embeddings.append(translation.reshape(emb_size,))
                
	    if ttype == 'distance':
                #metrics.distances(embeddings, samples, self.sv, self.tv, 'NN1')
		metrics.distance_from_algorithm(samples, self.sv, self.tv, 'NN1')
            else:
                metrics.top5(embeddings, samples, self.sv, self.tv, 'NN1')


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
    en = tmls.generateVectors(sbin, svocab, 300, args['source'])
    es = tmls.generateVectors(tbin, tvocab, 300, args['target'])
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
