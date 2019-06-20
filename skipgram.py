# -*- coding:utf-8 -*-
# Author: @Jonathan Wang

import numpy as np
import tensorflow as tf

tf.reset_default_graph()

sentences = ["tensorflow is good programming language", "apples are common seen fruit", "jonathan is fancy coding lover"]

word_list = list(set(" ".join(sentences).split()))
print(word_list)

NUM_STEPS = 2
NUM_HIDDEN = 2
NUM_CLASS = len(word_list)

word2idx = {w: i for i, w in enumerate(word_list)}
idx2word = {i: w for i, w in enumerate(word_list)}

def get_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        inputs = [word2idx[w] for w in word[:2]] + [word2idx[w] for w in word[3:]]
        target = word2idx[word[2]]

        input_batch.append(np.eye(NUM_CLASS)[inputs])
        target_batch.append(np.eye(NUM_CLASS)[target])

    return input_batch, target_batch

X = tf.placeholder(tf.float32, [None, NUM_STEPS*2, NUM_CLASS])
Y = tf.placeholder(tf.float32, [None, NUM_CLASS])

inputs = tf.reshape(X, [-1, NUM_STEPS*NUM_CLASS*2])
W = tf.Variable(tf.random_normal([2*NUM_CLASS*NUM_STEPS, NUM_HIDDEN]))
b1 = tf.Variable(tf.random_normal([NUM_HIDDEN]))
V = tf.Variable(tf.random_normal([NUM_HIDDEN, NUM_CLASS]))
b2 = tf.Variable(tf.random_normal([NUM_CLASS]))

tanh = tf.nn.tanh(tf.matmul(inputs, W) + b1)
model = tf.matmul(tanh, V) + b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction = tf.argmax(model, 1)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, target_batch = get_batch(sentences)

for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Predict
predict =  sess.run([prediction], feed_dict={X: input_batch})

# Test
inputs = [sen.split()[:2] for sen in sentences] + [sen.split()[3:] for sen in sentences]
print([sen.split()[:2] for sen in sentences], [idx2word[n] for n in predict[0]], [sen.split()[3:] for sen in sentences])
