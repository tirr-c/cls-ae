
# coding: utf-8

# In[1]:


import random

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")


# In[3]:


train_x = mnist.train.images
train_y = mnist.train.labels

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

idx = np.argsort(train_y)
train_x = train_x[idx]
train_y = train_y[idx]

print(train_x.shape)
print(train_y.shape)


# In[4]:


cnt = -1
train_x_per_class = {}
for i in range(train_x.shape[0]):
    if cnt == train_y[i]:
        train_x_per_class[cnt].append(train_x[i])
    else:
        cnt += 1
        train_x_per_class[cnt] = []

for i in range(len(train_x_per_class)):
    print(len(train_x_per_class[i]))


# In[5]:


total_epochs = 1000
batch_size = 100
learning_rate = 0.001
random_size = 100
image_size = 28*28
z_dim = 120


# In[6]:


init = tf.random_normal_initializer(mean=0, stddev=0.15)

def encoder(x, reuse=False):
    l = [image_size, 50, 30, z_dim]
    with tf.variable_scope(name_or_scope="mnist_encoder", reuse=reuse) as scope:
        out1 = tf.layers.dense(x, l[1], activation=tf.nn.relu)
        out2 = tf.layers.dense(out1, l[2], activation=tf.nn.relu)
        output = tf.layers.dense(out2, l[3], activation=tf.nn.sigmoid)
        return output

def decoder(z, reuse=False):
    l = [z_dim, 30, 50, image_size]
    with tf.variable_scope(name_or_scope="mnist_decoder", reuse=reuse) as scope:
        out1 = tf.layers.dense(z, l[1], activation=tf.nn.relu)
        out2 = tf.layers.dense(out1, l[2], activation=tf.nn.relu)
        output = tf.layers.dense(out2, l[3], activation=tf.nn.sigmoid)
        return output


# In[7]:


def random_z():
    return np.random.normal(size=[1, z_dim])

def dist(x1, x2):
    d = 0
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            l = x1[i] - x2[j]
            d += np.sqrt(np.dot(l, l))
    return d / (x1.shape[0] * x2.shape[0])


# In[8]:


g = tf.Graph()

with g.as_default():
    X1 = tf.placeholder(tf.float32, [None, image_size])
    X2 = tf.placeholder(tf.float32, [None, image_size])
    Z = tf.placeholder(tf.float32, [1, z_dim])

    enc1 = encoder(X1)
    dec1 = decoder(enc1)

    enc2 = encoder(X2, True)
    dec2 = decoder(enc2, True)

    loss1 = tf.reduce_mean(tf.square(X1 - dec1))
    loss2 = tf.reduce_mean(tf.square(X2 - dec2))
    avg_loss = -100 * tf.square(tf.reduce_mean(X1) - tf.reduce_mean(X2))
    var_loss1 = tf.reduce_mean(tf.reduce_mean(tf.square(X1)) - tf.square(tf.reduce_mean(X1)))
    var_loss2 = tf.reduce_mean(tf.reduce_mean(tf.square(X2)) - tf.square(tf.reduce_mean(X2)))
    loss = loss1 + loss2 + avg_loss + var_loss1 + var_loss2

    t_vars = tf.trainable_variables()
    e_vars = [var for var in t_vars if "encoder" in var.name]
    d_vars = [var for var in t_vars if "decoder" in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss, var_list=e_vars + d_vars)


# In[9]:


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(total_epochs):
        class1 = random.randint(0, 9)
        class2 = random.randint(0, 9)
        while class1 == class2:
            class1 = random.randint(0, 9)
            class2 = random.randint(0, 9)

        data1 = np.copy(train_x_per_class[class1])
        data2 = np.copy(train_x_per_class[class2])
        np.random.shuffle(data1)
        np.random.shuffle(data2)

        total_batchs = int(min(len(data1), len(data2)) / batch_size)
        for batch in range(total_batchs):
            batch_x1 = data1[(batch * batch_size):((batch + 1) * batch_size)]
            batch_x2 = data2[(batch * batch_size):((batch + 1) * batch_size)]

            sess.run(train, feed_dict={X1: batch_x1, X2: batch_x2})

        if epoch % 50 == 0:
            print("=== Epoch ", epoch, " ===")
            loss_r = sess.run(loss, feed_dict={X1: batch_x1, X2: batch_x2})
            print("loss: ", loss_r)
            
            x_list = []
            for cls in range(10):
                candidate = train_x_per_class[cls]
                x_list.append(candidate[random.randint(0, len(candidate) - 1)])
            x_list = np.asarray(x_list)
            
            gen_list = sess.run(dec1, feed_dict={X1: x_list})
            origimg = x_list.reshape([-1, 28, 28])
            genimg = gen_list.reshape([-1, 28, 28])
            
            _, axes = plt.subplots(5, 4)
            for cls in range(10):
                axes[cls // 2, (cls % 2) * 2 + 0].imshow(origimg[cls])
                axes[cls // 2, (cls % 2) * 2 + 0].axis('off')
                axes[cls // 2, (cls % 2) * 2 + 1].imshow(genimg[cls])
                axes[cls // 2, (cls % 2) * 2 + 1].axis('off')
            plt.show()

        if epoch % 200 == 0:
            print('=== Distances ===')
            for i in range(10):
                for j in range(i, 10):
                    gen1, gen2 = sess.run(
                        [enc1, enc2],
                        feed_dict={X1: train_x_per_class[i][0:100], X2: train_x_per_class[j][0:100]}
                    )
                    print(i, j, dist(gen1, gen2))
                print()

