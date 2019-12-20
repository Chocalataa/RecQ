# coding:utf8
from baseclass.DeepRecommender import DeepRecommender
from baseclass.SocialRecommender import SocialRecommender
import numpy as np
from random import randint, choice
from collections import defaultdict
import tensorflow as tf
from tool import config

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Agent():
    def __init__(self, userNum, emb_dim, learning_rate=0.05):

        self.userNum = userNum
        self.emb_dim = emb_dim

        self.learning_rate = learning_rate


        initializer = tf.contrib.layers.xavier_initializer()

        self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[userNum, emb_dim], stddev=0.005),dtype=tf.float32)
        self.context_embeddings = tf.Variable(tf.truncated_normal(shape=[userNum, emb_dim], stddev=0.005),dtype=tf.float32)
        self.u = tf.placeholder(tf.int32, name="u")
        self.c = tf.placeholder(tf.int32)
        self.mask = tf.placeholder(tf.float32, [None, userNum])
        self.negSample = tf.placeholder(tf.float32, [None, userNum])
        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u, name='u_e')
        self.c_embedding = tf.nn.embedding_lookup(self.context_embeddings, self.c, name='u_c')
        self.g_params = [self.user_embeddings, self.context_embeddings]

        self.reward = tf.placeholder(tf.float32)

        layer = tf.matmul(self.u_embedding,self.context_embeddings,transpose_a=False,transpose_b=True)

        self.output = tf.multiply(layer,self.mask)
        self.neg = tf.multiply(layer,self.negSample)

        self.output = tf.maximum(1e-6, self.output)
        self.neg = tf.maximum(1e-6, self.neg)

        self.cross_entropy = -tf.multiply(self.mask,tf.log(tf.sigmoid(self.output)))-tf.multiply(self.negSample,tf.log(tf.sigmoid(self.neg)))#???这里1-的部分可能有点问题

        #就是要让朋友的那些维概率接近1，让不是朋友的那些维概率接近0
        #交叉熵损失函数最小化这一步是为了让layer转换成采样概率的时候，对该用户的朋友那些维度的概率接近1，不是朋友的那些维度的概率接近0，少了个sigmoid

        self.pretrain_loss = tf.reduce_mean(self.cross_entropy) #+ 0.01*tf.nn.l2_loss(self.u_embedding)+0.01*tf.nn.l2_loss(self.context_embeddings))



        self.i_prob = tf.gather(
            tf.reshape(tf.nn.softmax(tf.reshape(self.output, [1, -1])), [-1]),
            self.c)

        self.gan_loss = tf.reduce_mean(-tf.log(self.i_prob) * self.reward)+0.01*tf.nn.l2_loss(self.u_embedding)+0.01*tf.nn.l2_loss(self.c_embedding)

        g_opt = tf.train.AdamOptimizer(self.learning_rate)

        pre_opt = tf.train.AdamOptimizer(self.learning_rate)
        #还可以换换别的optimizer

        self.pretrain = pre_opt.minimize(self.pretrain_loss,var_list=self.g_params)#这里看一下收敛情况，好像不是在损失函数最小的时候停止预训练的
        self.gan_updates = g_opt.minimize(self.gan_loss, var_list=self.g_params)

        # for test stage, self.u: [self.batch_size]


    # def save_model(self, sess, filename):
    #     param = sess.run(self.g_params)
    #     cPickle.dump(param, open(filename, 'w'))


class Environment():
    def __init__(self, itemNum, userNum, emb_dim, lamda, reg, mean,learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda
        self.reg = reg

        self.learning_rate = learning_rate

        self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[userNum, emb_dim], stddev=0.01),dtype=tf.float32)
        self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[itemNum, emb_dim], stddev=0.01),dtype=tf.float32)
        self.user_biases = tf.Variable(tf.truncated_normal(shape=[userNum], stddev=0.3),dtype=tf.float32)
        self.item_biases = tf.Variable(tf.truncated_normal(shape=[itemNum], stddev=0.3),dtype=tf.float32)
        self.d_params = [self.user_embeddings, self.item_embeddings,self.user_biases,self.item_biases]

        # placeholder definition
        self.u = tf.placeholder(tf.int32,name="u")
        self.f = tf.placeholder(tf.int32, name="f")
        self.i = tf.placeholder(tf.int32, name="i")
        self.r = tf.placeholder(tf.float32, name="r")
        self.friends = tf.placeholder (tf.float32,shape=[None,userNum])


        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u,name='u_e')
        self.f_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.f, name='u_e')
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i,name='i_e')
        self.u_biases = tf.nn.embedding_lookup(self.user_biases, self.u,name='u_b')
        self.i_biases = tf.nn.embedding_lookup(self.item_biases, self.i,name='i_b')

        self.friends_embedding = tf.matmul(self.friends, self.user_embeddings,transpose_a=False,transpose_b=False)

        u_pred = tf.reduce_sum(tf.multiply(self.u_embedding,self.i_embedding),1)+self.u_biases+self.i_biases+mean


        self.loss = tf.nn.l2_loss(self.r-u_pred)+ lamda*tf.nn.l2_loss(self.u_embedding-self.friends_embedding)

        self.loss = self.loss+ self.reg *(tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding))+\
                    0.01*(tf.nn.l2_loss(self.u_biases)+tf.nn.l2_loss(self.i_biases))

        self.pretrain_loss = tf.nn.l2_loss(self.r-u_pred)+ self.reg *(tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding))+\
                             0.01*(tf.nn.l2_loss(self.u_biases)+tf.nn.l2_loss(self.i_biases))#这里看一下收敛情况，好像不是在损失函数最小的时候停止预训练的

        pre_opt = tf.train.AdamOptimizer(self.learning_rate)

        self.pretrain = pre_opt.minimize(self.pretrain_loss, var_list=self.d_params)

        d_opt = tf.train.AdamOptimizer(self.learning_rate)

        self.d_updates = d_opt.minimize(self.loss, var_list=self.d_params)


        self.reward = tf.nn.softmax(1.0/(tf.reduce_sum(tf.abs(tf.reduce_sum(tf.multiply(self.u_embedding,self.i_embedding),1)-
                                                              tf.matmul(self.f_embedding,self.i_embedding,transpose_a=False,transpose_b=True)),1)))

        self.rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),1)#+self.u_biases+self.item_biases




class SRRL(SocialRecommender,DeepRecommender):
    
    #SRRL: Social Recommendation based on Reinforcement Learning
    
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):

        DeepRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)
        self.reward_baseline = 0

        args = config.LineConfig(self.config['SRRL'])
        self.SR = float(args['-SR'])
        self.samplingNum = int(args['-sample'])


    def generate_friends(self, model,n):
        mat_friends = np.zeros((self.num_users,self.num_users))
        friends_idx= {}
        for user in self.data.user:
            if self.social.followees.has_key(user):
                f_idx = [self.data.user[f] for f in self.social.followees[user]]
                mask = np.zeros((1,self.num_users))
                mask[0][f_idx] = 1
                prob = self.sess.run(model.output, feed_dict={model.u: [self.data.user[user]],model.mask:mask})
                prob = np.array(prob[0]) / 0.2  # Temperature
                exp_rating = np.exp(prob)
                prob = exp_rating / np.sum(exp_rating)
                selected = np.random.choice(np.arange(self.num_users), size=n, p=prob,replace=True)
                friends_idx[user]=selected
                row = np.zeros(self.num_users)
                row[selected]=prob[selected]/np.sum(prob[selected])
                mat_friends[self.data.user[user]]=row

        return np.array(mat_friends),friends_idx

    def next_batch(self,friends):
        batch_idx = np.random.randint(self.train_size, size=self.batch_size)
        users = [self.data.trainingData[idx][0] for idx in batch_idx]
        items = [self.data.trainingData[idx][1] for idx in batch_idx]
        ratings = [self.data.trainingData[idx][2] for idx in batch_idx]
        #friends = []
        user_idx,item_idx=[],[]

        for i,user in enumerate(users):
            user_idx.append(self.data.user[user])
            #friends.append([self.data.user[friend] for friend in self.social.followees[user]])
            item_idx.append(self.data.item[items[i]])

        return user_idx,item_idx,ratings,friends[user_idx]



    def initModel(self):
        super(SRRL, self).initModel()
        self.agent = Agent(self.num_users, self.k,  learning_rate=0.01)
        self.Environment = Environment(self.num_items, self.num_users, self.k, lamda=self.SR, reg=self.regU,mean=self.data.globalMean,learning_rate=self.lRate)

    def buildModel(self):
        # minimax training

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.mat_friends, friends_idx = self.generate_friends(self.agent, self.samplingNum)
        for i in range(300):
            if i % 50 == 0 and i>0:

                self.P = self.sess.run(self.Environment.user_embeddings)
                self.Q = self.sess.run(self.Environment.item_embeddings)
                self.Bu = self.sess.run(self.Environment.user_biases)
                self.Bi = self.sess.run(self.Environment.item_biases)
                self.isConverged(i)

            u_idx, i_idx, ratings, f_idx = self.next_batch(self.mat_friends)

            _, self.loss = self.sess.run([self.Environment.pretrain, self.Environment.pretrain_loss],
                                         feed_dict={self.Environment.u: u_idx,
                                                    self.Environment.i: i_idx,
                                                    self.Environment.r: ratings,
                                                    })
            print 'Pretraining epoch:', i + 1, 'd_epoch:', i + 1, 'loss:', self.loss

        #pretraining(agent)
        #这里可不可以加入isconverged
        userList = self.data.user.keys()
        for i in range(300):
            mask = np.zeros((128, self.num_users))
            neg = np.zeros((128, self.num_users))
            userBatch = np.random.choice(userList,size=128)
            for m,user in enumerate(userBatch):
                if self.social.followees.has_key(user):
                    f_idx = [self.data.user[f] for f in self.social.followees[user]]
                    mask[m][f_idx] = 1
                    negUsers = np.random.choice(userList,size=10)
                    for negUser in negUsers:
                        if not self.social.followees[user].has_key(negUser):
                            neg[m][self.data.user[negUser]]=1
            userBatch = [self.data.user[ub] for ub in userBatch]
            _,p_loss = self.sess.run([self.agent.pretrain,self.agent.pretrain_loss],
                          {self.agent.u: userBatch,self.agent.negSample:neg,
                           self.agent.mask: mask})
            print 'Pretraining epoch:', i+1, 'loss:', p_loss




        for iteration in range(self.maxIter):

            print 'Update agent...'
            for g_epoch in range(3):
                self.mat_friends, friends_idx = self.generate_friends(self.agent, self.samplingNum)
                for user in self.data.user:
                    if self.social.followees.has_key(user):
                        u = self.data.user[user]
                        items, ratings = self.data.userRated(user)
                        i_idx = [self.data.item[item] for item in items]
                        f_idx = friends_idx[user]
                        reward = self.sess.run(self.Environment.reward,
                                               feed_dict={self.Environment.u: [u],
                                                          self.Environment.i: i_idx,
                                                          self.Environment.f: f_idx,
                                                          })

                        self.reward_baseline = (np.average(reward) + self.reward_baseline) / 2.0
                        new_reward = reward - self.reward_baseline

                        #print new_reward

                        mask = np.zeros((1, self.num_users))
                        mask[0][f_idx] = 1

                        _ = self.sess.run(self.agent.gan_updates,
                                          {self.agent.u: [u], self.agent.c: f_idx,
                                           self.agent.mask: mask, self.agent.reward: new_reward*10})

                print 'epoch:', iteration + 1, 'd_epoch:', g_epoch + 1

            print 'Update Environment...'
            for d_epoch in range(300):
                if d_epoch % 50 == 0:
                    self.mat_friends,friends_idx = self.generate_friends(self.agent,self.samplingNum)
                    self.P = self.sess.run(self.Environment.user_embeddings)
                    self.Q = self.sess.run(self.Environment.item_embeddings)
                    self.Bu = self.sess.run(self.Environment.user_biases)
                    self.Bi = self.sess.run(self.Environment.item_biases)
                    self.isConverged(iteration)
                u_idx,i_idx,ratings,f_idx = self.next_batch(self.mat_friends)


                _, self.loss = self.sess.run([self.Environment.d_updates,self.Environment.loss],
                                        feed_dict={self.Environment.u:u_idx,
                                                   self.Environment.i:i_idx,
                                                   self.Environment.r:ratings,
                                                   self.Environment.friends: f_idx})
                print 'epoch:', iteration + 1, 'd_epoch:', d_epoch + 1, 'loss:', self.loss


    def predict(self,u,i):
        if self.data.containsUser(u) and self.data.containsItem(i):
            u = self.data.user[u]
            i = self.data.item[i]
            return self.P[u].dot(self.Q[i])+self.data.globalMean +self.Bi[i]+self.Bu[u]
        else:
            return self.data.globalMean
