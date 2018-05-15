#!/usr/bin/env python
#coding=utf-8

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)    
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
#import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA,ARMA
from sklearn.metrics import mean_squared_error
from math import sqrt,log,exp
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
#sys.path.append("C:/Anaconda3/Lib/site-packages/nnet_ts/")
#from nnet_ts import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.models.rnn import *
from tensorflow.contrib import rnn
import argparse

def calMAPE(test,pre):
	error = list()
	for i in range(len(test)):
		error.append(100*np.abs(test[i]-pre[i])/test[i])
	return error, np.mean(error)

class ts_prediction(object):
    
    def __init__(self, input_dim, time_step, n_hidden, d_hidden, batch_size):

        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.d_hidden = d_hidden
        self.o_hidden = 32
        
        self.input_dim = input_dim
        self.time_step = time_step
        
        self.seq_len = tf.placeholder(tf.int32,[None])
        self.input_x = tf.placeholder(dtype = tf.float32, shape = [None, None, input_dim])
        self.input_y = tf.placeholder(dtype = tf.float32,shape = [None,self.time_step])
        self.label = tf.placeholder(dtype = tf.float32)
        
        self.encode_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        self.decode_cell = tf.contrib.rnn.LSTMCell(self.d_hidden, forget_bias=1.0, state_is_tuple=True)
        self.output_cell = tf.contrib.rnn.LSTMCell(self.o_hidden, forget_bias=1.0, state_is_tuple=True)
        
        self.loss = tf.constant(0.0)
            ## ===========  build the model =========== ##
            
            ## ==== encoder ===== ## 
        out = self.en_RNN(self.input_x) # out[0]: (b*T*2d)
        self.out = out
        out = tf.transpose(out,[0,2,1]) # (b,2d,T)
        with tf.name_scope('encoder') as scope:
            stddev = 1.0/(self.n_hidden*self.time_step)
            Ue = tf.Variable(dtype=tf.float32,
                             initial_value = tf.truncated_normal(shape = [self.time_step,self.time_step], 
                                                                 mean = 0.0, stddev = stddev),name = 'Ue')
        var = tf.tile(tf.expand_dims(Ue,0),[self.batch_size,1,1]) #(b,T,T)
        
        batch_mul = tf.matmul(var,self.input_x) # (b*T*T)*(b*T*d) = (b,T,d)
        self.out = batch_mul
        e_list = []

        for k in range(self.input_dim):
            series_k = tf.reshape(batch_mul[:,:,k],[self.batch_size,self.time_step,1]) #(b,T,1)
            e_k = self.attention(out,series_k, scope = 'encoder')
            e_list.append(e_k)
        e_list = tf.concat(e_list,axis = 1)
        soft_attention = tf.nn.softmax(e_list,dim = 1)
        input_attention = tf.multiply(self.input_x,tf.transpose(soft_attention,[0,2,1])) #(b,T,d)
        
        with tf.variable_scope('fw_lstm') as scope:
            tf.get_variable_scope().reuse_variables()
            h,_ = tf.nn.dynamic_rnn(self.encode_cell, input_attention, self.seq_len, dtype = tf.float32)
        # h: (b,T,d)
        
            # ===== decoder ===== ## 
        d, dec_out = self.de_RNN(h) ## d: (b,T,q); dec_out:(b,T,2q)
        self.out = d
        dec_out = tf.transpose(dec_out,[0,2,1])
        with tf.name_scope('decoder') as scope:
            stddev = 1.0/(self.d_hidden*self.time_step)
            Ud = tf.Variable(dtype=tf.float32,
                             initial_value = tf.truncated_normal(shape = [self.n_hidden,self.n_hidden], 
                                                                 mean = 0.0, stddev = stddev),name = 'Ud')
        de_var = tf.tile(tf.expand_dims(Ud,0),[self.batch_size,1,1]) # (b,d,d)
        batch_mul_de = tf.matmul(h,de_var) #(b, T, d)
        batch_mul_de = tf.transpose(batch_mul_de,[0,2,1])
        e_de_list = []
        for t in range(self.time_step):
            series_t = tf.reshape(batch_mul_de[:,:,t],[self.batch_size,self.n_hidden,1])
            e_t = self.de_attention(dec_out,series_t, scope = 'decoder')
            e_de_list.append(e_t)
        e_de_list = tf.concat(e_de_list,axis = 1) # b,T,T
        de_soft_attention = tf.nn.softmax(e_de_list,dim = 1)
        #self.out = de_soft_attention
        
            # ===== context c_t ===== ##
        c_list = []
        for t in range(self.time_step):
            Beta_t = tf.expand_dims(de_soft_attention[:,:,0],-1)
            weighted = tf.reduce_sum(tf.multiply(Beta_t,h),1)
            c_list.append(tf.expand_dims(weighted,1))
        c_t = tf.concat(c_list,axis = 1) ## (b,T,d)
        self.out = c_t
        c_t_hat = tf.concat([c_t,tf.expand_dims(self.input_y,-1)],axis = 2) # b,T,(d+1), where +1 for concatenation
        
            # ===== y_hat ===== ##
        with tf.variable_scope('temporal'):
            mean = 0.0
            stddev = 1.0/(self.n_hidden*self.time_step)
            W_hat = tf.get_variable(name = 'W_hat',shape = [self.n_hidden+1,1],dtype = tf.float32,
                                    initializer=tf.truncated_normal_initializer(mean,stddev)) 
        
        W_o = tf.tile(tf.expand_dims(W_hat,0),[self.batch_size,1,1])
        y_hat = tf.matmul(c_t_hat,W_o) ## b,T,1
        
            ## ==== final step ==== ##
        d_y_concat = tf.concat([d,y_hat],axis = 2) ## b,T,q+1
        with tf.variable_scope('out_lstm') as scope:
            d_final,_ = tf.nn.dynamic_rnn(self.output_cell, d_y_concat, self.seq_len, dtype = tf.float32) # b,T,o_hidden
            
            ## ==== output y_T ==== ##
        ## only concat the last state d_T and c_T
        d_c_concat = tf.concat([d_final[:,-1,:],c_t[:,-1,:]],axis = 1) #b,o_hidden+q
        d_c_concat = tf.expand_dims(d_c_concat,-1) # b,d+q,1
        
        with tf.variable_scope('predict'):
            mean = 0.0
            stddev = 1.0/(self.n_hidden*self.time_step)
            Wy = tf.get_variable(name = 'Wy',shape = [self.o_hidden,self.o_hidden+self.n_hidden],dtype = tf.float32,
                                 initializer=tf.truncated_normal_initializer(mean,stddev)) 
            Vy = tf.get_variable(name = 'Vy',shape = [self.o_hidden],dtype = tf.float32,
                                 initializer=tf.truncated_normal_initializer(mean,stddev)) 
            bw = tf.get_variable(name = 'bw',shape = [self.o_hidden],dtype = tf.float32,
                                initializer = tf.constant_initializer(0.1))
        W_y = tf.tile(tf.expand_dims(Wy,0),[self.batch_size,1,1]) # b,q,q+d
        b_w = tf.expand_dims(tf.tile(tf.expand_dims(bw,0),[self.batch_size,1]),-1) #b,q -> b,q,1
        V_y = tf.tile(tf.expand_dims(Vy,0),[self.batch_size,1]) #b,q
        V_y = tf.expand_dims(V_y,1) #b,1,q
        self.y_predict = tf.squeeze(tf.matmul(V_y,tf.matmul(W_y,d_c_concat)+b_w)) #(b,1,q) * (b,q,1) -> squeeze -> (b,)
        
        self.loss += tf.reduce_mean(tf.square(self.label - self.y_predict))
        self.params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(1e-3)
        #self.train_op = optimizer.minimize(self.loss)
        grad_var = optimizer.compute_gradients(loss = self.loss, var_list = self.params, aggregation_method = 2)
        #self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = optimizer.apply_gradients(grad_var, global_step=self.global_step)
        
        
    def en_RNN(self,input_x):
        
        with tf.variable_scope('fw_lstm') as scope:
            cell = self.encode_cell  ## this step don't create variable 
            out, states = tf.nn.dynamic_rnn(cell, input_x, self.seq_len, dtype = tf.float32) ## this step does create variables
            
        tmp = tf.tile(states[1],[1,self.time_step])
        tmp = tf.reshape(tmp,[self.batch_size, self.time_step, self.n_hidden])
        
        concat = tf.concat([out,tmp],axis = 2)
        return concat ## shape shoule be (b,T,2*n_hidden)
    
    def de_RNN(self,h):
        
        with tf.variable_scope('dec_lstm') as scope:
            cell = self.decode_cell
            d,s = tf.nn.dynamic_rnn(cell, h, self.seq_len, dtype = tf.float32)
        tmp = tf.tile(s[1],[1,self.time_step])
        tmp = tf.reshape(tmp,[self.batch_size, self.time_step, self.d_hidden])
        concat = tf.concat([d,tmp],axis = 2)
        return d,concat
        
    def attention(self, out, series_k, scope = None):
   
        with tf.variable_scope('encoder') as scope:
            try:
                mean = 0.0
                stddev = 1.0/(self.n_hidden*self.time_step)
                We = tf.get_variable(name = 'We', dtype=tf.float32,shape = [self.time_step, 2*self.n_hidden],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
                Ve = tf.get_variable(name = 'Ve',dtype=tf.float32,shape = [1,self.time_step],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
            except ValueError:
                scope.reuse_variables()
                We = tf.get_variable('We')
                Ve = tf.get_variable('Ve')       
        W_e = tf.tile(tf.expand_dims(We,0),[self.batch_size,1,1])  # b*T*2d
        brcast = tf.nn.tanh(tf.matmul(W_e,out) + series_k) # b,T,T + b,T,1 = b, T, T
        V_e = tf.tile(tf.expand_dims(Ve,0),[self.batch_size,1,1]) # b,1,T
        
        return tf.matmul(V_e,brcast) # b,1,T
    
    def de_attention(self,out,series_k,scope = None):
        
        with tf.variable_scope('decoder') as scope:
            try:
                mean = 0.0
                stddev = 1.0/(self.d_hidden*self.time_step)
                Wd = tf.get_variable(name = 'Wd', dtype=tf.float32,shape = [self.n_hidden, 2*self.d_hidden],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
                Vd = tf.get_variable(name = 'Vd',dtype=tf.float32,shape = [1,self.n_hidden],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
            except ValueError:
                scope.reuse_variables()
                Wd = tf.get_variable('Wd')
                Vd = tf.get_variable('Vd') 
        W_d = tf.tile(tf.expand_dims(Wd,0),[self.batch_size,1,1])
        brcast = tf.nn.tanh(tf.matmul(W_d,out) + series_k) # b,d,2q * b,2q*T = b,d,T
        #return brcast
        V_d = tf.tile(tf.expand_dims(Vd,0),[self.batch_size,1,1]) # b,1,d
        
        return tf.matmul(V_d,brcast) # b,1,d * b,d,T = b,1,T
    
    def predict(self,x_test,y_test,sess):
        
        train_seq_len =  np.ones(self.batch_size) * self.time_step
#        feed = {model.input_x: x_test, 
#                model.seq_len: train_seq_len,
#                model.input_y: y_test}
        feed = {self.input_x: x_test, 
                self.seq_len: train_seq_len,
                self.input_y: y_test}

        y_hat = sess.run(self.y_predict,feed_dict = feed)
        return y_hat

FLAGS = None

def main(_):
    print("FLAGS: %s" % FLAGS)    
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
   
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
            idx1 = np.arange(1,8)
            print(idx1)
            idx1 = np.reshape(idx1,[7,1])
            print(idx1)
            encode1 = OneHotEncoder()
            encode1.fit(idx1)
            #print(encode.transform([[7]]).toarray())
            #p = encode.transform(np.reshape(1,[1,1])).toarray()
            #print(p)
            idx2 = np.arange(1,13)
            idx2 = np.reshape(idx2,[len(idx2),1])
            encode2 = OneHotEncoder()
            encode2.fit(idx2)
            #print(encode2.transform([[11]]).toarray())

            idx3 = np.arange(1,5)
            idx3 = np.reshape(idx3,[len(idx3),1])
            encode3 = OneHotEncoder()
            encode3.fit(idx3)

            #unit = 12
            #total = 36

            #df = pd.read_csv('revenuedaily.csv', sep = ',',encoding='utf-8',header = 0)
            df = pd.read_csv(FLAGS.train_file_name, sep = ',',encoding='utf-8',header = 0)

            total = 2608
            revenue = np.array(df['revenue'][0:total])
            weeknum = np.array(df['weeknum'][0:total])
            month = np.array(df['month'][0:total])
            season = np.array(df['season'][0:total])
            holiday = np.array(df['holiday'][0:total])

            revenue = np.reshape(revenue,[revenue.shape[0],1])

            lag = 56
            mid = 2506
            revenue_train = revenue[0:mid,:]
            revenue_test = revenue[mid:,:]

            print(revenue.shape)

            #scaler = StandardScaler()
            #revenue_train = np.reshape(revenue_train,[revenue_train.shape[0],revenue_train.shape[1]])
            #print(revenue_train)
            #scaler.fit(revenue_train)
            #revenue_train = scaler.transform(revenue_train)
            #print(revenue_train)

            trainlen = revenue_train.shape[0]

            #Add categorical feature
            X = np.zeros((trainlen-lag,lag),dtype = 'float64')
            AX = np.zeros((trainlen-lag,lag+12),dtype = 'float64')
            FY = np.zeros((trainlen-lag,lag),dtype = 'float64')
            Y = np.zeros((trainlen-lag,1),dtype = 'float64')

            for i in range(trainlen-lag):
                X[i,0:lag] = revenue_train[i:i+lag,0]
                AX[i, 0:lag] = revenue_train[i:i+lag,0]
                w = encode1.transform(np.reshape(int(weeknum[i+lag]),[1,1])).toarray()
                s = encode3.transform(np.reshape(int(season[i+lag]),[1,1])).toarray()
                #print(m,p)
                AX[i, lag:lag+7] = w[0, :]
                AX[i, lag+7:lag+11] = s[0, :]
                AX[i, lag+11:lag+12] = int(holiday[i+lag])
                FY[i,0:lag] = np.log(revenue_train[i:i+lag,0])
                Y[i] = np.log(revenue_train[i+lag,0])
            print('-----1')
            #AX = np.reshape(AX,[AX.shape[0],AX.shape[1],1])
            scaler = StandardScaler()
            X = np.reshape(X,[X.shape[0],X.shape[1]])
            #print(revenue_train)
            scaler.fit(X)
            X = scaler.transform(X)

            #print(AX)
            AX[:,0:lag] = X

            Y = np.reshape(Y,[Y.shape[0],])
            print(AX.shape)
            print(Y.shape)
            #print(AX)
            #print(Y)
            print('------2, %s' % datetime.now())

            duaX = np.zeros((trainlen-lag,lag,12))
            for i in range(trainlen-lag):
                for j in range(lag):
                    w = encode1.transform(np.reshape(int(weeknum[i+j]),[1,1])).toarray()
                    s = encode3.transform(np.reshape(int(season[i+j]),[1,1])).toarray()
                    duaX[i,j,0:7] = w[0,:]
                    duaX[i,j,7:11] = s[0,:]
                    duaX[i,j,11:] = int(holiday[i+j])
            duaY = FY
            dualabel = Y

            print('-------3, %s' % datetime.now())

            #dual attention model
            input_dim = 12
            time_step = lag
            n_hidden = 128
            d_hidden = 64
            batch_size = 256

            current_episode = 0
            # total_episodes = 500
            total_episodes = FLAGS.total_episodes
            
            model = ts_prediction(input_dim, time_step = time_step, n_hidden= n_hidden, d_hidden = d_hidden, batch_size = batch_size)
            init = tf.global_variables_initializer()

        last_step = total_episodes * (int((trainlen-lag)/batch_size)+1)
        print('last step: %d' % last_step)
        #hooks=[tf.train.StopAtStepHook(last_step=(last_step * 2))]
        hooks=[]

        with tf.train.MonitoredTrainingSession(master=server.target,
                                                       is_chief=(FLAGS.task_index == 0),
                                                       checkpoint_dir=FLAGS.train_output_dir,
                                                       save_checkpoint_secs=600,
                                                       hooks=hooks) as sess:
            sess.run(init)
            should_stop = False

            while not should_stop: #sess.should_stop(): 
                #shuffle
                index = np.array(range(trainlen-lag))
                np.random.shuffle(index)
                duaX = duaX[index]
                duaY = duaY[index]
                dualabel = dualabel[index]
                t = int((trainlen-lag)/batch_size)+1
                current_episode = current_episode + 1
                for i in range(t):
                    if i < t-1:
                        X_b = duaX[i*batch_size:(i+1)*batch_size,:]
                        Y_b = duaY[i*batch_size:(i+1)*batch_size,:]
                        label_b = dualabel[i*batch_size:(i+1)*batch_size]
                        train_seq_len =  np.ones(batch_size) * time_step
                    if i == t-1:
                        X_b = duaX[trainlen-lag-batch_size:trainlen-lag,:]
                        Y_b = duaY[trainlen-lag-batch_size:trainlen-lag,:]
                        label_b = dualabel[trainlen-lag-batch_size:trainlen-lag]
                        train_seq_len =  np.ones(batch_size) * time_step
                    feed = {model.input_x: X_b, 
                            model.seq_len: train_seq_len,
                            model.input_y: Y_b,
                            model.label: label_b}
                    g_step, loss,_ = sess.run([model.global_step, model.loss,model.train_op],feed_dict = feed)
                    print ("%s, current_episode %i, step %i, losses are %f, global_step %d, task_index %d" % (datetime.now(), current_episode-1, i, loss, g_step, FLAGS.task_index))

                    #if g_step > last_step or loss < FLAGS.target_loss:
                    if g_step > last_step:
                        should_stop = True

            #predict

            duaXtest = np.zeros((total-mid,lag,12))
            duaYtest = np.zeros((total-mid,lag))
            realv = revenue[mid:total,:]
            prey = np.zeros((total-mid,1),dtype = 'float64')

            for i in range(0,total-mid):
                for j in range(lag):
                    w = encode1.transform(np.reshape(int(weeknum[i+mid-lag+j]),[1,1])).toarray()
                    s = encode3.transform(np.reshape(int(season[i+mid-lag+j]),[1,1])).toarray()
                    duaXtest[i,j,0:7] = w[0,:]
                    duaXtest[i,j,7:11] = s[0,:]
                    duaXtest[i,j,11:] = int(holiday[i+mid-lag+j])
                    duaYtest[i,j] = np.log(revenue[i+mid-lag+j,0])

            st = int((total-mid)/batch_size)+1
            for i in range(st):
                if i < st-1:
                    Xt_b = duaXtest[i*batch_size:(i+1)*batch_size,:]
                    Yt_b = duaYtest[i*batch_size:(i+1)*batch_size,:]
                    py = model.predict(Xt_b,Yt_b,sess)
                    prey[i*batch_size:(i+1)*batch_size,:] = np.reshape(np.exp(py),[np.exp(py).shape[0],1])
                if i == st-1:
                    if total-mid >= batch_size:
                        Xt_b = duaXtest[total-mid-batch_size:,:]
                        Yt_b = duaYtest[total-mid-batch_size:,:]
                        py = model.predict(Xt_b,Yt_b,sess)
                        prey[total-mid-batch_size:,:] = np.exp(py)
                    if total-mid < batch_size:
                        Xt_b = np.zeros((batch_size,lag,12))
                        Yt_b = np.zeros((batch_size,lag))
                        Xt_b[0:total-mid,:] = duaXtest[0:total-mid,:]
                        Yt_b[0:total-mid,:] = duaYtest[0:total-mid,:]
                        py = model.predict(Xt_b,Yt_b,sess)
                        prey[0:total-mid,:] = np.reshape(np.exp(py[0:total-mid]),[np.exp(py[0:total-mid]).shape[0],1])

                        
            rprey1 = prey
            print(rprey1)
            rprey1 = np.array(rprey1)
            print(rprey1.shape)
            rprey1 = np.reshape(rprey1,[rprey1.shape[0],1])
            realv = revenue[mid:,0]
            perror, MAPE = calMAPE(realv,rprey1)
            print('Test MAPE: %.3f' % MAPE)
            print('Test Max APE: %.3f' % max(perror))
            print('Test Min APE: %.3f' % min(perror))
            print(perror)
            #plt.title("Predictions of  " + dkeys[i] + " monthly")
            #plt.plot(range(total-unit), data[i,0:total-unit], '-g',  label='2015-2016 Original series')
            # -- plt.plot(range(0,realv.shape[0]), realv, '-b', label='2017 Realvalues')
            # -- plt.plot(range(0,realv.shape[0]), rprey1, '-r', label='2017 Predictions', linewidth=1)
            #plt.plot(range(total,total+unit), pre18, '-y',label='2018 Predictions')
            # -- leg = plt.legend(loc='upper left')
            # -- leg.get_frame().set_alpha(0.3)
            
            # -- plt.savefig('sodexo_revenue_darnn.png')
            # -- plt.close('all')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.register("type", "bool", lambda v: v.lower() == "true")

    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      required=True,
      help="Comma-separated list of hostname:port pairs"
    )

    parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      required=True,
      help="Comma-separated list of hostname:port pairs"
    )

    parser.add_argument(
      "--job_name",
      type=str,
      default="",
      required=True,
      help="One of 'ps', 'worker'"
    )

    parser.add_argument(
      "--train_file_name",
      type=str,
      default="revenuedaily.csv",
      help="Train file name"
    )

    parser.add_argument(
      "--train_output_dir",
      type=str,
      default="",
      required=True,
      help="Train file name"
    )

    # Flags for defining the tf.train.Server
    parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      required=True,
      help="Index of task within the job"
    )

    parser.add_argument(
      "--total_episodes",
      type=int,
      default=5,
      help="Total epochs"
    )

    parser.add_argument(
      "--target_loss",
      type=float,
      default=0.0,
      help="Target loss"
    )

    FLAGS, unparsed = parser.parse_known_args()
    #print FLAGS, unparsed
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

