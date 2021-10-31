import tensorflow as tf
from ddpg.common.models import get_network_builder
import tensorflow.keras as K
import numpy as np
import pickle
from ddpg.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch


class Model(object):
    def __init__(self, name, network='mlp', **network_kwargs):
        self.name = name
        self.network_builder = get_network_builder(network)(**network_kwargs)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            obs=tf.reshape(obs,[-1,20])
            x = self.network_builder(obs)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
         
        return tf.expand_dims(tf.reshape(x,[-1,1000]),-1)


class Actor_mean(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            obs_all=tf.reshape(obs,[-1,5014])

            obs = obs_all[:,:14]

            stru = obs_all[:,-5000:]
            stru = tf.reshape(stru,[-1,1000,5000])

            x = self.network_builder(obs)

            # add entire info to each variable
            x = tf.reshape(x,[-1,1000,128])

            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),x)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc_c = tf.nn.tanh(xc)   #modify
            xc = tf.layers.dense(xc_c, 128)  ###modify
            xc = tf.matmul(stru,xc)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc = tf.nn.tanh(xc)


            xc_v = xc+x  ###add
            xc = tf.layers.dense(xc_v, 128)  ###modify
            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),xc)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc = tf.nn.tanh(xc)+xc_c  ###modify
            xc = tf.layers.dense(xc, 128)
            x = tf.matmul(stru,xc)
            xc = tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=-1, begin_params_axis=-1)  #add
            x = tf.nn.tanh(xc)+xc_v  #add


            x = tf.reshape(x,[-1,128])
            x = tf.layers.dense(x, 256)

            x = tf.nn.tanh(x)  
            x = tf.layers.dense(x, 128)

            x = tf.nn.tanh(x)                 
            x = tf.layers.dense(x, 1)

            x = tf.nn.sigmoid(x)

            x = x*0.6+0.2
         
        return tf.expand_dims(tf.reshape(x,[-1,1000]),-1)


class Critic_mean(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            xa = tf.tile(action, tf.constant([1,1,128], tf.int32))

            feats = obs[:,:,:14]

            stru = obs[:,:,-5000:]

            obs = tf.concat([feats, action], axis=-1) # this assumes observation and action can be concatenated

            x = tf.reshape(obs,[-1,15]) #######18

            x = self.network_builder(x)

            x = tf.reshape(x,[-1,1000,128])

            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),x)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc_c = tf.nn.tanh(xc)   #modify
            xc = tf.layers.dense(xc_c, 128)  ###modify
            xc = tf.matmul(stru,xc)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc = tf.nn.tanh(xc)


            xc_v = xc+x  ###add
            xc = tf.layers.dense(xc_v, 128)  ###modify
            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),xc)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc = tf.nn.tanh(xc)+xc_c  ###modify
            xc = tf.layers.dense(xc, 128)
            x = tf.matmul(stru,xc)
            xc = tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=-1, begin_params_axis=-1)  #add
            x = tf.nn.tanh(xc)+xc_v  #add


            xaa = xa*x 

            xaa = tf.reduce_sum(xaa, 1)/tf.reduce_sum(xa,axis=1)

            x = tf.reduce_mean(x, 1)

#            act = fc(tf.reshape(tf.squeeze(action),[-1,1000]), 'mlp_fc{}'.format(100), nh=128, init_scale=np.sqrt(2))

            x = tf.concat([x, xaa],axis=-1)

            x = fc(x, 'mlp_fc{}'.format(13), nh=256, init_scale=np.sqrt(2))     ### ADD to advance graph embedding

            x = tf.nn.tanh(x)            ### ADD to advance graph embedding

            x = fc(x, 'mlp_fc{}'.format(14), nh=128, init_scale=np.sqrt(2))     ### ADD to advance graph embedding

            x = fc(x, 'output', nh=1, init_scale=np.sqrt(2))

        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class Critic(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = tf.concat([obs, action], axis=-1) # this assumes observation and action can be concatenated
            x = tf.reshape(x,[-1,21])            
            x = self.network_builder(x)
            x = tf.layers.dense(x, 64, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
            x = tf.reshape(x,[-1,1000,64])
            x = tf.reduce_mean(x, 1)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output2')
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
    
    
    
