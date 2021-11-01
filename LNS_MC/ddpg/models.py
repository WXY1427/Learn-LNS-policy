import tensorflow as tf
from ddpg.common.models import get_network_builder
import tensorflow.keras as K
import numpy as np
import pickle
from ddpg.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, fc3d


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

class imitator(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x=tf.reshape(obs,[-1,310])
            x = tf.layers.dense(x, 100)
            x = tf.nn.relu(x)
            
            #### compute scores for each variable
            x = tf.layers.dense(x, 2)
         
        return tf.reshape(x,[-1,11975, 2])


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


class Actor_mean_pre(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            obs_all=tf.reshape(obs,[-1,4956])

            obs = obs_all[:,:6]

            stru = obs_all[:,-4950:]
            stru = tf.reshape(stru,[-1,2975,4950])

            x = self.network_builder(obs)
            # add entire info to each variable
            x = tf.reshape(x,[-1,2975,128])

            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),x)
            xc = tf.contrib.layers.layer_norm(xc, center=True, scale=True)

            xc = tf.nn.tanh(xc)
            xc = tf.layers.dense(xc, 128)
            xc = tf.matmul(stru,xc)
            xc = tf.contrib.layers.layer_norm(xc, center=True, scale=True)

            x = tf.nn.tanh(x)

            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),x)
            xc = tf.contrib.layers.layer_norm(xc, center=True, scale=True)

            xc = tf.nn.tanh(xc)
            xc = tf.layers.dense(xc, 128)
            x = tf.matmul(stru,xc)

#            g = tf.reduce_mean(x, 1, keepdims=True)

#            g = g[:,0,:]   ###ADD to advance graph embedding
#            g = fc(g, 'mlp_fc{}'.format(9), nh=128, init_scale=np.sqrt(2))   ###ADD to advance graph embedding
#            g = tf.nn.tanh(g)                                                                                                ### ADD to advance graph embedding
#            g = fc(g, 'mlp_fc{}'.format(10), nh=64, init_scale=np.sqrt(2))     ### ADD to advance graph embedding
#            g = g[:,np.newaxis,:]   ###ADD to advance graph embedding
            
#            x = tf.concat([x,tf.tile(g,tf.constant([1,2975,1], tf.int32))],axis=-1)

            x = tf.reshape(x,[-1,128])
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True)

            x = fc(x, 'mlp_fc{}'.format(11), nh=64, init_scale=np.sqrt(2))

            x = tf.contrib.layers.batch_norm(x, center=True, scale=True)

            x = tf.nn.tanh(x)
            
            #### compute scores for each variable

            x = fc(x, 'mlp_fc{}'.format(12), nh=1, init_scale=np.sqrt(2))
            x = tf.nn.sigmoid(x)
         
        return tf.expand_dims(tf.reshape(x,[-1,2975]),-1)

class Actor_mean(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            obs_all=tf.reshape(obs,[-1,4965])

            obs = obs_all[:,:15]

            stru = obs_all[:,-4950:]
            stru = tf.reshape(stru,[-1,2975,4950])

            x = self.network_builder(obs)
            # add entire info to each variable
            x = tf.reshape(x,[-1,2975,128])

            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),x)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc = tf.nn.tanh(xc)
            xc = tf.layers.dense(xc, 128)
            xc = tf.matmul(stru,xc)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc = tf.nn.tanh(xc)

            xc = tf.layers.dense(tf.concat([xc, x], axis=-1), 128)
            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),xc)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc = tf.nn.tanh(xc)
            xc = tf.layers.dense(xc, 128)
            x = tf.matmul(stru,xc)

            x = tf.reshape(x,[-1,128])
            x = tf.layers.dense(x, 256)

            x = tf.nn.tanh(x)  
            x = tf.layers.dense(x, 128)

            x = tf.nn.tanh(x)                 
            x = tf.layers.dense(x, 1)

            x = tf.nn.sigmoid(x)

#            x = x*0.6+0.2
         
        return tf.expand_dims(tf.reshape(x,[-1,2975]),-1)


class Actor_mean_semi(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            obs_all=tf.reshape(obs,[-1,4965])

            obs = obs_all[:,:15]

            stru = obs_all[:,-4950:]
            stru = tf.reshape(stru,[-1,2975,4950])

            x = self.network_builder(obs)
            # add entire info to each variable
            x = tf.reshape(x,[-1,2975,128])

            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),x)
#            xc = tf.transpose(xc, perm=[0, 2, 1])
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
#            xc = tf.transpose(xc, perm=[0, 2, 1])
            xc = tf.nn.tanh(xc)
            xc = tf.layers.dense(xc, 128)
            xc = tf.matmul(stru,xc)
#            xc = tf.transpose(xc, perm=[0, 2, 1])
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
#            xc = tf.transpose(xc, perm=[0, 2, 1])
            xc = tf.nn.tanh(xc)

            xc = tf.layers.dense(tf.concat([xc, x], axis=-1), 128)
            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),xc)
#            xc = tf.transpose(xc, perm=[0, 2, 1])
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
#            xc = tf.transpose(xc, perm=[0, 2, 1])
            xc = tf.nn.tanh(xc)
            xc = tf.layers.dense(xc, 128)
            x = tf.matmul(stru,xc)

            x = fc3d(x, 'mlp_fc{}'.format(11), nh=256, init_scale=np.sqrt(2))
            x = tf.nn.tanh(x)

            x = fc3d(x, 'mlp_fc{}'.format(12), nh=128, init_scale=np.sqrt(2))
            x = tf.nn.tanh(x)
            
            x = fc3d(x, 'mlp_fc{}'.format(13), nh=1, init_scale=np.sqrt(2))

            x = tf.nn.sigmoid(x)
         
        return x

class Critic_mean_pre(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            xa = tf.tile(action, tf.constant([1,1,128], tf.int32))

            x = tf.reshape(obs,[-1,4956]) 
       
            x = self.network_builder(x)

            x = fc(x, 'mlp_fc{}'.format(113), nh=256, init_scale=np.sqrt(2))     ### ADD to advance graph embedding

            x = tf.nn.tanh(x)            ### ADD to advance graph embedding

            x = fc(x, 'mlp_fc{}'.format(114), nh=128, init_scale=np.sqrt(2))     ### ADD to advance graph embedding

            x = tf.reshape(x,[-1,2975,128])


            xa = xa*x 

            xa = tf.reduce_mean(xa, 1)

            x = tf.reduce_mean(x, 1)

            act = fc(tf.reshape(tf.squeeze(action),[-1,2975]), 'mlp_fc{}'.format(100), nh=128, init_scale=np.sqrt(2))

            x = tf.concat([x, act],axis=-1)

            x = fc(x, 'mlp_fc{}'.format(13), nh=256, init_scale=np.sqrt(2))     ### ADD to advance graph embedding

            x = tf.nn.tanh(x)            ### ADD to advance graph embedding

            x = fc(x, 'mlp_fc{}'.format(14), nh=128, init_scale=np.sqrt(2))     ### ADD to advance graph embedding

            x = fc(x, 'output', nh=1, init_scale=np.sqrt(2))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class Critic_mean(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            feats = obs[:,:,:15]

            stru = obs[:,:,-4950:]

            obs = tf.concat([feats, action], axis=-1) # this assumes observation and action can be concatenated

            x = tf.reshape(obs,[-1,16]) #######18
            x = self.network_builder(x)
            x = tf.reshape(x,[-1,2975,128])

            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),x)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc = tf.nn.tanh(xc)
            xc = tf.layers.dense(xc, 128)
            xc = tf.matmul(stru,xc)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc = tf.nn.tanh(xc)


            xc = tf.layers.dense(tf.concat([xc, x], axis=-1), 128)
            xc = tf.matmul(tf.transpose(stru, perm=[0, 2, 1]),xc)
            xc = tf.contrib.layers.layer_norm(inputs=xc, begin_norm_axis=-1, begin_params_axis=-1)
            xc = tf.nn.tanh(xc)
            xc = tf.layers.dense(xc, 128)
            x = tf.matmul(stru,xc)

            x = tf.reduce_mean(x, 1)

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
    
    
    
