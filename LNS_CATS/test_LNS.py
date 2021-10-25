import os
import argparse
import multiprocessing as mp
import pickle
import glob
import numpy as np
import shutil
import gzip
import tensorflow as tf
import csv

import pyscipopt as scip
import utilities

import time
from collections import deque
import pickle
import random

from ddpg.ddpg_learner import DDPG
from ddpg.models import Actor_mean, Critic_mean
from ddpg.memory import Memory
from ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from ddpg.common import set_global_seeds
import ddpg.common.tf_util as U

import generate_instances_fly

import shutil

def make_samples(in_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

    episode, instance, obs, actions, seed, exploration_policy, eval_flag, time_limit, out_dir = in_queue
    print('[w {}] episode {}, seed {}, processing instance \'{}\'...'.format(os.getpid(),episode,seed,instance)) 

    if eval_flag==1:

        seed=0
    else:

#        seed=seed

        seed=0
  
    m = scip.Model()
    m.setIntParam('display/verblevel', 0)
    m.readProblem('{}'.format(instance))
    utilities.init_scip_paramsR(m, seed=seed)  ####params
    m.setIntParam('timing/clocktype', 1)
    m.setRealParam('limits/time', time_limit) 

#         branchrule = SamplingAgent0(
#             episode=episode,
#             instance=instance,
#             seed=seed,
#             out_queue=out_queue,
#             exploration_policy=exploration_policy,
#             query_expert_prob=query_expert_prob,
#             out_dir=out_dir)

#         m.includeBranchrule(
#             branchrule=branchrule,
#             name="Sampling branching rule", desc="",
#             priority=666666, maxdepth=-1, maxbounddist=1)
        
    varss = [x for x in m.getVars()]             
        
        # decide the fixed variables based on actions                              
#        minimum_k = tf.nn.top_k(-actions, k=700)[1].numpy()       #???? 

#    minimum_k = np.argpartition(-actions.squeeze(), -700)[-700:]       #???? 

#    minimum_k = np.where(np.array(actions.squeeze())<0.5)

#    minimum_k = np.where(np.random.binomial(1, actions) <0.5)

    if eval_flag==1:

#        minimum_k = np.argpartition(-actions.squeeze(), -700)[-700:]       #???? 
#        for i in minimum_k:
#            a,b = m.fixVar(varss[i],obs[i])      

#        minimum_k = np.where(np.random.binomial(1, actions) <0.5)

        minimum_k = np.where(np.array(actions.squeeze())<0.5)
        max_k = np.where(np.array(actions.squeeze())>0.5)[0]
        min_k = minimum_k[0]
        for i in min_k:
            a,b = m.fixVar(varss[i],obs[i])        
        
    else:

#        minimum_k = np.where(np.random.binomial(1, actions) <0.5)
        minimum_k = np.where(np.array(actions.squeeze())<0.5)
        max_k = np.where(np.array(actions.squeeze())>0.5)[0]
        min_k = minimum_k[0]
        for i in min_k:
            a,b = m.fixVar(varss[i],obs[i])         


#     ### strong
#     for i in minimum_k[0]:
#         a,b = m.fixVar(varss[i],obs[i])                  # ???????
   
    m.optimize()
    
    K = [m.getVal(x) for x in m.getVars()]  
        
#    obj = m.getObjVal()

    obj = m.getPrimalbound()

    print(m.getStatus())
        
    m.freeProb() 
        
    out_queue = {
        'type': 'solution',
        'episode': episode,
        'instance': instance,
        'sol' : np.array(K),
        'obj' : obj,
        'seed': seed,
        'mask': min_k[0],
    }               

    print("[w {}] episode {} done".format(os.getpid(),episode))
    
    return out_queue


def send_orders(instances, epi, obs, actions, seed, exploration_policy, eval_flag, time_limit, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.RandomState(seed)

#    episode = 0
    orders_queue = []
    for i in range(len(instances)):
#        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        orders_queue.append([epi[i], instances[i], obs[i], actions[i], seed, exploration_policy, eval_flag, time_limit, out_dir])
#        episode += 1

    return orders_queue


def collect_samples(instances, epi, obs, actions, out_dir, rng, n_samples, n_jobs,
                    exploration_policy, eval_flag, time_limit):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """

    # start workers
#     orders_queue = mp.Queue(maxsize=2*n_jobs)
#     answers_queue = mp.SimpleQueue()
#     workers = []
    
#     for i in range(n_jobs):
#         abc=time.time()
#         p = mp.Process(
#                 target=make_samples,
#                 args=(orders_queue, answers_queue),
#                 daemon=True)
#         workers.append(p)
#         p.start()
#         print(time.time()-abc) 

    tmp_samples_dir = '{}/tmp'.format(out_dir)
    os.makedirs(tmp_samples_dir, exist_ok=True)
    
    # start dispatcher
#     dispatcher = mp.Process(
#             target=send_orders,
#             args=(orders_queue, instances, epi, obs, actions, rng.randint(2**32), exploration_policy, query_expert_prob, time_limit, tmp_samples_dir),
#             daemon=True)
#     dispatcher.start()

    pars = send_orders(instances, epi, obs, actions, rng.randint(2**32), exploration_policy, eval_flag, time_limit, tmp_samples_dir) 
    
    out_Q = []
    for n in range(n_samples):
        out_queue = make_samples(pars[n])
        out_Q.append(out_queue)        


    # record answers 
    i = 0
    collecter=[]
    epi=[]
    obje=[]
    instances=[]
    mask=[]

    for sample in out_Q:
        
        collecter.append(sample['sol'])
        
        epi.append(sample['episode'])
        
        obje.append(sample['obj'])

        instances.append(sample['instance'])

        mask.append(sample['mask'])
        
        i += 1

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)
   
    return np.stack(collecter), np.stack(epi), np.stack(obje), instances, mask
    
##########  initial formulation features    
class SamplingAgent0(scip.Branchrule):

    def __init__(self, episode, instance, seed, exploration_policy, out_dir):
        self.episode = episode
        self.instance = instance
        self.seed = seed
        self.exploration_policy = exploration_policy
        self.out_dir = out_dir

        self.rng = np.random.RandomState(seed)
        self.new_node = True
        self.sample_counter = 0

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}

    def branchexeclp(self, allowaddcons):

        # custom policy branching           
        if self.model.getNNodes() == 1:    
            
            # extract formula features
            self.state = utilities.extract_state(self.model, self.state_buffer)              

            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)
                               
        elif self.model.getNNodes() != 1:
               
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)   
            
        else:
            raise NotImplementedError

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}


def make_samples0(in_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

#     while True:
    episode, instance, seed, exploration_policy, eval_flag, time_limit, out_dir = in_queue
    print('[w {}] episode {}, seed {}, processing instance \'{}\'...'.format(os.getpid(),episode,seed,instance))

    if eval_flag==1:
        seed=0
    else:
#        seed=seed

        seed=0
    
    m = scip.Model()
    m.setIntParam('display/verblevel', 0)
    m.readProblem('{}'.format(instance))
    utilities.init_scip_paramsR(m, seed=seed)   ###H
    m.setIntParam('timing/clocktype', 2)
#    m.setLongintParam('limits/nodes', 1) 
#    m.setRealParam('limits/time', 20) 
    m.setParam('limits/solutions', 1)


    branchrule = SamplingAgent0(
        episode=episode,
        instance=instance,
        seed=seed,
        exploration_policy=exploration_policy,
        out_dir=out_dir)

    m.includeBranchrule(
        branchrule=branchrule,
        name="Sampling branching rule", desc="",
        priority=666666, maxdepth=-1, maxbounddist=1)
    abc=time.time()    
    m.optimize()       
    print(time.time()-abc)    
    # extract formula features
#     state_buffer={}
#     state = utilities.extract_state(m, state_buffer)  

    b_obj = m.getObjVal()

    K = [m.getVal(x) for x in m.getVars()]     

    out_queue = {
        'type': 'formula',
        'episode': episode,
        'instance': instance,
        'state' : np.random.rand(16000, 17), #branchrule.state,
        'seed': seed,
        'b_obj': b_obj,
        'sol' : np.array(K),        
    }  
       
    m.freeTransform()  
        
    obj = [x.getObj() for x in m.getVars()]  
    
    out_queue['obj'] = sum(obj) 
    
    m.freeProb()  
        
    print("[w {}] episode {} done".format(os.getpid(),episode))
    
    return out_queue


def send_orders0(instances, n_samples, seed, exploration_policy, batch_id, eval_flag, time_limit, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.RandomState(seed)

    episode = 0
    st = batch_id*n_samples
    orders_queue = []
    for i in instances[st:st+n_samples]:     
        seed = rng.randint(2**32)
        orders_queue.append([episode, i, seed, exploration_policy, eval_flag, time_limit, out_dir])
        episode += 1
    return orders_queue



def collect_samples0(instances, out_dir, rng, n_samples, n_jobs,
                    exploration_policy, batch_id, eval_flag, time_limit):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    os.makedirs(out_dir, exist_ok=True)

    # start workers
#     orders_queue = mp.Queue(maxsize=2*n_jobs)
#     answers_queue = mp.SimpleQueue()
#     workers = []
#     for i in range(n_jobs):
#         p = mp.Process(
#                 target=make_samples0,
#                 args=(orders_queue, answers_queue),
#                 daemon=True)
#         workers.append(p)
#         p.start()

    tmp_samples_dir = '{}/tmp'.format(out_dir)
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
#     dispatcher = mp.Process(
#             target=send_orders0,
#             args=(orders_queue, instances, n_samples, rng.randint(2**32), exploration_policy, batch_id, time_limit, tmp_samples_dir),
#             daemon=True)
#     dispatcher.start()
    
    pars = send_orders0(instances, n_samples, rng.randint(2**32), exploration_policy, batch_id, eval_flag, time_limit, tmp_samples_dir)  
    
    out_Q = []
    for n in range(n_samples):
        out_queue = make_samples0(pars[n])
        out_Q.append(out_queue)        
        

    # record answers and write samples
    i = 0
    collecter=[]
    epi=[]
    instances=[]
    obje=[]
    bobj=[]
    ini_sol=[]

    
    for sample in out_Q:
        
        ini_sol.append(sample['sol'])         
        
#        collecter.append(sample['state'][2]['values'])

        collecter.append(sample['state'])


        
        epi.append(sample['episode'])
        
        instances.append(sample['instance'])
        
        obje.append(sample['obj'])

        bobj.append(sample['b_obj'])
        
        i += 1

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)
    
    return np.stack(collecter), np.stack(epi), np.stack(obje), np.stack(bobj), instances, np.stack(ini_sol)
                    
                    
def learn(args,network='mlp',
          seed=None,
          total_timesteps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=25,
          nb_rollout_steps=20,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type=None, #'adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=False,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,  #0.9 #0.96
          clip_norm=None,
          nb_train_steps=1, # per epoch cycle and MPI worker,  50 10 30   100  10  3
          nb_eval_steps=400, #20 #50 # 100
          batch_size=500, # per MPI worker  64 32  64   128   64  128
          tau=0.01,
          eval_env=None,
          load_path='models/RL_model/model50012.joblib',
          save_path=None,
          param_noise_adaption_interval=30):                    
                    

    print("seed {}".format(args.seed))

    batch_sample = 10  #8
    batch_sample_eval = 10 #8
    exploration_strategy = 'relpscost' ###'pscost'
    eval_val = 0
    time_limit = 2  # 5  #2

    instances_valid = []

    if args.problem == 'cauctions':
#        instances_train = glob.glob('data/instances/setcover/train_500r_1000c_0.05d/*.lp')

#        instances_valid += ["data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{}.lp".format(i+1) for i in range(4)]

        instances_train = glob.glob('data/instances/cauctions/train_2000_4000/*.lp')

#        instances_valid += ["data/instances/setcover/validation5000/instance_{}.lp".format(i+1) for i in range(10)]
        instances_valid = glob.glob('data/instances/cauctions/test800016000/*.lp')

#        instances_test = glob.glob('data/instances/setcover/test_500r_1000c_0.05d/*.lp')
        out_dir = 'data/samples/setcover/500r_1000c_0.05d'
    else:
        raise NotImplementedError
        
    ### number of epochs, cycles, steps
    nb_epochs = 20
    nb_epoch_cycles = len(instances_train)//batch_sample
    nb_rollout_steps = 50   # 30 #50   

    print("{} train instances for {} samples".format(len(instances_train),nb_epoch_cycles*nb_epochs*batch_sample))
                    
                    
    # define parameters                
    nb_actions = 16000  #set covering

    memory = Memory(limit=int(5e2), action_shape=(16000,1,), observation_shape=(16000,20,))
    critic = Critic_mean(network=network)
    actor = Actor_mean(nb_actions, network=network)

    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    agent = DDPG(actor, critic, memory, (16000,), (16000,),
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)



    ### TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)                    
                    
    sess = U.get_session()
    # Prepare everything.
    agent.initialize(sess)

    if load_path is not None:
        agent.load(load_path)

    sess.graph.finalize()

    rng = np.random.RandomState(args.seed)
    tf.set_random_seed(rng.randint(np.iinfo(int).max))


    agent.reset()
    nenvs = batch_sample
                    
    # create output directory, throws an error if it already exists                  
     
    episodes = 0 #scalar
    t = 0 # scalar

    max_obj = 0       
    #### start train    
    for epoch in range(1): 
        
        random.shuffle(instances_train)

        fieldnames = [
            'instance',
            'obj',
            'initial',
            'bestroot',
            'imp',
            'mean',
            'time',
        ]
        result_file = "{}_{}.csv".format(args.problem,time.strftime('%Y%m%d-%H%M%S'))    
        os.makedirs('ddpg_test_results', exist_ok=True)
        with open("ddpg_test_results/{}".format(result_file), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()    
        
            for cycle in range(1):        # nb_epoch_cycles                             
                # Evaluate  
                if cycle%1==0:
                    episodes = 0 #scalar
                    t = 0 # scalar    
#                    obj_lis = []
                
                    for cyc in range(len(instances_valid)//batch_sample_eval):
                        ### initial formulation features
                        obj_lis = []
                        abcd=time.time()

                        formu_feat, epi, ori_objs, best_root, instances, ini_sol=collect_samples0(instances_valid, out_dir + '/train', rng, batch_sample_eval,
                                        args.njobs, exploration_policy=exploration_strategy,
                                        batch_id=cyc,
                                        eval_flag=1,
                                        time_limit=None)
                     
                        ### initial solution
                        if args.problem == 'cauctions':     
                            init_sols = ini_sol  
#                            init_sols = np.stack([np.zeros(16000) for _ in range(batch_sample_eval)])    

                        ori_objs=np.copy(best_root)  
#                        ori_objs=np.zeros(batch_sample_eval)

                        cur_sols=init_sols                                
                        record_ini=np.copy(best_root)
                        
                        rec_inc = [[] for r in range(batch_sample)]  #ADD
                        [rec_inc[r].append(init_sols[r]) for r in range(batch_sample)]  #ADD    
                        rec_best = np.copy(best_root)                 #ADD
                        inc_val = np.stack([rec_inc[r][-1] for r in range(batch_sample)])#ADD
                        avg_inc_val = np.stack([np.array(rec_inc[r]).mean(0) for r in range(batch_sample)])#ADD 

                        pre_sols = np.ones([2,batch_sample_eval,16000])
                        cur_obs = np.concatenate((formu_feat, inc_val[:,:,np.newaxis], avg_inc_val[:,:,np.newaxis], pre_sols.transpose(1,2,0), cur_sols[:,:,np.newaxis]), axis=-1)         
                                        
                        mask = None
              
                        # Perform rollouts.                
                        for t_rollout in range(nb_eval_steps):     
                        
                            action, q, _, _ = agent.step(cur_obs, apply_noise=False, compute_Q=False)    

                            pre_sols = np.concatenate((pre_sols,cur_sols[np.newaxis,:,:]), axis=0) 
                            
                            if t_rollout==10:
                                print(action)     

                            if t_rollout>20:

#                            if np.random.uniform()<0.5:
                                action=np.ones((batch_sample,nb_actions,1))*0.5 
#                               action=np.random.uniform(low=0.0, high=1.0,size=(batch_sample,nb_actions,1))  

                         

                            action = np.random.binomial(1, action)
                            action=np.where(action > 0.5, action, 0.)  
                            action=np.where(action == 0., action, 1.)  
 
#                            if mask is not None:
#                                for m in range(batch_sample_eval):
#                                    action[m][mask[m]]=0.
                 
                            # Execute next action. derive next solution(state)
                            cur_sols, epi, cur_objs, instances, mask = collect_samples(instances, epi, cur_sols, action, out_dir + '/train', rng, batch_sample,
                                    args.njobs, exploration_policy=exploration_strategy,
                                    eval_flag=1,
                                    time_limit=time_limit) 
                                       
                            inc_ind = np.where(cur_objs > rec_best)[0]          
                            [rec_inc[r].append(cur_sols[r]) for r in inc_ind]                         
                            rec_best[inc_ind] = cur_objs[inc_ind]          
                                       
                            # compute rewards
                            r = cur_objs-ori_objs    
                            print(r)        
                            # note these outputs are batched from vecenv
                            t += 1
                    
                            inc_val = np.stack([rec_inc[r][-1] for r in range(batch_sample)])
                            avg_inc_val = np.stack([np.array(rec_inc[r]).mean(0) for r in range(batch_sample)])                

                            next_obs = np.concatenate((formu_feat, inc_val[:,:,np.newaxis], avg_inc_val[:,:,np.newaxis], pre_sols[-2:].transpose(1,2,0), cur_sols[:,:,np.newaxis]), axis=-1) 
      
                            cur_obs = next_obs.copy()               
                            ori_objs = cur_objs.copy()                   

                            # Book-keeping.
                            obj_lis.append(cur_objs)

                        tim = time.time()-abcd          
                        miniu = np.stack(obj_lis).max(axis=0)  
                        ave = np.mean(miniu)
                        for j in range(batch_sample_eval):                 
                            writer.writerow({
                                'instance': instances[j],
                                'obj':miniu[j],
                                'initial':record_ini[j],
                                'bestroot':best_root[j],
                                'imp':miniu[j]-best_root[j],
                                'mean':ave,
                                'time':tim,
                            })
                            csvfile.flush()    

             
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    
    parser.add_argument(
        '-t', '--total_timesteps',
        help='Number of total_timesteps.',
        type=int,
        default=1e4,
    )
                    
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=3,
    )
    
    arg = parser.parse_args()
       
    learn(args=arg)                
                    

