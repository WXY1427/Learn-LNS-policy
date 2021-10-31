import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import pickle

import pyscipopt as scip

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import utilities

from pyscipopt import Model, quicksum


class PolicyBranching(scip.Branchrule):

    def __init__(self, policy):
        super().__init__()

        self.policy_type = policy['type']
        self.policy_name = policy['name']

        if self.policy_type == 'gcnn':
            model = policy['model']
            model.restore_state(policy['parameters'])
            self.policy = tfe.defun(model.call, input_signature=model.input_signature)

        elif self.policy_type == 'internal':
            self.policy = policy['name']

        elif self.policy_type == 'ml-competitor':
            self.policy = policy['model']

            # feature parameterization
            self.feat_shift = policy['feat_shift']
            self.feat_scale = policy['feat_scale']
            self.feat_specs = policy['feat_specs']

        else:
            raise NotImplementedError

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):

        # SCIP internal branching rule
        if self.policy_type == 'internal':
            result = self.model.executeBranchRule(self.policy, allowaddcons)

        # custom policy branching
        else:
            
            if self.policy_type == 'gcnn' and self.model.getNNodes() == 1:        

#                print(self.model.getNConss())
#                result = scip.SCIP_RESULT.DIDNOTRUN
                result = self.model.executeBranchRule('relpscost', allowaddcons)
               
 
               
                
                
            elif self.policy_type == 'gcnn' and self.model.getNNodes() != 1:


               
                result = self.model.executeBranchRule('relpscost', allowaddcons)   

            

            else:
                raise NotImplementedError

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=3,
    )

    parser.add_argument(
        '-t', '--time',
        help='timelimit.',
        type=int,
        default=1,
    )


    args = parser.parse_args()

    result_file = "{}_{}.csv".format(args.problem,time.strftime('%Y%m%d-%H%M%S'))
    instances = []
    seeds = [0]

    times = [(i+1)*2 for i in range(50)]

    gcnn_models = ['baseline']
    other_models = ['extratrees_gcnn_agg', 'lambdamart_khalil', 'svmrank_khalil']
    internal_branchers = ['relpscost']
    time_limit = args.time

    if args.problem == 'setcover':
#        instances += [{'type': 'small', 'path': "data/instances/setcover/transfer_500r_1000c_0.05d/instance_{}.lp".format(i+1)} for i in range(20)]
#        instances += [{'type': 'medium', 'path': "data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{}.lp".format(i+1)} for i in range(20)]
#        instances += [{'type': 'big', 'path': "data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{}.lp".format(i+1)} for i in range(20)]
#        instances += [{'type': 'big', 'path': "data/instances/setcover/validation/instance_{}.lp".format(i+1)} for i in range(4)]

        instances += [{'type': 'big', 'path': "data/instances/setcover/test_5000r_1000c_0.05d/instance_{}.lp".format(i+101)} for i in range(0,50)] ##test small

#        instances += [{'type': 'big', 'path': "data/instances/setcover/test_5000r_2000c_0.05d/instance_{}.lp".format(i+1)} for i in range(50)]  ###test large


#        gcnn_models += ['mean_convolution', 'no_prenorm']


    else:
        raise NotImplementedError

    branching_policies = []

    # SCIP internal brancher baselines
    for brancher in internal_branchers:
        for seed in seeds:
            for timel in times:
                branching_policies.append({
                      'type': 'internal',
                      'name': brancher,
                      'seed': seed,
                      'timel': timel,
                 })


#     # ML baselines
#     for model in other_models:
#         for seed in seeds:
#             branching_policies.append({
#                 'type': 'ml-competitor',
#                 'name': model,
#                 'seed': seed,
#                 'model': f'trained_models/{args.problem}/{model}/{seed}',
#             })

#    # GCNN models
#    for model in gcnn_models:
#        for seed in seeds:
#            branching_policies.append({
#                'type': 'gcnn',
#                'name': model,
#                'seed': seed,
#                'parameters': 'trained_models/{}/{}/{}/best_params.pkl'.format(args.problem,model,0)
#            })

    print("problem: {}".format(args.problem))
    print("gpu: {}".format(args.gpu))
    print("time limit: {} s".format(time_limit))

    ### TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)
    tf.executing_eagerly()

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    for policy in branching_policies:
        if policy['type'] == 'gcnn':
            if policy['name'] not in loaded_models:
                sys.path.insert(0, os.path.abspath("models/{}".format(policy['name'])))
                import model
                importlib.reload(model)
                loaded_models[policy['name']] = model.GCNPolicy()
                del sys.path[0]
            policy['model'] = loaded_models[policy['name']]

    # load ml-competitor models
#     for policy in branching_policies:
#         if policy['type'] == 'ml-competitor':
#             try:
#                 with open(f"{policy['model']}/normalization.pkl", 'rb') as f:
#                     policy['feat_shift'], policy['feat_scale'] = pickle.load(f)
#             except:
#                 policy['feat_shift'], policy['feat_scale'] = 0, 1

#             with open(f"{policy['model']}/feat_specs.pkl", 'rb') as f:
#                 policy['feat_specs'] = pickle.load(f)

#             if policy['name'].startswith('svmrank'):
#                 policy['model'] = svmrank.Model().read(f"{policy['model']}/model.txt")
#             else:
#                 with open(f"{policy['model']}/model.pkl", 'rb') as f:
#                     policy['model'] = pickle.load(f)

    print("running SCIP...")

    fieldnames = [
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'stime',
        'gap',
        'status',
        'ndomchgs',
        'ncutoffs',
        'walltime',
        'proctime',
        'obj',
    ]


    os.makedirs('results', exist_ok=True)
    with open("results/{}".format(result_file), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            print("{}: {}...".format(instance['type'],instance['path']))
            solut = []
            for policy in branching_policies:
                tf.set_random_seed(policy['seed'])

                m = scip.Model()
                m.setIntParam('display/verblevel', 0)
                m.readProblem("{}".format(instance['path']))
                utilities.init_scip_params(m, seed=policy['seed'])
                m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
                if policy['type'] == 'gcnn':
                    m.setLongintParam('limits/nodes', 1) 
#                    utilities.init_scip_params(m, seed=policy['seed'],separating=False)                                 

                else:
                    m.setRealParam('limits/time', policy['timel'])

                brancher = PolicyBranching(policy)
                m.includeBranchrule(
                    branchrule=brancher,
                    name="{}:{}".format(policy['type'],policy['name']),
                    desc="Custom PySCIPOpt branching policy.",
                    priority=666666, maxdepth=-1, maxbounddist=1)

                walltime = time.perf_counter()
                proctime = time.process_time()
          
                m.optimize() 

               

                if policy['type'] == 'gcnn':
                        model = policy['model']
                        model.restore_state(policy['parameters'])
                        policy_ = tfe.defun(model.call, input_signature=model.input_signature)
                        state_buffer={}
                        state = utilities.extract_state(m, state_buffer)

                        # convert state to tensors
                        c, e, v = state
                        state = (
                            tf.convert_to_tensor(c['values'], dtype=tf.float32),
                            tf.convert_to_tensor(e['indices'], dtype=tf.int32),
                            tf.convert_to_tensor(e['values'], dtype=tf.float32),
                            tf.convert_to_tensor(v['values'], dtype=tf.float32),
                            tf.convert_to_tensor([c['values'].shape[0]], dtype=tf.int32),
                           tf.convert_to_tensor([v['values'].shape[0]], dtype=tf.int32),
                        )

                        var_logits = policy_(state, tf.convert_to_tensor(False)).numpy().squeeze(0)                               
                        minimum_k = tf.nn.top_k(-var_logits, k=700)[1].numpy()

                       
            
#                        m = scip.Model()
#                        m.setIntParam('display/verblevel', 0)
#                        m.readProblem("{}".format(instance['path']))
#                        utilities.init_scip_params(m, seed=policy['seed'])
#                        m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
                        m.freeTransform()
#                        utilities.init_scip_params(m, seed=policy['seed'])
                                     
                        m.setLongintParam('limits/nodes', -1)                        
                        m.setRealParam('limits/time', time_limit)

                

#                        brancher = PolicyBranching(policy)
#                        m.includeBranchrule(
#                            branchrule=brancher,
#                            name="{}:{}".format(policy['type'],policy['name']),
#                            desc="Custom PySCIPOpt branching policy.",
#                            priority=666666, maxdepth=-1, maxbounddist=1)




                        varss = [x for x in m.getVars()]  

                    

                

                        ### strong
                        for i in minimum_k:
#                            m.addCons(varss[i]==0)
                            a,b = m.fixVar(varss[i],0)
#                            print(a,b)
                      

                        ### soft
#                        m.addCons(quicksum(varss[i] for i in minimum_k)==0)



                        m.optimize()   




                walltime = time.perf_counter() - walltime
                proctime = time.process_time() - proctime
                      
                stime = m.getSolvingTime()
                nnodes = m.getNNodes()
                nlps = m.getNLPs()
                gap = m.getGap()
                status = m.getStatus()
                ndomchgs = brancher.ndomchgs
                ncutoffs = brancher.ncutoffs
                obj = m.getObjVal()
#                K = [m.getVal(x) for x in m.getVars()]        
#                print(K)


                solut.append([m.getVal(x) for x in m.getVars()])

                writer.writerow({
                    'policy': "{}:{}".format(policy['type'],policy['name']),
                    'seed': policy['seed'],
                    'type': instance['type'],
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'stime': stime,
                    'gap': gap,
                    'status': status,
                    'ndomchgs': ndomchgs,
                    'ncutoffs': ncutoffs,
                    'walltime': walltime,
                    'proctime': proctime,
                    'obj': obj,
                })

                csvfile.flush()
                m.freeProb()

                print("  {}:{} {} - {} ({}) nodes {} lps {:.2} ({:.2} wall {:.2} proc) s. {}".format(policy['type'],policy['name'],policy['seed'],nnodes,nnodes+2*(ndomchgs+ncutoffs),nlps,stime,walltime,proctime,status))
#            zc=0
#            for i in range(3):
                                
#                if solut[3]==solut[i+3]:
#                    zc+=1
#            print(zc)
