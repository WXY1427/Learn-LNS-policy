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
        default=2,
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
    gcnn_models = ['baseline']
    other_models = ['extratrees_gcnn_agg', 'lambdamart_khalil', 'svmrank_khalil']
    internal_branchers = ['relpscost']
    time_limit = 200

    if args.problem == 'setcover':

        instances += [{'type': 'big', 'path': "data/instances/setcover/test_5000r_4000c_0.05d/instance_{}.lp".format(i+1)} for i in range(50)] 
    else:
        raise NotImplementedError

    branching_policies = []

    # SCIP internal brancher baselines
    for brancher in internal_branchers:
        for seed in seeds:
            branching_policies.append({
                    'type': 'internal',
                    'name': brancher,
                    'seed': seed,
             })

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
        'dual',
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
                utilities.init_scip_paramsR(m, seed=policy['seed'])
                m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time

#                m.setHeuristics(1) 

#                m.setHeuristics(scip.SCIP_PARAMSETTING.AGGRESSIVE)


                if policy['type'] == 'gcnn':
                    m.setLongintParam('limits/nodes', 1)                              

                else:
                    m.setRealParam('limits/time', time_limit)

                brancher = PolicyBranching(policy)
                m.includeBranchrule(
                    branchrule=brancher,
                    name="{}:{}".format(policy['type'],policy['name']),
                    desc="Custom PySCIPOpt branching policy.",
                    priority=666666, maxdepth=-1, maxbounddist=1)

                walltime = time.perf_counter()
                proctime = time.process_time()
          
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

                dual = m.getDualbound()

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
                    'dual': dual,
                })

                csvfile.flush()
                m.freeProb()

                print("  {}:{} {} - {} ({}) nodes {} lps {:.2} ({:.2} wall {:.2} proc) s. {}".format(policy['type'],policy['name'],policy['seed'],nnodes,nnodes+2*(ndomchgs+ncutoffs),nlps,stime,walltime,proctime,status))                               