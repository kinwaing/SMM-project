from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor

import os

def main(args):
    
    #Pick GPU to use. Activate tensorflow_gpu conda env
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_instance
    
    ### Open model parameters file
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        
        ### Load adjacency matrix
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        
        ### load the graph look at /lib/utils.py for this function
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        
        ### Call the DCRNN supervisor class and start training
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

            supervisor.train(sess=sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    ### Argument takes model parameters look at /data/model/dcrnn_la.yaml
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    
    parser.add_argument('--gpu_instance', default='1', type=str, help='Set GPU instance')
    args = parser.parse_args()
    main(args)
