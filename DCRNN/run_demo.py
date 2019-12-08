import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor

'''This script loads a pretrained model and predicts/forecasts. It saves predictions alongside ground truth as npz compressed numpy.'''
def run_dcrnn(args):
    
    #Pick GPU to use. Activate tensorflow_gpu conda env
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_instance
    
    with open(args.config_filename) as f:
        config = yaml.load(f)
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    
    ### From the yaml file get the adjacency matrix
    graph_pkl_filename = config['data']['graph_pkl_filename']
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
        
        ### Load the current trained model, access filename from yaml file
        supervisor.load(sess, config['train']['model_filename'])
        
        ### Evaluate or perform prediction
        outputs = supervisor.evaluate(sess)
        np.savez_compressed(args.output_filename, **outputs)
        print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    
    ### Output filename to save predictions 
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    
    parser.add_argument('--gpu_instance', default='1', type=str, help='Set GPU instance')
    args = parser.parse_args()
    run_dcrnn(args)
