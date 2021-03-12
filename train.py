import argparse
import os
import pickle
import random
import sys
import time
import traceback
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, Callback
import keras.backend as K
from utils.model import create_model
from utils.myutils import batch_gen, init_tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=200)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--modeltype', dest='modeltype', type=str, default='codegnngru')
    parser.add_argument('--data', dest='dataprep', type=str, default='../data')
    parser.add_argument('--outdir', dest='outdir', type=str, default='./modelout')
    parser.add_argument('--asthops', dest='hops', type=int, default=2)
    args = parser.parse_args()

    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    modeltype = args.modeltype
    asthops = args.hops

    # set gpu here
    init_tf(gpu)

    config = dict()
    # gnn hops
    config['asthops'] = asthops
    # code vocab size
    config['tdatvocabsize'] = code_vocab_size
    # comment vocab size
    config['comvocabsize'] = comment_vocab_size
    # ast vocab size
    config['smlvocabsize'] = ast_vocab_size

    # set sequence length for our input
    # code seq len
    config['tdatlen'] = 100
    # ast seq len
    config['maxastnodes'] = 100
    # comment seq len
    config['comlen'] = 50

    config['batch_size'] = batch_size
    config['epochs'] = epochs

    # Load data
    # model parameters
    steps = int(len(seqdata['ctrain'])/batch_size)+1
    valsteps = int(len(seqdata['cval'])/batch_size)+1


    # Print information
    print('tdatvocabsize {}'.format(code_vocab_size))
    print('comvocabsize {}'.format(comment_vocab_size))
    print('smlvocabsize {}'.format(ast_vocab_size))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps*batch_size))
    print('vaidation data size {}'.format(valsteps*batch_size))
    print('------------------------------------------')

    # create model
    config, model = create_model(modeltype, config)

    print(model.summary())

    # set up data generators
    gen = batch_gen(seqdata, 'train', config, nodedata=node_data, edgedata=edges)

    checkpoint = ModelCheckpoint(outdir + "/models/" + modeltype + "_E{epoch:02d}.h5")

    valgen = batch_gen(seqdata, 'val', config, nodedata=seqdata['sval_nodes'], edgedata=seqdata['sval_edges'])
    callbacks = [checkpoint]

    model.fit_generator(gen, steps_per_epoch=steps, epochs=epochs, verbose=1, max_queue_size=4,
                        callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)
