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
from utils.myutils import BatchGen, init_tf, load_data
from vocab import Vocab

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

    data_dir = '../data_set/py'

    # load vocab
    vocab = Vocab(data_dir=data_dir)
    vocab.load_vocab()

    # load data
    train_code_data, train_ast_data, train_edges, train_nl = load_data(data_dir, 'train')
    val_code_data, val_ast_data, val_edges, val_nl = load_data(data_dir, 'dev')

    config = dict()
    # gnn hops
    config['asthops'] = asthops
    # code vocab size
    config['tdatvocabsize'] = len(vocab.code2index)
    # comment vocab size
    config['comvocabsize'] = len(vocab.nl2index)
    # ast vocab size
    config['smlvocabsize'] = len(vocab.ast2index)

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
    steps = int(len(train_code_data)/batch_size)+1
    valsteps = int(len(val_code_data)/batch_size)+1


    # Print information
    print('tdatvocabsize {}'.format(config['tdatvocabsize']))
    print('comvocabsize {}'.format( config['comvocabsize']))
    print('smlvocabsize {}'.format(config['smlvocabsize']))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps*batch_size))
    print('vaidation data size {}'.format(valsteps*batch_size))
    print('------------------------------------------')

    # create model
    config, model = create_model(modeltype, config)

    print(model.summary())

    # set up data generators
    gen = BatchGen(config, 'train', train_code_data, train_ast_data, train_nl, train_edges, vocab)

    checkpoint = ModelCheckpoint(outdir + "/models/" + modeltype + "_E{epoch:02d}.h5")

    valgen = BatchGen(config, 'val', val_code_data, val_ast_data, val_nl, val_edges, vocab)
    callbacks = [checkpoint]

    model.fit_generator(gen, steps_per_epoch=steps, epochs=epochs, verbose=1, max_queue_size=4,
                        callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)
