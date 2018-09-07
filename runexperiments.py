import numpy as np 

import keras
from keras.optimizers import SGD, Adam, Nadam, Adadelta
from keras.utils import plot_model
from keras.callbacks import History, TensorBoard, ModelCheckpoint

from data_utils import interface_groundtruth_1d
from data_utils import interface_groundtruth_max
import argparse

from blocks.DenseBlock import DenseBlock
from blocks.ConvBlock import ConvBlock, ResBlock
from meta_arch.UNet import UNet
from meta_arch.AutoEncoder import AutoEncoder
from meta_arch.ConvNet import CNN

import json
import os 

MODEL_TYPES = {
        'UNet': UNet,
        'AE': AutoEncoder,
        'CNN': CNN
        }

OPTIMIZERS = {
        'sgd': SGD,
        'nadam': Nadam,
        'adam': Adam,
        'adadelta':Adadelta
        }

BLOCKS = {
        'dense': DenseBlock,
        'res': ResBlock,
        'conv': ConvBlock
}

LABEL_FN = {
        'interface_max':interface_groundtruth_max,
        'interface_1d':interface_groundtruth_1d
        }

choices_msg = "Expected {} to be from {} but got {}"
if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('--config',
                        help='Path to experiment config json file',
                        type = str)
        parser.add_argument('--save-dir',
                        help = 'The directory to save the model and experiments config file',
                        type = str,
                        default = 'test')
        parser.add_argument('--label-fn',
                        help = 'A function to preprocess the labels',
                        type = str,
                        default = interface_groundtruth_max,
                        choices = list(LABEL_FN.keys())+[None])

        args = parser.parse_args()

        with open(args.config,'r') as fc:
                exp_config = json.load(fc)

        # Get the individual configs
        model_config = exp_config['model']
        meta_arch_config = model_config['meta_arch']
        block_config = model_config['block']

        train_config = exp_config['train']
        eval_config = exp_config['eval']

        model_type = model_config['meta_arch']['name']
        block_type = model_config['block']['name']
        
        assert model_type in MODEL_TYPES.keys(), choices_msg.format('meta_arch name',MODEL_TYPES.keys(),model_type)
        assert block_type in BLOCKS.keys(), choices_msg.format('block name', BLOCKS.keys(), block_type)
        
        # Handle the data
        # Load Data
        ds_fact = train_config['downsample']
        epochs = train_config['epochs'] 
        shuffle = train_config['shuffle']
        batch_size = train_config['batch_size']

        x_train = np.load(train_config['data']).astype(np.float32)
        y_train = np.load(train_config['labels']).astype(np.float32)
        
        x_eval = np.load(eval_config['data']).astype(np.float32)
        y_eval = np.load(eval_config['labels']).astype(np.float32)
        
        # Downsample temporal resolution
        assert x_train.shape[1] % ds_fact == 0, 'The downsample factor, %d, must divide the initial sample size %d'%(ds_fact,x_train.shape[1])
        x_train = x_train[:,::ds_fact,...]
        x_eval = x_eval[:,::ds_fact,...]

        assert x_train.shape[0] == y_train.shape[0], 'Number of samples does not match between station data and their labels'

        # Process labels
        # TODO: Make the label processor a choosable from the config. 
        # Process data if a function is given.
        if args.label_fn:
                label_fn = args.label_fn
                y_train = label_fn(y_train, output_shape=x_train.shape[1])
                y_eval = label_fn(y_eval, output_shape=x_train.shape[1])

        # Initiate block and model instance
        #
        
        # Block instance
        block = BLOCKS[block_type](block_config)
        #block = DenseBlock()

        # Create meta_arch instance        
        model = MODEL_TYPES[model_type](block = block, meta_config = meta_arch_config)
        
        if model_type == 'CNN':
                red_factor = 2**(meta_arch_config['num_layers'])

                y_train = y_train[:,::red_factor,...]
                y_eval = y_eval[:,::red_factor,...]

                model = model.build_model(input_shape=x_train.shape[1:], output_shape=y_train.shape[1])
        
        else:
                model = model.build_model(input_shape=x_train.shape[1:])

        # Get the optimizer
        optimizer_type = train_config['optimizer']['algorithm']
        assert optimizer_type in OPTIMIZERS.keys(), choices_msg.format('optimizer',OPTIMIZERS.keys(),optimizer_type)
        optimizer_config = train_config['optimizer']['parameters']
        optimizer = OPTIMIZERS[optimizer_type](**optimizer_config)
        
        mdl_chkpt_path = os.path.join(args.save_dir,'model_chkpt.hdf5')
        if os.path.exists(mdl_chkpt_path):
                print("Loading model from existing checkpont ", mdl_chkpt_path)
                model.load_weights(mdl_chkpt_path)


        model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        
        # Record training loss and initialize callbacks for logging and saving
        history = History()

        if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)

        # Make logs directory for tensorboard if needed
        if not os.path.isdir(args.save_dir +'/logs/'):
                os.makedirs(args.save_dir + '/logs/')
        
        model_path = os.path.join(args.save_dir,'model.h5')
        config_path = os.path.join(args.save_dir,'config.json')
        loss_path = os.path.join(args.save_dir,'loss.json')
        
        if 'save_every' in train_config.keys():
                save_every = train_config['save_every']
        else:
                save_every = 100
        tensorboard = TensorBoard(log_dir=args.save_dir + '/logs/', batch_size=batch_size, write_images=True)
        mdl_chkpt = ModelCheckpoint(mdl_chkpt_path, monitor='val_acc',verbose=1,  period=save_every, save_best_only=True)

        from contextlib import redirect_stdout

        with open(os.path.join(args.save_dir,'summary.txt'), 'w') as f:
                with redirect_stdout(f):
                        model.summary()


        model.fit(x_train,y_train,
                epochs=epochs,
                shuffle=shuffle,
                batch_size=batch_size,
                validation_data=(x_eval,y_eval),
                callbacks = [history, tensorboard, mdl_chkpt])
        
        model.evaluate(x_eval, y_eval)
        
        # Specify experiment outputs
        
        model.save(model_path)

        with open(config_path,'w') as fc, open(loss_path,'w') as fl:
                json.dump(exp_config,fc, indent= 2)
                json.dump(history.history, fl, indent= 2)
