import numpy as np 
import keras 
from data_utils import interface_groundtruth_1d
from data_utils import interface_groundtruth_max
import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load-path',
                        help='Path to the .h5 file conataining the saved model',
                        type = str,
                        required = True)
    parser.add_argument('--data-path',
                        help='Directory to the evaluation data',
                        type = str,
                        default = 'fga_data_set/eval/')

    args = parser.parse_args()
    if (args.load_path[-1] != '/'):
        args.load_path = args.load_path + '/'

    # Load the Keras Model
    model_load_path = args.load_path
    print('Loading Keras model ' + model_load_path)
    model = keras.models.load_model(model_load_path + 'model.h5')

    if (args.data_path[-1] != '/'):
        args.data_path = args.data_path + '/'

    # Load the eval dataset 
    data_load_path = args.data_path + 'data_bucket_2.npy'
    label_load_path = args.data_path + 'label_bucket_2.npy'
    print('Loading Eval data and labels for inference')
    x_eval = np.load(data_load_path).astype(np.float32)
    y_eval = np.load(label_load_path).astype(np.float32)

    # Load the downsample factor
    #config_path = 'test/config.json'
    config_path = model_load_path + 'config.json'
    with open(config_path,'r') as fc:
            exp_config = json.load(fc)

    ds_fact = exp_config['train']['downsample']
    model_type = exp_config['model']['meta_arch']['name']

    # Downsample the x data
    assert x_eval.shape[1] % ds_fact == 0, 'The downsample factor, %d, must divide the initial sample size %d'%(ds_fact,x_eval.shape[1])
    x_eval = x_eval[:,::ds_fact,...]

    assert x_eval.shape[0] == y_eval.shape[0], 'Number of samples does not match between station data and their labels'

    # Apply the label transformation
    y_eval = interface_groundtruth_max(y_eval, output_shape=x_eval.shape[1])
    #y_eval = interface_groundtruth_1d(y_eval, output_shape=x_eval.shape[1])

    if model_type == 'CNN':
        red_factor = 2**(exp_config['model']['meta_arch']['num_layers'])
        y_eval = y_eval[:,::red_factor,...]

    # Make the predictions
    print('Making the predictions')
    eval_metrics = model.evaluate(x=x_eval, y=y_eval)
    print('Evalutaion Loss: ', eval_metrics[0])
    print('Evalutaion Accuracy: ', eval_metrics[1])
    y_pred = model.predict(x_eval, verbose=1)

    # Save output for analysis
    print('Saving actual and predicted output')
    np.save(model_load_path + '/y_pred_test.npy', y_pred)
    np.save(model_load_path + '/y_actual_test.npy', y_eval)