# Training script for the various neural networks

python3 runexperiments.py --config samples/UNet_conv_config.json --save-dir trained_models/UNet_conv
python3 runexperiments.py --config samples/UNet_dense_config.json --save-dir trained_models/UNet_dense
python3 runexperiments.py --config samples/UNet_res_config.json --save-dir trained_models/UNet_res

python3 runexperiments.py --config samples/AE_conv_config.json --save-dir trained_models/AE_conv
python3 runexperiments.py --config samples/AE_dense_config.json --save-dir trained_models/AE_dense
python3 runexperiments.py --config samples/AE_res_config.json --save-dir trained_models/AE_res