# Training script for the various neural networks
PREFIX=trained_models/batch_norm_off/


python3 runexperiments.py --config samples/UNet_conv_config.json --save-dir $PREFIX"UNet_conv"
python3 runexperiments.py --config samples/UNet_dense_config.json --save-dir $PREFIX"UNet_dense"
python3 runexperiments.py --config samples/UNet_res_config.json --save-dir $PREFIX"UNet_res"

python3 runexperiments.py --config samples/AE_conv_config.json --save-dir $PREFIX"AE_conv"
python3 runexperiments.py --config samples/AE_dense_config.json --save-dir $PREFIX"AE_dense"
python3 runexperiments.py --config samples/AE_res_config.json --save-dir $PREFIX"AE_res"