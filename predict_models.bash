# Predict script for the models

PREFIX=trained_models/batch_norm_off/

python3 predict.py --load-path $PREFIX"UNet_conv"
python3 predict.py --load-path $PREFIX"UNet_dense"
python3 predict.py --load-path $PREFIX"UNet_res"

python3 predict.py --load-path $PREFIX"AE_conv"
python3 predict.py --load-path $PREFIX"AE_dense"
python3 predict.py --load-path $PREFIX"AE_res"