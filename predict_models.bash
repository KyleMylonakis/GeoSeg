# Predict script for the models

#python3 predict.py --load-path trained_models/UNet_conv
python3 predict.py --load-path trained_models/UNet_dense
python3 predict.py --load-path trained_models/UNet_res

python3 predict.py --load-path trained_models/AE_conv
python3 predict.py --load-path trained_models/AE_dense
python3 predict.py --load-path trained_models/AE_res