# Predict script for the models

PREFIX=experiments/trained_models/
EXPERIMENTS=(bn_on/two_layer/ bn_on/three_layer/ bn_off/two_layer/ bn_off/three_layer/)
NETWORKS=(UNet_conv UNet_dense UNet_res AE_conv AE_dense AE_res)

for EXP in ${EXPERIMENTS[*]}; do
    for NET in ${NETWORKS[*]}; do
        python3 predict.py --load-path $PREFIX$EXP$NET
    done
done