# Training script for the various neural networks
SAVE_PREFIX=experiments/trained_models/
LOAD_PREFIX=experiments/config/
EXPERIMENTS=(bn_on/two_layer/ bn_on/three_layer/ bn_off/two_layer/ bn_off/three_layer/)
NETWORKS=(UNet_conv UNet_dense UNet_res AE_conv AE_dense AE_res)

for EXP in ${EXPERIMENTS[*]}; do
    for NET in ${NETWORKS[*]}; do
        python3 runexperiments.py --config $LOAD_PREFIX$EXP$NET"_config.json" --save-dir $SAVE_PREFIX$EXP$NET
    done
done