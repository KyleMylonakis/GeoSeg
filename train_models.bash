# Training script for the various neural networks
SAVE_PREFIX=experiments/trained_models/SP_Wave/
#SAVE_PREFIX=experiments/trained_models/P_wave/
LOAD_PREFIX=experiments/config/SP_Wave/
#LOAD_PREFIX=experiments/config/P_Wave/
#EXPERIMENTS=(bn_on/two_layer/ bn_on/three_layer/ bn_off/two_layer/ bn_off/three_layer/)
EXPERIMENTS=(bn_on/two_layer/ bn_on/three_layer/)
NETWORKS=(UNet_dense AE_dense)

for EXP in ${EXPERIMENTS[*]}; do
    for NET in ${NETWORKS[*]}; do
        python3 runexperiments.py --config $LOAD_PREFIX$EXP$NET"_config.json" --save-dir $SAVE_PREFIX$EXP$NET
    done
done