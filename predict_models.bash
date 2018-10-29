# Predict script for the models

PREFIX=experiments/trained_models/SP_Wave/
#PREFIX=experiments/trained_models/P_wave/
#EXPERIMENTS=(bn_on/two_layer/ bn_on/three_layer/ bn_off/two_layer/ bn_off/three_layer/)
EXPERIMENTS=(bn_on/two_layer/ bn_on/three_layer/)
#EXPERIMENTS=(bn_on/three_layer/)
NETWORKS=(UNet_dense AE_dense)
#LOAD=fga_data_set/SP_wave_data_set/FGA/eval
LOAD=fga_data_set/SP_wave_data_set/SEM/eval
#LOAD=fga_data_set/Pwave_data_set/FGA/eval/
#LOAD=fga_data_set/Pwave_data_set/SEM/eval/

for EXP in ${EXPERIMENTS[*]}; do
    for NET in ${NETWORKS[*]}; do
        python3 predict.py --load-path $PREFIX$EXP$NET --data-path $LOAD
    done
done
