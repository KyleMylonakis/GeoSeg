# A script to convert Jim's binary fortran data into a single numpy array

#SPLIT=.7

matlab -nodesktop -nodisplay -nojvm -r "open_fortran_binary"
python3 convert_mat_to_numpy.py
rm *.mat

echo Numpy files generated