# A script to convert Jim's binary fortran data into a single numpy array

#matlab -nodesktop -nodisplay -nojvm -r "open_fortran_binary"
matlab -nodesktop -nodisplay -nojvm -r "multi_output_open_fortran_binary"
python3 multi_output_convert_mat_to_numpy.py
rm *.mat

echo Numpy files generated