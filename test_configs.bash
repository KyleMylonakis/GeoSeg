# Test a directory full of config jsons.
# Uses 1D mulitclass label function.

config_dir=$1
label_fn=$2

dir_tag=${config_dir%"/"}

echo "dir_tag: $dir_tag"

configs=$(ls $config_dir | grep .json)

touch results.txt
echo "Running tests" > results.txt
for config in ${configs[@]};
    do 
    echo "On Config: $config"
    save_dir=$config"_"$dir_tag
    save_dir=${save_dir/".json"/""}
    
    python  main.py --config $config_dir"/"$config --save-dir $save_dir --label-fn $label_fn

    if [ -d $save_dir"/logs" ] 
    then 
        echo "$config: Success" >> results.txt
        rm -r $save_dir
    else
        echo "$config: Failure" >> results.txt
    fi
done
