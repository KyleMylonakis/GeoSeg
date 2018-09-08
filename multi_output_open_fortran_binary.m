%ordered_fortran_data = dir('./fga_data_set/L7/Stations*');
%ordered_fortran_data = dir('./fga_data_set/original_data/Stations*');
ordered_fortran_data = dir('./fga_data_set/Pwave_data_set/FGA/Stations*');
fortran_data = ordered_fortran_data(randperm(length(ordered_fortran_data)));

% 20833 files corresponds to ~3 gigs of memory for the mat file
max_bucket_size = 20833;
%max_bucket_size = 89;
%max_bucket_size = 300;
%max_bucket_size = 1000;
num_files = length(fortran_data);
num_labels = 3;

num_buckets = ceil(num_files/max_bucket_size);
for current_bucket = 1:num_buckets

    if current_bucket == num_buckets
       if mod(num_files,max_bucket_size) ~= 0
        max_bucket_size = mod(num_files,max_bucket_size); 
       end
    end

    data_bucket = zeros(max_bucket_size,6000,3,3);
    label_bucket = zeros(max_bucket_size,num_labels);

    %count = 1;
    start = (current_bucket-1)*max_bucket_size;
    for ii = 1:max_bucket_size

        fprintf(1,'Opening %s\n',fortran_data(start + ii).name);

        name = fortran_data(start + ii).name(10:end);
        numbers = string(strsplit(name,'_'));

        y_value = zeros(1,num_labels);
        for iii = 1:num_labels
            number = numbers(1,iii);
            y_value(iii) = str2num(number);
        end

        label_bucket(ii,1:num_labels) = y_value;


        %fid = fopen(strcat('./fga_data_set/L7/' , fortran_data(start + ii).name),'rb');
        fid = fopen(strcat('./fga_data_set/Pwave_data_set/FGA/' , fortran_data(start + ii).name),'rb');
        tmp = fread(fid,inf,'double');
        normalize_tmp = tmp./max(max(max(abs(tmp))));
        data_bucket(ii,1:end) = normalize_tmp;
        %max(max(max(data_bucket(ii,:))))
        %size(data_bucket)
        fclose(fid);
        %count = count + 1;
    end

    save(strcat('data_bucket_', num2str(current_bucket)),'data_bucket', '-v7.3');
    save(strcat('label_bucket_', num2str(current_bucket)),'label_bucket', '-v7.3');
end
    
exit;