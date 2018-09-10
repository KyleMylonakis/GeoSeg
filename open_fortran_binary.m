%fortran_data = dir('./fga_data_set/L7/Stations*');
fortran_data = dir('./fga_data_set/Pwave_data_set/FGA/Stations*');
num_files = length(fortran_data);
data_bucket = zeros(num_files,6000,3,3);
label_bucket = zeros(num_files,3);

%count = 1;
for ii = 1:length(fortran_data)
    
    fprintf(1,'Opening %s\n',fortran_data(ii).name);
    
    name = fortran_data(ii).name(10:end);
    numbers = string(strsplit(name,'_'));

    y_value = zeros(1,3);
    for iii = 1:3
        number = numbers(1,iii);
        y_value(iii) = str2num(number);
    end
    
    label_bucket(ii,1:3) = y_value;
    

    fid = fopen(strcat('./fga_data_set/Pwave_data_set/FGA/' , fortran_data(ii).name),'rb');
    tmp = fread(fid,inf,'double');
    normalize_tmp = tmp./max(max(max(abs(tmp))));
    data_bucket(ii,1:end) = normalize_tmp;
    %max(max(max(data_bucket(ii,:))))
    %size(data_bucket)
    fclose(fid);
    %count = count + 1;

end
%max(max(max(max(data_bucket))))

save('data_bucket','data_bucket');
save('label_bucket','label_bucket');
exit;

