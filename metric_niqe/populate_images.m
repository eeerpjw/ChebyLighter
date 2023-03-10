function sub_folder_names = populate_images(folder_path)
    sub_folder_list = dir(folder_path);
    sub_folder_names_length = 0;
    sub_folder_names = cell(1,sub_folder_names_length+1);
    for k=1:length(sub_folder_list)
        if( isequal( sub_folder_list( k ).name, '.' )||...
            isequal( sub_folder_list( k ).name, '..')||...
            isequal( sub_folder_list( k ).name, '.DS_Store'))
            continue;
        end 
        sub_folder_names{sub_folder_names_length+1} = sub_folder_list(k).name;
        sub_folder_names_length = sub_folder_names_length+1;
    end
end