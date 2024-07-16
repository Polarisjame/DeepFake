import os
import shutil

file_path = r'/data/lingfeng/full_data/phase1'
set_index = ['trainset', 'valset']

for set in set_index:
    file_count = 0
    sub_dir_count = 1
    parent_dir_path = os.path.join(file_path, set)
    sub_dir = os.path.join(parent_dir_path, f"sub_dir{sub_dir_count}")
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    for index,file in enumerate(os.listdir(parent_dir_path)):
        if index % 1000 == 0:
            print(index)
        old_path = os.path.join(parent_dir_path, file)
        new_path = sub_dir
        shutil.move(old_path, new_path)
        file_count+=1
        if file_count % 10000 == 0:
            sub_dir_count+=1
            sub_dir = os.path.join(parent_dir_path, f"sub_dir{sub_dir_count}")
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
    print(f"Stage:{set}, FileCount:{file_count}, Create{sub_dir}Subdirs")