

num_exp=12
num_fov=30
file_path_over_10=[]
for i in range (num_exp):
    for j in range (num_fov):
        public_data_path = '/data1/jiangy/andi_tcu/challenge_results/0710/oyX_new/track_1_vip' + '/exp_{}/'.format(i)  # make sure the folder has this name or change it
        csv_data_path = 'fov_{}.txt'.format(j)
        file_path=public_data_path+csv_data_path
        with open(file_path, 'r') as file:
            lines = file.readlines()
            num_lines = len(lines)
        if num_lines != 10:
            file_path_over_10.append(file_path)
print(file_path_over_10)

a=0