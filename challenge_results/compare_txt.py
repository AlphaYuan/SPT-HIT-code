import os

def get_subdirectories(directory):
    """
    获取目录下所有子文件夹的路径
    """
    return [os.path.join(directory, sub_dir) for sub_dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, sub_dir))]

def compare_files(file1, file2):
    """
    比较两个文件的内容是否一致
    """
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        content1 = f1.read()
        content2 = f2.read()
    # return content1 == content2
    for idx, (i, j) in enumerate(zip(content1, content2)):
        i_split = i.split(',')
        j_split = j.split(',')
        if len(i_split) != len(j_split):
            print(i, j)
            return False
    return True

def compare_directories(dir1, dir2):
    """
    比较两个目录中的所有 ensemble.txt 文件内容是否一致
    """
    subdirs1 = get_subdirectories(dir1)
    subdirs2 = get_subdirectories(dir2)
    
    # 按子目录名称排序，确保文件比较是按相同顺序进行的
    subdirs1.sort()
    subdirs2.sort()

    print(len(subdirs1))
    print(len(subdirs2))
    print()
    
    if len(subdirs1) != len(subdirs2):
        print("子文件夹数量不一致")
        # return False
    
    for subdir1, subdir2 in zip(subdirs1, subdirs2):
        # for i in range(30):
        #     file1 = os.path.join(subdir1, 'fov_{}.txt'.format(i))
        #     file2 = os.path.join(subdir2, 'fov_{}.txt'.format(i))
        #     # print(file1, file2)
        #     if not os.path.exists(file1) or not os.path.exists(file2):
        #         print(f"文件缺失: {file1} 或 {file2}")
        #         return False
        #     if not compare_files(file1, file2):
        #         print(f"文件内容不一致: {file1} 和 {file2}")
        #         return False

        file1 = os.path.join(subdir1, 'ensemble_labels.txt')
        file2 = os.path.join(subdir2, 'ensemble_labels.txt')
        # print(file1, file2)
        if not os.path.exists(file1) or not os.path.exists(file2):
            print(f"文件缺失: {file1} 或 {file2}")
            return False
        if not compare_files(file1, file2):
            print(f"文件内容不一致: {file1} 和 {file2}")
            # return False
    
    print("所有文件内容一致")
    return True

# 示例用法
# dir1 = '/Users/alphayuan/Desktop/Lab/project/AnDiChallenge/结果/challenge/0701/daR_new/track_1_all/ensemble'
# dir2 = '/Users/alphayuan/Desktop/Lab/project/AnDiChallenge/结果/challenge/0706/daR_new/track_1_all/ensemble'

dir1 = '0713/track_1_all/ensemble'
# dir1 = '0713/track_1_vip'
dir2 = 'track_1'
compare_directories(dir1, dir2)
