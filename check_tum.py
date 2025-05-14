import json

def find_duplicate_keys(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    
    duplicates = set(data1.keys()) & set(data2.keys())
    if duplicates:
        print("重复的键：", duplicates)
    else:
        print("没有重复的键")

# 用法示例
file1 = "/home/ubuntu/xwy/dataset/TumEmo/train_7.json"
file2 = "/home/ubuntu/xwy/dataset/TumEmo/dev_7.json"
find_duplicate_keys(file1, file2)
