import os
import shutil
import random


def split(src, dest, splits, shuffle=False):
    '''
    Split image files in train, validation, and test directorys
    
    src = dir of files to split
    dest = root of new directories
    splits = tuple of percentages of train, validation, and test
             sets. Should add to 1. E.g. (.8, .1, .1)
    '''
    
    if (sum(splits) != 1):
        print('Invalid split')
        return
    
    files = os.listdir(src)
    file_count = len(files)
    if shuffle:
        random.shuffle(files)
        
    if not os.path.isdir(dest):
        os.mkdir(dest)
        os.mkdir(os.path.join(dest, 'train'))
        os.mkdir(os.path.join(dest, 'validation'))
        os.mkdir(os.path.join(dest, 'test'))
    
    subdir = os.path.join(dest, 'train', src)
    os.mkdir(subdir)
    for file in files[:int(file_count * splits[0])]:
        shutil.copy(os.path.join(src, file), os.path.join(subdir, file))
    train_set = set(os.listdir(subdir))
    
    mid = int(file_count * splits[0]) + int(file_count * splits[1])
    subdir = os.path.join(dest, 'validation', src)
    os.mkdir(subdir)
    for file in files[int(file_count * splits[0]): mid]:
        shutil.copy(os.path.join(src, file), os.path.join(subdir, file))
    valid_set = set(os.listdir(subdir))
    
    subdir = os.path.join(dest, 'test', src)
    os.mkdir(subdir)
    for file in files[mid:]:
        shutil.copy(os.path.join(src, file), os.path.join(subdir, file))
    test_set = set(os.listdir(subdir))
                          
        
    print(f'{len(train_set)} files in training')
    print(f'{len(valid_set)} files in validation')
    print(f'{len(test_set)} files in test')
    
    clean = not (train_set & valid_set & test_set)
    print(f'Sets are unique: {clean}')
    
def makeData(items, dest, splits, shuffle=False):
    if os.path.isdir('data'):
        shutil.rmtree(dest)

    for item in items:
        print(f'{item} images:')
        split(item, dest, splits, shuffle=shuffle)
        print('\n')                          
    