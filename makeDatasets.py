import os
import shutil
import sys
import cv2


def split(src, splits):    
    if (sum(splits) != 1):
        print('Invalid split')
        return
    
    files = os.listdir(src)
    file_count = len(files)

    mid = int(file_count * splits[0]) + int(file_count * splits[1])
    train_files = [file for file in files[:int(file_count * splits[0])]]
    valid_files = [file for file in files[int(file_count * splits[0]): mid]]
    test_files = [file for file in files[mid:]]

    print(f'{len(train_files)} files in training')
    print(f'{len(valid_files)} files in validation')
    print(f'{len(test_files)} files in test')
    return train_files, valid_files, test_files

def make_files(src, dest, files):
    splits = ['train', 'valid', 'test']

    if not os.path.isdir(dest):
        os.mkdir(dest)
        for split in splits:
            os.mkdir(os.path.join(dest, split))
    
    count = 0
    total = sum([len(file_list) for file_list in files])
    for i, split in enumerate(splits):
        subdir = os.path.join(dest, split, src)
        os.mkdir(subdir)
        for file_name in files[i]:
            process_image(src, subdir, file_name)
            count += 1 
            percent = int(count / total * 100)
            print(f'{percent}% processed', end='\r')  
    print('\n')         
        
def process_image(src, dest, file_name):
    in_path = os.path.join(src, file_name)
    out_path = os.path.join(dest, file_name)
    img = cv2.imread(in_path)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cahle = cv2.createCLAHE()
    img_clahe = cahle.apply(img_gs)
    cv2.imwrite(out_path, img_clahe)   
    

if __name__ == '__main__':
    dest = sys.argv[1]
    splits = [float(split) for split in sys.argv[2:5]]
    items = sys.argv[5:]

    if os.path.isdir(dest):
        shutil.rmtree(dest)

    for item in items:
        print(f'Inventory {item} files')
        files = split(item, splits)
        print(f'Process {item} files')
        make_files(item, dest, files)
        print('\n')
                          
    