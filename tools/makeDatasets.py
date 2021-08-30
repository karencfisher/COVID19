import os
import shutil
import sys
import cv2

from tools import parseArgs


def split(src, item, splits):
    '''
    List image files and divide list into portions according to percentages
    for "train, validate, and test" sets.

    parameters:
    src: parent directory
    item: subdirectory with image files for that class
    splits: tuple of three floats, proportion of each set. 
            E.g.: (0.8, 0.1, 0.1) splits files into 80% for
            training, 10% for validation, and 10% for test.

    returns:
    file lists for train, validate, and test

    effects:
    None
    '''    
    if (sum(splits) != 1):
        print('Invalid split')
        return
    
    # lists and counts image files in src/item direcotry
    src_dir = os.path.join(src, item)
    files = os.listdir(src_dir)
    file_count = len(files)

    # split files into three proportionate portions according to splits
    mid = int(file_count * splits[0]) + int(file_count * splits[1])
    train_files = [file for file in files[:int(file_count * splits[0])]]
    valid_files = [file for file in files[int(file_count * splits[0]): mid]]
    test_files = [file for file in files[mid:]]

    print(f'{len(train_files)} files in training')
    print(f'{len(valid_files)} files in validation')
    print(f'{len(test_files)} files in test')
    return train_files, valid_files, test_files

def make_files(src, item, dest, files):
    '''
    Retrieve files from source, process and store each in destination

    parameters:
    src: source path
    item: specfic class
    dest: destination path
    files: lists of file lists (for train, validate, and test sets)

    return:
    None

    Effects: creates and populates the destination path
    '''
    parts = ['train', 'valid', 'test']

    # Create destination directories if not already existing
    if not os.path.isdir(dest):
        os.mkdir(dest)
    dest_dir = os.path.join(dest, item)
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    
    # for each of train, validate, and test portions
    count = 0
    total = sum([len(file_list) for file_list in files])
    for i, part in enumerate(parts):
        part_dir = os.path.join(dest_dir, part)
        src_dir = os.path.join(src, item)
        # retrieve files, process and save in the destination
        for file_name in files[i]:
            if enhancement is not None:
                process_image(src_dir, part_dir, file_name, enhancement)
            count += 1 
            percent = int(count / total * 100)
            print(f'{percent}% processed', end='\r')  
        print('\n')         
        
def process_image(src_dir, part_dir, file_name, enhancement):
    '''
    Process and store an individual image file

    parameters:
    src_dir: source directory (e.g., src_dir/classA)
    part_dir: destination directory for processed files
              (e.g., dest_dir/classA/train)
    file_name: image file name

    returns:
    None

    effects:
    stores processed image to destination path
    '''
    in_path = os.path.join(src_dir, file_name)
    out_path = os.path.join(part_dir, file_name)

    #Read original image file
    img = cv2.imread(in_path)

    # convert to gray scale
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if enhancement.lower() == 'clahe':
        # apply adaptive histogram equalization to enhance contrast
        cahle = cv2.createCLAHE()
        img = cahle.apply(img_gs)
    else:
        raise ValueError('Invalid or unsupported enhancement method')

    # write enhanced file to destination
    cv2.imwrite(out_path, img)   
    

if __name__ == '__main__':
    # Get parameters as dictionary
    params = parseArgs(sys.argv[1:])

    # Get paths from arguments and properly encode for
    # platform
    src = params.get('src_path', None)
    if src is not None:
        src = os.path.normpath(src)
    dest = os.path.normpath(params['dest_path'])

    # Get other parameters
    items = params['items']
    if not isinstance(items, list):
        items = [items]
    splits = [float(split) for split in params['splits']]
    enhancement = params.get('enhance', None)

    # remove destination if already exists
    if os.path.isdir(dest):
        shutil.rmtree(dest)

    # process by each class
    for item in items:
        print(f'Inventory {item} files')
        files = split(src, item, splits)
        print(f'Process {item} files')
        make_files(src, item, dest, files, enhancement)
        print('\n')
                          
    