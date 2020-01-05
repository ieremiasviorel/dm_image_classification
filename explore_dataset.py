import pathlib
import matplotlib.pyplot as plt

codes = {}
idx = 0
for path in pathlib.Path('./input/stanford-dogs-dataset/images/Images/').iterdir():
    code = path.name.split('-')[0]
    codes[code] = idx 
    idx += 1

def path_to_label(path):
    code = path.stem.split('_')[0]
    return codes[code]

def get_all_file_path(directory, file_pattern=''):
    paths = list(f for f in directory.rglob('**/*{}'.format(file_pattern)) if f.is_file())
    return sorted(paths, key=str) 

# get path to all images ; sorted by name
all_image_cropped_paths = get_all_file_path(pathlib.Path("./data"), '.jpg') # PosixPath
# converts each path to a breed index : 0, 1, 2
all_image_cropped_labels = [path_to_label(path) for path in all_image_cropped_paths] 

LABEL = all_image_cropped_labels

_ = plt.hist(LABEL, bins=120)
plt.xlabel('Label index')
plt.ylabel('Count')
plt.title('Label Distribution')
plt.show()