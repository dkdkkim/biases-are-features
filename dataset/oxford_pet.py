
import torch, os, json
from glob import glob
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import transforms
# import torch.optim as optim
# from torch import nn
# import matplotlib.pyplot as plt
from PIL import Image
# import numpy as np
import xmltodict, collections, cv2


# from torch_lr_finder import LRFinder

def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('RGB')
    return img

def load_data(path):
    filenames = glob(path + '/*.jpg')

    classes = set()

    data = []
    labels = []

    # Load the images and get the classnames from the image path
    for image in filenames:
        class_name = image.rsplit("\\", 1)[1].rsplit('_', 1)[0]
        classes.add(class_name)
        img = load_image(image)

        data.append(img)
        labels.append(class_name)

    # convert classnames to indices
    class2idx = {cl: idx for idx, cl in enumerate(classes)}        
    labels = torch.Tensor(list(map(lambda x: class2idx[x], labels))).long()

    data = list(zip(data, labels))

    return data, classes

# Since the data is not split into train and validation datasets we have to 
# make sure that when splitting between train and val that all classes are represented in both

def generateVOC2Json(rootDir,xmlFiles):
    attrDict = dict()
    attrDict["categories"]=[{"supercategory":"none","id":0,"name":"cat"},
                    {"supercategory":"none","id":1,"name":"dog"}
                  ]
    images = list()
    annotations = list()
    for root, dirs, files in os.walk(rootDir):
        image_id = 0
        # print(root)
        for file in xmlFiles:
            # print(file)
            image_id = image_id + 1
            if file in files:
                try:
                    annotation_path = os.path.abspath(os.path.join(root, file))
                    # print(annotation_path)
                    image = dict()
                    doc = xmltodict.parse(open(annotation_path).read())
                    # print(doc)
                    # input('enter')
                    image['file_name'] = str(doc['annotation']['filename'])
                    image['height'] = int(doc['annotation']['size']['height'])
                    image['width'] = int(doc['annotation']['size']['width'])
                    image['sem_seg_file_name'] = 'trimaps/' + file[:-4] + '.png'
                    image['id'] = image_id
                    print("File Name: {} and image_id {}".format(file, image_id))
                    images.append(image)

                    id1 = 1
                    if 'object' in doc['annotation']:
                        obj = doc['annotation']['object']
                        for value in attrDict["categories"]:
                            annotation = dict()
                            if str(obj['name']) == value["name"]:
                                annotation["iscrowd"] = 0
                                annotation["image_id"] = image_id
                                x1 = int(obj["bndbox"]["xmin"])  - 1
                                y1 = int(obj["bndbox"]["ymin"]) - 1
                                x2 = int(obj["bndbox"]["xmax"]) - x1
                                y2 = int(obj["bndbox"]["ymax"]) - y1
                                annotation["bbox"] = [x1, y1, x2, y2]
                                annotation["area"] = float(x2 * y2)
                                annotation["category_id"] = value["id"]
                                annotation["ignore"] = 0
                                annotation["id"] = image_id
                                
                                image_mask = cv2.imread(os.path.join(root[:-5], "trimaps/") + file[:-4] + ".png")
                    
                                xmin, xmax, ymin, ymax = mask_to_bbox(image_mask[:, :, 0])
                        
                                image_mask = np.where(image_mask==3, 1, image_mask)
                                image_mask = np.where(image_mask==2, 0, image_mask)
                                image_mask = image_mask.astype('uint8')
                                segmask = mask.encode(np.asarray(image_mask, order="F"))
                                
                                for seg in segmask:
                                    seg['counts'] = seg['counts'].decode('utf-8')
                                
                                x1 = int(xmin)
                                y1 = int(ymin)
                                x2 = int(xmax - x1)
                                y2 = int(ymax - y1)
                                annotation["bbox"] = [x1, y1, x2, y2]
                                annotation["area"] = float(x2 * y2)
                                
                                annotation["segmentation"] = segmask[0]
                                id1 +=1
                                # print(annotation)
                                annotations.append(annotation)

                    else:
                        print("File: {} doesn't have any object".format(file))
                except:
                    pass
                
            else:
                print("File: {} not found".format(file))
            

    attrDict["images"] = images    
    attrDict["annotations"] = annotations
    attrDict['info'] = {
        'contributor': 'DK',
        'date_created': '2021.04.29',
        'description': 'Pets',
        'url': '',
        'version': '1.0',
        'year': 2021
    }

    attrDict['licenses'] = [{'id': 1, 'name': 'DK', 'url': ''}]
    jsonString = json.dumps(attrDict)
    
    return jsonString

def load_annotation(txt_path='./annotations/test.txt', img_dir='./images', label_col=2):
    """
    Image CLASS-ID SPECIES BREED ID
		ID: 1:37 Class ids          --> label_col = 1
		SPECIES: 1:Cat 2:Dog        --> label_col = 2
		BREED ID: 1-25:Cat 1:12:Dog --> label_col = 3
    """
    f = open(txt_path)
    iter_f = iter(f)
    annotation_dict = collections.defaultdict(list)
    for line in iter_f:
        cur_anno = line.split() # [Image_name, CLASS-ID, SPECIES, BREED_ID]
        id = int(cur_anno[2])-1
        img_name = cur_anno[0]
        annotation_dict[id].append(('/'.join([img_dir,img_name+'.jpg']),id))
    return annotation_dict

class Databasket():
    "Helper class to ensure equal distribution of classes in both train and validation datasets"
    
    def __init__(self, data, num_cl, ):
        self.class_values = [[] for x in range(num_cl)]
        # create arrays for each class type
        for d in data:
            self.class_values[d]+=data[d]

    def gen_dataset_split(self, val_split=0.2, train_transforms=None, val_transforms=None):
        self.train_data = []
        self.val_data = []
        # put (1-val_split) of the images of each class into the train dataset
        # and val_split of the images into the validation dataset
        for class_dp in self.class_values:
            split_idx = int(len(class_dp)*(1-val_split))
            self.train_data += class_dp[:split_idx]
            self.val_data += class_dp[split_idx:]
        train_ds = PetDataset(self.train_data, transforms=train_transforms)
        val_ds = PetDataset(self.val_data, transforms=val_transforms)

        return train_ds, val_ds
    
    def gen_dataset(self, transforms=None):
        self.data = []
        for class_dp in self.class_values:
            self.data += class_dp
        ds = PetDataset(self.data, transforms=transforms)
        return ds

class PetDataset(Dataset):
    "Dataset to serve individual images to our model"
    
    def __init__(self, data, transforms=None):
        self.data = data
        self.len = len(data)
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = cv2.imread(img_path)
        if self.transforms:
            img = self.transforms(img)
            
        return img, label
    
    def __len__(self):
        return self.len


