'''
1.多线程
2.去掉labelme
3.ujson
python labelme2yolo_fast.py './labelmeDataset' --save_dir='./YOLODataset' --label_path='label.txt' --val_size=0.2 --thread_num=15
'''
import ujson
import os
import sys
import shutil
import random
from multiprocessing.pool import ThreadPool
import argparse

#NAMES = ["antenna","cam","car","chair","dent","dirt","exhaust","floor","good","handle","hole","lamp","logo","mirror","paint","part","reflection","ribbon","scratch","sensor","shadow","tire","void","wheel","wiper"]

class Labelme2Yolo(object):
    def __init__(self, labelme_dir, save_dir, val_size, thread_num):
        self.labelme_dir = labelme_dir
        if save_dir == 'default':
            self.save_dir = os.path.join(os.path.split(labelme_dir.rstrip('/'))[0],'YOLODataset')
        else:
            self.save_dir = save_dir
        self.labels = NAMES     
        self.thread_num = thread_num
        self.train_list,self.val_list = [],[]
        self.make_train_val_dir()
        self.split_train_val(val_size)
        self.convert()
        
    def make_train_val_dir(self):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        self.train_dir_path = os.path.join(self.save_dir,'train')
        self.val_dir_path = os.path.join(self.save_dir,'val')

        for yolo_path in [os.path.join(self.train_dir_path,'images'), 
                          os.path.join(self.train_dir_path,'labels'),
                          os.path.join(self.val_dir_path,'images'), 
                          os.path.join(self.val_dir_path,'labels')]:            
            os.makedirs(yolo_path) 
            
    def split_train_val(self,val_size):
        all_name = []
        for name in os.listdir(self.labelme_dir):
            if name.endswith('json'):
                all_name.append(name.split('.')[0])
        random.shuffle(all_name)
        train_num = int(len(all_name)*val_size)
        self.val_list = all_name[:train_num]
        self.train_list = all_name[train_num:]
        
    #将单个的json转成yolo所需的txt
    def convert_json_to_yolo(self, json_path, yolo_path):
        yolo_obj_list = []
        with open (json_path,'r') as f:
            json_data = ujson.load(f)
        img_h, img_w= json_data['imageHeight'],json_data['imageWidth']
        
        for shape in json_data['shapes']:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape['shape_type'] == 'circle':
                yolo_obj = self.circle_to_box(shape, img_h, img_w)
            else:
                yolo_obj = self.other_to_box(shape, img_h, img_w)
            yolo_obj_str = [str(x) for x in yolo_obj]
            yolo_obj_list.append(' '.join(yolo_obj_str))
        
        with open (yolo_path,'w') as f:
            f.write('\n'.join(yolo_obj_list))
            
        src_img = json_path.replace('json','jpg')
        opt_img = yolo_path.replace('txt','jpg').replace('labels','images')
        try:
            shutil.copyfile(src_img, opt_img)
        except:
            sign = 0
            for attr in ['JPG','jpeg','png','JPEG','PNG']:
                src_img_attr = src_img.replace('jpg',attr)
                if os.path.exists(src_img_attr):
                    shutil.copyfile(src_img_attr,opt_img)
                    sign = 1
                    break
            if sign == 0:
                print('ERROR : The image must be jpg or jpeg or png format!')
                print(src_img)
                    
    def circle_to_box(self, shape, img_h, img_w):
        obj_center_x, obj_center_y = shape['points'][0]
        
        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 + 
                           (obj_center_y - shape['points'][1][1]) ** 2)
        obj_w = 2 * radius
        obj_h = 2 * radius
        
        yolo_center_x= round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)
            
        label_id = self.labels.index(shape['label'])
        
        return [label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h]
    
    def other_to_box(self, shape, img_h, img_w):
        x_lists = [point[0] for point in shape['points']]        
        y_lists = [point[1] for point in shape['points']]
        x_min = min(x_lists)
        y_min = min(y_lists)
        box_w = max(x_lists) - min(x_lists)
        box_h = max(y_lists) - min(y_lists)
        
        yolo_center_x= round(float((x_min + box_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((y_min + box_h / 2.0) / img_h), 6)
        yolo_w = round(float(box_w / img_w), 6)
        yolo_h = round(float(box_h / img_h), 6)
        
        label_id = self.labels.index(shape['label'])

        return [label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h]
        
    def save_yaml(self):
        yaml_path = os.path.join(self.save_dir, 'dataset.yaml')
        
        with open(yaml_path, 'w+') as yaml_file:
            yaml_file.write('train: ' + os.path.join(self.save_dir,'train') + '\n\n')
            yaml_file.write('val: ' + os.path.join(self.save_dir,'val') + '\n\n')
            yaml_file.write('nc: ' + str(len(NAMES)) + '\n\n')
            yaml_file.write('names: ' + ujson.dumps(NAMES))
        
    def convert(self):
        p = ThreadPool(self.thread_num) 
        for name in self.train_list:
            json_path = os.path.join(self.labelme_dir, name+'.json')
            yolo_path = os.path.join(self.train_dir_path, 'labels', name+'.txt')
            #print(yolo_path)
            #self.convert_json_to_yolo(json_path,yolo_path)
            p.apply_async(self.convert_json_to_yolo,args=(json_path,yolo_path))
        p.close()
        p.join()
        
        p = ThreadPool(self.thread_num) 
        for name in self.val_list:
            json_path = os.path.join(self.labelme_dir, name+'.json')
            yolo_path = os.path.join(self.val_dir_path, 'labels', name+'.txt')
            #print(yolo_path)
            self.convert_json_to_yolo(json_path,yolo_path)
            p.apply_async(self.convert_json_to_yolo,args=(json_path,yolo_path))
        p.close()
        p.join()
        self.save_yaml()
if __name__ == '__main__':
    # labelme_dir = '/media/data/training/data/orgDataset'
    # Labelme2Yolo(labelme_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument('labelme_dir',type=str, default = 'labelmeDataset',
                        help='Please input the path of the labelme dataset.')
    parser.add_argument('--save_dir',type=str, default='default',
                        help='Please input the path of the saving yolo dataset')
    parser.add_argument('--label_path',type=str, default='label.txt',
                        help='Please input the path of the saving yolo dataset')
    parser.add_argument('--val_size',type=float, default=0.2,
                        help='Please input the validation dataset size, for example 0.1 ')
    parser.add_argument('--thread_num',type=int, default=4,
                        help='Please input the thread number,the default number is 4')
    args = parser.parse_args()
    
    
    label_path = args.label_path
    with open(label_path,'r') as f:
        data = f.read()
    NAMES = eval(data)
    Labelme2Yolo(args.labelme_dir, args.save_dir, args.val_size, args.thread_num)
    