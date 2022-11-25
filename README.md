### 说明
该脚本可将labelme格式的数据转成yolo格式，并划分训练集测试集，同时生成yolo数据配置文件dataset.yaml。    
在原有的labelme2yolo的基础上做了以下优化。
1. 支持多线程，速度更快。
2. 不用导入labelme模块，不读取图片，只读json，速度更快
3. 用了ujson模块，速度更快

### 用法
#### 1.多线程
'./labelmeDataset'是原始的labelme格式的数据集。save_dir是输出yolo数据集的位置。 label_path是标签文件label.txt的位置。 val_size是划分测试集的比例。thread_num是开的线程数。
``` bash
python labelme2yolo_fast.py './labelmeDataset' --save_dir='./YOLODataset' --label_path='label.txt' --val_size=0.2 --thread_num=15
```
#### 2.单线程
相比多线程，单线程不用提供标签文件label.txt。
``` bash
python labelme2yolo_one.py './labelmeDataset' --save_dir='./YOLODataset' --val_size=0.2
```
### 参考
1.https://github.com/rooneysh/Labelme2YOLO
