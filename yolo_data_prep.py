''' Converts from data labeled in labelme JSON format into darknet format
described here: https://github.com/ultralytics/yolov5/issues/12

Use labelme to label your training data, then run this script with --data where your images are
'''

import json
import os
from shutil import copyfile
import cv2
import argparse

classes = ['black','blue','brown','cyan','green','lime','orange','pink','purple','red','white','yellow']

def data_prep(data_dir, outdir, copy_images = True):
	os.makedirs(os.path.join(outdir,'images','train'), exist_ok = True)
	os.makedirs(os.path.join(outdir,'labels','train'), exist_ok = True)

	file_pair_list = [(file[:-5]+'.jpg', file) for file in os.listdir(data_dir) if file.endswith('.json')]

	for img_name, json_name in file_pair_list:
		#copy images
		if copy_images:
			copyfile(os.path.join(data_dir,img_name), os.path.join(outdir,'images','train',img_name))

		img = cv2.imread(os.path.join(data_dir,img_name))

		#read json file
		with open(os.path.join(data_dir,json_name), 'r') as file:
			data = json.load(file)['shapes']

		#generate label files per: https://github.com/ultralytics/yolov5/issues/12
		with open(os.path.join(outdir,'labels','train',json_name[:-5]+'.txt'),'w') as file:
			for label in data:
				cl = label['label']
				class_id = classes.index(label['label'])
				bbox = label['points']
				normalized_bbox = [ (bbox[1][0] + bbox[0][0]) / (2 * img.shape[1]),
									(bbox[1][1] + bbox[0][1]) / (2 * img.shape[0]),
									abs(bbox[1][0] - bbox[0][0]) / img.shape[1],
									abs(bbox[1][1] - bbox[0][1]) / img.shape[0],
									]

				#print(f'{cl}, {class_id}, {normalized_bbox}')
				file.writelines('{} {} {} {} {}\n'.format(class_id, *normalized_bbox))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, default='./data', help='path with the jpg images and json files from labelme')
	parser.add_argument('--out', type=str, default='./data_darknet_format', help='data to output darknet format labels')  # file/folder, 0 for webcam

	opt = parser.parse_args()
	print(opt)

	data_prep(opt.data,opt.out)