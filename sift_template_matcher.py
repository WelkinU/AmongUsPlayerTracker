import cv2
import os
import numpy as np
import pandas as pd
import argparse
from VideoLoader import VideoLoader


#for knowing what color to draw the player dots on the video/image output
color_dict = {0: (0,0,0), #black
			  1: (0,0,255),#blue
			  2: (165,42,42),#brown
			  3: (0,255,255),#cyan
			  4: (0,128,0), #green
			  5: (0,255,0), #lime
			  6: (255,165,0), #orange
			  7: (255,0,147), #pink
			  8: (128,0,128), #purple
			  9: (255,0,0), #red
			  10:(255,255,255), #white
			  11:(255,255,0), #yellow
			}
#switch color dict from rgb to bgr for weird opencv BGR nonsense
color_dict = {key:(val[2],val[1],val[0]) for key,val in color_dict.items()}

class SIFTTemplateMatcher():
	''' Match one or more query images to a preset map image
	Uses SIFT keypoint matching with homography transform
	Based on https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html'''

	def __init__(self,map_image):
		self.map_image = map_image

		self.sift = cv2.xfeatures2d.SIFT_create()

		# find the keypoints and descriptors with SIFT
		self.kp_map, self.des_map = self.sift.detectAndCompute(map_image,None)

		#initialize matcher
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

		self.center_list = []

	def query(self,query_image, MIN_MATCH_COUNT = 10, debug = False):
		kp_query, des_query = self.sift.detectAndCompute(query_image,None)
		
		matches = self.flann.knnMatch(des_query,self.des_map,k=2)

		good = []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append(m) 

		if len(good)>MIN_MATCH_COUNT:
			src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ self.kp_map[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			self.last_homography_matrix = M

			matchesMask = mask.ravel().tolist()
			h,w,d = query_image.shape

			try:
				transform_points = cv2.perspectiveTransform(np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2),M)
			except:
				#we can get here if it's a bad SIFT match, and all the template features go to the same map feature
				print('Bad SIFT Match')
				center,height,width = None, None, None
				matchesMask = None
				temp = self.map_image.copy()
			else:
				topleft,bottomleft,bottomright,topright = [np.int32(x[0]) for x in transform_points]

				center = tuple(np.int32((topleft + bottomright)/2))
				width = bottomright[0]-topleft[0]
				height = bottomright[1]-topleft[1]

				#draw output image polylines
				temp = cv2.polylines(self.map_image.copy(),[np.int32(transform_points)],True,(255,0,0),3, cv2.LINE_AA)

		else:
			print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
			center,height,width = None, None, None
			matchesMask = None
			temp = self.map_image.copy()
			
		self.center_list.append(center)

		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)

		img = cv2.drawMatches(query_image,kp_query,temp,self.kp_map,good,None,**draw_params)
		img = cv2.resize(img, (0,0), fx =0.5, fy = 0.5)

		return center,width,height,img

	def apply_last_homography_matrix(self, input_pts):
		return cv2.perspectiveTransform(input_pts,self.last_homography_matrix)


def process_video(map_image,query_video_path, yolo_output_dir, downsample_rate = 1):
	print('Initializing SIFTTemplateMatcher')
	stm = SIFTTemplateMatcher(map_image)
	print('Initializing VideoLoader')
	vid = VideoLoader(query_video_path)
	print(vid)
	print('Initializing YOLOOutputReader')

	yolo_reader = YOLOOutputReader(os.path.join(yolo_output_dir,os.path.basename(query_video_path)[:-4] + '_{}.txt'),
								video_width = vid.width, video_height = vid.height )

	print('Processing frames...')
	center_prev = None
	vid_writer = None
	#for idx,vid_idx in enumerate(range(0,len(vid),30)):
	for idx,frame in enumerate(vid[0::downsample_rate]):
		center,width,height,img = stm.query(frame, debug = True)

		if vid_writer is None:
			save_path = './videos/output.mp4'
			vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), vid.fps, (img.shape[1], img.shape[0]))

		if (center is None) or (height is None) or (width is None):
			print(f'Frame {idx*downsample_rate}: No matching map area! (Not enough SIFT features to match with any map location)')
			vid_writer.write(img)
			continue

		if  (map_image.shape[1] * 0.1 > width) or (map_image.shape[1] * 0.3 < width) or \
			(map_image.shape[0] * 0.1 > height) or (map_image.shape[0] * 0.3 < height):
			print(f'Frame {idx*downsample_rate}: No matching map area! width/height outside allowed range.')
			vid_writer.write(img)
			continue

		print(f'Frame {idx*downsample_rate}: center:{center},w={width},h={height}')

		#draw arrowed lines
		'''
		if (center is not None) and (center_prev is not None):
			if np.linalg.norm(np.array(center) - np.array(center_prev)) < map_image.shape[1]*0.2:
				map_image = cv2.arrowedLine(map_image, center_prev, center, (0,0,255), 5, tipLength = 0.1) #tiplen as percentage of line length
			else:
				print(f'Frame {idx*downsample_rate}: Distance between frame centers too far from prev frame')
		'''
		df = yolo_reader[idx*downsample_rate]

		if df is not None:
			for row in df.itertuples():
				transform_points = tuple(stm.apply_last_homography_matrix(np.float32([[row.x,row.y]]).reshape(-1,1,2))[0][0])
				map_image=cv2.circle(map_image,transform_points,5,color_dict[row.class_num],-1)

		vid_writer.write(img)
		center_prev = center

		#if idx * downsample_rate > 1185:
		#	break

	vid_writer.release()

	cv2.imshow('im',cv2.resize(map_image, (0,0), fx = 0.5, fy = 0.5))
	cv2.waitKey()


class YOLOOutputReader():

	def __init__(self, txt_format, video_width = 1280, video_height = 720):
		self.txt_format = txt_format
		self.width = video_width
		self.height = video_height
		c =['black','blue','brown','cyan','green','lime','orange','pink','purple','red','white','yellow']
		self.class_dict = {i:cl for i,cl in enumerate(c)}

	def __getitem__(self,idx):
		try:
			df = pd.read_csv(self.txt_format.format(idx+1), sep = ' ', index_col = False, names = ['class_num','x','y','w','h'])
		except:
			print(f'File does not exist for idx: {idx+1}')
			return None

		df['class'] = [self.class_dict[x] for x in df['class_num'].values]
		df['x'] = np.int32(df['x'].values * self.width)
		df['y'] = np.int32(df['y'].values * self.height)
		df['w'] = np.int32(df['w'].values * self.width)
		df['h'] = np.int32(df['h'].values * self.height)

		return df

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
	#copied from ultralytics yolov5 repo's utils/general.py

	# Plots one bounding box on image img
	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
	cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--map-img', type=str, default='map_images/the_skeld.png', help='path to the map image')
	parser.add_argument('--video', type=str, default='videos/Skeld With Tasks.mp4', help='path to video to run player tracker on')  # file/folder, 0 for webcam
	parser.add_argument('--yolo-output', type=str, default = r'C:\Users\W\Desktop\dev\yolov5\inference\output', help = 'path to yolo output')
	parser.add_argument('--query-image',type=str, default=None, help='for single image testing, overrides video input, intended for debugging/demo')

	opt = parser.parse_args()
	print(opt)

	map_image = cv2.imread(opt.map_img)
	map_image = cv2.resize(map_image, (0,0), fx = 0.5, fy = 0.5)
	#query_video_path = 'videos/Skeld With Tasks.mp4'

	if opt.query_image is None:
		query_video_path = opt.video
		process_video(map_image,query_video_path, opt.yolo_output)
	else:
		#process single image
		query_image = cv2.imread(opt.query_image)
		SIFTTemplateMatcher(map_image).query(query_image,debug = True)
	