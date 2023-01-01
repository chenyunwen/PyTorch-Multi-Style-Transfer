import os
import cv2
import time
import numpy as np
import torch
from torch.autograd import Variable

from net import Net
from option import Options
import utils
from utils import StyleLoader
import imageio

def run_demo(args, mirror=False):
	style_model = Net(ngf=args.ngf)
	model_dict = torch.load(args.model)
	model_dict_clone = model_dict.copy()
	for key, value in model_dict_clone.items():
		if key.endswith(('running_mean', 'running_var')):
			del model_dict[key]
	style_model.load_state_dict(model_dict, False)
	style_model.eval()
	if args.cuda:
		style_loader = StyleLoader(args.style_folder, args.style_size)
		style_model.cuda()
	else:
		style_loader = StyleLoader(args.style_folder, args.style_size, False)

	# Define the codec and create VideoWriter object
	# input_video = cv2.VideoCapture(args.input_video)
	cam = cv2.VideoCapture(args.input_video) # loadVideo
	height =  args.demo_size
	width = int(4.0/3*args.demo_size)
	frame_width = int(cam.get(3))
	frame_height = int(cam.get(4))
	swidth = int(width/4)
	sheight = int(height/4)
	video = imageio.get_reader(args.input_video)
	fps = video.get_meta_data()['fps']
	print("fps = ", fps)
	if args.record:
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter(args.output_video , fourcc, fps, (frame_width,frame_height))
	# cam = cv2.VideoCapture(args.input_video) # loadVideo
	# cam.set(3, width)
	# cam.set(4, height)
	key = 0
	idx = 0
	style_v = style_loader.get(int(idx/20))
	style_v = Variable(style_v.data)
	style_model.setTarget(style_v)
	while True:
		# read frame
		idx += 1
		ret_val, img = cam.read()
		if (img is None):
			print("img is None")
			break
		# if mirror: 
		# 	img = cv2.flip(img, 1)
		cimg = img.copy()
		img = np.array(img).transpose(2, 0, 1)
		# changing style 
		# if idx%20 == 1:
		# 	style_v = style_loader.get(int(idx/20))
		# 	style_v = Variable(style_v.data)
		# 	style_model.setTarget(style_v)

		img=torch.from_numpy(img).unsqueeze(0).float()
		if args.cuda:
			img=img.cuda()

		img = Variable(img)
		img = style_model(img)

		if args.cuda:
			simg = style_v.cpu().data[0].numpy()
			img = img.cpu().clamp(0, 255).data[0].numpy()
		else:
			simg = style_v.data.numpy()
			img = img.clamp(0, 255).data[0].numpy()
		simg = np.squeeze(simg)
		img = img.transpose(1, 2, 0).astype('uint8')
		simg = simg.transpose(1, 2, 0).astype('uint8')

		# display
		simg = cv2.resize(simg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
		cimg[0:sheight,0:swidth,:]=simg
		# img = np.concatenate((cimg,img),axis=1)
		cv2.imshow('MSG Demo', img)
		#cv2.imwrite('stylized/%i.jpg'%idx,img)
		key = cv2.waitKey(1)
		if args.record:
			out.write(img)
		if key == 27: 
			break
	cam.release()
	if args.record:
		out.release()
	cv2.destroyAllWindows()

def main():
	# getting things ready
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")

	# run demo
	start = time.time()
	run_demo(args, mirror=True)
	stop = time.time()
	print(f"transfer time: {(stop - start)}s")

if __name__ == '__main__':
	main()
