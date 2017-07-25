import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import pandas as pd

with open('index.txt','r') as f:
  temp_list=f.readlines()

index_list=[]

for a in temp_list:
   temp=a.split('/')
   if(len(temp)>=4):
      temp[2]=temp[2]+'_'+temp[3]
   temp=temp[2].split(' ')
   index_list.append(temp[0])

os.environ['GLOG_minloglevel'] = '3'
caffe_root = '/home/user/Caffe/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

DEMO_DIR = '.'
cur_net_dir = 'Scene_Recognition_models'

mean_filename=os.path.join(DEMO_DIR,cur_net_dir,'places365CNN_mean.binaryproto')

proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]
mean=mean.mean(1).mean(1)
net_pretrained = os.path.join(DEMO_DIR,cur_net_dir,'vgg16_places365.caffemodel')
net_model_file=os.path.join(DEMO_DIR,cur_net_dir,'deploy_vgg16_places365.prototxt')

VGG_S_Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

def main(image_path,VGG_S_Net,index_list):
 try:
   input_image = caffe.io.load_image(image_path)
   prediction = VGG_S_Net.predict([input_image],oversample=False)
   os.environ['GLOG_minloglevel'] = '0'
   return_list=[]
   for a in np.array(prediction[0]).argsort()[::-1][:5]:
      return_list.append(index_list[int(a)])
      return_list.append(prediction[0][int(a)])
   return(return_list)
 except:
   print("Not a valid image")
   return(-1)

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", help="path to input image")
  ap.add_argument("-d", "--dir", help="path to the dir of image files")

  args = vars(ap.parse_args())
  if args["image"]  and os.path.isfile(args["image"]):
     print(main(args["image"],VGG_S_Net,index_list))
  elif args["dir"] and os.path.isdir(args["dir"]):
    df=pd.DataFrame(columns=["Image Name","Class SmartPic","Category 1","Prob 1","Category 2","Prob 2","Category 3","Prob 3","Category 4","Prob 4","Category 5","Prob 5"])
    for dir, subdir, files in os.walk(args["dir"]):
     for file in files:
        print(file)
        l=main(os.path.join(dir, file),VGG_S_Net,index_list)
        if(l!=-1):
            df=df.append(pd.Series([file]+[dir.split('/')[1]]+l, index=["Image Name","Class SmartPic","Category 1","Prob 1","Category 2","Prob 2","Category 3","Prob 3","Category 4","Prob 4","Category 5","Prob 5"]),ignore_index=True)
    df.to_csv("result.csv", sep=',')
  else:
     print("No Image found")
     print("Use python Scene.py -i path_to_image")




