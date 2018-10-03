#!user/bin/python
from scipy.misc import imread
import cPickle as pickle
import numpy as np
import os

data_path = r'../tensor/TT_RNN-master/Datasets/UCF11_yolo/'

#classes = ['basketball', 'biking']
classes = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding', 'soccer_juggling',
          'swing', 'tennis_swing', 'trampoline_jumping', 'volleyball_spiking', 'walking']


for k in range(11):
    class_name = classes[k]
    files = os.listdir(data_path + class_name )
    # os.system(r"cd ../darknet_new")
    for this_file in files:

        for filename in os.listdir(data_path + class_name + '/' + this_file):

            if ( filename.endswith(".mpg") ):

                os.system("./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights "+ data_path + class_name + '/' + this_file + '/' + filename)
                #os.system("ffmpeg -i {0} -r 24 -s 160*120 %d.jpg".format(data_path + class_name + '/' + this_file + '/' + filename))

                count = 0
                for txt_filename in os.listdir('./test/'):
                    if ( txt_filename.endswith(".txt") ):
                        count = count + 1
                    else:
                        continue



                ndarray = np.zeros([count, 19, 19, 425])
                for i in range(0,count):
                    file = './test/'+str(i) + '.txt'
                    txtdata=np.loadtxt(file)
                    txtdata=txtdata.reshape((19,19,425))
                    ndarray[i] = txtdata
                # print ndarray[:,1,1,1]


                write_out = open(data_path + class_name + '/' + this_file + '/' + filename + '.pkl', 'wb')
                pickle.dump(ndarray, write_out)
                write_out.close()


                for txt_filename in os.listdir('./test/'):
                    if ( txt_filename.endswith(".txt") ):
                        os.remove('./test/'+txt_filename)
                    else:
                        continue


            else:
                continue


# delete the .mpg file
for k in range(11):
    class_name = classes[k]
    files = os.listdir(data_path + class_name )
    for this_file in files:

        for filename in os.listdir(data_path + class_name + '/' + this_file):
            #print filename
            if ( filename.endswith(".mpg") ):
                os.remove(data_path + class_name + '/' + this_file + '/' + filename)
            else:
                continue
