import sys
import os
import argparse
import cv2
import numpy as np
import scipy.misc
from models import baseline_model
from keras.models import load_model

from keras.utils import to_categorical 

from sklearn.model_selection import train_test_split

from keras import regularizers 

parser = argparse.ArgumentParser()

parser.add_argument("-train", action='store_true')
parser.add_argument("-test", action='store_true')
parser.add_argument("-test_and_vis", action='store_true')

parser.add_argument("-use_training_data", action='store_true')

train_images_opt_folder = 'data/training_images_opt/'
train_images_rgb_folder = 'data/training_images_rgb/'

# set up some constants for easy file i/o
test_images_opt_folder = 'data/test_images_opt/'
test_images_rgb_folder = 'data/test_images_rgb/'

BATCH_SIZE = 64 #128  #32
EPOCHS = 1

HEIGHT = 60 
WIDTH = 160 

debug = False

if not debug:
    EPOCHS = 50

class_dict = {1:"palm" , 2:"l" , 3:"fist" , 4:"fist_moved" , 5:"thumb" , 6:"index" ,
             7:"ok" , 8:"palm_moved" , 9:"c" , 10 : "down" }     
    
# i wanted to the files arranged numerically, which os.listdir doesn't inherently do.
def sort_files_numerically(path_to_files):
    files = os.listdir(path_to_files)
    for file in files:
        if(file.split(".")[1] != "jpg"):
            files.remove(file)

    return sorted(files, key=lambda x: int(x.split(".")[0]))

# calculate the optical flow between two frames and return a BGR image.
def get_dense_opt_flow(first, second):
    hsv = np.zeros_like(first)

    # don't care about saturation.
    hsv[...,1] = 255

    first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    # https://funvision.blogspot.cz/2016/02/opencv-31-tutorial-optical-flow.html
    flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.4, 1, 12, 2, 8, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

# load up the video and actually save the frames/optical flow output locally for faster training/testing.
def load_video(training=True):
    video = None
    if not training:
        video = cv2.VideoCapture('data/test.mp4')
    else:
        video = cv2.VideoCapture('data/train.mp4')

    success, first = video.read()
    
    ####veda added 
    #cropping the relevant part 
    
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Loading video %d seconds long with FPS %d and total frame count %d ' % (total_frame_count/fps, fps, total_frame_count))

    count = 0
    while success:
        success, second = video.read()
        if not success:
            break
        # go get the optical flow and save both the rgb image and optical flow image locally. 
        
        ####veda added 
        #cropping the relevant part 
        
        
        flow = get_dense_opt_flow(first, second)
        
        print(flow.shape())
        if training:
            cv2.imwrite(train_images_opt_folder + str(count) + '.jpg', flow)
            cv2.imwrite(train_images_rgb_folder + str(count) + '.jpg', second)
        else:
            cv2.imwrite(test_images_opt_folder + str(count) + '.jpg', flow)
            cv2.imwrite(test_images_rgb_folder + str(count) + '.jpg', second)

        first = second
        sys.stdout.write("\rCurrently on frame %d of video. Processing with optical flow." % count)
        count += 1

    print('Saved %d frames' % (count) )
    video.release()

# simple function to handle all the image preprocessing stuff in one fucntion.
def process_image(file_name, training):

    #image = scipy.misc.imread( file_name) 
    image = scipy.misc.imread( file_name)
 
    #cropping is not done at the moment 
    #image = image[200:400 ]
    
    #image = image[ : , 80 : 560   ] # cropping  
    #img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    image = scipy.misc.imresize(   image, [HEIGHT, WIDTH ]) / 255 #resize and normalization simple 
    image = np.reshape(image , ( HEIGHT, WIDTH , 1 ) )   # reshape  to maintain channel input 
    # if debug: scipy.misc.imsave('data/debug_' + "file_name", image)
    return image

# go load data (images and speeds) for either the training set or the test set.
def get_data(training):

    if  training:
        dir = os.listdir('data/leapGestRecog/') 
        
        if(len(dir) != 10):
            print("No data present or missing some classes")
            sys.exit() 
            
        
    images = []
    labels = []

    count = 0
    data_path = "data/leapGestRecog/" 
    
    num_of_sets = 10 
    num_of_classes = 10 
    num_of_imgs_per_class = 200 
    
    count = 0 
    
    for i in range(num_of_sets) :
        for j in range(num_of_classes):
            for k in range(num_of_imgs_per_class): 
                name_file = data_path + "0" + str(i) + "/" + str(j+1).zfill(2) + "_" + class_dict[j+1]  \
                                    + "/" + "frame" + "_" + "0" + str(i) + "_" + str(j+1).zfill(2) + "_"  + \
                str(k+1).zfill(4) + "." + "png" 
                 
                image = process_image(name_file, training)   
                images.append(image) 
                labels.append(j) 
                
                #flip horizontal the image  is necessary   
                
                image_flip = np.flip(image , 1 ) 
                images.append(image_flip) 
                labels.append(j) 
                
                
                
                if count % 1000 == 0 :
                    print("processed images count " , count ) 
                if debug and count == 1000:
                    break
                count += 1 
                
    print('\n')

    return np.asarray(images), np.asarray(labels )


def train(training):
    X, y = get_data(training) 
    #if os.path.exists('model.h5') :
        #model = load_model('model.h5') 
    #else :
    
    perm = np.random.permutation(X.shape[0])
    
    X = X[perm] 
    y=y[perm]
    y_onehot = to_categorical(y) 
    
    model = baseline_model(X.shape[1], X.shape[2], X.shape[3]) 
    #model = load_model('model.h5') 
   
    model.fit(X, y_onehot   , batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.2    , shuffle=True    )  

    model.save('model.h5')

def test_and_visualize(training):
    X, y = get_data(training)
    model = load_model('model.h5')

    if not training:
        opt_file_names = sort_files_numerically(test_images_opt_folder)
        rgb_file_names = sort_files_numerically(test_images_rgb_folder)
    else:
        opt_file_names = sort_files_numerically(train_images_opt_folder)
        rgb_file_names = sort_files_numerically(train_images_rgb_folder)

    index = 0

    # credit: https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2/34273603
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.9
    fontColor              = (255,255,255)
    lineType               = 2

    for image_for_model, speed, opt_file, rgb_file in zip(X, y, opt_file_names, rgb_file_names):
        if not training:
            full_opt = scipy.misc.imread(test_images_opt_folder + opt_file)[200:400]
            full_rgb = scipy.misc.imread(test_images_rgb_folder + rgb_file)[200:400]
        else:
            full_opt = scipy.misc.imread(train_images_opt_folder + opt_file)[200:400]
            full_rgb = scipy.misc.imread(train_images_rgb_folder + rgb_file)[200:400]

        predicted_speed = model.predict(np.expand_dims(image_for_model, axis=0))[0][0]


        cv2.putText(full_rgb, "Actual: " + str(round(speed, 2)),
            (10,30),
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.putText(full_rgb, "Predicted: " + str(round(predicted_speed,2)),
            (10,60),
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.putText(full_rgb, "Error: " + str(round(abs(predicted_speed - speed), 2)),
            (10,90),
            font,
            fontScale,
            fontColor,
            lineType)

        cv2.imshow('frame', np.concatenate((full_opt, cv2.cvtColor(full_rgb, cv2.COLOR_BGR2RGB)), axis=0))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        index += 1

def evaluate_model(training):
    X, y = get_data(training)
    print("Loading/ Evaluating model... ")
    model = load_model('model.h5')
    loss_and_metrics = model.evaluate(X, y, batch_size=32)

    print("Done evaluating, found MSE to be...")
    print(loss_and_metrics)

def get_test_txt(training):
    X, y = get_data(training)
    print("Loading/ Evaluating model... ")
    text_file = open("test.txt", "w")
    model = load_model('model.h5')

    for image_for_model, speed in zip(X,y):
        pred = model.predict(np.expand_dims(image_for_model, axis=0))[0][0]
        text_file.write("%s\n" % pred)

    text_file.close()


if __name__ == '__main__':
    args = parser.parse_args()
    use_training_data = args.use_training_data

    
    # get_test_txt(False)
    # sys.exit()

    if args.train:
        print("You chose to train a new model...")

        dir = os.listdir('data/') 
        
        if(len(dir) == 0):
            print("No data present")
            sys.exit() 
            

        train(use_training_data)

    elif args.test:
        print("You chose to test the model...")
        evaluate_model(use_training_data)

    elif args.test_and_vis:
        print("You chose to test and visualize the output of the model...")
        test_and_visualize(use_training_data)
