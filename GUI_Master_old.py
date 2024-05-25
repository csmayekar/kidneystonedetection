import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import CNNModel 
import h5py
from sklearn import svm
import svm as svm
import pickle
from skimage import feature
import sqlite3
#import tfModel_test as tf_test
global fn
fn=""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Kidney Stone Detection system Using ML")

#####For background Image
#loading the images
img=ImageTk.PhotoImage(Image.open("image1.jpg"))

img2=ImageTk.PhotoImage(Image.open("image2.jpg"))

img3=ImageTk.PhotoImage(Image.open("image3.jpeg"))


logo_label=tk.Label()
logo_label.place(x=0,y=0)

x=1


# function to change to next image
def move():
    global x
    if x == 4:
            x = 1
    if x == 1:
        logo_label.config(image=img)
    elif x == 2:
        logo_label.config(image=img2)
    elif x == 3:
        logo_label.config(image=img3)
    x = x+1
    root.after(2000, move)
  
# calling the function
move()

# calling the function
def shift():
    x1,y1,x2,y2 = canvas.bbox("marquee")
    if(x2<0 or y1<0): #reset the coordinates
        x1 = canvas.winfo_width()
        y1 = canvas.winfo_height()//2
        canvas.coords("marquee",x1,y1)
    else:
        canvas.move("marquee", -2, 0)
    canvas.after(1000//fps,shift)

canvas=tk.Canvas(root,bg="light blue")
canvas.pack()
canvas.place(x=0, y=0)
text_var="Kidney Stone Detection system Using ML"
text=canvas.create_text(0,-2000,text=text_var,font=('Raleway',25,'bold'),fill='white',tags=("marquee",),anchor='w')
x1,y1,x2,y2 = canvas.bbox("marquee")
width = 1600
height = 50
canvas['width']=width
canvas['height']=height
fps=40    #Change the fps to make the animation faster/slower
shift()   #Function Calling


#frame_display = tk.LabelFrame(root, text=" --Display-- ", width=900, height=250, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display.grid(row=0, column=0, sticky='nw')
#frame_display.place(x=300, y=100)

#frame_display1 = tk.LabelFrame(root, text=" --Result-- ", width=900, height=200, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display1.grid(row=0, column=0, sticky='nw')
#frame_display1.place(x=300, y=430)

#frame_display2 = tk.LabelFrame(root, text=" --Calaries-- ", width=900, height=50, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display2.grid(row=0, column=0, sticky='nw')
#frame_display2.place(x=300, y=380)

frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=220, height=400, bd=5, font=('times', 14, ' bold '),bg="grey")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=10, y=120)



def update_label1(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=300, y=550)
    
    
    
################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
def update_cal(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=350, y=400)
    
    
    
###########################################################################
def train_model():
 
    update_label("Model Training Start...............")
    
    start = time.time()

    X= CNNModel.main()
    
    end = time.time()
        
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
    msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

    print(msg)

import functools
import operator


def convert_str_to_tuple(tup):
    s = functools.reduce(operator.add, (tup))
    return s

def test_model_proc(fn):
    from tensorflow.keras.models import load_model
    #from keras.optimizers import Adam

#    global fn
    
    IMAGE_SIZE = 64
    LEARN_RATE = 1.0e-4
    CH=3
    print(fn)
    if fn!="":
        # Model Architecture and Compilation
       
        model = load_model('kidney_model.h5')
            
        # adam = Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
        img = np.array(img)
        
        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
        
        img = img.astype('float32')
        img = img / 255.0
        print('img shape:',img)
        prediction = model.predict(img)
        print(np.argmax(prediction))
        Nutrient=np.argmax(prediction)
        print(Nutrient)
        
        
        
        if Nutrient == 0:
             Cd="Kidney Cyst Detect"
             # result_label = tk.Label(root, text="Healthy knee image",height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
             # result_label.place(x=300, y=650)
        elif Nutrient == 1:
            Cd="Kidney Normal Detect"
            # result_label = tk.Label(root, text="Doubtful joint narrowing with possible osteophytic lipping", height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
            # result_label.place(x=300, y=650)

        elif Nutrient == 2:
            Cd="Kidney Stone Detect"
            # result_label = tk.Label(root, text="Definite presence of osteophytes and possible joint space narrowing", height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
            # result_label.place(x=300, y=650)
        
        elif Nutrient == 3:
            Cd="Kidney Tumor Detect"
            # result_label = tk.Label(root, text="Multiple osteophytes, definite joint space narrowing, with mild sclerosis.", height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
            # result_label.place(x=300, y=650)
        
       
    A = Cd 
        
    return A
  
               
               
               
# def clear_img():
    
#     img11 = tk.Label(frame_display, background='lightblue4',width=160,height=120)
#     img11.place(x=0, y=0)

def update_label(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=300, y=450)
# def train_model():
    
#     update_label("Model Training Start...............")
    
#     start = time.time()

#     X=Model_frm.main()
    
#     end = time.time()
        
#     ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
#     msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

#     update_label(msg)

def test_model():
    global fn
    if fn!="":
        update_label("Model Testing Start...............")
        
        start = time.time()
    
        X=test_model_proc(fn)
        
        X1="{0} ".format(X)
        
        end = time.time()
            
        ET="Execution Time: {0:.4} seconds \n".format(end-start)
        
        msg="Image Testing Completed.."+'\n'+ X1 + '\n'+ ET
        fn=""
    else:
        msg="Please Select Image For Prediction...."
        
    update_label(msg)
    
    
def openimage():
   
    global fn
    fileName = askopenfilename(initialdir='Grape Leaf Diasease Detection', title='Select image for Aanalysis ',
                               filetypes=[("all files", "*.*")])
    IMAGE_SIZE=200
    imgpath = fileName
    fn = fileName


#        img = Image.open(imgpath).convert("L")
    img = Image.open(imgpath)
    
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
#        img = img / 255.0
#        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)


    x1 = int(img.shape[0])
    y1 = int(img.shape[1])



    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(root, image=imgtk, height=250, width=250)
    

    img.image = imgtk
    img.place(x=300, y=100)
   # out_label.config(text=imgpath)

def convert_grey():
    global fn    
    IMAGE_SIZE=200
    
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
    
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)

    gs = cv2.resize(gs, (x1, y1))

    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(threshold)

    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250, font=("bold", 25), bg='bisque2', fg='black',height=250)
    #result_label1.place(x=300, y=400)
    img2 = tk.Label(root, image=imgtk, height=250, width=250,bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)

    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)

    img3 = tk.Label(root, image=imgtk, height=250, width=250)
    img3.image = imgtk
    img3.place(x=880, y=100)
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250, font=("bold", 25), bg='bisque2', fg='black')
    #result_label1.place(x=300, y=400)
    
    
def SVMModel_test(pth):
    def fd_hu_moments(image):
    #For Shape of signature Image
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    def quantify_image(image):
        features = feature.hog(image, orientations=9,
            pixels_per_cell=(10, 10), cells_per_block=(2, 2),
            transform_sqrt=True, block_norm="L1")
    
        # return the feature vector
        return features

    with open('clf_SVM.pkl', 'rb') as f:
        SVM_Cl = pickle.load(f)

    image = cv2.imread(pth)
    
    # pre-process the image in the same manner we did earlier
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
       
    # quantify the image and make predictions based on the extracted
    # features using the last trained Random Forest
    
    features1 = quantify_image(image)
    features2 = fd_hu_moments(image)
    global_feature = np.hstack([features1,features2])
    
    Nutrient =SVM_Cl.predict([global_feature])
    
    if Nutrient == 0:
         Cd="Kidney Cyst Detect"
         # result_label = tk.Label(root, text="Healthy knee image",height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
         # result_label.place(x=300, y=650)
    elif Nutrient == 1:
        Cd="Kidney Normal Detect"
        # result_label = tk.Label(root, text="Doubtful joint narrowing with possible osteophytic lipping", height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
        # result_label.place(x=300, y=650)

    elif Nutrient == 2:
        Cd="Kidney Stone Detect"
        # result_label = tk.Label(root, text="Definite presence of osteophytes and possible joint space narrowing", height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
        # result_label.place(x=300, y=650)
    
    elif Nutrient == 3:
        Cd="Kidney Tumor Detect"
        # result_label = tk.Label(root, text="Multiple osteophytes, definite joint space narrowing, with mild sclerosis.", height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
        # result_label.place(x=300, y=650)
        
    
    return Cd



def testSVM_model():
    global fn

    if fn!="":
        update_label("SVM Model Testing Start...............")
        
        start = time.time()
    
        X=SVMModel_test(fn)
        
        X1="Selected  {0}".format(X)
        
        end = time.time()
            
        ET="Execution Time: {0:.2} seconds \n".format(end-start)
        
        msg="Image SVM Testing Completed.."+'\n'+ X1 + '\n'+ ET
#        fn=""
    else:
        msg="Please Select Image For Prediction...."
        
    update_label(msg)

def SVM_Cl():


    update_label("SVM Training Process Start...............")
    
    start = time.time()

    X=SVM_Cl()
    
    end = time.time()
        
    ET="Execution Time: {0:.2} seconds \n".format(end-start)
    
    msg=X+'\n'+ET

    update_label(msg)


def DTModel_test(pth):
    def fd_hu_moments(image):
    #For Shape of signature Image
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    def quantify_image(image):
        features = feature.hog(image, orientations=9,
            pixels_per_cell=(10, 10), cells_per_block=(2, 2),
            transform_sqrt=True, block_norm="L1")
    
        # return the feature vector
        return features

    with open('clf_DT.pkl', 'rb') as f:
        RF_Cl = pickle.load(f)

    image = cv2.imread(pth)
    
    # pre-process the image in the same manner we did earlier
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
       
    # quantify the image and make predictions based on the extracted
    # features using the last trained Random Forest
    
    features1 = quantify_image(image)
    features2 = fd_hu_moments(image)
    global_feature = np.hstack([features1,features2])
    
    Nutrient =RF_Cl.predict([global_feature])
    
    if Nutrient == 0:
         Cd="Kidney Cyst Detect"
         # result_label = tk.Label(root, text="Healthy knee image",height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
         # result_label.place(x=300, y=650)
    elif Nutrient == 1:
        Cd="Kidney Normal Detect"
        # result_label = tk.Label(root, text="Doubtful joint narrowing with possible osteophytic lipping", height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
        # result_label.place(x=300, y=650)

    elif Nutrient == 2:
        Cd="Kidney Stone Detect"
        # result_label = tk.Label(root, text="Definite presence of osteophytes and possible joint space narrowing", height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
        # result_label.place(x=300, y=650)
    
    elif Nutrient == 3:
        Cd="Kidney Tumor Detect"
        # result_label = tk.Label(root, text="Multiple osteophytes, definite joint space narrowing, with mild sclerosis.", height=2, width=50, font=("bold", 25), bg='bisque2', fg='black')
        # result_label.place(x=300, y=650)
    
    return Cd



def testDT_model():
    global fn

    if fn!="":
        update_label("DT Model Testing Start...............")
        
        start = time.time()
    
        X=DTModel_test(fn)
        
        X1="Selected  {0}".format(X)
        
        end = time.time()
            
        ET="Execution Time: {0:.2} seconds \n".format(end-start)
        
        msg="Image DT Testing Completed.."+'\n'+ X1 + '\n'+ ET
#        fn=""
    else:
        msg="Please Select Image For Prediction...."
        
    update_label(msg)

def DT_Cl():


    update_label("DT Training Process Start...............")
    
    start = time.time()

    X=DT_Cl()
    
    end = time.time()
        
    ET="Execution Time: {0:.2} seconds \n".format(end-start)
    
    msg=X+'\n'+ET

    update_label(msg)
    
    


def process():
    global fn
    
    from subprocess import call
    call(['python','precautions.py'])
    
        


#################################################################################################################
def window():
    root.destroy()




button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button1.place(x=10, y=40)

button2 = tk.Button(frame_alpr, text="Image_preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button2.place(x=10, y=100)

#button3 = tk.Button(frame_alpr, text="Train Model", command=train_model, width=12, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
#button3.place(x=10, y=160)

button3 = tk.Button(frame_alpr, text="CNN_Prediction", command=test_model,width=15, height=1,bg="white",fg="black", font=('times', 15, ' bold '))
button3.place(x=10, y=170)

button3 = tk.Button(frame_alpr, text="SVM_Prediction", command=test_model,width=15, height=1,bg="white",fg="black", font=('times', 15, ' bold '))
button3.place(x=10, y=230)

button3 = tk.Button(frame_alpr, text="DT_Prediction", command=test_model,width=15, height=1,bg="white",fg="black", font=('times', 15, ' bold '))
button3.place(x=10, y=280)

button4 = tk.Button(frame_alpr, text="Precautions", command=process, width=12, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button4.place(x=10, y=330)



exit = tk.Button(frame_alpr, text="Exit", command=window, width=15, height=1, font=('times', 15, ' bold '),bg="black",fg="white")
exit.place(x=10, y=370)




root.mainloop()