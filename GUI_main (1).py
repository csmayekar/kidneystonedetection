import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk  #imgae
from tkinter.filedialog import askopenfilename #select ann open file
from tkinter import messagebox as ms
import sqlite3
import os
import numpy as np
import time
global fn
fn = ""



##############################################+=============================================================
#root: main top level window

root = tk.Tk()    
root.configure(background="brown")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Kidney Stone Detection system Using ML")



# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image and top bar


image2 = Image.open('./k3.jpg')
image2 = image2.resize((1380, 700))

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=10)  # , relwidth=1, relheight=1)

label_l1 = tk.Label(root, text="Kidney Stone Detection system Using ML",font=("Times New Roman", 35, 'bold'),
                    background="#131e3a", fg="white", width=60, height=1)
label_l1.place(x=-10, y=0)

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Create a Label widget to describe the purpose of the textbox
# Create labels with text and no background
label1 = tk.Label(root, text="YOUR HEALTH", font=('times', 55, 'bold'), bg="#6faebb", fg="white")
label1.place(x=100, y=100+30)

label2 = tk.Label(root, text="IS OUR", font=('times', 55, 'bold'), bg="#72b3bf", fg="#131e3a")
label2.place(x=100, y=200+30)

label3 = tk.Label(root, text="PRIORITY!", font=('times', 55, 'bold'), bg="#7eb6c7", fg="white")
label3.place(x=100, y=300+30)

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#login and registration buttons

def reg():
    root.destroy()
    from subprocess import call
    call(["python","registration.py"])

def log():
    root.destroy()
    from subprocess import call
    call(["python","login.py"])
    
def window():
  root.destroy()



button1 = tk.Button(root, text="Login", command=log, width=14, height=1,font=('times', 20, ' bold '), bg="#131e3a", fg="white")
button1.place(x=100, y=410+80)

button2 = tk.Button(root, text="Register",command=reg,width=14, height=1,font=('times', 20, ' bold '), bg="#131e3a", fg="white")
button2.place(x=350, y=410+80)

button3 = tk.Button(root, text="Click here to exit!",command=window,width=14, bd=0, height=1,font=('times', 20), bg="#72b3bf", fg="white")
button3.place(x=200, y=580)

root.mainloop()