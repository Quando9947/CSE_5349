import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import Menu
from tkinter import Spinbox
from tkinter import messagebox as mBox
import time
import math
import numpy as np
from MLAlgo import *
from playsound import playsound
alert_sound = './FallDownAlert.wav'

#tkinter don't have ToolTip，so it is defined as
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0


    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, _cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 27
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))


        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)


    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()
            
#===================================================================    
def createToolTip( widget, text):
    toolTip = ToolTip(widget)
#    def enter(event):
#        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
#    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

# Create instance
win = tk.Tk()   

# Add a title       
win.title("Python GUI")

# Disable resizing the GUI
win.resizable(0,0)

# Tab Control introduced here --------------------------------------
tabControl = ttk.Notebook(win)          # Create Tab Control


tab1 = ttk.Frame(tabControl)            # Create a tab 
tabControl.add(tab1, text='Window')      # Add the tab

tabControl.pack(expand=1, fill="both")  # Pack to make visible
# ~ Tab Control introduced here -----------------------------------------


#---------------Tab1------------------#
# We are creating a container tab3 to hold all other widgets
monty = ttk.LabelFrame(tab1, text='User Interface')
monty.grid(column=0, row=0, padx=8, pady=4)

'''
def get_data_size(*args):   
        Data_Size = name.get()
        if (Data_Size == ''):
            Data_Size = 0
        print('Data_Size is',Data_Size)
'''

# Changing our Label
ttk.Label(monty, text="0 means not falling, \n1 means falling,\n").grid(column=0, row=0, sticky='W')

# Adding a Textbox Entry widget
#name = tk.StringVar()
#nameEntered = ttk.Entry(monty, width=12, textvariable=name)
#nameEntered.grid(column=0, row=1, sticky='W')
#nameEntered.bind("<Button-1>",get_data_size)  


def clickMe():
    #action.configure(text='Click\n ' + name.get())
    algo_name = book.get()  
    #Data_Size = name.get()
    print('algo_name is',algo_name)
    start = time.time()
    end = time.time()
    running_time = math.ceil(end-start)
    print('running time is',running_time)
    ml_prediction = randomforest()
    if(ml_prediction == 3 ):
        ml_prediction = 1
    else:
        ml_prediction = 0 

    print(supportvectormachine())
    print(knearestneigbour())
    scr.insert(tk.INSERT,"Predict result is:"+str(ml_prediction)+'\n')

    if(ml_prediction == 3):
        playsound(alert_sound)
    #action.configure(state='disabled')    # Disable the Button Widget
#Adding a Button
action = ttk.Button(monty,text="Click",width=10,command=clickMe)   
action.grid(column=2,row=1,rowspan=2,ipady=7)

ttk.Label(monty, text="Select methods:").grid(column=1, row=0,sticky='W')

def go(*args):   #solve event
        algo_name = bookChosen.get()  
        print('algo_name is ',algo_name)

# Adding a Combobox
book = tk.StringVar()
bookChosen = ttk.Combobox(monty, width=12, textvariable=book)
bookChosen['values'] = ('Machine Learning', 'FFT')
bookChosen.grid(column=1, row=1)
bookChosen.current(0)  
bookChosen.config(state='readonly')  #ReadOnly
bookChosen.bind("<<ComboboxSelected>>",go)  #bind event

# Using a scrolled Text control    
scrolW  = 30; scrolH  =  5
scr = scrolledtext.ScrolledText(monty, width=scrolW, height=scrolH, wrap=tk.WORD)
scr.grid(column=0, row=3, sticky='WE', columnspan=3)

# Add Tooltip
#createToolTip(nameEntered,'This is an Entry.')
createToolTip(bookChosen, 'This is a Combobox.')
createToolTip(scr,        'This is a ScrolledText.')

#---------------Tab1------------------#

# create three Radiobuttons using one variable
radVar = tk.IntVar()

#----------------Menu-------------------#    
# Exit GUI cleanly
def _quit():
    win.quit()
    win.destroy()
   # exit()
    
# Creating a Menu Bar
menuBar = Menu(win)
win.config(menu=menuBar)


# Add menu items
fileMenu = Menu(menuBar, tearoff=0)
fileMenu.add_separator()
fileMenu.add_command(label="Exit", command=_quit)
menuBar.add_cascade(label="Option", menu=fileMenu)


# Place cursor into name Entry
#nameEntered.focus()      
#======================
# Start GUI
#======================
win.mainloop()

