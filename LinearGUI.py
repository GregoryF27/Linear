#import tkinter as tk#tkinter is tk interface, glue code / binding to tk language/script
from tkinter import *
#other color names are red,orange, yellow green, blue, purple
# can also specify hexadecimal RGB value
#width/height specified in text units (dimensions of 0 char)

#basic Tkinter widges
#Label: used to display image/text on screen
# Button: adds buttons
# Canvas: to draw pictures + layouts like texts, graphics
#ComboBox: contains down arrow to select from avail options
# CheckButton: toggle buttons 
#RadioButton One-of-many selection; allows only one option selection
# Entry: used to input single line text entry from user
# Frame: container to hold / organize widgets
#Message: same as Label but multi-line and non-editable text
#Scale: provides graphical slider, allows select any value from scale
#Scrollbar: scool down contents, provices slide controller
#SpinBox: allows user to select from given set
#Text: allows user to edit multiline text + format the way its displayed
# Menu: used to create menus used in applications


#other color names are red,orange, yellow green, blue, purple
# can also specify hexadecimal RGB value
#width/height specified in text units (dimensions of 0 char)
from Linear import Vector, Matrix
root = Tk()
root.title("Linear Algebra is Fun!")
root.geometry('600x400')

m1 = Matrix(3,3,[[7,8,9],[4,5,6,],[1,2,3]])

l1 = Label(root, text=m1)
l1.grid() # .grid() puts it on the GUI

def clickedL1():
    
    l1.configure(text=m1.rref()) #changes l1
    
button = Button(root, text="RREF this matrix!", bg='red', fg='white', command=clickedL1) # command 
button.grid(column=2,row=3)

root.mainloop() #executes tkinter

# visual representation of vectors, linear transformations (hype!)