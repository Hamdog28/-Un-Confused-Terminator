

#Librerias interfaz
from __future__ import division
from math import *
from Tkinter import *
import Tkinter,tkFileDialog
import tkMessageBox

from Main import Neural_Network
import os
  
top = Tk()
top.title("Progra 2")
top.configure(background='grey90')
top.geometry("200x400")
top.maxsize(400, 300)
top.minsize(350, 300)

Txt = ""
Img = ""
NN = Neural_Network()

  
def ejecutar():
    L4['text']="Calculando..."
    respuesta = NN.test(Img,None,Txt)
    L4['text']="Completado, \nEl numero ingresado es: "+ str(respuesta)
    return 0

def agregarTxt():
    global Txt
    file = tkFileDialog.askopenfile(parent=top,mode='rb',title='Choose a file')
    if file != None:
        data = file.read()
        Txt = file.name
        file.close()
        print "I got %d bytes from this file." % len(data)
        T1['text']=os.path.basename(file.name)
    return 0

def agregarImg():
    global Img
    file = tkFileDialog.askopenfile(parent=top,mode='rb',title='Choose a file')
    if file != None:
        data = file.read()
        Img = file.name
        file.close()
        print "I got %d bytes from this file." % len(data)
        T2['text']=os.path.basename(file.name)
    return 0

L=Label(top,text="\n (Un)Confused Terminator \n\n",
         font = "Verdana 13 bold", background = 'grey90').pack()
L1=Label(top,text="Ingrese archivo con los pesos:", background = 'grey90')
L2=Label(top,text="Ingrese imagen de prueba:", background = 'grey90')
L4=Label(top, text="",font = "Arial 18", background = 'grey90',width=30)
T1 = Label(top, height=1, width=23)
T2 = Label(top, height=1, width=23)
B = Button(top, text ="Iniciar", command = ejecutar, background = 'grey90')
B1 = Button(top, text ="Buscar", command = agregarTxt, background = 'grey90')
B2 = Button(top, text ="Buscar", command = agregarImg, background = 'grey90')
  
           
L1.place(x=30,y=60)
T1.place(x=35,y=80)
B1.place(x=270,y=80)

L2.place(x=30,y=130)
T2.place(x=35,y=150)
B2.place(x=270,y=150)



L4.place(x=20,y=190) 
B.place(x=260,y=250)
  
  
top.mainloop()