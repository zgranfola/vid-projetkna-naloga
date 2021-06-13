import cv2
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tkinter import * #doda vse (button, Frame, Entry...)
from tkinter import filedialog
import tkinter.font as font
import threading
from PIL import Image
import math
import imageio
import json
import random
from numpy import savez_compressed
import os
import keyboard
import requests
import webbrowser

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

import mysql.connector

# povezava na MySQL database. Ni najboljse ker hardcodamo uporabnisko in geslo ampak za zdaj je vredu.
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="geslo",
  database="mydb"
)

#funkcija za sobela
def sobel_filters(img):
    Kx = np.array(np.mat('-1 0 1; -2 0 2; -1 0 1'))
    Ky = np.array(np.mat('-1 -2 -1; 0 0 0; 1 2 1'))

    M, N = img.shape
    magnituda = np.full((M,N),0)
    koti = np.full((M, N), 0)

    sumx = 0
    sumy = 0
    #konvolucija izračun magnitude sobelova (G)§
    for i in range(0, M-2):
        for j in range(0, N-2):
            newPixelX = 0
            for a in range(i, i+3):
                newPixelY = 0
                for b in range(j, j+3):
                    sumx += Kx[newPixelX][newPixelY] * img[a][b]
                    sumy += Ky[newPixelX][newPixelY] * img[a][b]
                    newPixelY+=1
                newPixelX+=1
            magnituda[i][j] = math.sqrt(math.pow(sumx,2) + math.pow(sumy,2))

            #calculate angles if sumx = 0 then divide by 1 else error
            if (a == 0): 
                a = 1
            
            magnituda[i][j] = ((math.atan(b/a))*180)/np.pi

            if koti[i,j] < 0:
                koti[i,j] = 180 + (180 + koti[i,j])
            sumx=0
            sumy=0

    return koti, magnituda


def funDodajUporabnik():
    
    print("Dodajanje uporabnika")
    #odpre meni za izbiranje slike
    filetypes = [("Izberite sliko", ".jpg .png .jpeg .jpe .tif .gif .jfif")]
    filepath = filedialog.askopenfilename(title="Izberite sliko", filetypes=filetypes)
    # preberemo sliko
    img = cv2.imread(filepath)

    # pretvorimo v sivinsko
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        cv2.imshow("face",faces)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    facesGray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)

    blur1 = cv2.GaussianBlur(facesGray,(5,5),0) #Gaussian Filtering

    #creating hog features 
    fd, hog_image = hog(blur1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
    znacilnica = (fd[0]+fd[10]+fd[20]+fd[25]+fd[30])/5

    #vpisemo podatke za db, ime in starost
    name = input("Enter your name: ")
    age = input("Enter your age: ")
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    mycursor = mydb.cursor()

    #sql poizvedba za vpis v database
    sql = "INSERT INTO user (name, age, znacilnice, username, password) VALUES (%s, %s, %s, %s, %s)"
    val = (name, age, znacilnica, username, password)
    mycursor.execute(sql, val)
    mydb.commit()


    #simpleDB on visualstudiocode
    #savez_compressed("simpleDB/"+str(random.randint(1,10000))+".npz", data=fd) 


#funckija katera nam prebere celotno tabelo user, in nam returna vse vrednosti
def readFromDatabase():
    mycursor = mydb.cursor()
    sql = "SELECT * FROM user"
    mycursor.execute(sql)
    return mycursor.fetchall()



def funNajdiUporabnik():
    #snemanje
    cameraVideo = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if cameraVideo.isOpened == False:
        print("Neki te j**e")

    stop=True
    while stop:
        ret, frame = cameraVideo.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            # Load the cascade
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangle around the faces and crop the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                faces = frame[y:y + h, x:x + w]
            
            if cv2.waitKey(1) & 0xFF == ord('s'): #s tipko q zapreme kamero
                facesGray2= cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)

                blur2 = cv2.GaussianBlur(facesGray2,(5,5),0) #Gaussian Filtering

                fd1, hog_image1 = hog(blur2, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
                
                read = readFromDatabase() #klicemo funkcijo katera nam naredi sql poizvedbo
                print(read)
                fd = 0.0
                for znacilnica in read: #gremo skoz celo tabelo
                    #print(znacilnica[3])
                    koeficientM = 0.9 
                    koeficientV = 1.1
                    test1 = znacilnica[3]
                    test2 = (fd1[0]+fd1[10]+fd1[20]+fd1[25]+fd1[30])/5

                    if test1*koeficientM < test2 and test2 < test1*koeficientV:
                        print("enaki sliki")
                        username = znacilnica[4]
                        password = znacilnica[5]
                        
                        print("Ime: " + znacilnica[1] + ", age: " + str(znacilnica[2]))
                        #post metoda z
                        url = 'http://localhost:3000/HomeFaceScan'
                        #myobj = {'username', 'password'}
                        x = requests.post(url, data = {'Uporabniško_ime': username, 'Geslo': password})
                        chrome_path = "C://Program Files (x86)//Google//Chrome//Application/chrome.exe %s"
                        webbrowser.get(chrome_path).open("http://localhost:3000/HomeFaceScan?"+username+"&"+password)
                        #print(x.text)  
                    else:
                        print("bol slaba")
                    break
                            
        if keyboard.is_pressed("q"):
            stop = False
            
        # Display the output
        cv2.imshow('Frame', frame)


    cameraVideo.release()
    cv2.destroyAllWindows()

#tukaj spodaj je glavni program

#vmesnik oz. okence kjer imamo gumbe
window = Tk()
window.title("ORV") #določimo ime okenca
window.geometry("400x200")

#mal igranja z gumbami pa tak...
myFont = font.Font(family='Helvetica', size=15)

btnDodajUporabnika = Button(window, text="Dodaj uporabnika",bg='#0052cc', fg='#Ffffff', command=funDodajUporabnik, width=20, height=3, font=myFont)
btnDodajUporabnika.pack(pady=1)

btnPrepoznajUporabnika = Button(window, text="Prepoznaj uporabnika",bg='#0052cc', fg='#Ffffff', command=funNajdiUporabnik, width=20, height=3, font=myFont)
btnPrepoznajUporabnika.pack(pady=1)

window.mainloop()