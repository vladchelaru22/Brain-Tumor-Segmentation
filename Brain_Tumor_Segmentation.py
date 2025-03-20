import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as sc

def rgb2gri(img_in, format):
    img_in=img_in.astype('float')
    s=img_in.shape
    if len(s)==3 and s[2]==3:
        if format=='png':
            img_out=(0.299*img_in[:,:,0]+0.587*img_in[:,:,1]+0.114*img_in[:,:,2])*255
        elif format=='jpg':
            img_out=0.299*img_in[:,:,0]+0.587*img_in[:,:,1]+0.114*img_in[:,:,2]
        img_out=np.clip(img_out, 0,255)
        img_out=img_out.astype('uint8')
        return img_out
    else:
        print('Conversia nu a putut fi realizata deoarece imaginea de intrare nu este color!')
        return img_in
    
def contrast_liniar_portiuni(img_in,L,a,b,Ta,Tb):
    s=img_in.shape #un tuplu cu proportiile imaginii, (linii, coloane)
    img_out=np.empty_like(img_in) #o imagine goala de aceeasi dimensiune
    img_in=img_in.astype(float) #aducem valorile in float pt inmultiri si impartiri
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            if (img_in[i,j]<a):
                img_out[i,j]=(Ta/a)*img_in[i,j]
            if(img_in[i,j]>=a and img_in[i,j]<=b):
                img_out[i,j]=Ta+((Tb-Ta)/(b-a))*(img_in[i,j]-a)
            if(img_in[i,j]>b):
                img_out[i,j]=Tb+((L-1-Tb)/(L-1-b))*(img_in[i,j]-b)
                
    img_out=np.clip(img_out,0,255) 
    img_out=img_out.astype('uint8') 
    return img_out

def binarizare (img_in, L, a):
    s=img_in.shape #un tuplu cu proportiile imaginii, (linii, coloane)
    img_out=np.empty_like(img_in) #o imagine goala de aceeasi dimensiune
    img_in=img_in.astype(float) #aducem valorile in float pt inmultiri si impartiri
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            if(img_in[i,j]<=a):
                img_out[i,j]=0;
            else:
                img_out[i,j]=L-1;
                
    img_out=np.clip(img_out,0,255) 
    img_out=img_out.astype('uint8') 
    return img_out

def putere_pct_fix(img_in,L,r,a):
    s=img_in.shape #un tuplu cu proportiile imaginii, (linii, coloane)
    img_out=np.empty_like(img_in) #o imagine goala de aceeasi dimensiune
    img_in=img_in.astype(float) #aducem valorile in float pt inmultiri si impartiri
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            if(img_in[i,j]<=a):
                img_out[i,j]=a*((img_in[i,j]/a)**r)
            if(img_in[i,j]>=a and img_in[i,j]<=L-1):
                img_out[i,j]=L-1-(L-1-a)*((L-1-img_in[i,j])/(L-1-a))**r
    img_out=np.clip(img_out,0,255) 
    img_out=img_out.astype('uint8') 
    return img_out

def exponential(img_in,L):
    s=img_in.shape #un tuplu cu proportiile imaginii, (linii, coloane)
    img_out=np.empty_like(img_in) #o imagine goala de aceeasi dimensiune
    img_in=img_in.astype(float) #aducem valorile in float pt inmultiri si impartiri
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            img_out[i,j]=L**((img_in[i,j])/(L-1))-1
    img_out=np.clip(img_out,0,255) 
    img_out=img_out.astype('uint8') 
    return img_out


cale=r'C:\Users\kellyy\Desktop\poze_proiect'
files = os.listdir(cale) #lista de obiecte din calea specificata mai sus
for i in files:          #se citeste fiecare imagine
    cale_img = os.path.join(cale, i)
    img_plt = plt.imread(cale_img)
    img_gray_tones = rgb2gri(img_plt, 'jpg')
    elem1 = np.ones((9,9))
    elem2 = np.ones((5,5))
    img_contrast_liniar=contrast_liniar_portiuni(img_gray_tones,256,60,100,20,180)
    img_pct_fix=putere_pct_fix(img_gray_tones, 256, 2, 100)
    img_exponential=exponential(img_gray_tones, 256)
    img_binarizata=binarizare(img_pct_fix, 256, 100)/255*img_gray_tones
    img_binarizata = (img_binarizata * 255).astype(np.uint8)
    _, otsu_thresholded = cv2.threshold(img_binarizata, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_erodata=sc.binary_erosion(otsu_thresholded, structure = elem1)
    img_dilatata=sc.binary_dilation(img_erodata,structure=elem2)
    
    plt.figure('Imagini')
    plt.subplot(1,4,1), plt.imshow(img_gray_tones, cmap='gray'), plt.title('originala')
    plt.subplot(1,4,2), plt.imshow(img_pct_fix, cmap='gray'), plt.title('pct fix')
    plt.subplot(1,4,3), plt.imshow(otsu_thresholded, cmap='gray'), plt.title('otsu')
    plt.subplot(1,4,5), plt.imshow(img_dilatata, cmap='gray'), plt.title('morfologie')
    plt.subplot(3,2,4), plt.imshow(img_binarizata, cmap='gray'), plt.title('Binarizare')
    
    plt.show()