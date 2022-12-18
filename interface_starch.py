import tkinter
import tkinter.filedialog
from PIL import Image,ImageTk
from torchvision import transforms as transforms
import os
import cv2
import numpy as np





win = tkinter.Tk()
win.title("Image Process")
win.geometry("1920x1080")


original = Image.new('RGB', (1024, 1360))
save_img = Image.new('RGB', (1024, 1360))

img2 = tkinter.Label(win)
img2_1 = tkinter.Label(win)
img2_2 = tkinter.Label(win)
img2_3 = tkinter.Label(win)
img2_4 = tkinter.Label(win)
img2_5 = tkinter.Label(win)
img2_6 = tkinter.Label(win)


pointsCount = 0
tpPointsChoose = []
lsPointsChoose = []
tpPointsChoose_display = []
lsPointsChoose_display = []
roi=np.array([])
roi1=np.array([])
roi2=np.array([])
roi3=np.array([])
roi4=np.array([])
roi5=np.array([])
roi6=np.array([])
points=[]
n=-1

def choose_file():
    select_file = tkinter.filedialog.askopenfilename(title='Select Image')
    e.set(select_file)
    global load
    load = Image.open(select_file)
    load_vision = transforms.Resize((512,680))(load)

    global original
    original = load
    global im
    im = np.asarray(original)
    global im_mark
    im_mark=np.copy(im)

    global output
    output = np.zeros((1024,1360,3), np.uint8)
    
    render = ImageTk.PhotoImage(load_vision)
    img  = tkinter.Label(win,image=render)
    img.image = render
    img.place(x=20,y=120)
    img2.destroy()
    img2_1.destroy()
    img2_2.destroy()
    img2_3.destroy()
    img2_4.destroy()
    img2_5.destroy()
    img2_6.destroy()
    
    global n
    n=n+1
    


    
#draw circle
def draw_mark(event,x,y,flags,param):
    global points,Cur_point,Start_point,pointsCount,im_mark
    global lsPointsChoose, tpPointsChoose,lsPointsChoose_display,tpPointsChoose_display,output
    #Circle
    if event==cv2.EVENT_LBUTTONDBLCLK:
        pointsCount = pointsCount + 1
        cv2.circle(im_mark, (x,y), 1, (255, 255, 255), -1)
        if roi.size>0:
            tpPointsChoose_display.append([x,y])
            lsPointsChoose_display.append((x,y))
            lsPointsChoose.append([x, y])  # 用于转化为darry 提取多边形ROI
            tpPointsChoose.append([x, y])           
        else:
            tpPointsChoose_display.append([x,y])
            lsPointsChoose_display.append((x,y))
            lsPointsChoose.append([x, y]) 
            tpPointsChoose.append((x, y))
        
        
    if event==cv2.EVENT_MBUTTONDOWN:
        draw_mask()
        lsPointsChoose = []
        lsPointsChoose_display = []   
        
    if event == cv2.EVENT_RBUTTONDBLCLK:  # 右键点击
        if  roi.size>0:
            pointsCount = 0
            tpPointsChoose_display= []
            lsPointsChoose_display = []
            tpPointsChoose = []
            lsPointsChoose = []
            output=np.zeros((1024,1360,3), np.uint8)
            im_mark=np.copy(roi)
        else:
            pointsCount = 0
            tpPointsChoose_display = []
            lsPointsChoose_display = []
            tpPointsChoose = []
            lsPointsChoose = []
            output=np.zeros((1024,1360,3), np.uint8)
            im_mark=np.copy(im)  

            
    '''if event==cv2.EVENT_MBUTTONDWON: 
        global radius    
        cv2.circle(im_mark,(x,y),radius,(255,255,255),-1)'''
 
    '''if  event == cv2.EVENT_LBUTTONDOWN:
        pointsCount = pointsCount + 1
        Start_point = [x,y]
        tpPointsChoose.append((x, y))
        points.append(Start_point)

    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        pointsCount = pointsCount + 1
        Cur_point = [x,y]
        tpPointsChoose.append((x, y))
        # print(points)
        cv2.line(im_mark,tuple(points[-1]),tuple(Cur_point),(255,255,255),1)
        tpPointsChoose.append((x, y))
        points.append(Cur_point)
        
    if event == cv2.EVENT_LBUTTONUP:
        Cur_point=Start_point
        cv2.line(im_mark,tuple(points[-1]),tuple(Cur_point),(255,255,255),1)'''
          
def draw_mask():
    global pts,display,im_mark,output,output_mask
    
    pts = np.array([lsPointsChoose], np.int32)  # pts是多边形的顶点列表（顶点集）
    pts = pts.reshape((-1, 1, 2))
    
    pts_display = np.array([lsPointsChoose_display], np.int32)  
    pts_display = pts_display.reshape((-1, 1, 2))
    
    display = cv2.fillPoly(im_mark, [pts_display], (255, 255, 255))
    output_mask = cv2.fillPoly(output, [pts_display], (255, 255, 255))
    
  
'''def roi_crop():
    global roi,roi1,roi2,roi3,im_mark,im_crop
    roi=im_crop[170:300,0:640]
    roi1=roi[0:130,0:240]
    roi2=roi[0:130,2000:440]
    roi3=roi[0:130,400:640]
    im_mark=np.copy(roi)

def roi_crop(event,x,y,flags,param): 
    global roi,im_mark,crop_Start_point,crop_Cur_point,crop_End_point
    global im_crop

    
    if  event == cv2.EVENT_LBUTTONDOWN:
        crop_Start_point = [x,y]
        #points.append(Start_point)

    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        crop_Cur_point = [x,y]
        #print(points)
        #points.append(Cur_point)
        
    if event == cv2.EVENT_LBUTTONUP:
        crop_End_point=crop_Cur_point
        cv2.rectangle(im_crop,(crop_Start_point[0],crop_Start_point[1]),(crop_Cur_point[0],crop_Cur_point[1]),(255,255,255),1)
        roi=im_crop[(crop_Start_point[1]+1):(crop_End_point[1]-1),(crop_Start_point[0]+1):(crop_End_point[0]-1)]
        im_mark=np.copy(roi)
    
    if event== cv2.EVENT_RBUTTONDOWN:
        im_crop=np.copy(im)'''
        
'''def circle_radius(x):
    global radius
    radius=x'''



#mark
def mark(): 
    global img2
    global save_img

    cv2.namedWindow('mark',cv2.WINDOW_NORMAL)
    #cv2.createTrackbar('mark_r','mark',5,50,circle_radius)
    cv2.setMouseCallback('mark',draw_mark)
    while(1):
        cv2.imshow('mark',im_mark)
        if cv2.waitKey(20) & 0xFF==27:
            break
    cv2.destroyAllWindows() 
    
    output_1=output[0:512,0:512]
    output_2=output[0:512,424:936]
    output_3=output[0:512,848:1360]
    output_4=output[512:1024,0:512]
    output_5=output[512:1024,424:936]
    output_6=output[512:1024,848:1360]
    
    save= Image.fromarray(output.astype('uint8')).convert('RGB')
    save1= Image.fromarray(output_1.astype('uint8')).convert('RGB')
    save2= Image.fromarray(output_2.astype('uint8')).convert('RGB')
    save3= Image.fromarray(output_3.astype('uint8')).convert('RGB')
    save4= Image.fromarray(output_4.astype('uint8')).convert('RGB')
    save5= Image.fromarray(output_5.astype('uint8')).convert('RGB')
    save6= Image.fromarray(output_6.astype('uint8')).convert('RGB')
    
    vision1 = Image.fromarray(output_1.astype('uint8')).convert('RGB')
    vision2 = Image.fromarray(output_2.astype('uint8')).convert('RGB')
    vision3 = Image.fromarray(output_3.astype('uint8')).convert('RGB')
    vision4 = Image.fromarray(output_4.astype('uint8')).convert('RGB')
    vision5 = Image.fromarray(output_5.astype('uint8')).convert('RGB')
    vision6 = Image.fromarray(output_6.astype('uint8')).convert('RGB')
    
    render1 = ImageTk.PhotoImage(vision1)
    render2 = ImageTk.PhotoImage(vision2)
    render3 = ImageTk.PhotoImage(vision3)
    render4 = ImageTk.PhotoImage(vision4)
    render5 = ImageTk.PhotoImage(vision5)
    render6 = ImageTk.PhotoImage(vision6)
    #save=transforms.Resize((load.size[1],load.size[0]))(save)

    
    global img2_1
    img2_1.destroy()  
    img2_1  = tkinter.Label(win,image=render1)
    img2_1.image = render1
    img2_1.place(x=850,y=100)
    
    global img2_2
    img2_2.destroy()  
    img2_2  = tkinter.Label(win,image=render2)
    img2_2.image = render2
    img2_2.place(x=850,y=200)
    
    global img2_3
    img2_3.destroy()  
    img2_3  = tkinter.Label(win,image=render3)
    img2_3.image = render3
    img2_3.place(x=850,y=300)
    
    global img2_4
    img2_4.destroy()  
    img2_4  = tkinter.Label(win,image=render4)
    img2_4.image = render4
    img2_4.place(x=850,y=400)
    
    global img2_5
    img2_5.destroy()  
    img2_5  = tkinter.Label(win,image=render5)
    img2_5.image = render5
    img2_5.place(x=850,y=500)
    
    global img2_6
    img2_6.destroy()  
    img2_6  = tkinter.Label(win,image=render6)
    img2_6.image = render6
    img2_6.place(x=850,y=600)
    
    global save_img
    save_img = save
    global save1_img
    save1_img = save1
    global save2_img
    save2_img = save2
    global save3_img
    save3_img = save3
    global save4_img
    save4_img = save4
    global save5_img
    save5_img = save5
    global save6_img
    save6_img = save6
    

         
def crop():
    '''global im_crop
    im_crop=np.copy(im)
    
    cv2.namedWindow('crop',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('crop',roi_crop)
    while(1):
        cv2.imshow('crop',im_crop)
        if cv2.waitKey(20) & 0xFF==27:
            break
    cv2.destroyAllWindows()'''
    global roi,roi1,roi2,roi3,roi4,roi5,roi6,im_mark,im_crop
    im_crop=np.copy(im)
    roi=im_crop[0:1024,0:1360]
    roi1=roi[0:512,0:512]
    roi2=roi[0:512,424:936]
    roi3=roi[0:512,848:1360]
    roi4=roi[512:1024,0:512]
    roi5=roi[512:1024,424:936]
    roi6=roi[512:1024,848:1360]
    
    im_mark=np.copy(roi)   
    
    save= Image.fromarray(roi.astype('uint8')).convert('RGB')
    save1= Image.fromarray(roi1.astype('uint8')).convert('RGB')
    save2= Image.fromarray(roi2.astype('uint8')).convert('RGB')
    save3= Image.fromarray(roi3.astype('uint8')).convert('RGB')
    save4= Image.fromarray(roi4.astype('uint8')).convert('RGB')
    save5= Image.fromarray(roi5.astype('uint8')).convert('RGB')
    save6= Image.fromarray(roi6.astype('uint8')).convert('RGB')
    
    vision1 = Image.fromarray(roi1.astype('uint8')).convert('RGB')
    vision2 = Image.fromarray(roi2.astype('uint8')).convert('RGB')
    vision3 = Image.fromarray(roi3.astype('uint8')).convert('RGB')
    vision4 = Image.fromarray(roi4.astype('uint8')).convert('RGB')
    vision5 = Image.fromarray(roi5.astype('uint8')).convert('RGB')
    vision6 = Image.fromarray(roi6.astype('uint8')).convert('RGB')
    
    render1 = ImageTk.PhotoImage(vision1)
    render2 = ImageTk.PhotoImage(vision2)
    render3 = ImageTk.PhotoImage(vision3)
    render4 = ImageTk.PhotoImage(vision4)
    render5 = ImageTk.PhotoImage(vision5)
    render6 = ImageTk.PhotoImage(vision6)

    
    global img2_1
    img2_1.destroy()  
    img2_1  = tkinter.Label(win,image=render1)
    img2_1.image = render1
    img2_1.place(x=850,y=100)
    
    global img2_2
    img2_2.destroy()  
    img2_2  = tkinter.Label(win,image=render2)
    img2_2.image = render2
    img2_2.place(x=850,y=200)
    
    global img2_3
    img2_3.destroy()  
    img2_3  = tkinter.Label(win,image=render3)
    img2_3.image = render3
    img2_3.place(x=850,y=300)

    global img2_4
    img2_4.destroy()  
    img2_4  = tkinter.Label(win,image=render4)
    img2_4.image = render4
    img2_4.place(x=850,y=400)
    
    global img2_5
    img2_5.destroy()  
    img2_5  = tkinter.Label(win,image=render5)
    img2_5.image = render5
    img2_5.place(x=850,y=500)
    
    global img2_6
    img2_6.destroy()  
    img2_6  = tkinter.Label(win,image=render6)
    img2_6.image = render6
    img2_6.place(x=850,y=600)
    
    global save_img
    save_img = save
    global save1_img
    save1_img = save1
    global save2_img
    save2_img = save2
    global save3_img
    save3_img = save3
    global save4_img
    save4_img = save4
    global save5_img
    save5_img = save5
    global save6_img
    save6_img = save6
    
    #im_crop=np.copy(im)

'''#def set_bright():
	
	def show_bright(ev=None):
		temp = original
		new_im = transforms.ColorJitter(brightness=scale.get())(temp)
		render = ImageTk.PhotoImage(new_im)
		global img2
		img2.destroy()
		img2  = tkinter.Label(win,image=render)
		img2.image = render
		img2.place(x=650,y=90)
		global save_img
		save_img = new_im
		
	top = tkinter.Tk()
	top.geometry('250x150+420+350')
	top.title('set brightness')
	scale = tkinter.Scale(top, from_=0, to=100, orient=tkinter.HORIZONTAL,command=show_bright)
	scale.set(1)
	scale.pack()
	

#def set_contrast():
	
	def show_contrast(ev=None):
		temp = original
		new_im = transforms.ColorJitter(contrast=scale.get())(temp)
		render = ImageTk.PhotoImage(new_im)
		global img2
		img2.destroy()
		img2  = tkinter.Label(win,image=render)
		img2.image = render
		img2.place(x=650,y=90)
		global save_img
		save_img = new_im
		
	top = tkinter.Tk()
	top.geometry('250x150+420+350')
	top.title('set contrast')
	scale = tkinter.Scale(top, from_=0, to=100, orient=tkinter.HORIZONTAL,command=show_contrast)
	scale.set(1)
	scale.pack()
'''
	

def save():
    global pointsCount,lsPointsChoose, tpPointsChoose,lsPointsChoose_display,tpPointsChoose_display

    if pointsCount!=0:
        save_img.save('mask\mask_'+str(int(n))+'.png')
        save1_img.save('train_mask\mask_'+str(int(6*n+0))+'.png')
        save2_img.save('train_mask\mask_'+str(int(6*n+1))+'.png')
        save3_img.save('train_mask\mask_'+str(int(6*n+2))+'.png')
        save4_img.save('train_mask\mask_'+str(int(6*n+3))+'.png')
        save5_img.save('train_mask\mask_'+str(int(6*n+4))+'.png')
        save6_img.save('train_mask\mask_'+str(int(6*n+5))+'.png')
        save_successfully()
        print('image_'+str(int(n/2)),'totally marked',pointsCount, 'points:',[i for i in tpPointsChoose])
        records='image'+str(int(n/2)),[i for i in tpPointsChoose]
        f=open('points.txt','a')
        f.write(str(records)+'\n')
        
        pointsCount = 0
        tpPointsChoose = []
        lsPointsChoose = []
        tpPointsChoose_display = []
        lsPointsChoose_display = []
        
    else:
        save_img.save('crop\crop_'+str(int((n)))+'.jpg')
        save1_img.save('train_crop\crop_'+str(int(6*n+0))+'.jpg')
        save2_img.save('train_crop\crop_'+str(int(6*n+1))+'.jpg')
        save3_img.save('train_crop\crop_'+str(int(6*n+2))+'.jpg')
        save4_img.save('train_crop\crop_'+str(int(6*n+3))+'.jpg')
        save5_img.save('train_crop\crop_'+str(int(6*n+4))+'.jpg')
        save6_img.save('train_crop\crop_'+str(int(6*n+5))+'.jpg')
        
        save_successfully()

    
def save_successfully():
      winNew = tkinter.Toplevel(win)
      winNew.geometry('320x120+420+350')
      winNew.title('Save Image')
      lb2 = tkinter.Label(winNew,text='Save Successfully',font=('Arial',15))
      lb2.place(relx=0.25,rely=0.2)
      btClose=tkinter.Button(winNew,text='OK',command=winNew.destroy)
      btClose.place(relx=0.45,rely=0.5)
	

e = tkinter.StringVar()
e_entry = tkinter.Entry(win,width=68, textvariable=e)
e_entry.pack()

button1 = tkinter.Button(win, text ="Select", font=('Arial',10),command = choose_file)
button1.place(x=300,y=0)

#label
label1 = tkinter.Label(win,text="Original Picture",font=('Arial',10))
label1.place(x=300,y=50)

label2 = tkinter.Label(win,text="Modified Picture",font=('Arial',10))
label2.place(x=920,y=50)



button2 = tkinter.Button(win,text="Mark",font=('Arial',10),command=mark)
button2.place(relx=0.58,rely=0.4)


button3 = tkinter.Button(win,text="Save",font=('Arial',10),command=save)
button3.place(relx=0.58,rely=0.75)

#bightness
#button4 = tkinter.Button(win,text="Brightness",command=set_bright)
#button4.place(relx=0.47,rely=0.5)


#button5 = tkinter.Button(win,text="Contrast",command=set_contrast)
#button5.place(relx=0.475,rely=0.65)

#crop
button6 = tkinter.Button(win,text="Crop",command=crop)
button6.place(relx=0.58,rely=0.25)

win.mainloop()