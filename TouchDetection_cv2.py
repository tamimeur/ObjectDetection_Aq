import cv2.cv as cv
import cv2
from datetime import datetime
import time
import numpy as np
import math
import itertools
import sys


top = 0
bottom = 1
left = 0
right = 1

step_type = ""
take_time_start = 0
take_time_end = 0
object_name = ""

#input_step = "2"
input_step = sys.argv[1]
input_file = "/Users/Leli/Documents/opencv/OpenCV-Python/Other_Examples/vob/step" + input_step

class MotionDetectorAdaptative():
    
    def __init__(self,threshold=40, doRecord=False, showWindows=True):
        self.writer = None
        self.font = None
        self.doRecord=False #Either or not record the moving object
        self.show = showWindows #Either or not show the 2 windows
        self.frame = None
	self.start = True   
	 
	self.capture=cv2.VideoCapture(input_file + ".mp4")
	self.video=cv2.VideoWriter("/Users/Leli/Documents/opencv/OpenCV-Python/Other_Examples/video.mp4", -1, 25, (640,480))
	
        _,self.frame = self.capture.read()
	
	
	self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)        

	self.average_frame = np.float32(self.frame)	

	self.absdiff_frame = None
        self.previous_frame = None
        self.surface = (np.size(self.frame, 1)) * (np.size(self.frame, 0))
	
	self.currentsurface = 0
        self.currentcontours = None
	self.currentgloves = None
        self.threshold = threshold
        self.isRecording = False
        self.trigger_time = 0 #Hold timestamp of the last detection
        
       
    def initRecorder(self): #Create the recorder
        codec = cv.CV_FOURCC('M', 'J', 'P', 'G')
        self.writer=cv.CreateVideoWriter(datetime.now().strftime("%b-%d_%H:%M:%S")+".wmv", codec, 5, cv.GetSize(self.frame), 1)
        #FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) #Creates a font

    def run(self):
        started = time.time()
	bounding_box_list = []
	matched_list = []
	big_box = ()
        f = open(input_file + ".txt")
        step_type = f.readlines()
        out = step_type[1].split("Time: ")
        out = out[1].split(",")
        take_time_start = int(out[0])
        out = out[1].split("\n")
        take_time_end = int(out[0])
        print step_type
        print take_time_start
        print take_time_end 
        f.close()

        object_found = False

	while True:
            
            _,currentframe = self.capture.read()
	    
	    instant = time.time() #Get timestamp o the frame
            
            self.processImage(currentframe) #Process the image
            
	    box_areas = []
	    test_this = True

	    if test_this:
		#TODO: call track gloves function and return the bounding box of each glove

		glove_circle = self.findGloves(currentframe)
		cv2.circle(currentframe,glove_circle,75,255,3)

		#hold up ill uncomment this in a bit
		#cv.DrawContours (currentframe, self.currentgloves, (0, 0, 255), (0, 255, 0), 1, 2, cv.CV_FILLED)
		

		#if((instant-started > 12) and (instant-started < 14)):
		if((instant-started > take_time_start) and (instant-started < take_time_end)):
		    bounding_box_list = self.somethingHasMoved()
		    
		if((instant-started > take_time_end)):
		    object_found = True

		#print bounding_box_list
                if (bounding_box_list):
		    #print bounding_box_list
                    self.trigger_time = instant #Update the trigger_time
                    if instant > started +10:#Wait 5 second after the webcam start for luminosity adjusting etc..
                        #print "Something is moving !"
                        if self.doRecord: #set isRecording=True only if we record a video
                            self.isRecording = True
                    ###################
		    for box in bounding_box_list:
			box_width = box[right][0] - box[left][0]
			box_height = box[bottom][0] - box[top][0]
			box_areas.append( box_width* box_height )

		    average_box_area = 0.0
		    if len(box_areas): average_box_area = float( sum(box_areas) ) / len(box_areas)

		    #print "Glove circle = "
		    #print glove_circle
		    #print "Average box area = "
		    #print average_box_area

		    trimmed_box_list = []
		    for box in bounding_box_list:
			box_width = box[right][0] - box[left][0]
			box_height = box[bottom][0] - box[top][0]
			
			x_left_to_circle = math.fabs(box[left][0] - glove_circle[0])
			y_left_to_circle = math.fabs(box[left][1] - glove_circle[1])
			#print "distance to circle="
			#print x_left_to_circle
			#print y_left_to_circle

			#TODO: check that the glove bbox and the current object bbox's interect and only then
			#add the current object's bbox to trimmed_box_list if its area is appropriate
			#if( (box_width * box_height) > 5000 and (box_width * box_height) < 10000 ): 
			if(((box_width * box_height) > 300) and (x_left_to_circle < 25) and (y_left_to_circle < 25) ): 

			    trimmed_box_list.append( box )
			#if( (box_width * box_height) > 1000 ): trimmed_box_list.append( box )
		    
		    #bounding_box_list = merge_collided_bboxes( trimmed_box_list )
		    bounding_box_list = trimmed_box_list
		    print "BOX LIST="
		    print bounding_box_list
		    for box in bounding_box_list:
			cv2.rectangle( currentframe, box[0], box[1], (0,255,0), 1 )
	  	    ###################

		if (object_found == True):
		    matched_list = self.match(currentframe)
		    #print  "MATCHED FEATURES:"
		    #print matched_list
		    text_color = (0,0,255)
		    if(int(input_step) == 2):
			cv2.putText(currentframe, "1_L_Bottle", (45, 20), cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=1, lineType=cv2.CV_AA)
		    elif(int(input_step)==3):
			cv2.putText(currentframe, "800_mL_LB_sterile", (45, 20), cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=1, lineType=cv2.CV_AA)
		    elif(int(input_step)==4):
			cv2.putText(currentframe, "Petri_Dish", (45, 20), cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=1, lineType=cv2.CV_AA)
		    elif(int(input_step)==5):
			cv2.putText(currentframe, "Falcon_Tube", (45, 20), cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=1, lineType=cv2.CV_AA)
		    for match in matched_list:
			cv2.circle(currentframe,match,3,255,3)	

 		#cv.DrawContours (currentframe, self.currentcontours, (0, 0, 255), (0, 255, 0), 1, 2, cv.CV_FILLED)
            #else:
            #    if instant >= self.trigger_time +10: #Record during 10 seconds
            #        #print "Stop recording"
            #        self.isRecording = False
            #    else:
            #        cv.PutText(currentframe,datetime.now().strftime("%b %d, %H:%M:%S"), (25,30),self.font, 0) #Put date on the frame
            #        cv.WriteFrame(self.writer, currentframe) #Write the frame
           
 
            if self.show:
                #cv.ShowImage("Image", currentframe)
		text_color = (255, 0, 0)  # color as (B,G,R)
		print "Time = "
		print (instant-started)
		#cv2.putText(currentframe, (instant-started), (45, 20), cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=1, lineType=cv2.CV_AA)
		cv2.imshow('Image', currentframe)                
		self.video.write(currentframe)
            #c=cv.WaitKey(1) % 0x100
            #if c==27 or c == 10: #Break if user enters 'Esc'.
            #    break            
	
	    if cv2.waitKey(33)== 27:
		break
   
    #def processImage2(self,curframe):
	#    curframe = cv2.blur(curframe,(3,3))
	     

    def processImage(self, curframe):
            #cv.Smooth(curframe, curframe) #Remove false positives
	    curframe = cv2.blur(curframe,(3,3))            

            #if not self.absdiff_frame: #For the first time put values in difference, temp and moving_average
            if self.start: #For the first time put values in difference, temp and moving_average
                self.start = False
		
		#self.absdiff_frame = cv.CloneImage(curframe)
		self.absdiff_frame = curframe.copy()

                #self.previous_frame = cv.CloneImage(curframe)
		self.previous_frame = curframe.copy()

                #cv.Convert(curframe, self.average_frame) #Should convert because after runningavg take 32F pictures
		#self.average_frame = cv2.convertScaleAbs(curframe)

            else:
                #cv.RunningAvg(curframe, self.average_frame, 0.05) #Compute the average
                cv2.accumulateWeighted(self.frame, self.average_frame, 0.05) #Compute the average

            #cv.Convert(self.average_frame, self.previous_frame) #Convert back to 8U frame
	    self.previous_frame = cv2.convertScaleAbs(self.average_frame)
            
            #cv.AbsDiff(curframe, self.previous_frame, self.absdiff_frame) # moving_average - curframe
	    self.absdiff_frame = cv2.absdiff(curframe, self.previous_frame)            
	
	    cv2.imshow('absdiff',self.absdiff_frame)

            #cv.CvtColor(self.absdiff_frame, self.gray_frame, cv.CV_RGB2GRAY) #Convert to gray otherwise can't do threshold
            self.gray_frame = cv2.cvtColor(self.absdiff_frame, cv2.COLOR_BGR2GRAY)

	    #cv.Threshold(self.gray_frame, self.gray_frame, 50, 255, cv.CV_THRESH_BINARY)
	    ret,self.gray_frame = cv2.threshold(self.gray_frame, 50, 255, cv2.THRESH_BINARY)
#HERE
            #cv.Dilate(self.gray_frame, self.gray_frame, None, 15) #to get object blobs
	    #self.gray_frame = cv2.dilate(self.gray_frame,None,15) )
	    dilation_size = 15
	    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation_size,dilation_size))
	    dilated = cv2.dilate(self.gray_frame,kernel)

            #cv.Erode(self.gray_frame, self.gray_frame, None, 10)
	    erosion_size = 10
	    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erosion_size,erosion_size))
    	    eroded = cv2.erode(self.gray_frame,kernel)
           
    def findGloves(self, curframe):	
        #_,self.frame = self.capture.read()
	#self.frame = cv2.blur(self.frame,(3,3))
	#self.hsv_frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2HSV)    	
	
	curframe = cv2.blur(curframe,(3,3))
	self.hsv_frame = cv2.cvtColor(curframe,cv2.COLOR_BGR2HSV)    	
	thresh = cv2.inRange(self.hsv_frame,np.array((100,80,80)), np.array((204,255,196)))
	thresh2 = thresh.copy()

	
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	
	self.currentgloves = contours
	
	max_area = 0
	best_cnt = contours[0]
	for cnt in contours:
	    area = cv2.contourArea(cnt)
	    if area > max_area:
		max_area = area
		best_cnt = cnt
	
	# finding centroids of best_cnt and draw a circle there
	M = cv2.moments(best_cnt)
	glove_coord = int(M['m10']/M['m00'])+10, int(M['m01']/M['m00'])+10

	##draw contour?
	
	cv2.imshow('gloves', thresh2)

	return glove_coord

    def findKeyPoints(self, img, template, distance=200):
	detector = cv2.FeatureDetector_create("SIFT")
	descriptor = cv2.DescriptorExtractor_create("SIFT")

	skp = detector.detect(img)
	skp, sd = descriptor.compute(img, skp)
 
 	tkp = detector.detect(template)
 	tkp, td = descriptor.compute(template, tkp)
 
 	flann_params = dict(algorithm=1, trees=4)
 	flann = cv2.flann_Index(sd, flann_params)
 	idx, dist = flann.knnSearch(td, 1, params={})
 	del flann
 
 	dist = dist[:,0]/2500.0
 	dist = dist.reshape(-1,).tolist()
 	idx = idx.reshape(-1).tolist()
 	indices = range(len(dist))
 	indices.sort(key=lambda i: dist[i])
 	dist = [dist[i] for i in indices]
 	idx = [idx[i] for i in indices]
 	skp_final = []
 	for i, dis in itertools.izip(idx, dist):
	    if dis < distance:
 	        skp_final.append(skp[i])
 
 	flann = cv2.flann_Index(td, flann_params)
 	idx, dist = flann.knnSearch(sd, 1, params={})
 	del flann
	dist = dist[:,0]/2500.0
 	dist = dist.reshape(-1,).tolist()
 	idx = idx.reshape(-1).tolist()
 	indices = range(len(dist))
 	indices.sort(key=lambda i: dist[i])
 	dist = [dist[i] for i in indices]
 	idx = [idx[i] for i in indices]
 	tkp_final = []
 	for i, dis in itertools.izip(idx, dist):
 	    if dis < distance:
 	        tkp_final.append(tkp[i])
 
 	return skp_final, tkp_final
 
    def drawKeyPoints(self, img, template, skp, tkp, num=-1):
 	match_points = []
	h1, w1 = img.shape[:2]
 	h2, w2 = template.shape[:2]
 	nWidth = w1+w2
 	nHeight = max(h1, h2)
 	hdif = (h1-h2)/2
 	#newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
 	#newimg[hdif:hdif+h2, :w2] = template
 	#newimg[:h1, w2:w1+w2] = img
 
 	maxlen = min(len(skp), len(tkp))
 	if num < 0 or num > maxlen:
 	    num = maxlen
 	for i in range(num):
 	    pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]+hdif))
 	    #pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]))
 	    pt_b = (int(skp[i].pt[0]), int(skp[i].pt[1]))
	    match_points.append( (pt_b) )
 	    #cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
	return match_points
 	#return newimg


    def match(self, curframe):
	points = []
	temp = cv2.imread('/Users/Leli/Documents/opencv/OpenCV-Python/Other_Examples/vob/1_L_Bottle.png')
	
	if(int(input_step) == 2):
	    temp = cv2.imread('/Users/Leli/Documents/opencv/OpenCV-Python/Other_Examples/vob/1_L_Bottle.png')
	elif(int(input_step) == 3):
	    temp = cv2.imread('/Users/Leli/Documents/opencv/OpenCV-Python/Other_Examples/vob/800_mL_LB_sterile.png')
	elif(int(input_step) == 4):
	    temp = cv2.imread('/Users/Leli/Documents/opencv/OpenCV-Python/Other_Examples/vob/Petri_Dish2.png')
	elif(int(input_step) == 5):
	    temp = cv2.imread('/Users/Leli/Documents/opencv/OpenCV-Python/Other_Examples/vob/Falcon_Tube.png')

	cv2.imshow("objfound", temp)

	img = curframe
	dist = 200
	num = -1
	skp,tkp = self.findKeyPoints(img, temp, dist)
	points = self.drawKeyPoints(img, temp, skp, tkp, num)
	#cv2.imshow("objfound", newimg)
 	return points

    def somethingHasMoved(self):
        
	bounding_box_list = [] 
        # Find contours
        #storage = cv.CreateMemStorage(0)
        #contours = cv.FindContours(self.gray_frame, storage, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(self.gray_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        self.currentcontours = contours #Save contours

	for cnt in contours:
            self.currentsurface += cv2.contourArea(cnt)
        

        #while contours: #For all contours compute the area

	    ####################
	    #bounding_rect = cv2.boundingRect( list(contours) )
	    bounding_rect = cv2.boundingRect(cnt)
            point1 = ( bounding_rect[0], bounding_rect[1] )
            point2 = ( bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3] )

            bounding_box_list.append( ( point1, point2 ) )
	    #print bounding_box_list
            #polygon_points = cv.ApproxPoly( list(contours), storage, cv.CV_POLY_APPROX_DP )
                      # Draw the contours:
            #cv.FillPoly( self.gray_frame, [ list(polygon_points), ], cv.CV_RGB(255,255,255), 0, 0 )
            #cv.PolyLine( display_image, [ polygon_points, ], 0, cv.CV_RGB(255,255,255), 1, 0, 0 )
	    ###################

            #contours = contours.h_next()
        
        avg = (self.currentsurface*100)/self.surface #Calculate the average of contours area on the total size
        self.currentsurface = 0 #Put back the current surface to 0
        
        #if avg > self.threshold:
	    #return "Average is large enough"
        return bounding_box_list
        #else:
        #    return False

        
if __name__=="__main__":
    detect = MotionDetectorAdaptative(doRecord=False)
    detect.run()
