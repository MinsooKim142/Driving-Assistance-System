import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


#Converts the image into a gray scale image
def grayImage(image):

    #Converts image to gray scale image.
    convtGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #Applies a GaussianBlur to a gray scale image which reduces image noise.
    imgBlur = cv2.GaussianBlur(convtGray, (5 , 5), 0)


    #Finds the edges of the gray image
    # if it is below low_threshold rejected, if higher than higher_threshold
    # as accepted as edge pixel, If between the two thresholds only accepted if
    # connected to a strong edge.
    cannyImage = cv2.Canny(imgBlur, 50, 150)

    return cannyImage

#Shows the region of interest and masks the image except for the region of interest.
def maskRegionInterest(image):

    #Creates a polygon which is the region of interest
    height = image.shape[0]
    regionInterest = np.array([[(100,height), (1300, height), (650, 400)]])

    #Creates an array filled with zeros.
    maskImage = np.zeros_like(image)

    #Fills the maskImage completly white
    cv2.fillPoly(maskImage, regionInterest, 255)

    onlyShowRegion = cv2.bitwise_and(image, maskImage)

    return onlyShowRegion

#Displays the lines
def showLines(image, line):

    #Declares the arrays as zeros. Clears the image.
    imageLine = np.zeros_like(image)

    #Checks if it detected any lines
    if line is not None:

        #Loops through the lines
        for eachLine in line:

            #Unpacks the elements into 4 different variables
            x1, y1, x2, y2 = eachLine.reshape(4)


            #draws a line segment that connects two points on the
            #imageLine
            cv2.line(imageLine, (x1, y1), (x2, y2), (5, 200,50), 10)

    return imageLine

#Detects pedestrians
cvhog = cv2.HOGDescriptor()
cvhog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def pedestrianDetect(humanFrame):

    #Declares the array of zeros which clears the image.
    imageOfPeople = np.zeros_like(humanFrame)

    rects, weights = cvhog.detectMultiScale(humanFrame, winStride=(8, 8), padding=(16, 16), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    c = 1
    for x, y, w, h in pick:
        cv2.rectangle(imageOfPeople, (x, y), (w, h), (139, 34, 104), 2)

        c += 1

    return imageOfPeople


#Captures the video
capVid = cv2.VideoCapture('obstacleTest6.mp4')

while(capVid.isOpened()):
    _, pic = capVid.read()

    #Creates an image that is converted to the gray scale.
    gray = grayImage(pic)

    # Shows only the region of interest and masks the image except
    # the region of interest.
    maskedImage = maskRegionInterest(gray)

    #Detects the lines
    lines = cv2.HoughLinesP(maskedImage, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    #Displays the lines
    imageLines = showLines(pic, lines)

    #Detects pedestrians
    cutImage = maskRegionInterest(pic)
    humanImage = pedestrianDetect(cutImage)

    # Blends the lane lines to the original image
    # Gives pic a weight of 0.8 and gives imageLines a weight of 1
    # which makes the lines more weight which makes the lanes more defined
    combineImage = cv2.addWeighted(pic, 0.8, imageLines, 1, 1)

    #Adds the detected pedestrians and combines it with combineImage
    totalImage = cv2.addWeighted(combineImage, 1, humanImage, 0.8, 1)

    cv2.imshow("output", totalImage)

    cv2.waitKey(1)







