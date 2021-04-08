from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import pytesseract


net = cv2.dnn.readNet("frozen_east_text_detection.pb")

pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def preprocess_small_image(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    cv2_img = cv2.threshold(cv2_img, 200, 255, 
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2_img = cv2.resize(cv2_img, (300, 100), 
                         interpolation = cv2.INTER_CUBIC)

    mask = np.zeros(cv2_img.shape[:2], np.uint8)
    mask[0:10, 0:300] = 255
    cv2_hist = cv2.calcHist([cv2_img], [0], mask, [256], [0, 256])
    black_hist = cv2_hist[0]+cv2_hist[1]+cv2_hist[2]
    white_hist = cv2_hist[253]+cv2_hist[254]+ cv2_hist[255]
    if (black_hist > white_hist):
        cv2_img=255-cv2_img

    cv2_img = cv2.GaussianBlur(cv2_img, (5, 5), 0)
    
    return cv2_img


# function notation
def text_detector(img_path):
    image = cv2.imread(img_path)
    # image = cv2.resize(image, (640,320), interpolation = cv2.INTER_AREA)

    orig = image
    (H, W) = orig.shape[:2]

    (newW, newH) = (320, 320)
    image = cv2.resize(image, (newW, newH), interpolation = cv2.INTER_AREA)
    
    rW = W / float(newW)
    rH = H / float(newH)
    
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    words = []
    
    for (startX, startY, endX, endY) in boxes:

        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        boundary = 2

        text = orig[startY-boundary:endY+boundary, 
                    startX - boundary:endX + boundary]
        
        text = preprocess_small_image(text)
        textRecongized = pytesseract.image_to_string(text, 
                                                     lang='eng',
                                                     config='--psm 7')
        words.append(textRecongized.replace('\x0c', ''))
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # orig = cv2.putText(orig, textRecongized, (endX,endY+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    return words[::-1], orig


