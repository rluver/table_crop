# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:28:02 2021

@author: MJH
#refer: https://towardsdatascience.com/a-table-detection-cell-recognition-and-text-extraction-algorithm-to-convert-tables-to-excel-files-902edcf289ec
"""
import pandas as pd
import numpy as np 
import cv2
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
from pdf2image import convert_from_path




def PILtoarray(image):
    return np.array(image.getdata(), np.uint).reshape(image.size[1], image.size[0], 3)


def convert_pdf_to_image(file):   
    return convert_from_path(file, poppler_path = r"C:\Program Files (x86)\poppler-21.03.0\Library\bin", dpi = 600)
  



class TableCrop:
    
    def __init__(self, save_path):
        self.save_path = save_path
        self.table_houghlinep = ''
        self.table_kernel = ''
        self.image = ''
                
        
    def get_table_by_houghlinep(self, image_path):
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        plt.imshow(image)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength = 300, maxLineGap = 200)
            
        for line in lines:
            x1, y1, x2, y2 = line[0]        
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            plt.imshow(image)
            
        self.table_houghlinep = image
    
        
    def get_table_by_kernel(self, image_path):
            
        image = cv2.imread(image_path, 0)
        self.image = image
        
        # threshold the image
        threshold, image_binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        image_binary = 255 - image_binary
        
        plt.imshow(image_binary, cmap = 'gray')
        plt.show()
            
        # define a kernel length
        kernel_length = np.array(image).shape[1] // 50
        # vertical kernel of (1 x kernel_length), whichwill detect all the vertical lines from the image
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        # horizontal kernel of (kernel_length x 1), which will help to detect all the horizontal line from the image.
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        # kernel of (3 x 3) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    
        # morphologial operation to detect vertical lines from an image
        img_temp1 = cv2.erode(image_binary, vertical_kernel, iterations = 3)
        vertical_line_image = cv2.dilate(img_temp1, vertical_kernel, iterations = 3)
        # plt.imshow(vertical_line_image, cmap = 'gray')
        # cv2.imwrite('c:/etc/test_img/vertial_lines.jpg', vertical_line_image)
        
        img_temp2 = cv2.erode(image_binary, horizontal_kernel, iterations = 3)
        horizontal_line_image = cv2.dilate(img_temp2, horizontal_kernel, iterations = 3)
        # plt.imshow(horizontal_line_image, cmap = 'gray')
        # cv2.imwrite('c:/etc/test_img/horizontal_lines.jpg', horizontal_line_image)
        
        alpha = 0.5
        beta = 1.0 - alpha    
        
        img_final_bin = cv2.addWeighted(vertical_line_image, alpha, horizontal_line_image, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations = 2)
        threshold, img_final_bin = cv2.threshold(img_final_bin, 122, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        plt.imshow(img_final_bin, cmap = 'gray')
        # cv2.imwrite('d:/img_final_bin2.jpg', img_final_bin)
        
        self.table_kernel = img_final_bin
        
        
    def save_table_by_houghlinep(self, save_name):
        cv2.imwrite(''.join([self.save_path, save_name]), self.table_houghlinep)
        
        
    def save_table_by_kernel(self, save_name):
        cv2.imwrite(''.join([self.save_path, save_name]), self.table_kernel)
    
    
    def save_table_cell_by_kernel(self):
        contours, img2 = cv2.findContours(self.table_kernel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        idx = 0
        for c in contours[::-1]:
            x, y, w, h = cv2.boundingRect(c)
            if (w > 80 and h > 20) and (w > 3 * h):
                idx += 1
                new_img = self.image[y: y + h, x: x + w]
                cv2.imwrite(''.join([self.save_path, str(idx), '.jpg']), new_img)
                
                
    def sort_contours(self, cnts, method = 'left-to-right'):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)                
                
                
    def save_table_by_kernel_to_csv(self, save_name):
        
        contours, hierarchy = cv2.findContours(self.table_kernel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, boundingBoxes = self.sort_contours(contours, method = 'top-to-bottom')
                
        # Creating a list of heights for all detected boxes
        heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
        #Get mean of heights
        mean = np.mean(heights)
        
        
        # Create list box to store all boxes in
        box = []
        
        # Get position (x,y), width and height for every contour and show the contour on image
        for c in contours:            
            x, y, w, h = cv2.boundingRect(c)

            if (w < 1000 and h > 500):
                image = cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                box.append([x, y, w, h])
          
            
        # Creating two lists to define row and column in which cell is located
        row = []
        column = []
        j = 0
        
        for i in range(len(box)):
            if ( i == 0 ):
                column.append(box[i])
                previous=box[i]
            else:
                if ( box[i][1] <= previous[1] + mean / 2 ):
                    column.append(box[i])
                    previous = box[i]
                    if (i == len(box) - 1):
                        row.append(column)
                else:
                    row.append(column)
                    column = []
                    previous = box[i]
                    column.append(box[i])
                    
        
        #calculating maximum number of cells
        countcol = 0
        for i in range(len(row)):
            countcol = len(row[i])
            if countcol > countcol:
                countcol = countcol
                
                
        # Retrieving the center of each column
        center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
        center=np.array(center)
        center.sort()
        
        
        # Regarding the distance to the columns center, the boxes are arranged in respective order
        finalboxes = []
        for i in range(len(row)):
            lis=[]
            for k in range(countcol):
                lis.append([])
            for j in range(len(row[i])):
                diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
                minimum = min(diff)
                indexing = list(diff).index(minimum)
                lis[indexing].append(row[i][j])
            finalboxes.append(lis)
            
            
        # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
        outer=[]
        for i in range(len(finalboxes)):
            for j in range(len(finalboxes[i])):
                inner = ''
                if (len(finalboxes[i][j]) == 0 ):
                    outer.append(' ')
                else:
                    for k in range(len(finalboxes[i][j])):
                        y,x,w,h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                        finalimg = self.image[x:x+h, y:y+w]
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                        border = cv2.copyMakeBorder(finalimg,2,2,2,2,   cv2.BORDER_CONSTANT,value=[255,255])
                        resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        dilation = cv2.dilate(resizing, kernel,iterations=1)
                        erosion = cv2.erode(dilation, kernel,iterations=1)
        
                        
                        out = pytesseract.image_to_string(erosion)
                        if(len(out)==0):
                            out = pytesseract.image_to_string(erosion, config = '--psm 3')
                        inner = inner + ' ' + out
                    outer.append(inner)
                    
                    
        # Creating a dataframe of the generated OCR list
        arr = np.array(outer)
        dataframe = pd.DataFrame(arr.reshape(len(row),countcol))
        data = dataframe.style.set_properties(align = 'left')
        # Converting it in a excel-file        
        data.to_excel(''.join([self.save_path, save_name]))
                                        
                                
        


if __name__ == '__main__':
    
    financial_statement_image_path = 'financial_statement.jpg'
    report_image_path = 'report.jpg'
    
    table_crop = TableCrop('save/')
    
    # financial statement
    # kernel
    table_crop.get_table_by_kernel(financial_statement_image_path)
    table_crop.save_table_by_kernel('financial_statement_kernel.jpg')
    table_crop.save_table_by_kernel_to_csv('financial_statement_kernel_excel.xlsx')
    
    # houghlinep
    table_crop.get_table_by_houghlinep(financial_statement_image_path)
    table_crop.save_table_by_houghlinep('financial_statement_houghlinep.jpg')
    
    
    # analyst_report
    table_crop.get_table_by_kernel(report_image_path)
    table_crop.save_table_by_kernel('report_kernel.jpg')
    # houghlinep
    table_crop.get_table_by_houghlinep(report_image_path)
    table_crop.save_table_by_houghlinep('report_houghlinep.jpg')