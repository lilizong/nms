# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 11:48:20 2022

@author: 李立宗
计算机视觉40例，电子工业出版社，第24章 深度学习应用实践
"""

import cv2
import numpy as np

image = cv2.imread("image/x.jpg")
            
boxes=[[508, 26, 288, 300], [11, -47, 680, 531], [50, -60, 667, 551], [509, 42, 288, 301], [9, -6, 681, 532], [49, -25, 668, 573], [5, 19, 673, 540], [44, 10, 687, 558], [21, 79, 638, 500], [59, 59, 672, 547], [433, 372, 324, 233]]

resultIDS=[16, 1, 1, 16, 1, 1, 1, 1, 1, 1, 17]
confidences=[0.5467395782470703, 0.8727190494537354, 0.7452741861343384, 0.6148457527160645, 0.9958364367485046, 0.978717565536499, 0.995242178440094, 0.9866145253181458, 0.9464831948280334, 0.7797505855560303, 0.9638967514038086]
# indexes=


# =============================================================================
# 显示函数            
# =============================================================================
def display(boxes,resultIDS,image,frame_name,indexes):
    # ===========绘制边框===========
    # 给每个分类随机分配一个颜色
    classesCOLORS = np.random.randint(0, 255, size=(80, 3), dtype="uint8") 
    # 绘制边框及置信度、分类
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = [int(c) for c in classesCOLORS[resultIDS[i]]]  #边框颜色
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            result = "{:.0f}%".format(confidences[i]*100)
            cv2.putText(image, result, (x, y+35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  
    cv2.imshow(frame_name, image)
    
            
# =============================================================================
# 不进行非极大值抑制（NMS）处理结果 
# =============================================================================
            
display(boxes,resultIDS,image.copy(),"noNMS",range(len(boxes)))

# =============================================================================
# 进行非极大值抑制（NMS）处理结果 
# =============================================================================            
# 非极大值抑制，将众多重合的边框保留一个最关键的（去重处理）
indexes = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.4)
# indexes，所有可能边框的序号集合（需要注意，indexes表示的是boxes内的序号）
display(boxes,resultIDS,image.copy(),"NMS",indexes)

cv2.waitKey(0)
cv2.destroyAllWindows()