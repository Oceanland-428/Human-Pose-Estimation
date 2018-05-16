
# coding: utf-8

# In[1]:


def drawLines(image,original_labels,predict_labels,lineThickness = 2,color_origin =(0,255,0),color_pred =(0,0,255)):
    '''
    image is a (227*227*3) image
    original_labels is a 2*14 original label
    predict_labels is a 2*14 predicted label
    '''
    import matplotlib.pyplot as plt
    import cv2 as cv
    original = original_labels
    predict = predict_labels
    # original label
    # right-left ankle
    a = cv.line(image,(int(original[0,0]),int(original[1,0])),(int(original[0,5]),int(original[1,5])),color_origin,lineThickness)
    # right-left knee
    a = cv.line(image,(int(original[0,1]),int(original[1,1])),(int(original[0,4]),int(original[1,4])),color_origin,lineThickness)
    # right-left hip
    a = cv.line(image,(int(original[0,2]),int(original[1,2])),(int(original[0,3]),int(original[1,3])),color_origin,lineThickness)
    # right-left wrist
    a = cv.line(image,(int(original[0,6]),int(original[1,6])),(int(original[0,11]),int(original[1,11])),color_origin,lineThickness)
    # right-left elbow
    a = cv.line(image,(int(original[0,7]),int(original[1,7])),(int(original[0,10]),int(original[1,10])),color_origin,lineThickness)
    # right-left shoulder
    a = cv.line(image,(int(original[0,8]),int(original[1,8])),(int(original[0,9]),int(original[1,9])),color_origin,lineThickness)
    # neck
    a = cv.circle(image,(int(original[0,12]),int(original[1,12])),2,color_origin,lineThickness)
    # head top
    a = cv.circle(image,(int(original[0,13]),int(original[1,13])),2,color_origin,lineThickness)
    
    # predicted label
    # right-left ankle
    a = cv.line(image,(int(predict[0,0]),int(predict[1,0])),(int(predict[0,5]),int(predict[1,5])),color_pred,lineThickness)
    # right-left knee
    a = cv.line(image,(int(predict[0,1]),int(predict[1,1])),(int(predict[0,4]),int(predict[1,4])),color_pred,lineThickness)
    # right-left hip
    a = cv.line(image,(int(predict[0,2]),int(predict[1,2])),(int(predict[0,3]),int(predict[1,3])),color_pred,lineThickness)
    # right-left wrist
    a = cv.line(image,(int(predict[0,6]),int(predict[1,6])),(int(predict[0,11]),int(predict[1,11])),color_pred,lineThickness)
    # right-left elbow
    a = cv.line(image,(int(predict[0,7]),int(predict[1,7])),(int(predict[0,10]),int(predict[1,10])),color_pred,lineThickness)
    # right-left shoulder
    a = cv.line(image,(int(predict[0,8]),int(predict[1,8])),(int(predict[0,9]),int(predict[1,9])),color_pred,lineThickness)
    # neck
    a = cv.circle(image,(int(predict[0,12]),int(predict[1,12])),2,color_pred,lineThickness)
    # head top
    a = cv.circle(image,(int(predict[0,13]),int(predict[1,13])),2,color_pred,lineThickness)
    
    plt.imshow(image)


# In[ ]:


# usage:
# drawLines(test_list[0].copy(),test_label[0].copy(),pred[0].copy())

