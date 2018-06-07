
# coding: utf-8

# In[1]:


import numpy as np
from drawLines_v2 import drawLines
from parse_flic_data import *


# In[2]:


train_list1, train_label1, val_list1, val_label1, test_list, test_label = getFLICData()
print("train X: ", train_list1.shape, "val X: ", val_list1.shape, "test X: ", test_list.shape)
print("train Y: ", train_label1.shape, "val Y: ", val_label1.shape, "test Y: ", test_label.shape)


# In[3]:


train_data = np.loadtxt('train_pred_vF_FLIC.txt')
print(train_data.shape)
train_data = train_data.reshape((-1,2,11))
print(train_data.shape)


# In[6]:


print(train_data[0])


# In[7]:


print(train_label1[0])


# In[8]:


drawLines(train_list1[0].copy(),train_label1[0].copy(),train_data[0])


# In[4]:


val_data = np.loadtxt('val_pred_vF_FLIC.txt')
print(val_data.shape)
val_data = val_data.reshape((-1,2,11))
print(val_data.shape)


# In[9]:


thresh = 0.5
max_acc = 0
acc_list = []
acc_6 = []
for i in range(int(val_label1.shape[0])):
    torsor_xy = val_label1[i, :, 0] - val_label1[i, :, 7]
    torsor_dist = np.sqrt(np.sum(np.square(torsor_xy),keepdims = True))
    torsor_frac = torsor_dist * thresh
    joint_diff = np.square(val_label1[i] - val_data[i])
    joint_dist = np.sqrt(np.sum(joint_diff, axis = 0))
    valid_mask = (joint_dist <= torsor_frac)
    cur_acc = np.sum(valid_mask) / 11
    if cur_acc > 0.6:
        acc_6.append(cur_acc)
    acc_list.append(cur_acc)
    if cur_acc > max_acc:
        max_acc = cur_acc
        max_ind = i
acc_list = np.array(acc_list)
sort_ind = np.argsort(acc_list)
sort_acc = np.sort(acc_list)
print(max_acc, max_ind)
print(sort_acc[::-1])
print(sort_ind[::-1])
acc_6 = np.array(acc_6)
sort_ind6 = np.argsort(acc_6)
sort_acc6 = np.sort(acc_6)
print(sort_acc6[::-1])
print(sort_ind6[::-1])


# In[52]:


# 1: 60 66 84 72 77 88 89 71 40 7 94 48 57 82
# 0.6: 33 37 21 32  8 11 12 15 29
j = 71
print(val_data[j])
print(val_label1[j])
drawLines(val_list1[j].copy(),val_label1[j].copy(),val_data[j])


# In[5]:


#thresh = 0.5
def get_acc(targets, predictions, torsor_frac):
    joint_diff = np.square(targets - predictions)
    joint_dist = np.sqrt(np.sum(joint_diff, axis = 1))   # (N, 2, 14) -> (N, 14)
    loss = np.sum(joint_diff)
    valid_mask = (joint_dist <= torsor_frac) * 1.0
    accuracy = np.sum(valid_mask) / (int(joint_dist.shape[0]) * int(joint_dist.shape[1]))
    wrist_acc = np.sum(valid_mask[:, 2] + valid_mask[:, 5]) / (int(joint_dist.shape[0]) * 2)
    elbow_acc = np.sum(valid_mask[:, 1] + valid_mask[:, 4]) / (int(joint_dist.shape[0]) * 2)
    return accuracy, wrist_acc, elbow_acc

torsor_xy = val_label1[:, :, 2] - val_label1[:, :, 9] #distance left_shoulder <-> right_hip in xy, result (N, 2)
torsor_dist = np.sqrt(np.sum(np.square(torsor_xy), axis = 1, keepdims = True)) # distance scaler, (N, 2) -> (N)
#torsor_frac = torsor_dist * thresh # max error distance
acc = []
w_acc = []
e_acc = []
thresh_list = np.arange(0.1, 1.1, 0.1)
for thresh in thresh_list:
    cur_acc, cur_w_acc, cur_e_acc = get_acc(val_label1, val_data, torsor_dist * thresh)
    acc.append(cur_acc)
    w_acc.append(cur_w_acc)
    e_acc.append(cur_e_acc)
print(acc, w_acc, e_acc)


# In[7]:


fig = plt.figure()
plt.plot(thresh_list, np.squeeze(acc), label = "acc")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('torso fraction')
plt.title("Learning rate: " + str(0.00005))
fig.savefig('AlexNet_acc_FLIC_thresh.png')
plt.show()

fig = plt.figure()
plt.plot(thresh_list, np.squeeze(w_acc), label = "wrist")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('torso fraction')
plt.title("Learning rate: " + str(0.00005))
fig.savefig('AlexNet_acc_wrist_FLIC_thresh.png')
plt.show()

fig = plt.figure()
plt.plot(thresh_list, np.squeeze(e_acc), label = "elbow")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('torso fraction')
plt.title("Learning rate: " + str(0.00005))
fig.savefig('AlexNet_acc_elbow_FLIC_thresh.png')
plt.show()

fig = plt.figure()
plt.plot(thresh_list, np.squeeze(w_acc), label = "wrist")
plt.plot(thresh_list, np.squeeze(e_acc), label = "elbow")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('torso fraction')
plt.title("Learning rate: " + str(0.00005))
fig.savefig('AlexNet_acc_arm_FLIC_thresh.png')
plt.show()

