import pandas as pd
import numpy as np
import math



def div_zero(x,y,z,t):
    if z* t != 0:
        result = x*y /(z*t)
    else: 
        result = 0
    return result
    

#take the data 
sms_train_features = pd.read_csv("sms_train_features.csv").to_numpy()
sms_train_labels = pd.read_csv("sms_train_labels.csv").to_numpy()
sms_test_features = pd.read_csv("sms_test_features.csv").to_numpy()
sms_test_labels = pd.read_csv("sms_test_labels.csv").to_numpy()


# get rid of first column
sms_train_features = sms_train_features[:,1:]
sms_train_labels = sms_train_labels[:,1:]
sms_test_features = sms_test_features[:,1:]
sms_test_labels = sms_test_labels[:,1:]

#finding priors in log form
total_row = len(sms_train_features[:,0])
spam = np.sum(sms_train_labels)
real = total_row-spam

prior_spam = math.log10(spam/total_row)
prior_real = math.log10(real/total_row)


#find the indexes of classes
class_0_index = np.where(sms_train_labels==0)[0]
class_1_index = np.where(sms_train_labels!=0)[0]

#find the total words for class 0 and 1 
total_word_count_class_0 = np.sum(sms_train_features[class_0_index,:])
total_word_count_class_1 = np.sum(sms_train_features[class_1_index,:])

#find the probability of each word in its class
prob_of_word_i_class0 = []
prob_of_word_i_class1 = []
for k in range(3458):
    word_count_0 = np.sum(sms_train_features[class_0_index,k])
    word_count_1 = np.sum(sms_train_features[class_1_index,k])
    
    
    prob0 = word_count_0/total_word_count_class_0
    prob1 = word_count_1/total_word_count_class_1
    prob_of_word_i_class0.append(prob0)
    prob_of_word_i_class1.append(prob1)
  
#make a prediction numpy    
predictions = np.zeros(978)
#we have the probabilities we need the test data

for i in range(len(sms_test_labels)):
    sample = sms_test_features[i,:]
    
    #take the indexes of the words in 
    sample_word_indexes = list(np.where(sample != 0)[0])
    
    #find the probabilities of the words in train data
    word_prob_list_0 = []
    word_prob_list_1 = []
    for l in sample_word_indexes:
        word_prob_list_0.append(prob_of_word_i_class0[l])
        word_prob_list_1.append(prob_of_word_i_class1[l])
    
    count_0 = 0
    count_1 = 0
    #icount the number of zeros in probabilities
    for p in range(len(word_prob_list_0)):
        if  word_prob_list_0[p] > 0:
            count_0 += 1
        if  word_prob_list_1[p] > 0:
            count_1 += 1
    
    #get the predition
    #if both have zero, take the real one
    if count_0 < len(word_prob_list_0) and count_1 <len(word_prob_list_1):
        pass
    #if 1 has no zero and 0 has zero predict 1
    elif count_0 < len(word_prob_list_0) and count_1 == len(word_prob_list_1):
        predictions[i] = 1
        #if both have no zero predict the prob
    elif count_0 == len(word_prob_list_0) and count_1 == len(word_prob_list_1):
        count_of_words = sample[sample_word_indexes].reshape(len(word_prob_list_0),1)
        
        #because of log we will multipy the prob with the count of the word and multiply with the prior
        for k in range(len(word_prob_list_0)):
            word_prob_list_0[k] = word_prob_list_0[k] * count_of_words[k]
            prob_0 = np.sum(word_prob_list_0)*prior_real
            
            word_prob_list_1[k] = word_prob_list_1[k] * count_of_words[k]
            prob_1 = np.sum(word_prob_list_1)*prior_spam
            
        if prob_1 > prob_0  :
            predictions[i] = 1
        

#find the accuracy of multinomial
accuracy = 0
false_positive = 0
true_positive = 0
false_negative = 0
true_negative = 0
for i in range(len(predictions)):
    if sms_test_labels[i] == predictions[i]:
        accuracy = accuracy + 1
        if sms_test_labels[i] == 0:
            true_negative = true_negative + 1
        else :
            true_positive = true_positive + 1

    else:
        if sms_test_labels[i] == 0:
            false_positive = false_positive + 1
        else :
            false_negative = false_negative + 1
accuracy = accuracy/len(predictions)
print("accuracy = ",accuracy)
print("TP = ",true_positive, "FN = ", false_negative)
print("FP = ",false_positive, "TN = ", true_negative)  

#Start of Bernoulli


#We have to convert our data to bernoulli first
for i in range(3911):
    for k in range(3458):
        if sms_train_features[i][k] > 0.5:
            sms_train_features[i][k] =1
            
for i in range(978):
    for k in range(3458):
        if sms_test_features[i][k] > 0.5:
            sms_test_features[i][k] =1
            
            
#Mutual information 
I_list = []


column_no = 0
for column_no in range(3458):
    #get the sample column
    wi = sms_train_features[:][column_no]
    N10 = 0
    N11 = 0
    N01 = 0
    N00 = 0
    for i in range(len(wi)):
        if sms_train_labels[i] == wi[i]:
            if sms_train_labels[i] == 0:
                N00 = N00 + 1
            else :
                N11 = N11 + 1
    
        else:
            if sms_train_labels[i] == 0:
                N10 = N10 + 1
            else :
                N01 = N01 + 1
    
    N = N00 + N10 + N11 + N01
    N1x = N10 + N11
    Nx1 = N01 + N11
    N0x = N00 + N01
    Nx0 = N10 + N00
    

    I1 = div_zero(N,N11,N1x,Nx1)
    I2 = div_zero(N,N01,N0x,Nx1)
    I3 = div_zero(N,N10,N1x,Nx0)
    I4 = div_zero(N,N00,N0x,Nx0)
    
    
    if I1 == 0:
        pass
    else:
        I1 = math.log2(I1)*(N11/N)
        
    if I2 == 0:
        pass
    else:
        I2 = math.log2(I2) * (N01/N)
        
    if I3 == 0:
        pass
    else:
        I3 = math.log2(I3)*(N10/N)
        
        
    if I4 == 0:
        pass
    else:
        I4 = math.log2(I4)*(N00/N)
        
    I = I1 + I2 + I3 + I4
    I_list.append(I)
    
    
 
    
#sort the indexes
I_indexes = np.argsort(I_list)
predictions_bernoulli = []      
            



#priors for bernoulli
prior_spam = (spam/total_row)
prior_real = (real/total_row)



feature_number = 200


#take the last 100 of these indexes
I_indexes_100 = I_indexes[-feature_number:]
sms_train_100 = sms_train_features[:,I_indexes_100]
sms_train_100_class_0 = sms_train_100[class_0_index]
sms_train_100_class_1 = sms_train_100[class_1_index] 




prob_list_class_0 = []
prob_list_class_1 = []

#calculate the probability of word i for both classes and take it to a list
for k in range(feature_number):   
    feat_class_1 = sms_train_100_class_1[:,k]
    feat_class_0 = sms_train_100_class_0[:,k]
    #summations are for not getting zero
    ust1 = np.sum(feat_class_1)+1
    alt1 = len(feat_class_1)+2
    
    
    ust0 = np.sum(feat_class_0)+1
    alt0 = len(feat_class_0)+2
    
    #add he prob to an array
    prob_wi_1 = ust1/alt1
    prob_wi_0 = ust0/alt0
    
    prob_list_class_1.append(prob_wi_1)
    prob_list_class_0.append(prob_wi_0)
    
    
    

#make the test sample 
    
sms_test_100 = sms_test_features[:,I_indexes_100]






#select a sample from the test set
for l in range(978):
    sample_test = sms_test_100[l,:]
    for z in range(feature_number):
        prob_list_class_0_new = np.zeros((feature_number,1))
        prob_list_class_1_new = np.zeros((feature_number,1))
        if sample_test[z] == 0:
            #change the probabilities according to 0 or 1
            prob_list_class_1_new[z] = 1-prob_list_class_1[z]
            prob_list_class_0_new[z] = 1-prob_list_class_0[z]

    #multiply all the probabilities and prior
    final_mul_class_1 = 1  
    final_mul_class_0 = 1          
    for p in prob_list_class_1_new:
        final_mul_class_1 = final_mul_class_1*p
    for p2 in prob_list_class_0_new:
        final_mul_class_0 = final_mul_class_0*p2
        
        
    final_mul_class_0 = final_mul_class_0*prior_real
    final_mul_class_1 = final_mul_class_1*prior_spam   
    
    
    
    if final_mul_class_0 < final_mul_class_1:
        predictions_bernoulli.append(1)
    else:
        predictions_bernoulli.append(0)
        
        
#find the accuracy 
acc_bern = 0
for i in range(len(sms_test_labels)):
    if sms_test_labels[i] == predictions_bernoulli[i]:
        acc_bern = acc_bern + 1        
        
acc_bern = acc_bern/ len(predictions_bernoulli)
            







    



















    