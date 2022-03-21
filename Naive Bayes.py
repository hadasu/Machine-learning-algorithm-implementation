# -*- coding: utf-8 -*-
"""

@author: Hadas
Naive Bayes spam filter
"""
import numpy as np
from collections import defaultdict
import glob, os

#Read all words from a file into a list
def all_words_in_file_to_list(path):
    text_file = open(path, "r")
    words = text_file.read().split(' ')
    text_file.close()
    return words

#Count number of instances of words in a list
def count_num_of_words(words ,frequency):
    count_words_dict = defaultdict(int)
    count_words = 0
    for word in words:
        if word not in ["and", "the", "of", " ", ""]:
            count_words += 1
            key = word
            if key in count_words_dict:
                count_words_dict[key] += 1
            else:
                count_words_dict[key] = 1.0
                
#   return count_words_dict                
    return {key:val for key, val in count_words_dict.items() if val > frequency} 


#Compute the probability of word appearing in a spam or a ham (not a spam) message by
#counting the number of times it appears in all appropriate messages divided by the number
#of words in all those messages.    
def word_probabilities_in_dict(count_words_dict):
    for word in count_words_dict:
        if word in count_words_dict:
                count_words_dict[word] = count_words_dict[word] /sum(count_words_dict.values())
    return count_words_dict

#To combine the probability for all words in a message we inspect, we need to multiply them.
#Since we are multiplying many small numbers we are likely to run into floating pointing
#underflow . Hence we will calculate the log probabilities instead, and add them up
def calculate_probabilities(count_words_dict, count_words, input_words):
    sum_probabilities = 0
    
    for word in input_words:
        if word not in ["and", "the", "of", " "]:
            if word in count_words_dict:
#                probabilitie = count_words_dict[word] / count_words
#                sum_probabilities += probabilitie
                sum_probabilities += np.log1p(count_words_dict[word])
            else:
                sum_probabilities += 0
    return sum_probabilities

#the classifier which will simply compare the probability of message being ham or
#spam
def predict(count_nonspam_dict, count_spam_dict, input_words):
    count_nonspam  = sum(count_nonspam_dict.values())
    count_spam = sum(count_spam_dict.values())
    
    nonspam_probabilitie = (calculate_probabilities(count_nonspam_dict, count_nonspam, input_words) *
                            count_nonspam )/(count_spam + count_nonspam)
    spam_probabilitie = (calculate_probabilities(count_spam_dict, count_spam, input_words) * 
                         count_spam)/(count_spam + count_nonspam)
    
    if nonspam_probabilitie > spam_probabilitie:
        return 0  #"nonspam"
    else:
        return 1  #"spam"
      
def main():
    train_path_nonspam = input('Enter Path - nonspam-train')
    
    if train_path_nonspam == '':
         train_path_nonspam ="..\\nonspam-train"
         
    train_path_spam = input('Enter Path - spam-train')
    
    if train_path_spam == '':
        train_path_spam ="..\\spam-train"
        
    test_path_spam = input('Enter Path - spam-test')
    
    if test_path_spam == '':
            test_path_spam ="..\\spam-test"
            
    test_path_nonspam = input('Enter Path - nonspam-test')
    
    if test_path_nonspam == '':
        test_path_nonspam ="..\\nonspam-test"
    
    
    train_words_nonspam=[]
    os.chdir(train_path_nonspam)
    for file in glob.glob("*.txt"):
        words = all_words_in_file_to_list(train_path_nonspam +'\\'+ file)
        train_words_nonspam.extend(words)
        
    train_count_nonspam_dict  = count_num_of_words(train_words_nonspam, 5)
    
         
    train_words_spam=[]
    os.chdir(train_path_spam )
    for file in glob.glob("*.txt"):
        words = all_words_in_file_to_list(train_path_spam +'\\'+ file)
        train_words_spam.extend(words)
        
    train_count_spam_dict = count_num_of_words(train_words_spam, 4)
       
    train_count_nonspam_dict = word_probabilities_in_dict(train_count_nonspam_dict)
    train_count_spam_dict = word_probabilities_in_dict(train_count_spam_dict)
    
        
    predict_mail = 0
    count_mail_spam = 0
    os.chdir(test_path_spam)
    for file in glob.glob("*.txt"):
        count_mail_spam += 1
        words = all_words_in_file_to_list(test_path_spam +'\\'+ file)
        predict_mail += predict(train_count_nonspam_dict, train_count_spam_dict, words)
    
    predict_spam = (predict_mail / count_mail_spam) * 100
    print("predict spam - " + str(predict_spam) + "%")
    
    
    predict_mail = 0
    count_mail_nonspam = 0
    os.chdir(test_path_nonspam)
    for file in glob.glob("*.txt"):
        count_mail_nonspam += 1
        words = all_words_in_file_to_list(test_path_nonspam +'\\'+ file)
        predict_mail += predict(train_count_nonspam_dict, train_count_spam_dict, words)
    
    predict_nonspam = ((count_mail_nonspam - predict_mail) / count_mail_nonspam) * 100
    print("predict nonspam - " + str(predict_nonspam) + "%")
     
if __name__== "__main__":
    main() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    