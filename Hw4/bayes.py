import os
import glob
import math
import numpy as np

# Folder Path
path = "languageID"
languages = ['e', 'j', 's']
  
# Read text File
  
def read_text_file(file_path):
    """
    Reads a file from the given path and returns its string contents.
    """
    with open(file_path, 'r') as f:
        contents = f.read()
    return contents
  
# iterate through all training files
global_count_dict = {'e':{}, 'j':{}, 's': {}}

for language in languages:
    dict = {}
    for file in glob.glob(f"{path}/{language}[0-9].txt"):
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{file}"
            # call read text file function
            contents = read_text_file(file_path)
        for char in contents:
            if char == "\n":
                continue
            else:
                if dict.get(char) == None:
                    dict[char] = 1
                else:
                    dict[char] = dict[char] + 1   
    global_count_dict[language] = dict

"""
Part 4.2 and 4.3 -- Emitting the Class-Conditional Likelihood for each token in the vocabulary under the 3 classes.
"""
global_ccp = {'e':{}, 'j':{}, 's': {}}
for language in languages:
    ccp = {}
    total = 0
    for char in sorted(global_count_dict[language].keys()):
        total = total + global_count_dict[language][char]
    for char in sorted(global_count_dict[language].keys()):
        if ccp.get(char) == None:
            ccp[char] =  float(global_count_dict[language][char] + 0.5)/ (total + (27 * 0.5))
    global_ccp[language] = ccp
    # Uncomment next line to produced desired output for each language
    # print(f"For language : {language}, the CCP vector is {ccp}\n")

"""
Part 4.4 -- Representing X from e10.txt as a bag-of-words vector
"""
test_file = f"{path}/e10.txt"

def predict(test_file):
    x_vector = {}
    for char in read_text_file(test_file):
        if char == "\n":
            continue
        else:
            if x_vector.get(char) == None:
                x_vector[char] = 1
            else:
                x_vector[char] = x_vector[char] + 1   

    # Printing the required bag of characters vector X
    # Uncomment next line to produced desired output
    # print(sorted(x_vector.items()))

    # Printing Log-Likelihood for English, Japanese and Spanish respectively
    log_likelihood =  {'e': float(0), 'j': float(0), 's': float(0)}
    for language in languages:
        ccp = global_ccp[language]
        logsum = 0
        for char in x_vector:
            if ccp.get(char) == None:
                # The token does not occur in our training sample and hence we use the smoothing parameters.
                ccp[char] = 0.5 / 27*0.5
            logsum = logsum + math.log(ccp[char]) * x_vector[char]
        # Uncomment next line to produced desired output for each language
        #print(f"Log-Likelihood for language {language} is {logsum}")
        log_likelihood[language] = logsum

    # The prior is the same for each class as the number of samples (10) is the same.
    prior = float((10 + 0.5)) / (30 + 3*0.5)

    # Calculate posterior using Bayes rule
    posterior = [log_likelihood[i] * prior for i in log_likelihood]
    # print(posterior)

    # Estimate the prediciton based on max. value of posterior probability
    prediction = posterior.index(max(posterior))

    #print(prediction)
    return languages[prediction]

"""
Part 4.5 -- Calculating likelihood of X for each given class under multinomial model assumption (Is a part of the pipeline of Part 4.6)
"""

"""
Part 4.6 -- Printing the posterior probabilities and prediction for our single test vector X.
"""
predict(f"{path}/e10.txt")

"""
Part 4.7 -- Predicting the posterior probabilities and prediction for the entire test suite as described.
"""
for language in languages:
    dict = {}
    for file in glob.glob(f"{path}/{language}[1-9][0-9].txt"):
        if file.endswith(".txt"):
            file_path = f"{file}"
            # call read text file function
            print(f"Actual:  {language}")
            print(f"Prediction: {predict(file_path)}")
