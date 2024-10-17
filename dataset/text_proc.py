### IMPORTS ### 
import re

### LARGE MANAGER ###
def cleanText(text):
    text = standardise(text)
    text = expand_abbrevs(text)
    return text

### STANDARDISE FORMATTING ###
def cut_whitespace(text):
    return re.sub(r'\s+', ' ', text.strip())

def remove_punctuation(text):
    return re.sub("['`!(),.:;?\"]", '', text)

def lower_case(text):
    return text.lower()

def standardise(text):
    text = cut_whitespace(text)
    text = remove_punctuation(text)
    text = lower_case(text)
    return text

def expand_abbrevs(text):    
