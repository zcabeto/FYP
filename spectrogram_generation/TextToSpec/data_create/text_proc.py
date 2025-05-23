### IMPORTS ###
import re
from unidecode import unidecode
import torch


### CREATE ENUMERATED SET ###
alphabet = '-!\'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
alphabet_enum = {}
counter = 1
for char in alphabet:
    if not char.isupper():
        alphabet_enum[char] = counter
        # only use punctuation and lower characters
    counter += 1

# Metacharacters for modifications
# note: none of these are used in the training text
audio_feature_chars = r'{}\/*><^`_~' 
counter = -1
for char in audio_feature_chars:
    alphabet_enum[char] = counter
    counter -= 1

### LARGE MANAGER ###
def textFeatures(text):
    text = cleanText(text)
    text = text_to_sequence(text)
    return torch.IntTensor(text)

def cleanText(text):
    text = standardise(text)
    text = expand_abbrevs(text)
    return text

### STANDARDISE FORMATTING ###
def lower_case(text):
    return text.lower()

def to_english(text):
    return unidecode(text)

def standardise(text):
    text = lower_case(text)
    text = to_english(text)
    return text

### EXPAND ABBREVIATIONS ###
abbrev = {'mr'  :   'mister',
        'mrs'   :   'misess',
        'dr'    :   'doctor',
        'st'    :   'saint',
        'co'    :   'company',
        'jr'    :   'junior',
        'maj'   :   'major',
        'gen'   :   'general',
        'drs'   :   'doctors',
        'rev'   :   'reverend',
        'lt'    :   'lieutenant',
        'hon'   :   'honorable',
        'sgt'   :   'sergeant',
        'capt'  :   'captain',
        'esq'   :   'esquire',
        'ltd'   :   'limited',
        'col'   :   'colonel',
        'ft'    :   'fort'
}

def expand_abbrevs(text):
    # use regex to make pattern to match with
    pattern = [re.compile((r'\b%s\.' % key), re.IGNORECASE) for key in abbrev.keys()]
    # function for replacing the found matches
    def replace(match):
        abbrev_key = match.group(0)[:-1]    # remove the point
        expansion = abbrev[abbrev_key]
        return expansion
    # apply the function with created pattern and replacement
    for regex in pattern:
        text = regex.sub(replace, text)
    return text

### CONVERT TO NUMERIC ARRAY ###
def text_to_sequence(text):
    sequence = []
    for symbol in text:
        if symbol in alphabet_enum:
            sequence.append(alphabet_enum[symbol])
    return sequence

