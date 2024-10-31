### IMPORTS ### 
import re
import numpy as np
import nltk
from nltk.corpus import cmudict
from g2p_en import G2p
ARPAbet = cmudict.dict()
g2p = G2p()

### CREATE ENUMERATED SET ###
arpabet_symbols = set()
for word in ARPAbet:
    for pronunciation in ARPAbet[word]:
        for phoneme in pronunciation:
            arpabet_symbols.add(phoneme)
arpabet_symbols = sorted(arpabet_symbols)
arpabet_enum = {}
counter = 1
for symbol in arpabet_symbols:
    arpabet_enum[symbol] = counter
    counter += 1

### LARGE MANAGER ###
def getFeatures(text):
    text = cleanText(text)
    text = phrase_to_ARPAbet(text)
    return text

def cleanText(text):
    text = standardise(text)
    text = expand_abbrevs(text)
    return text

### STANDARDISE FORMATTING ###
def cut_whitespace(text):
    return re.sub(r'\s+', ' ', text.strip())

def remove_punctuation(text):
    text = re.sub("['`!(),.:;?\"]", '', text)
    return re.sub("-", " ", text)

def lower_case(text):
    return text.lower()

def standardise(text):
    text = cut_whitespace(text)
    text = remove_punctuation(text)
    text = lower_case(text)
    return text

### EXPAND ABBREVIATIONS ###
abbrev = {'mrs' :   'misses',
        'mr'    :   'mister',
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
        'ft'    :   'fort',
        'ie'    :   'i ee',
        'eg'    :   'ee gee'
}

def expand_abbrevs(text):    
    # use regex to make pattern to match with
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in abbrev.keys()) + r')\b', re.IGNORECASE)
    # function for replacing the found matches
    def replace(match):
        abbrev_key = match.group(0)
        expansion = abbrev[abbrev_key]
        return expansion
    # apply the function with created pattern and replacement
    return pattern.sub(replace, text)

### CONVERT TO ARPABET ###
def phrase_to_ARPAbet(phrase):
    words = phrase.split(' ')
    phonetic_phrase = []
    for word in words:
        phonetic_phrase += word_to_ARPAbet(word)
    return np.array(phonetic_phrase)

def word_to_ARPAbet(word):
    if word == '': return ''
    pronunciations = ARPAbet.get(word)
    if pronunciations:
        # Use the first pronunciation variant
        arpabet_word = [arpabet_enum[phoneme] for phoneme in pronunciations[0]]
    else:
        # use g2p to filter word into likliest equivalent form
        arpabet_word = g2p(word)
        arpabet_word = [arpabet_enum[phoneme] for phoneme in arpabet_word if phoneme.isalpha()]
    return arpabet_word