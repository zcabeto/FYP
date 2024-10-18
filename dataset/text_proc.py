### IMPORTS ### 
import re
import nltk
from nltk.corpus import cmudict
from g2p_en import G2p
ARPAbet = cmudict.dict()
g2p = G2p()

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
    words = [word_to_ARPAbet(word) for word in words]
    return ' '.join(words)

def word_to_ARPAbet(word):
    if word == '': return ''
    pronunciations = ARPAbet.get(word)
    if pronunciations:
        # Use the first pronunciation variant
        arpabet_word = pronunciations[0]
    else:
        # use g2p to filter word into likliest equivalent form
        arpabet_word = g2p(word)
        arpabet_word = [phoneme for phoneme in arpabet_word if phoneme.isalpha()]
    return ' '.join(arpabet_word)