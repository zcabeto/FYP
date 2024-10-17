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
        'ft'    :   'fort'
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
