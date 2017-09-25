import unicodedata
import sys

def decode_digits(string):
    new_digit = ''
    for character in string:
        try:
            new_digit += str(unicodedata.digit(character))
        except ValueError:
            pritn('have some problem')
            pass
    return new_digit

def is_arabic(character):
    try:
        return 'ARABIC LETTER' in unicodedata.name(character)
    except ValueError:
        return False

def get_arabic_letters():
    return [chr(character) for character in range(sys.maxunicode) if is_arabic(chr(character))]

def without_vocalization(character):
    return 'WITH' not in unicodedata.name(character)

def get_clean_arabic_letters():
    return [character for character in get_arabic_letters() if without_vocalization(character)]

def delete_diacritic(string):
    new_string = ''
    for character in string:
        new_string += unicodedata.normalize('NFD', character)[0]
    return new_string
