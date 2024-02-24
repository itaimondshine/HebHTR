from processFunctions import *
from predictWord import *
import os

class HebHTR:

    def __init__(self, img_path):
        self.img_path = img_path
        self.original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

    

    def imgToWord(self, iterations=5, decoder_type='word_beam'):
        transcribed_words = []
        model = getModel(decoder_type=decoder_type)
        transcribed_words.extend(predictWord(self.original_img, model))
        return transcribed_words
