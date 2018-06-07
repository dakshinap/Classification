'''
Created on Sep 22, 2017

@author: dakshina
'''

from .pca import PCA
from .multifileaudioframes import MultiFileAudioFrames
from .dftstream import DFTStream

import os.path
from datetime import datetime
import numpy as np


def pca_analysis_of_spectra(files, adv_ms, len_ms): 
    """"pca_analysis_of_spectra(files, advs_ms, len_ms)
    Conduct PCA analysis on spectra of the given files
    using the given framing parameters
    """
    
    framestream = MultiFileAudioFrames(files, adv_ms, len_ms)
    dftstream = DFTStream(framestream)
    
    spectra = []
    for s in dftstream:
        spectra.append(s)
    # Convert to matrix
    spectra = np.asarray(spectra)

    # principal components analysis
    pca = PCA(spectra)
    
    return pca

       
def get_corpus(dir, filetype=".wav"):
    """get_corpus(dir, filetype=".wav"
    Traverse a directory's subtree picking up all files of correct type
    """
    
    files = []
    
    # Standard traversal with os.walk, see library docs
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(filetype)]:
            files.append(os.path.join(dirpath, filename))
                         
    return files
    
def get_class(files):
    """get_class(files)
    Given a list of files, extract numeric class labels from the filenames
    """
    
    # TIDIGITS single digit file specific
    
    classmap = {'z': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'o': 10}

    # Class name is given by first character of filename    
    classes = []
    for f in files:        
        dir, fname = os.path.split(f) # Access filename without path
        classes.append(classmap[fname[0]])
        
    return classes
    
class Timer:
    """Class for timing execution
    Usage:
        t = Timer()
        ... do stuff ...
        print(t.elapsed())  # Time elapsed since timer started        
    """
    def __init__(self):
        "timer() - start timing elapsed wall clock time"
        self.start = datetime.now()
        
    def reset(self):
        "reset() - reset clock"
        self.start = datetime.now()
        
    def elapsed(self):
        "elapsed() - return time elapsed since start or last reset"
        return datetime.now() - self.start
    
    