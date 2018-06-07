'''
Created on Oct 3, 2017

@author: dakshina
'''

from .audioframes import AudioFrames
from .dftstream import DFTStream

import numpy as np

def get_features(file, adv_ms, len_ms, pca, components, offset_s):
    """get_features(file, adv_ms, len_ms, pca, components, offset_s)
    
    Given a file path (file), compute a spectrogram with
    framing parameters of adv_ms, len_ms.
    
    Reduce the dimensionality of the spectra to the specified number of components
    using a PCA analysis (dsp.PCA object in variable pca).
    
    """
    framestream = AudioFrames(file, adv_ms, len_ms)
    dftstream = DFTStream(framestream)
    
    # Assemble spectra into numpy array
    dlist = [d for d in dftstream]
    dlist = np.array(dlist)

    # Perform PCA analysis
    red_pca = pca.transform(dlist, components)

    # Find temporal center of the pca spectrum
    c = np.floor(len(red_pca)/2)
    
    # Fix the length
    offset_f = framestream.get_Fs()*offset_s/200
    
    red_pca =  red_pca[int(c - offset_f) : int(c + offset_f), :]

    # Reshape the array into a 1D vector called features1d (use .flatten()
    features1d = red_pca.flatten()

    return features1d