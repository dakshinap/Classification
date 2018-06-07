import numpy as np
import matplotlib.pyplot as plt
#from mydsp.audioframes import AudioFrames
#from mydsp.multifileaudioframes import MultiFileAudioFrames
#from mydsp.dftstream import DFTStream
from mydsp.utils import *
from mydsp.features import get_features
from myclassifier.classifier import CrossValidator

files = get_corpus('wav/train')

"""
framestream = MultiFileAudioFrames(files, 10, 20)
dftstream = DFTStream(framestream)

# Generate and Print Spectra prints all spectra in a single file    
spectra = []
for s in dftstream:
    spectra.append(s)
spectra = np.asarray(spectra)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Intensity')
spectra = [np.average(spectra[s,:]) for s,v in enumerate(spectra)]
plt.plot(spectra)
plt.show()
"""

# PCA analysis of spectra
pca = pca_analysis_of_spectra(files, 10, 20)


data = []
for f in files:
    features1d = get_features(f, 10, 20, pca, 40, 0.25)
    #print(len(features1d))
    data.append(features1d)

data = np.matrix(data)
labels = get_class(files)

classifier = CrossValidator(data, labels)
