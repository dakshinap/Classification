'''
Created on Sep 22, 2017

@author: dakshina
'''

from .audioframes import AudioFrames
import numpy as np

class MultiFileAudioFrames(object):
    '''
    MultiFileAudioFrames
    Class for creating a stream of audio frames across files   
    '''


    def __init__(self, filelist, adv_ms, len_ms):
        """"AudioFrames(filelist, adv_ms, len_ms)
        Create a stream of audio frames where each is in len_ms milliseconds long
        and frames are advanced by adv_ms.
        Frames are generated from the files in filelist.
        """
        
        # Store params
        self.files = filelist
        self.adv_ms = adv_ms
        self.len_ms = len_ms
        # Verify list of filenames is okay (basic checks)
        if not isinstance(self.files, list):
            raise RuntimeError("filelist must be a list")
        if len(filelist) < 1:
            raise RuntimeError("filelist must have at least one file")
        
        self.samplefile = AudioFrames(self.files[0], adv_ms, len_ms)
        self.Fs = self.samplefile.get_Fs()        
        
    def __iter__(self):
        """__iter__() - Return an iterator
        """
        return MultiFileIter(self.files, self.adv_ms, self.len_ms, self.Fs)
    
    def get_framelen_samples(self):
        "get_framelen_ms - Return frame length in samples"
        return self.samplefile.len_N
    
    def get_framelen_ms(self):
        "get_framelen_ms - Return frame length in ms"
        return self.samplefile.len_ms
    
    def get_frameadv_samples(self):
        "get_frameadv_ms - Return frame advance in samples"
        return self.samplefile.adv_N  

    def get_frameadv_ms(self):
        "get_frameadv_ms - Return frame advance in ms"
        return self.samplefile.adv_ms
    
    def get_Fs(self):
        "get_Fs() - Return sample rate"
        return self.Fs

    def get_Nyquist(self):
        "get_Nyquist() - Return Nyquist rate"
        return self.Fs/2.0
    
    def get_params(self):
        "Return dict with file parameters"
        
        params = {
            "filename" : self.files,
            "Fs" : self.Fs,
            "framing" : {"adv_ms" : self.samplefile.adv_ms, 
                         "len_ms" : self.samplefile.len_ms, 
                         "adv_N": self.samplefile.adv_N, 
                         "len_N" : self.samplefile.len_N},
            "format" : self.samplefile.format
            }
            
        return params
    
    def shape(self):
        "shape() - shape of tensor generated by iterator"
        return np.asarray([self.samplefile.len_N, 1])
    
    def size(self):
        "size() - number of elements in tensor generated by iterator"
        return np.asarray(np.product(self.samplefile.shape()))
    
class MultiFileIter:
    "MultiFileIter - Iterator over multiple files"
    
    def __init__(self, filelist, adv_ms, len_ms, Fs):
        """MFIter(filelist, adv_ms, len_ms, Fs)
        Create an iterator that iterates through the frames
        of each of the files
        """
        self.idx = 0
        # If calling object is modified, we want to use the files the
        # iterator was created with.  If we don't copy we just get a
        # pointer
        self.files = list(filelist)
        
        # Save parameters
        self.adv_ms = adv_ms
        self.len_ms = len_ms
        self.Fs = Fs
        
        # Set up the first frame stream
        self.frame_obj = AudioFrames(self.files[self.idx], adv_ms, len_ms)
        self.frame_it = iter(self.frame_obj)
        
    def __next__(self):
        try:
            frame = next(self.frame_it)
        except StopIteration:
            # Finished this file, try next
            frame = self.frame_next_file()
        return frame
    
    def frame_next_file(self):
        "frame_next_file - Get first frame from next file"
        
        # Some files may have zero frames...
        frame = None
        while frame is None:
            self.idx = self.idx + 1
            if self.idx >= len(self.files):
                # No more, propagate the exception
                raise StopIteration
            else:
                # more files, get next one
                self.frame_obj = AudioFrames(self.files[self.idx],
                                          self.adv_ms, self.len_ms)
                if self.frame_obj.get_Fs() != self.Fs:
                    raise RuntimeError(
                        "Initial files have Fs={}, file {} has {}".format(
                            self.Fs, self.files[self.idx],
                            self.frame_obj.get_Fs()))
                    
                self.frame_it = iter(self.frame_obj)
                try:
                    # Get first frame
                    frame = next(self.frame_it)
                except StopIteration:
                    # No frames in file...
                    # Catch the exception and move on to the next file
                    pass
        return frame
                 
                 
        
                  
             
        
        
                
        
        
        