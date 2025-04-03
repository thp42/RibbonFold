import os, sys, random, time
import torch

class Profiler:
    """
    Profile the time spend and CUDA memory increase
    """
    def __init__(self, profile_cuda_memory=False, profile_max_cuda_memory=False, cuda_device='cuda:0'):
        """
        Parameter
        ---------
        profile_cuda_memory: bool
            Profile the CUDA memory
        profile_max_cuda_memory: bool
            Profile the max CUDA memory, max memory will be reset each called
        cuda_device: str
            CUDA device
        """
        self.profile_cuda_memory = profile_cuda_memory
        self.profile_max_cuda_memory = profile_max_cuda_memory
        self.cuda_device = cuda_device
        self.activate = True
        
        self.time_list = []
        self.cuda_memory = []
        self.max_cuda_memory = []
        self.user_string = []
    
    def _get_memory(self):
        return torch.cuda.memory_allocated(self.cuda_device) / 1024**3
    
    def _get_max_memory(self):
        memory = torch.cuda.torch.cuda.max_memory_allocated(self.cuda_device) / 1024**3
        torch.cuda.reset_max_memory_allocated()
        return memory
    
    @staticmethod
    def get_abs_time_str(seconds):
        """
        Parameter
        ----------
        seconds: float
            Return by time.time()
        """
        tobj = time.gmtime(seconds)
        return f"{tobj.tm_year}-{tobj.tm_mon}-{tobj.tm_mday} {tobj.tm_hour}:{tobj.tm_min}:{tobj.tm_sec}"
    
    def disable(self):
        """
        Disable the profiler
        """
        self.activate = False
        
    def enable(self):
        """
        Enable the profiler
        """
        self.activate = True
        
    def reset(self):
        """
        Reset the states
        """
        self.time_list.clear()
        self.cuda_memory.clear()
        self.user_string.clear()
    
    def start(self):
        """
        Start to record
        """
        if not self.activate:
            return
        self.reset()
        self.record()
    
    def print_log(self, file=sys.stdout, flush=True):
        """
        Print the current information
        """
        if not self.activate:
            return
        N_rec = len(self.time_list)
        
        if N_rec == 0:
            info = "Empty record, please start the profile process"
        else:
            time_str = Profiler.get_abs_time_str(self.time_list[-1])
            info = time_str + f" | {N_rec} records"
            if self.profile_cuda_memory:
                info += f" -- CUDA memory: {self.cuda_memory[-1]:.2f}G"
            if N_rec > 1:
                time_increase = self.time_list[-1] - self.time_list[-2]
                info += f"; Time increase: {time_increase:5f}s"
                if self.profile_cuda_memory:
                    cuda_increase = self.cuda_memory[-1] - self.cuda_memory[-2]
                    info += f"; CUDA increase: {cuda_increase:3f}G"
                if self.profile_max_cuda_memory:
                    max_cuda_increase = self.max_cuda_memory[-1] - self.max_cuda_memory[-2]
                    info += f"; Max CUDA increase: {max_cuda_increase:3f}G"
            if self.user_string[-1] is not None:
                info += f"; User string: {self.user_string[-1]}"
        print(info, file=file, flush=flush)
    
    def record(self, log=False, txt=None):
        """
        Record the time, memory usage at this point. Print the information if log is True
        
        Parameter
        ---------
        log: bool
            Print the information
        txt: string 
            Text to record
        """
        if not self.activate:
            return
        torch.cuda.synchronize()
        self.time_list.append(time.time())
        if self.profile_cuda_memory:
            self.cuda_memory.append( self._get_memory() )
        if self.profile_max_cuda_memory:
            self.max_cuda_memory.append( self._get_max_memory() )
        self.user_string.append(txt)
        
        if log:
            self.print_log()


def_profiler = Profiler(torch.cuda.is_available())
            
