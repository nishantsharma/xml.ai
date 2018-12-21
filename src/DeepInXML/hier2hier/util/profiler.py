"""
File containing methods which help profile python code..
"""
import os, time, contextlib, requests, functools, warnings, json
from collections import OrderedDict

# When profiling function calls for performance, this object stores the timing information
# for later retrieval.
profilingData = [OrderedDict({ "Label": "Root", "BreakUp":[] })]

def lastCallProfile(doPop=False):
    """
    This method returns the profiling information of last profiler call made.
    """
    retval = profilingData[-1]["BreakUp"][-1]
    if doPop:
        profilingData[-1]["BreakUp"].pop()

    return retval

def appendProfilingData(key, metadata):
    """
    Metadata is sent to frontend for debugging. If part of the application wants to
    send some data for debugging, it can call this function with string key and json
    metadata and the data will be sent to frontend.
    """
    profilingData[-1].setdefault("Metadata", {})[key] = metadata

def methodProfiler(method):
    """
    Decorator to profile each call of a method.
    To use this profiler, annotate the function with this decorator.
    Then, every call to that method will get profiled and data stored in profilingData
    variable above. The data can be retrieved by using lastCallProfile() function.

    @methodProfiler
    def funcToProfile():
        pass

    """
    def wrapper(*args, **kw):
        with blockProfiler(method.__qualname__):
            result = method(*args, **kw)
        return result
    return wrapper

class blockProfiler(object):
    """
    Any code block can be profiled by using blockProfiler class.
    Use it as follows.

    with blockProfiler("labelOfBlock"):
        CodeBeingProfiled1()
        CodeBeingProfiled2()
    """
    def __init__(self, label):
        """
        Initialization.
        """
        self.curNode = OrderedDict({ "Label": label, "BreakUp":[] })

    def __enter__(self):
        """
        When we enter the with block, this code is executed.
        """
        # Append cur node
        profilingData.append(self.curNode)
        # Start the timer.
        self.start = time.time()

    def __exit__(self, exception_type, exception_value, traceback):
        """
        When we exit the with block, this code is executed.
        """
        # Stop the timer.
        end = time.time()

        # Record time measurement.
        self.curNode["MilliSeconds"] = ((end - self.start) * 1000.0)
        self.curNode.move_to_end("BreakUp")

        # Curnode building complete. Now remove from stack and insert it under the calling last node.
        profilingData.pop()
        profilingData[-1]["BreakUp"].append(self.curNode)