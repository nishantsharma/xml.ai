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

def appendProfilingLabel(labelExt):
    """
    Metadata is sent to frontend for debugging. If part of the application wants to
    send some data for debugging, it can call this function with string key and json
    metadata and the data will be sent to frontend.
    """
    profilingData[-1]["Label"] += labelExt

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

def getAllLabelNodes(profileData, label):
    if isinstance(profileData, list):
        return sum(
            [
                getAllLabelNodes(profileItem, label)
                for profileItem in profileData
            ],
            []
        )
    retval = []
    if profileData:
        if profileData["Label"] == label:
            retval.append(profileData)

        if "BreakUp" in proileData:
            retval = sum(
                [
                    getAllLabelNodes(breakUpNode, label)
                    for breakUpNode in proileData["BreakUp"]
                ],
                retval
           )
    return retval

def summarizeStats(labelNodes):
    summaryStats = {}
    
    if labelNodes:
        summaryStats["MilliSeconds"] = 0
        summaryStats["Gap"] = 0
        summaryStats["Count"] = len(labelNodes)

        for labelNode in labelNodes:
            summaryStats["MilliSeconds"] += labelNode["MilliSeconds"]
            summaryStats["Gap"] += labelNode["MilliSeconds"]
            if "BreakUp" in labelNode:
                summaryStats["Gap"] -= sum([
                    childNode["MilliSeconds"]
                    for childNode in labelNode["BreakUp"]
                ])

        summaryStats["AvgMilliSeconds"] = summaryStats["MilliSeconds"] /summaryStats["Count"]
        summaryStats["AvgGap"] = summaryStats["Gap"] /summaryStats["Count"]

    return summaryStats
    

def summarizeLabelNodes(labelDesc):
    if isinstance(labelDesc, dict):
        # Singleton case.
        labelNodes = [labelDesc]
    elif isinstance(labelDesc, str):
        labelNodes = getAllLabelNodes(lastCallProfile(), label)
    elif isinstance(labelDesc, list):
        labelNodes = labelDesc
    else:
        raise NotImplemented("Support for {0}".format(labelDesc))

    childNodes = sum(
        [
            labelNode["BreakUp"]
            for labelNode in labelNodes
            if "BreakUp" in labelNode
        ],
        []
    )

    breakUp = { childNode["Label"]:[] for childNode in childNodes }
    for childNode in childNodes:
        breakUp[childNode["Label"]].append(childNode)

    for childNode in breakUp.keys():
        breakUp[childNode] = summarizeLabelNodes(breakUp[childNode])

    retval = summarizeStats(labelNodes)
    retval["BreakUp"] = breakUp
    return retval


def summarizeLabel(profileData, label):
    retval = {}

    labelNodes = getAllLabelNodes(profileData, label)
    retval[label] = { "Stats": summarizeStats(labelNodes) }

    retval["Children"] = summarizeLabelNodes    

    return retval

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