#!/usr/bin/env python


# Built in modules
from __future__ import division
import sys
import os
import pdb
import glob

# Third party modules
import numpy as np
import matplotlib.pyplot as plt


def main():
    """Plotter for the delta autoretreat simulation."""


    home = os.path.expanduser("~")
    savepath = (home + '/Documents/SCarolina-classes/Spring-2014/delta-autoretreat/')
    dirname = '20140526-191349'
    wd = os.path.join(savepath, dirname)
    os.chdir(wd)

    # Given the name of the directory where the output is stored, return a list
    # of text files. 
    print 'Looking in directory {}'.format(wd)
    print 'These are all the output files'
    output_files = sorted(glob.glob('*.txt'), key=os.path.getctime)
    print output_files

    
    # Given a list of text files, return a matrix with the relevant columns to
    # be plot. 

    for datafile in output_files:
        data = np.loadtxt(datafile)
        print datafile
        plt.figure(1)    # 
        plt.subplot(311) # First subplot of the first figure.
        plt.plot(data[:,0]/1000, data[:,2], 'o')    #Plot the bed elevation profile.

        plt.subplot(312)    # Second subplot in the first figure
        plt.plot(data[:,0]/1000, data[:,4], 'o--') # plotting the slopes vs. distance

        plt.subplot(313)    # Third subplot in the first figure
        plt.plot(data[:,0]/1000, data[:,5], 'o')    # Plot the transport capacity.
        
#        plt.subplot(414)
#        plt.plot(data[-1,4])

    plt.figure(1)
    plt.subplot(311)
#    plt.title('Bed elevation')   
    plt.xlabel('Distance / km')
    plt.ylabel('Bed Elevation / m')
    plt.plot(data[:,0]/1000, data[:,3])    #Plot the bed elevation profile.


    plt.subplot(312)
    plt.yscale('log')
#    plt.title('Slopes')
    plt.xlabel('Distance / km')
    plt.ylabel('Slope / m/m')

    plt.subplot(313)
#    plt.title('Transport capacity')
    plt.xlabel('Distance / km')
    plt.ylabel('Bedload transport\n capacity / m$^2$/s')

    

    plt.show()
    # given x-y axes, plot them.

    return

    


# Allow for this script to be imported as a module.
if __name__ == "__main__":
    main()
