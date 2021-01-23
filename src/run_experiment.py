#!/usr/bin/env python
# coding: utf-8

# # Psychophysics experiment
# 
# When prompted at the start of experiment, for Experiment Type enter either 'ensemble' or 'single'


# from __future__ import print_function
from psychopy import gui
from psychopy.tools.filetools import toFile

from builtins import next
from psychopy import visual, core, data, event
from numpy.random import shuffle
import copy  #from the std python libs

# sys.path.insert(0, '/Users/Klavdia/bml2020_project/src/')

from src.tools.perceptual_experiment import *

def run_exp():
    '''Runs psychophysics experiment to determine threshold for distortion for ensemble
    and individual mode. 
    
    after starting, you will be prompted for subject name and experiment type
    for experiment type put either "ensemble" or "single"
    
    '''
    
    # DIR_DATA = '/Users/Klavdia/bml2020_project/data/'
    DIR_DATA = '../data'
    save_dir = op.join(DIR_DATA, 'perceptual_data')

    # create folder for psychophysics data if not there already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # params 
    numTrials = 60
    alphaMin = 0
    alphaMax= 60
    alphaStart = 40
    alphaSteps = [10, 5, 5, 2, 2]
    images = ['church', 'dog', 'fish', 'horn', 'truck']


    # set file name
    info = {'Subject':'ld', 'Experiment Type':'ensemble'}
    info['dateStr'] = data.getDateStr()  # add the current time
    # present a dialogue to change params
    dlg = gui.DlgFromDict(info, title='Percpetual Exp', fixed=['dateStr'])
    if dlg.OK:
        toFile('lastParams.pickle', info)  # save params to file for next time
    else:
        core.quit()  # the user hit cancel so exit

    # make a text file to save data    
    fileName = save_dir + '/Experiment_'+info['Experiment Type']+'_'+info['Subject']+'_'+ info['dateStr']
    dataFile = open(fileName+'.csv', 'w')  # a simple text file with 'comma-separated-values'
    dataFile.write('distortedSide,trialNum,image, alpha, correct, experimentType\n')

    #create a window
    win = visual.Window([1000,600], monitor="testMonitor", units="deg")

    #create some stimuli
    dist_ind, dist_ensemble = create_distortions()
    preprocessed_images = load_images()
    fixation = visual.GratingStim(win=win, size=0.5, pos=[0,0], sf=0, color=-1)

    #display instructions
    message1 = visual.TextStim(win, pos=[0,+3],text='Hit a key when ready.')
    message2 = visual.TextStim(win, pos=[0,-3],text="Then press left or right to indicate which image was more distorted.")
    message3 = visual.TextStim(win, pos=[0,+3],text="press left or right")


    message1.draw()
    message2.draw()
    fixation.draw()
    win.flip()
    event.waitKeys()


    # setup stair procedure
    stairs=[]
    info['baseImages']= images
    for imgName in info['baseImages']:
        thisInfo = copy.copy(info)
        thisInfo['imgName'] = imgName
        thisStair = data.StairHandler(startVal=alphaStart, 
            extraInfo=thisInfo,
            nTrials=100, nUp=1, nDown=2,nReversals=5,
            minVal = alphaMin, maxVal=alphaMax, 
            stepSizes=alphaSteps)
        stairs.append(thisStair)



    for trial in np.arange(numTrials):

        # shuffle stairs
        shuffle(stairs)
        #print('Trial Num:', trial)

        for thisStair in stairs:

            thisAlpha = next(thisStair) # get the alpha value for this trial
            thisImage = thisStair.extraInfo['imgName'] # which image are we on

            dist_ind, dist_ensemble = create_distortions()
            preprocessed_images = load_images()


            if info['Experiment Type'] == 'ensemble':
                base_image, distorted_image = random_trial(preprocessed_images, dist_ind, dist_ensemble, alpha=thisAlpha, img=thisImage, ensemble=1)
            if info['Experiment Type'] == 'single':
                base_image, distorted_image = random_trial(preprocessed_images, dist_ind, dist_ensemble, alpha=thisAlpha, img=thisImage)


            distortedImg = visual.ImageStim(win=win, image=distorted_image, units='pix', size=260, ori=180)
            baseImg = visual.ImageStim(win=win, image=base_image, units='pix', size=260, ori=180)


            # randomly assign side on screen to images
            targetSide= np.random.choice([-1,1])
            distortedImg.setPos([250*targetSide, 0])
            baseImg.setPos([-250*targetSide, 0])


            # randomize which image is presented first
            order = np.random.choice([-1,1])


            #Trial
            if order ==1:
                # draw first stimulus
                fixation.draw()
                distortedImg.draw()
                win.flip()

                # wait 1s
                core.wait(1.0)

                # draw second stimulus
                fixation.draw()
                baseImg.draw()
                win.flip()

            else:
                # draw first stimulus
                fixation.draw()
                baseImg.draw()
                win.flip()

                # wait 1s
                core.wait(1.0)

                # draw second stimulus
                fixation.draw()
                distortedImg.draw()
                win.flip()


            # wait 1s
            core.wait(1.0)

            # blank screen
            message3.draw()
            fixation.draw()
            win.flip()


            # get response
            thisResp=None
            while thisResp==None:
                allKeys=event.waitKeys()
                for thisKey in allKeys:
                    if thisKey=='left':
                        if targetSide==-1: thisResp = 1  # correct
                        else: thisResp = -1              # incorrect
                    elif thisKey=='right':
                        if targetSide== 1: thisResp = 1  # correct
                        else: thisResp = -1              # incorrect
                    elif thisKey in ['q', 'escape']:
                        core.quit()  # abort experiment
                event.clearEvents()  # clear other (eg mouse) events - they clog the buffer

            #adjust stairs
            thisStair.addData(thisResp) 

            # save responses
            dataFile.write('%i,%i,%s,%.3f,%i, %s \n' %(targetSide, trial, thisImage, thisAlpha, thisResp, info['Experiment Type']))
            core.wait(0.5)

    # close file
    dataFile.close()

    win.clearBuffer() #make screen black
    endText = visual.TextStim(win, pos=[0,+3], text='Thanks!  You have finished the experiment.')
    endText.draw()
    win.flip()
    win.close()
    core.quit()
    return


if __name__ == '__main__':
    run_exp()

