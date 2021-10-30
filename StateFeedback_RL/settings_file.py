# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 2021

@author: Saehong Park
"""

# TRAINING settings
 #settings 
settings={}
settings['sample_time']= 30; 
settings['periodic_test']=10; # number of save points.

settings['number_of_training_episodes']= 3000
settings['number_of_training']= 4 # Number of training.
settings['episodes_number_test']=10 # Number of testing.

#reference for the state of charge
control_settings={}
control_settings['references']={}
control_settings['references']['soc']=0.8; 


# constraints    
control_settings['constraints']={}
control_settings['constraints']['temperature']={}
control_settings['constraints']['voltage']={}
control_settings['constraints']['etasLn']={}
control_settings['constraints']['temperature']['max']=313; #273+40 = 313
# control_settings['constraints']['voltage']['max']=4.2;
control_settings['constraints']['etasLn']['min'] = 0.0000
        
# negative score at which the episode ends
control_settings['max_negative_score']=-500