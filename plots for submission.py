# -*- coding: utf-8 -*-
"""
Created on Wed May 14 09:02:21 2025

@author: pfkin
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from decision_maker import DecisionAgent,GaussianCategory



def plot_A(cats, rewards):
    # A: plot impact of beta on a pair
    threshold = -0
    first_pair = ("wide_low", "narrow_high")
    
    ch_det_history=[]
    conf_det_history=[]
    for beta in np.logspace(-2,1.5,50):
    #for beta in [.1]:
        agent_det = DecisionAgent(cats, rewards, model="entropy", threshold=threshold, beta=beta, soft=False, verbose=True)
        
        ch_det, conf_det = agent_det.choose(first_pair)
        ch_det_history.append(ch_det)
        conf_det_history.append(conf_det)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ln1=ax1.plot(np.logspace(-2,1.5,50),ch_det_history, label="Choice (LH axis)", color='k')
    ln2=ax2.plot(np.logspace(-2,1.5,50),conf_det_history, label="Confidence (RH axis)", color='r')
    lns=ln1+ln2
    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc="upper left")
    ax1.set_xlabel("β")
    ax1.set_xscale('log')
    ax1.set_ylabel("Choice", color='k')
    ax1.set_yticklabels(['Left', 'Right', 'Right'])

    ax2.set_yticks([0, 0.5, 1])

    ax2.set_ylabel("Confidence", color='r')
    ax1.tick_params(axis='y', colors='k')  
    ax2.tick_params(axis='y', colors='r') 
    plt.show()

def plot_B():
    # B: plot impact of beta on a pair for 3 confidence measure types
    threshold = 0
    first_pair = ("wide_low", "narrow_high")
    
    ch_det_history_ent=[]
    conf_det_history_ent=[]
    for beta in np.logspace(-2,1,50):
    #for beta in [.1]:
        agent_det = DecisionAgent(cats, rewards, model="entropy", threshold=threshold, beta=beta, soft=False, verbose=False)
        
        ch_det, conf_det = agent_det.choose(first_pair)
        ch_det_history_ent.append(ch_det)
        conf_det_history_ent.append(conf_det)
    
    ch_det_history_diff=[]
    conf_det_history_diff=[]
    for beta in np.logspace(-2,1,50):
    #for beta in [.1]:
        agent_det = DecisionAgent(cats, rewards, model="diff", threshold=threshold, beta=beta, soft=False, verbose=False)
        
        ch_det, conf_det = agent_det.choose(first_pair)
        ch_det_history_diff.append(ch_det)
        conf_det_history_diff.append(conf_det)
    
    ch_det_history_map=[]
    conf_det_history_map=[]
    for beta in np.logspace(-2,1,50):
    #for beta in [.1]:
        agent_det = DecisionAgent(cats, rewards, model="map", threshold=threshold, beta=beta, soft=False, verbose=False)
        
        ch_det, conf_det = agent_det.choose(first_pair)
        ch_det_history_map.append(ch_det)
        conf_det_history_map.append(conf_det)
    
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ln1=ax1.plot(np.logspace(-2,1,50),ch_det_history_ent, label="Choice (LH axis)", color='k')
    ln2=ax2.plot(np.logspace(-2,1,50),conf_det_history_ent, label="Entropy as Confidence (RH axis)", color='r')
    ln3=ax2.plot(np.logspace(-2,1,50),conf_det_history_diff, label="Diff as Confidence (RH axis)", color='r', linestyle='dotted', marker="x")
    ln4=ax2.plot(np.logspace(-2,1,50),conf_det_history_map, label="Max as Confidence (RH axis)", color='r', linestyle='dashed')
    lns=ln1+ln2+ln3+ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left")
    ax1.set_xlabel("Beta")
    ax1.set_xscale('log')
    ax1.set_ylabel("Choice", color='k')
    ax2.set_ylabel("Confidence", color='r')
    ax1.tick_params(axis='y', colors='k')  
    ax2.tick_params(axis='y', colors='r') 
    plt.show()


def plot_C():
    # C: plot impact of threshold on a pair
    beta=0
    
    first_pair = ("wide_low", "narrow_high")
    
    ch_det_history=[]
    conf_det_history=[]
    for threshold in np.linspace(.1,.5,50):
        agent_det = DecisionAgent(cats, rewards, threshold=threshold, beta=beta, soft=False, verbose=False)
        # agent_soft = DecisionAgent(cats, rewards, threshold=0.31, beta=10, soft=True, verbose=True)
        
        ch_det, conf_det = agent_det.choose(first_pair)
        ch_det_history.append(ch_det)
        conf_det_history.append(conf_det)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ln1=ax1.plot(ch_det_history, label="Choice (LH axis)", color='k')
    ln2=ax2.plot(conf_det_history, label="Confidence (RH axis)", color='r')
    lns=ln1+ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")
    ax1.set_xlabel("Threshold")
    #ax1.set_xscale('log')
    ax1.set_ylabel("Choice", color='k')
    ax2.set_ylabel("Confidence", color='r')
    ax1.tick_params(axis='y', colors='k')  
    ax2.tick_params(axis='y', colors='r') 
    plt.show()



def plot_D():
    # # D: plot impact of threshold on a pair for different confidence measures
    beta=1
    first_pair = ("wide_low", "narrow_high")
    
    ch_det_history_ent=[]
    conf_det_history_ent=[]
    for threshold in np.linspace(.25,.5,50):
        agent_det = DecisionAgent(cats, rewards, model='entropy', threshold=threshold, beta=beta, soft=False, verbose=False)
        # agent_soft = DecisionAgent(cats, rewards, threshold=0.31, beta=10, soft=True, verbose=True)
        
        ch_det, conf_det = agent_det.choose(first_pair)
        ch_det_history_ent.append(ch_det)
        conf_det_history_ent.append(conf_det)
    
    ch_det_history_diff=[]
    conf_det_history_diff=[]
    for threshold in np.linspace(.25,.5,50):
        agent_det = DecisionAgent(cats, rewards, model='diff', threshold=threshold, beta=beta, soft=False, verbose=False)
        # agent_soft = DecisionAgent(cats, rewards, threshold=0.31, beta=10, soft=True, verbose=True)
        
        ch_det, conf_det = agent_det.choose(first_pair)
        ch_det_history_diff.append(ch_det)
        conf_det_history_diff.append(conf_det)
    
    ch_det_history_map=[]
    conf_det_history_map=[]
    for threshold in np.linspace(.25,.5,50):
        agent_det = DecisionAgent(cats, rewards, model='map', threshold=threshold, beta=beta, soft=False, verbose=False)
        # agent_soft = DecisionAgent(cats, rewards, threshold=0.31, beta=10, soft=True, verbose=True)
        
        ch_det, conf_det = agent_det.choose(first_pair)
        ch_det_history_map.append(ch_det)
        conf_det_history_map.append(conf_det)    
    
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ln1=ax1.plot(np.linspace(.1,.5,50),ch_det_history_ent, label="Choice (LH axis)", color='k')
    ln2=ax2.plot(np.linspace(.1,.5,50),conf_det_history_ent, label="Entropy as Confidence (RH axis)", color='r')
    ln3=ax2.plot(np.linspace(.1,.5,50),conf_det_history_diff, label="Diff as Confidence (RH axis)", color='r', linestyle='dotted', marker="x")
    ln4=ax2.plot(np.linspace(.1,.5,50),conf_det_history_map, label="Map as Confidence (RH axis)", color='r', linestyle='dashed')
    lns=ln1+ln2+ln3+ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")
    ax1.set_xlabel("Threshold")
    #ax1.set_xscale('log')
    ax1.set_ylabel("Choice", color='k')
    ax2.set_ylabel("Confidence", color='r')
    ax1.tick_params(axis='y', colors='k')  
    ax2.tick_params(axis='y', colors='r') 
    plt.show()
    
if __name__ == "__main__":    
    cats = {
        "narrow_low": GaussianCategory(mu=0.22, sigma=0.02),
        #"wide_low": GaussianCategory(mu=0.22, sigma=0.06),
        # "narrow_high": GaussianCategory(mu=0.30, sigma=0.02),
        
        
        "wide_low": GaussianCategory(mu=-0.5, sigma=0.5),
        "narrow_high": GaussianCategory(mu=0.5, sigma=0.5),
        
        
        "wide_high": GaussianCategory(mu=0.30, sigma=0.06),
    }
    # rewards = {
    #     "narrow_low": (10.0, 1e-6),
    #     "wide_low": (11.0, 1e-6),
    #     "narrow_high": (14.0, 1e-6),
    #     "wide_high": (18.0, 1e-6),
    # }
    rewards = {
        "narrow_low": (4.0, 1e-6),
        "wide_low": (2.0, 1e-6),
        "narrow_high": (4.0, 1e-6),
        "wide_high": (40.0, 1e-6),
    }

    plot_A(cats, rewards)
    #plot_B()
    # plot_C()
    # plot_D()
    
    