#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A method that always returns no change points

Author: G.J.J. van den Burg
Date: 2020-05-07
License: MIT
Copyright: 2020, The Alan Turing Institute

"""

import argparse
import time

from cpdbench_utils import exit_with_error, load_dataset, exit_success

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats

class DHMM():
    def __init__(self,dataori_diri, no_of_components,n_iter=100,printsome=True ) -> None:
        self.dataori_diri = dataori_diri
        self.no_of_components = no_of_components
        self.no_of_elements = dataori_diri.shape[0]
        self.n_iter = n_iter
        self.AlphaDiriGuess = np.random.uniform(low=1,high=5,size=(no_of_components,dataori_diri.shape[1]))
        ones = 10*np.ones(self.no_of_components)
        self.print_some = printsome
        # pi is one dimensional matrix where Pi[i] denotes 
        # probability that 1st element is from 'i'th state.
        self.Pi = np.zeros(self.no_of_components)
        self.Pi[0] = 1
        # self.A = np.random.dirichlet(ones,no_of_components)
        self.A = np.identity(self.no_of_components)
        
        # I am considering that the algorithm will divide this into
        # 2 parts of equal size. it will be in the same position until 
        # the middle . until middle, n/2 elements, inside it, the 
        # last one will have different position.
        self.A[0, 0] = (self.no_of_elements - 2 )/self.no_of_elements
        for i in range(1, self.no_of_components):
            self.A[0,i] = 1/(self.no_of_elements* (self.no_of_components-1))
        self.A = self.A/np.sum(self.A, 0)


        self.beta = np.zeros((self.no_of_elements,no_of_components))
        self.alpha = np.zeros((self.no_of_elements,self.no_of_components))
        self.c = np.zeros(self.no_of_elements)
        self.gamma = np.zeros((self.no_of_elements,no_of_components))
        self.Xi = np.zeros((self.no_of_elements,self.no_of_components,self.no_of_components))
        self.viterbi = np.zeros(shape= (self.no_of_components,self.no_of_elements))
        self.log_likel = []

    def digamma(self,x):
        return sp.special.digamma(x)

    def trigamma(self,x):
        return sp.special.polygamma(1,x)

    def inverse_digamma(self,x):
        if(x>-2.22):
            y= np.exp(x)+(1/2) 
        else:
            y= -1/((x)+0.5772156649015329)
        for _ in range(100):
            y= y- ((self.digamma(y)-x)/self.trigamma(y))
        return y

    def forward(self):
        # alpha(Zn) = p(x1,x2,x3,...,xn,zn) 
        # alpha(Zn) = p(xn/zn)*{(summation over all z_n-1) alpha[z_n-1]*A[zn|z_n-1]}
        
        for i in range(self.no_of_components):
            self.alpha[0,i] = self.Pi[i]*stats.dirichlet.pdf(self.dataori_diri[0],self.AlphaDiriGuess[i]) 
        self.c[0]= np.sum(self.alpha[0,:])
        self.alpha[0,:]/=np.sum(self.alpha[0,:])
        
        for i in range(1,self.no_of_elements):
            for j in range(self.no_of_components):
                marginal_sum = 0                            
                for x in range(self.no_of_components):
                    marginal_sum += self.alpha[i-1,x]*self.A[x,j]
                self.alpha[i,j] = stats.dirichlet.pdf(self.dataori_diri[i],self.AlphaDiriGuess[j])*(marginal_sum)
            self.c[i] = np.sum(self.alpha[i,:])
            self.alpha[i,:] /= self.c[i]

    def backward(self,):
        for i in range(self.no_of_components):
            self.beta[self.no_of_elements-1,i] = 1
        
        for i in range(1,self.no_of_elements):
            i = self.no_of_elements -i-1
            for j in range(self.no_of_components):
                marginal_sum = 0
                for x in range(self.no_of_components):
                    marginal_sum += self.beta[i+1,x]*self.A[j,x]*stats.dirichlet.pdf(self.dataori_diri[i+1],self.AlphaDiriGuess[x])
                self.beta[i,j] = marginal_sum
            self.beta[i,:] /= self.c[i+1]   

    def maximization(self):
        self.gammasum= np.sum(self.gamma[0,:]) 
        self.Pi = self.gamma[0,:]/self.gammasum
        self.gammasum = np.sum(self.gamma,axis=0,keepdims=True)
        # for j in range(self.no_of_components):
        #     Xi_individual= np.zeros((self.no_of_components))
        #     for k in range(self.no_of_components):    
        #         for i in range(1,self.no_of_elements):
        #             Xi_individual[k] += self.Xi[i,j,k]
        #     self.A[j,:] = Xi_individual/np.sum(Xi_individual)
            
        logdata= np.log(self.dataori_diri)
        dataterm = []
        for i in range(self.no_of_components):
            datater = np.sum(np.multiply(self.gamma[:,i].reshape(self.no_of_elements,1),logdata),axis=0)/self.gammasum[0,i]
            dataterm.append(datater)

        for i in range(self.no_of_components):
            for e in range(self.AlphaDiriGuess[0].shape[0]):
                    self.AlphaDiriGuess[i][e] = self.inverse_digamma(self.digamma(float(np.sum(self.AlphaDiriGuess[i])))+dataterm[i][e])
    
    def fit(self):

        for i in (range(self.n_iter)):
            self.forward()
            # print(self.alpha)
            # print(self.c)
            self.log_likel.append(self.c.sum())
            self.backward()
            # print(self.beta)
            
            if i%(int(self.n_iter/5))==0 and self.print_some:
                print(i)
                print(self.AlphaDiriGuess)

            self.gamma = np.zeros((self.no_of_elements,self.no_of_components))
            for i in range(self.no_of_elements):
                self.gamma[i] = np.multiply(self.alpha[i],self.beta[i])

            # for i in range(1,self.no_of_elements):
            #     for j in range(self.no_of_components):
            #         for k in range(self.no_of_components):
            #             self.Xi[i,j,k] = self.alpha[i-1,j]*stats.dirichlet.pdf(self.dataori_diri[i],self.AlphaDiriGuess[k])*self.A[j,k]*self.beta[i,k]/self.c[i]
            self.maximization()

    def Viterbi(self,):
        self.viterbi = np.zeros(shape= (self.no_of_components,self.no_of_elements))
        back_pointer = np.zeros((self.no_of_components,self.no_of_elements))
        state = []
        for i in range(self.no_of_components):
            self.viterbi[i,0] = np.log(self.Pi[i])+stats.dirichlet.logpdf(self.dataori_diri[0],self.AlphaDiriGuess[i])
            
        for i in range(1,self.no_of_elements):
            for j in range(self.no_of_components):
                maximum_prob= float('-inf')
                # maxindex = 0
                for k in range(self.no_of_components):
                    prob = self.viterbi[k,i-1]+stats.dirichlet.logpdf(self.dataori_diri[i],self.AlphaDiriGuess[j])+np.log(self.A[k,j])
                    if (maximum_prob<prob):
                        maximum_prob = prob
                        index = k
                self.viterbi[j,i] = maximum_prob
                back_pointer[j,i] = index
        # maxElem = np.amax(self.viterbi[:,-1])
        res = np.argmax(self.viterbi[:,-1])
        self.tags = []
        self.tags.append(res)
        for j in range(2,self.no_of_elements+1):
            self.tags.append(int(back_pointer[int(res),-j+1]))
            res = int(back_pointer[res,-j+1]) 
        self.tags = self.tags[::-1]
        return self.tags
    
    def label(self,):
        self.fit()
        tags = self.Viterbi()
        return tags


def multivariate_to_composite(data):
    """
    data: Multinomial Data
    returns Compositional data using the transformation.
    """
    # first transformation: scale by std dev and mean.

    datanew = data

    # datanew = (data - np.mean(data,axis=0))/(np.std(data,axis=0))
    
    datanew += 0.1*np.random.randn(datanew.shape[0],datanew.shape[1])

    # second transformation: inverse sigmoid transform.
    datanew = np.exp(datanew)
    datasum = (np.sum(datanew,axis =1)+1).reshape(data.shape[0],1)
    datanew =  datanew/datasum
    datanew = np.concatenate((datanew,1-datanew.sum(axis=1,keepdims=True)),axis=1)
    return datanew


def plot_batch(batch,start_index,end_index):
    plt.scatter(range(start_index,end_index),batch[:,0])
    plt.title(f'{start_index}, {end_index}')
    plt.show()

def find_cps(data, parameters, normalized = False, n_iter = 40):
    increment_size = parameters['increment_size']
    init_size = parameters['init_size']
    
    if not normalized:
        data_diri = multivariate_to_composite(data)
    else:
        data_diri = data
    leftInd = 0
    rightInd = init_size
    # print(f"init_size = {init_size}")
    total_points = data.shape[0]
    
    cp_list = []
    
    while ( rightInd <= total_points and rightInd-leftInd+1 >= init_size -10 ):
        # print("(i,j) = "+ str(leftInd)+", "+ str(rightInd))
        
        batch = data_diri[leftInd: rightInd]

        # plot_batch(batch,leftInd,rightInd)
        minBIC = float('inf')
    
        hmms = []
        for no_of_classes in range(1,3):
            # print(f"For no of classes = {no_of_classes}")    
            hmm = DHMM(batch,no_of_classes,n_iter=n_iter,printsome = False)
            hmm.fit()
            hmms.append(hmm)
            n = no_of_classes
            bic = (n*n+n+ n*hmm.AlphaDiriGuess.shape[1])*np.log(batch.shape[0]) - 2*hmm.log_likel[-1]
            # print(f"BIC = {bic}")  
            if(bic<minBIC):
                minBIC = bic
                best_no_of_classes = no_of_classes
        # print(f"best_no_of_classes={best_no_of_classes}")
        if(best_no_of_classes > 1):
            tags = hmms[best_no_of_classes-1].Viterbi()
            cur_change_points = []
            for points in range(len(tags)-1):
                if(tags[points]!= tags[points+1]):
                    # print(f"Change point detected at {points+leftInd} !!")
                    cur_change_points.append(points+ leftInd )
            
            cp_list += cur_change_points
            if(len(cur_change_points)>0):
                leftInd = cur_change_points[-1] +1
                rightInd = min(total_points, leftInd + init_size-1)
            else:
                if(rightInd == total_points):
                    return cp_list
                rightInd = min(rightInd + increment_size, total_points)
            

        else:
            if(rightInd== total_points):
                return cp_list
            rightInd = min(rightInd + increment_size, total_points)
    return cp_list



def parse_args():
    parser = argparse.ArgumentParser(description="Wrapper for None-detector")
    parser.add_argument(
        "-i", "--input", help="path to the input data file", required=True,   
    )
    parser.add_argument("-o", "--output", help="path to the output file")
    parser.add_argument("-s","--initsize", help="minimum window size")
    parser.add_argument("-j", "--jumpsize", help="jump size")
    return parser.parse_args()

def make_param_dict(args, defaults = None):
    parameters = {}
    parameters['init_size'] = int(args.initsize) if args.initsize!= None else 50
    parameters['increment_size'] = int(args.jumpsize) if args.jumpsize!= None else 50
    return parameters

def main():
    args = parse_args()

    data, mat = load_dataset(args.input)
    # defaults = {
        
    # }

    # combine command line arguments with defaults
    parameters = make_param_dict(args)

    # start the timer
    start_time = time.time()
    error = None
    status = 'fail' # if not overwritten, it must have failed
    
    # run the algorithm in a try/except
    try:
        locations = find_cps(mat, parameters)
        status = 'success'
    except Exception as err:
        error = repr(err)
        error = error+args.input
        exit_with_error(data,args=args,parameters=parameters,error=error, script_filename=__file__)

    stop_time = time.time()
    runtime = stop_time - start_time

    exit_success(data, args, parameters, locations, runtime, __file__)


if __name__ == "__main__":
    main()
