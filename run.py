'''Implementation of Self-Organizing Maps with Python for processing COVID 19 databases'''

# Subject = Analisys of Algorithm (AOA)
# @author = Darío Sebastián Cabezas Erazo - 0402019749
# email = dario.cabezas@yachaytech.edu.ec
# Overleaf = https://www.overleaf.com/project/60a984611325762c6504fdef

'''url = 'https://drive.google.com/file/d/1VXq08OkQQBrS_iYe5UmJ38_A5dHjMQ8-'
output = 'Epidemic-Data-for-Novel-Coronavirus-COVID-19.csv'
gdown.download(url, output, quiet=False)'''
'''LIBRARIES'''

import math 
import numpy as np
import csv
import webbrowser
import timeit
import os

'''UTILITY'''

def dimension (matriz):                                                                 # 0(c) complexity constant
    filas = len(matriz) 
    if filas > 0: 
        columnas = len(matriz[0]) 
    else: 
        columnas = 0 
    return filas, columnas
# Dimension Function 
chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
webbrowser.register('chrome', None)
# Creating Path for opening pages in internet

'''DATASET'''

with open('Epidemic-Data-for-Novel-Coronavirus-COVID-19.csv', newline='') as csvfile:
    # t= v*i + v + C                                                                          # 0(n^2)
    reader = csv.DictReader(csvfile)
    data = list(reader)
    countries=[]
    data_to_work=[]
    for v in data:
        for i in v.keys():
            k=[]
            if i=="Country":
                k.append(v[i])
                k.append(v['GeoPosition'])
            if k!=[]:
                countries.append(k)                
            else:
                continue
    # Saving Countries
    for v in data:
        for i in v.keys():
            k=[]
            if i=="ConfirmedCases":
                k.append(v["ConfirmedCases"])
                k.append(v["RecoveredCases"])
                k.append(v["Deaths"])
            if k!=[]:
                data_to_work.append(k)                
            else:
                continue
    # Saving Raw Data for working with
# Partitioning Raw Data 
def clean_countries():                                                                  # 0(n^2) complexity
    # t= i*j + i + C
    k=0
    X=[]
    for i in countries:
        x=i[0].split(",")
        x.append(k)
        if isinstance(x[1], str):
            temp=''
            for j in x[1]:
                if j=="]" or j=='\"' or j==' ':
                    continue
                temp= temp + j
            x[1]=temp
        k = k+1
        x.pop(0)
        X.append(x)
    return X
    # Cleaning countries for being used as strings
def search_country(string):                                                             # 0(n) complexitty
    # t= i + C    
    countries_cleaned=clean_countries()
    countries_list=[]
    for i in countries_cleaned:
        if i[0]==string:
            countries_list.append(i[1])
    return countries_list
    # Function to look for specific countries and return a list of databases found
def raw_data_manager(n):                                                                # 0(n^2) complexity
    #t = r*c + r + d + C
    raw_data=[]
    d,e=dimension(data_to_work)
    for i in range(e):
        raw_data.append((data_to_work[n][i].split("{{{")[1].split("}}")[0]).split(","))
    r,c=dimension(raw_data)
    data=np.zeros((c, r))
    for i in range(r):
        for j in range(c):
            if  raw_data[i][j]==' Missing["NotAvailable"]' or raw_data[i][j]=='Missing["NotAvailable"]':
                raw_data[i][j]=0
                data[j][i]=raw_data[i][j]
            else:
                raw_data[i][j]=int(raw_data[i][j])
                data[j][i]=raw_data[i][j]
        np.append(data, raw_data[i])
    return data
    # Function to replace not consistent data in raw data

'''EUCLIDEAN DISTANCE'''

def eu_distance(A,B):                                                                   #O(1) — Constant Time            
    return np.sqrt(np.sum((A-B)**2))
    # Euclidean Norm
'''WEIGHT UPDATE FORMULA'''

def learning(init_learning_rate,i,n_iter):                                              #O(1) — Constant Time
    return init_learning_rate* np.exp(-i/n_iter)
def topological_neighborhood(distance,radius):                                          #O(1) — Constant Time
    return np.exp(-distance/(2*(radius**2)))
def neighborhood_size(init_radius,i,time_constant):                                     #O(1) — Constant Time
    return init_radius*np.exp(-i/time_constant)

'''BMU'''

def find_bmu(t,net):                                                                    # T=K*N + C Where K is number of neurons in net and N is number of rows in each neuron
    min_dist=1000000                                                                    # C constant operation
    for x in range(net.shape[0]):                                                       # K times N operations
        for y in range(net.shape[1]):                                                   # N times constant operations
            unit=net[x,y].reshape(1,-1)
            t=t.reshape(1,-1)
            euc_dist=eu_distance(unit,t)
            if euc_dist < min_dist:
                min_dist=euc_dist
                bmu=net[x,y]
                bmu_idx=np.array([x,y])
    return bmu, bmu_idx

'''SOM FUNCTION'''

def SOM_COVID():
    '''SETUP PARAMETERS'''
    
    #User input data
    
    country=input("What country do you want to analyze?\n")                             # Constant
    print(search_country(country))
    n=int(input("Which country database do you want to analyze?\n"))                    # Constant
    
    # Initialization Parameters & Benchmark
    
    start = timeit.default_timer()                                                      # Benchmark Constant 
    #
    data=raw_data_manager(n)                                                            # Data pre-processed Constant()
    iterations=1000                                                                     # Constant
    learning_rate=0.1                                                                   # Constant
    row=data.shape[0]                                                                   # Constant
    columns=data.shape[1]                                                               # Constant
    # K is optimal number of neurons
    K=int(5*math.sqrt(row))                                                             # Constant
    # N is number of rows in neural network (Changing it make algorithm grows in cubic way)
    N=1
    network_dim=np.array([K,N])                                                         # Constant
    # Creating Neural Network 'net'
    net=np.random.random((network_dim[0],network_dim[1],columns))                       # Constant
    
    init_radius=max(network_dim[0],network_dim[1])/2                                    # Constant
    time_constant=iterations/np.log(init_radius)                                        # Constant
                                                                                        # All constants make a big constant "D"
    # TRAINING NEURAL NETWORK by Iterations
    # 0(n^3) cubic complexity, where I is the number of Iterations
    # T= 2(I*K*N) + I*B + I*C + D
    print("\nProcessing COVID 19 databases...")
    for i in range(iterations):                                                         # I times K*N operations + I times B, where I is number of iterations
        t=data[np.random.randint(0,row),:]                                              # Constant
        r=neighborhood_size(init_radius,i,time_constant)                                # Constant
        l=learning(learning_rate,i,iterations)                                          # Constant
                                                                                        # All constants make a big constant "B"
        bmu,bmu_idx=find_bmu(t,net)                                                     # I times T=K*N + C
        # Calculate Best Matching Unit
        for x in range(net.shape[0]):                                                   # K times N operations
            for y in range(net.shape[1]):                                               # N times constant operations, "N" is number of rows in each neuron
                w=net[x,y].reshape(1,columns)
                w_dist=eu_distance(np.array([[x,y]]),bmu_idx.reshape(1,2))
                # Calculing weight distance for being used in neighborhood
                if w_dist<=r:
                    influence=topological_neighborhood(w_dist,r)
                    new_w=w+(l*influence*(t.reshape(1,-1)-w))
                    net[x,y]=new_w.reshape(1,3)
    print(len(net))
    print("\nCOVID 19 databases processed. Saving...")
    #OPERATIVE SISTEM
    os.mkdir(str(country) + " Processed Data",0o777)
    os.chdir(str(country) + " Processed Data")
    for i in range(len(net)):
        np.savetxt(str(country) + str(i) + str(".txt"),net[i])
    os.chdir("../")
    #It Saves trained neurons
    
    #USEFULL DATA
    end = timeit.default_timer()
    print("\nSOM",end - start,"seconds")
    print('Geolocating',country)
    #GEOLOCATING
    webbrowser.get(chrome_path).open('https://earth.google.com/web/search/' + str(countries[n][1].split("{")[1].split("}")[0]))
SOM_COVID()