import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
r = 0.1
sigma = 0.3
T = 1
K = 90
S0 = 100

############### Exact Formulas ###############

def ExactPricing(r,T,K,S0,sigma):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)


def geometricExactPricing(r,T,K,S0,sigma):
    d1 = (np.log(S0/K) + (r+(sigma**2)/6)*T/2)/(sigma*np.sqrt(T/3))
    d2 = (np.log(S0/K) + (r-(sigma**2)/2)*T/2)/(sigma*np.sqrt(T/3))
    return S0*np.exp(-(r + (sigma**2)/6)*T/2) * norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)



############## Calcul de l'espérance pour la lognormal

def esperance(r = 0.1, sigma = 0.3, T = 1, S0 = 100):
    return ((np.exp(r*T)-1)*S0)/(r*T)


############## Calcul de la variance pour la lognormal

def variance(r = 0.1, sigma = 0.3, T = 1, S0 = 100):
    temp1 = (2*S0**2/((T**2)*(r+sigma**2)))
    temp2 = (np.exp((2*r+sigma**2)*T)-1)/(2*r+sigma**2)
    temp3 = (np.exp(r*T)-1)/r
    temp = temp1 * (temp2 - temp3)
    return temp - (((np.exp(r*T)-1)*S0)/(r*T))**2

def ecarttype(r = 0.1, sigma = 0.3, T = 1, S0 = 100):
    return np.sqrt(variance(r, sigma, T, S0))


esp = esperance()
var = variance()
EC = ecarttype()

mulog = np.log(esp**2/np.sqrt(var + esp**2))
sigmalog = np.sqrt(np.log(1 + (var/(esp**2))))


################# Monte Carlo Simulation ##########

def Monte_Carlo_lognormal(iterations = 1000000, K = 110, T = 1, r = 0.1):
    lognorm = np.random.lognormal(mulog, sigmalog, iterations)
    Price = np.exp(-r*T) * np.maximum(lognorm - K, 0)
    std = np.std(Price)
    mean = np.mean(Price)
    return mean+(1.96*std/np.sqrt(iterations)), mean, mean-(1.96*std/np.sqrt(iterations))


def Monte_Carlo_lognormal_it(iterations = 1000000, K = 110, T = 1, r = 0.1):
    lognorm = np.random.lognormal(mulog, sigmalog, iterations)
    Price = np.exp(-r*T) * np.maximum(lognorm - K, 0)
    lower_list = []
    upper_list = []
    mean_list = []
    for i in range(1, iterations, 100):
        std = np.std(Price[:i])
        mean = np.mean(Price[:i])
        mean_list.append(mean)
        upper_list.append(mean+(1.96*std/np.sqrt(i)))
        lower_list.append(mean-(1.96*std/np.sqrt(i)))
    return upper_list, mean_list, lower_list


################# Difference X-Y

def difference(iterations = 100000, K = 90, T = 1, r = 0.1,N = 150, S0=100, sigma = 0.3):
        
    # Generate N*iterations gaussian samples of variance T/N
    times=(T/N)*np.arange(1,N+1)
    accroissements=np.random.normal(0,np.sqrt(T/N),(N, iterations))
    brownian=np.dot(np.tril(np.ones((N,N))),accroissements)
    
    # Arithmetic average call
    St=S0*np.exp(sigma*np.transpose(brownian)+(r-sigma**2/2)*times)

    arithmetic_sampling=np.exp(-r*T)*np.maximum(np.mean(St, axis = 1)-K,0)
    arithmetic_call= [np.mean(arithmetic_sampling), (1.96*np.std(arithmetic_sampling)/np.sqrt(iterations))]
    # Geometric average call
    geometric_sampling=np.exp(-r*T)*np.maximum(S0*np.exp(sigma*np.mean(brownian,axis = 0)+(r-(sigma**2)/2)*(T/2))-K,0)

    difference=arithmetic_sampling-geometric_sampling
    std = np.std(difference)
    mean = np.mean(difference)
    #mean+(1.96*std/np.sqrt(iterations)), mean, mean-(1.96*std/np.sqrt(iterations))
    return  arithmetic_call[0], np.mean(geometric_sampling), mean




def difference_std(iterations = 100000, K = 110, T = 1, r = 0.1,N = 150, S0=100, sigma = 0.3):
        
    # Generate N*iterations gaussian samples of variance T/N
    times=(T/N)*np.arange(1,N+1)
    accroissements=np.random.normal(0,np.sqrt(T/N),(N, iterations))
    brownian=np.dot(np.tril(np.ones((N,N))),accroissements)
    
    # Arithmetic average call
    St=S0*np.exp(sigma*np.transpose(brownian)+(r-sigma**2/2)*times)

    arithmetic_sampling=np.exp(-r*T)*np.maximum(np.mean(St, axis = 1)-K,0)
    #arithmetic_call= [np.mean(arithmetic_sampling), (1.96*np.std(arithmetic_sampling)/np.sqrt(iterations))]
    # Geometric average call
    geometric_sampling=np.exp(-r*T)*np.maximum(S0*np.exp(sigma*np.mean(brownian,axis = 0)+(r-(sigma**2)/2)*(T/2))-K,0)
    
    difference = arithmetic_sampling-geometric_sampling
    std_ari = []
    std_diff = []
    for i in range(1, iterations, 100):
        std_ari.append((1.96*np.std(arithmetic_sampling[:i])/np.sqrt(i)))
        std_diff.append((1.96*np.std(difference[:i])/np.sqrt(i)))
    #std_diff = ((1.96*np.std(difference)/np.sqrt(iterations)))
    #std_ari = (1.96*np.std(arithmetic_sampling)/np.sqrt(iterations))
    return std_ari, std_diff




################# Timer ####################

def Monte_Carlo_Arithmetic(iterations = 1000000, K = 110, T = 1, r = 0.1,N = 150, S0=100, sigma = 0.3):
        
    # Generate N*iterations gaussian samples of variance T/N
    times=(T/N)*np.arange(1,N+1)
    accroissements=np.random.normal(0,np.sqrt(T/N),(N, iterations))
    brownian=np.dot(np.tril(np.ones((N,N))),accroissements)
    
    # Arithmetic average call
    St=S0*np.exp(sigma*np.transpose(brownian)+(r-sigma**2/2)*times)

    arithmetic_sampling=np.exp(-r*T)*np.maximum(np.mean(St, axis = 1)-K,0)
    # lower_list = []
    # upper_list = []
    # mean_list = []
    # for i in range(1, iterations, 100):
    #     std = np.std(arithmetic_sampling[:i])
    #     mean = np.mean(arithmetic_sampling[:i])
    #     mean_list.append(mean)
    #     upper_list.append(mean+(1.96*std/np.sqrt(i)))
    #     lower_list.append(mean-(1.96*std/np.sqrt(i)))
    mean_list = np.mean(arithmetic_sampling)
    std_ari = (1.96*np.std(arithmetic_sampling)/np.sqrt(iterations))
    return mean_list+std_ari, mean_list, mean_list-std_ari


    
def Monte_Carlo_diff(iterations = 100000, K = 110, T = 1, r = 0.1,N = 150, S0=100, sigma = 0.3):
        
    # Generate N*iterations gaussian samples of variance T/N
    times=(T/N)*np.arange(1,N+1)
    accroissements=np.random.normal(0,np.sqrt(T/N),(N, iterations))
    brownian=np.dot(np.tril(np.ones((N,N))),accroissements)
    
    # Arithmetic average call
    St=S0*np.exp(sigma*np.transpose(brownian)+(r-sigma**2/2)*times)

    arithmetic_sampling=np.exp(-r*T)*np.maximum(np.mean(St, axis = 1)-K,0)
    #arithmetic_call= [np.mean(arithmetic_sampling), (1.96*np.std(arithmetic_sampling)/np.sqrt(iterations))]
    # Geometric average call
    geometric_sampling=np.exp(-r*T)*np.maximum(S0*np.exp(sigma*np.mean(brownian,axis = 0)+(r-(sigma**2)/2)*(T/2))-K,0)
    
    difference = arithmetic_sampling-geometric_sampling
    # std_ari = []
    # std_diff = []
    # for i in range(1, iterations, 100):
    #     std_ari.append((1.96*np.std(arithmetic_sampling[:i])/np.sqrt(i)))
    #     std_diff.append((1.96*np.std(difference[:i])/np.sqrt(i)))
    std_diff = ((1.96*np.std(difference)/np.sqrt(iterations)))
    print(std_diff)


def timing_ari(it):
    start_time = time.time()
    Monte_Carlo_Arithmetic(iterations= it)
    elapsed_time = time.time() - start_time
    print('elapsed_time = ', elapsed_time)


def timing_diff(it):
    start_time = time.time()
    Monte_Carlo_diff(iterations= it)
    elapsed_time = time.time() - start_time
    print('elapsed_time = ', elapsed_time)














