#!/bin/python
import numpy as np
from astropy.convolution import convolve, Box1DKernel
from matplotlib import pylab as plt 
from scipy.special import erf


import argparse

import emcee
import triangle

lst_msp = ['J0348+0432','1738+0333','J1909-3744','J1910-5958','J1748-2446J','J1748-2446I']

def pdf_onepk(xx,mm,f0,m0,ms):
    sini0 = (((1 + xx/(mm-xx))**2. * (mm-xx)**(-1.))*f0)**(1./3.)
    cosi0 = (1-sini0**2.)**0.5

    fac = (sini0**2. / cosi0) / (3.*f0)
    t = fac*np.exp(-(m0 -mm)**2. / (2.*ms**2.))
    t = np.nan_to_num(t)
    t = t.sum(0)
    t = convolve(t,Box1DKernel(100))
    
    return t/t.sum()


def normal(x,mu,s):
    norm = (1./(s*(2.*np.pi)**0.5))*np.exp(-((mu-x)**2.)/(2.*s**2.))
    return norm/norm.sum()


def gauss(x,theta):
    mu,s = theta
    return normal(x,mu,s)
    mu,s,l = theta




def normal_prior(theta):
    mu,s = theta
    if 1.0 < mu < 2.5 and 0.001 < s < 1.0:
        return 0.0
    return -np.inf



def normal_exp(x,theta):
    mu,s,l = theta
    e = (l/2.)*np.exp((l/2.)*(2.*mu + l*s**2. - 2*x))
    er = erf((mu + l*s**2. -x)/((2.**0.5) *s))
    ne = e*(1.-er)
    return ne/ne.sum()
    
def normal_exp_prior(theta):
    mu,s ,l= theta
    if 1.0 < mu < 2.5 and 0.001 < s < 1.0 and 1.0 < l < 10.0:
        return 0.0
    return -np.inf




def bimodal(x,theta):
    mu1,s1,mu2,s2,r = theta
    sn1 = normal(x,mu1,s1)
    sn2 = normal(x,mu2,s2)
    sn_b = (1-r)*sn1 + r*sn2
    return sn_b/sn_b.sum()


def bimodal_prior(theta):
    mu1,s1,mu2,s2,r = theta
    if 1.0 < mu1 < 1.6 and 0.005 < s1 < 0.5 and  1.0 < mu2 < 2.5 and 0.005 < s2 < 0.5 and 0.000 < r < 1.000:
        return 0.0
    return -np.inf



def bimodal_cut(x,theta):
    mu1,s1,mu2,s2,r,mmax = theta
    sn1 = normal(x,mu1,s1)
    sn2 = normal(x,mu2,s2)
    sn_b = (1-r)*sn1 + r*sn2
    for i in range(len(x)):
        if x[i] >= mmax:
            sn_b[i] = 0

    return sn_b/sn_b.sum()


def bimodal_cut_prior(theta):
    mu1,s1,mu2,s2,r,mmax = theta
    if 1.0 < mu1 < 1.6 and 0.005 < s1 < 0.5 and  1.0 < mu2 < 2.5 and 0.005 < s2 < 0.5 and 0.000 < r < 1.000 and 1.700 < mmax < 3.000:
        return 0.0
    return -np.inf



def skewed_norm(x,theta):
    mu,s,a = theta 
    f = (1./(s*(2.*np.pi)**0.5))*np.exp(-((x-mu)**2.)/(2.*s**2.))
    F = (1./2.)*(1 + erf(a*(x-mu)/(s*2.**0.5)))
    sn = (2./s)*f*F
    return sn/sn.sum()


def skewed_norm_prior(theta):
    mu,s,a = theta
    if 1.0 < mu < 1.5 and 0.005 < s < 1.0 and  -10.0 < a < 30.0:
        return 0.0
    return -np.inf



def skewed_norm_cut(x,theta):
    mu,s,a,mmax = theta
    f = (1./(s*(2.*np.pi)**0.5))*np.exp(-((x-mu)**2.)/(2.*s**2.))
    F = (1./2.)*(1 + erf(a*(x-mu)/(s*2.**0.5)))
    sn = (2./s)*f*F
    for i in range(len(x)):
        if x[i] >= mmax:
            sn[i] = 0
    return sn/sn.sum()

def skewed_norm_cut_prior(theta):
    mu,s,a,mmax = theta
    if 1.0 < mu < 2.5 and 0.005 < s < 1.0 and  -10.0 < a < 30.0 and  1.7 < mmax < 3.0:
        return 0.0
    return -np.inf



def read_pdfs(x,msp_names):
    pdfs = np.zeros([len(msp_names),len(x)])
    i = 0
    for n in msp_names:
        name = '%.11s.pdf' % n
        ln = np.genfromtxt(open(name))
        pdfs[i,:] = ln
        i += 1
    return pdfs 



def lnlike(theta,x,pdfs,prob):
    ln1 = prob(x,theta)
    return np.log((ln1*pdfs).sum(1)).sum()






def lnprob(theta, x, y,like, prior):
    lp = prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return (lp + lnlike(theta, x, y, prob=like))



def plot_dist(samples,x,fnc=bimodal,cum=False):
    for theta  in samples[np.random.randint(len(samples),size=100)]:
        if cum:
            plt.plot(x,np.cumsum(fnc(x,theta)),color='k',alpha=0.1)
        else:
            plt.plot(x,fnc(x,theta),color='k',alpha=0.1)

    truths = [np.percentile(sam,[50]) for sam in samples.T]
    if cum:
        plt.plot(x,np.cumsum(fnc(x,truths)),color='b')
    else:
        plt.plot(x,fnc(x,truths),color='b')






def psrpdfs(x,m,filegauss,fileonepk=None,calculate=True,verbose=True):
    msp_names = []
    if fileonepk:
        d1pk = np.genfromtxt(open(fileonepk),dtype=[('name','|S11'),('f0','f8'),('m','f8'),('ms','f8')])
        if verbose:
                print
                print
                print
                print "Loading data for pulsars with 1 pK parameter from file {0}".format(fileonepk)
                print
                print
                print 
        for n,f0,m0,ms in d1pk:
               if calculate:
                   xx,mm = np.meshgrid(x,m)
                   t = pdf_onepk(xx,mm,f0,m0,ms)
                   np.savetxt('%.11s.pdf' % n,t)
                   if verbose:
                        print "Calculating Mass PDF for pulsar {0}".format(n)
                       

               msp_names = np.append(msp_names,n) 

    dgauss = np.genfromtxt(open(filegauss),dtype=[('name','|S11'),('m','f8'),('ms','f8')])
    if verbose:
        print
        print
        print
        print "Loading data for pulsars with 1 pK parameter from file {0}".format(fileonepk)
        print
        print
        print 
    for n,m0,ms in dgauss:
        if calculate:
            t = normal(x,m0,ms)
            np.savetxt('%.11s.pdf' % n,t)
            if verbose:
                        print "Saving results to  %.11s.pdf" % n
        msp_names = np.append(msp_names,n)
    pdfs = read_pdfs(x,msp_names)
    return pdfs, msp_names



def P_Parser():
    parser = argparse.ArgumentParser(prog='pymass.py', 
                                     description='Script for determining the distribution of NS masses using MCMC')

    parser.add_argument('-d1', '--data_meas',type=str, default='msps3.txt',
                        help='File containing NS mass measurements with gaussian uncertainties. The file must have 3 columns displaying the name, mean, and standard deviation (in Solar Masses)')
    parser.add_argument('-d2', '--data_1pk',type=str, default='onepk.data',
                        help='File containing the constraints on the total mass for systems that have only 1pk parameter determined \n The file must have 4 columns displaying the name, mass function, mean, and standard deviation on the Total Mass  (all in Solar Masses)')

    parser.add_argument('-c', '--calc',type=bool, default=False,help='Determines whether PDFs will be calculated or not')
    parser.add_argument('-v', '--verbose',type=bool, default=True,help='Verbosity')
    parser.add_argument('-n', '--ntrials',type=int, default=2000,help='Determines the number of MCMC iterations')
    parser.add_argument('-w', '--nwalkers',type=int, default=200,help='The number of MCMC walkers')
    parser.add_argument('-o', '--output',type=str, default='chain.dat',help='Name of the output file')
    parser.add_argument('--threads',type=int, default=4,help='Number of CPU threads to use')
    parser.add_argument('-t', '--thin',type=int, default=10,help='MCMC thinning factor')
    parser.add_argument('-f', '--lnprob',type=str, default='bimodal',help='Distribution to fit',
                        choices=['normal','bimodal','bimodal_cut','skewed_normal','skewed_normal_cut','normal_exp'])
    parser.add_argument('-i', '--init',type=str, default="1.4,0.1,1.96,0.1,0.5",help='Initial possition for the walkers')   
    parser.add_argument('-s', '--spread',type=float, default=1e-2,
                        help='Factor that regulates the scatter of the walkers around the initial parameters')  
    parser.add_argument('--plots',type=bool, default=True,
                        help='Produce diagnostic plots at the end of the simulation') 
    return parser


likelihoods = {'normal':gauss,
               'bimodal':bimodal,
               'bimodal_cut':bimodal_cut,
               'skewed_normal':skewed_norm, 
               'skewed_norm_cut':skewed_norm_cut, 
               'normal_exp': normal_exp}

priors = {'normal':normal_prior,
          'bimodal':bimodal_prior,
          'bimodal_cut':bimodal_cut_prior,
          'skewed_normal':skewed_norm_prior, 
          'skewed_norm_cut':skewed_norm_cut_prior,
          'normal_exp': normal_exp_prior}

labels={'normal': ["$\mu$","$\sigma$"],
        'bimodal': ["$\mu_1$","$\sigma_1$","$\mu_2$", "$\sigma_2$","$r$"],
        'bimodal_cut': ["$\mu_1$","$\sigma_1$","$\mu_2$","$\sigma_2$","$r$","$m_{max}$"],
        'skewed_normal': ["$\mu$","$\sigma$", "$a$"],
        'skewed_normal_cut': ["$\mu$","$\sigma$", "$a$","$m_{max}$"],
        'normal_exp': ["$\mu$","$\sigma$", "$\lambda$"]}


if __name__ == "__main__":

    parser = P_Parser()

    s_args = parser.parse_args()
    x = np.arange(0.3,3.03,0.0001)
    m = np.arange(0.3,4.001,0.001)

 
    plt.ion()


    pdfs, msp_names = psrpdfs(x,m,s_args.data_meas,s_args.data_1pk,calculate=s_args.calc,verbose=s_args.verbose)
    print "Runing MCMC...."
    print 
    print "Number of Walkers: {0}\nNumber of Samples per Walker: {1}".format(s_args.nwalkers,s_args.ntrials)  

    theta = map(float,s_args.init.split())
    ndim = len(theta)
    pos = [theta + s_args.spread*np.random.randn(ndim) for i in range(s_args.nwalkers)]

    likelihood = likelihoods[s_args.lnprob]
    prior = priors[s_args.lnprob]
  
    sampler = emcee.EnsembleSampler(s_args.nwalkers,ndim,lnprob,args=(x,pdfs,likelihood,prior),threads=s_args.threads)
 
    f = open(s_args.output, "w")
    f.close()

    i = 0
    for result in sampler.sample(pos, iterations=s_args.ntrials, storechain=False,thin=s_args.thin):
        i += 1
        res = result
        print "Iteration {0}".format(i)
        position = result[0]
        f = open(s_args.output, "a")
        for k in range(position.shape[0]):
            f.write("{0}\n".format(" ".join(map(str,position[k]))))
        f.close()
    if s_args.plots:
        samples = np.genfromtxt(open(s_args.output))
        truths = [np.percentile(sam,[50]) for sam in samples.T]
        label = labels[s_args.lnprob]
        plot_dist(samples,x,fnc=likelihoods[s_args.lnprob],cum=False)
        plt.show()

        fig = triangle.corner(samples[:,:],labels=label,quantiles=[0.01,0.023,0.159,0.5,0.841,0.977,0.99],plot_datapoints=False,truths=truths)
        plt.show()
        fig.savefig('corner_plot.pdf')

