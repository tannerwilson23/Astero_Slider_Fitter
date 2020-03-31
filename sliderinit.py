#
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button, RadioButtons
#
##import lightkurve as lk
#import scipy.ndimage
#
#import pymc3 as pm
#import theano.tensor as tt
#from exoplanet.gp import terms, GP
#
#import exoplanet as xo


import numpy as np
import matplotlib.pyplot as plt
Sg0 = np.exp(-23.6)
wg0 = np.exp(5.6)
S10 = np.exp(-24.3)
w10 = np.exp(5.5)
Q10 = np.exp(1.2)



nu = np.linspace(10,400,5000)
omega = nu

Sw = (np.sqrt(2/np.pi)*(Sg0*wg0**4)/((nu**2-wg0**2)**2+2*wg0**2*nu**2) + np.sqrt(2/np.pi)*(S10*w10**4)/((nu**2-w10**2)**2+(w10**2*nu**2)/Q10**2))


plt.loglog(nu,Sw)
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.title('Log log of amplitdue vs frequency (looks correct for solar like oscillations)')

plt.figure()
time = np.arange(5000)
ts = np.fft.ifft(np.abs(Sw))
inverse = np.abs(ts)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Inverse fourier transform of amplitude vs frequency')
plt.plot(time,inverse)


plt.show()


plt.figure()
plt.plot(time, ts.real, 'b-', time, ts.imag, 'r--')
plt.show()

#yerr = 0.005* np.ones_like(t)




#def run_gp_single(Sgv,wgv,S1v,w1v,Q1v):
#
#    with pm.Model() as model:
#
#
#        logSg = pm.Normal("logSg", mu=Sgv, sigma= 100.0, testval=Sgv)
#        logwg = pm.Normal("logwg", mu = wgv, sigma = 100.0, testval=wgv)
#        logS1 = pm.Normal("logS1", mu=S1v, sigma=100.0, testval=S1v)
#        logw1 = pm.Normal("logw1", mu =w1v, sigma = 100.0, testval=w1v)
#        logQ1 = pm.Normal("logQ1", mu=Q1v, sigma=100.0, testval=Q1v)
#
#
#        # Set up the kernel an GP
#        bg_kernel = terms.SHOTerm(log_S0=logSg, log_w0=logwg, Q=1.0 / np.sqrt(2))
#        star_kernel1 = terms.SHOTerm(log_S0=logS1, log_w0=logw1, log_Q=logQ1)
#        kernel = star_kernel1 + bg_kernel
#
#        gp = GP(kernel, t, yerr)
#        #gp_star1 = GP(star_kernel1, t, yerr ** 2 + pm.math.exp(logs2))
#        #gp_bg = GP(bg_kernel, t, yerr ** 2 + pm.math.exp(logs2))
#
#        gp_star1 = GP(star_kernel1, t, yerr)
#        gp_bg = GP(bg_kernel, t, yerr, mean = 0.)
#
#        # Condition the GP on the observations and add the marginal likelihood
#        # to the model
#        gp.marginal("gp")
#
#
#    with model:
#        val = gp.kernel.psd(omega)
#
#        psd_init = xo.eval_in_model(val)
#
#        bg_val = gp_bg.kernel.psd(omega)
#        star_val_1 = gp_star1.kernel.psd(omega)
#
#
#
#        bg_psd_init = xo.eval_in_model(bg_val)
#        star_1_psd_init = xo.eval_in_model(star_val_1)
#
#
#    #     print('done_init_plot')
#
#        map_soln = pm.find_MAP(start=model.test_point)
#
#
#        map_soln = model.test_point
#        print(model.test_point)
#
#        psd_final = xo.eval_in_model(gp.kernel.psd(omega),map_soln)
#
#
#        bg_psd_fin = xo.eval_in_model(bg_val,map_soln)
#        star_1_psd_fin = xo.eval_in_model(star_val_1,map_soln)
#
#
#
#
#
#    return psd_init, star_1_psd_init, bg_psd_init , psd_final, star_1_psd_fin, bg_psd_fin, map_soln
#
#


#a,b,c,d,e,f,g = run_gp_single(Sg0,wg0,S10,w10,Q10)
plt.show()
