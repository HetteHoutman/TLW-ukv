# General Thermodynamics functions
# Last updated 29/10/2020 by Peter Clark
#
import numpy as np
import thermodynamics_constants as tc

def esat(T):
    '''
     Input  : Temperature (K)
     Output : Vapour pressure over water (Pa)
    '''

    T_ref=tc.freeze_pt
    T_ref2=243.04-T_ref # Bolton uses 243.5
    es_Tref=610.94      # Bolton uses 611.2
    const=17.625        # Bolton uses 17.67
    res = np.clip(es_Tref * np.exp(const * (T - T_ref)/(T+T_ref2)), 
                  1e-18, None)
    return res

def esat_ice(T):
    '''
    Magnus Teten
    Murray (1967)
     Input  : Temperature (K)
     Output : Vapour pressure over ice (Pa)
    '''

    T_ref=tc.triple_pt
    T_ref2= -7.66
    es_Tref=610.87
    const= 21.8745584
    res = es_Tref * np.exp(const * (T - T_ref)/(T+T_ref2))
    return res

def inv_esat(es):
    '''
     Input  : Vapour pressure over water (Pa)
     Output : Temperature (K)
    '''
    T_ref=tc.freeze_pt
    T_ref2=243.04-T_ref
#
# This is how constants are derived:
#    es_Tref=610.94
#    const=17.625
#    ln_es_Tref = np.log(es_Tref)
#    C1 = const * T_ref - ln_es_Tref * T_ref2
#    C2 = const + ln_es_Tref
#
    ln_es_Tref = 6.41499875468
    C1 = 5007.4243625
    C2 = 24.039998754
    ln_es =  np.log(es)
    T = (T_ref2 * ln_es +  C1) / (C2 - ln_es)
    return T

def inv_esat_ice(es):
    '''
    Magnus Teten
    Murray (1967)
     Input  : vapour pressure over ice (Pa)
     Output : Temperature (K)
    '''
    T_ref=tc.tc.triple_pt
    T_ref2=-7.66
#
# This is how constants are derived:
#    es_Tref = 610.87
#    const = 21.8745584
#    ln_es_Tref = np.log(es_Tref)
#    C1 = const * T_ref - ln_es_Tref * T_ref2
#    C2 = const + ln_es_Tref
#
    ln_es_Tref = 6.4148841705762614
    C1 = 6024.3923852906155
    C2 = 28.289442570576263

    ln_es =  np.log(es)
    T = (T_ref2 * ln_es +  C1) / (C2 - ln_es)
    return T

def esat_over_Tkappa(T):
    '''
    From Bolton 1980
    Computes es/T^(1/kappa) (es in Pa)
    '''
    T_ref = tc.freeze_pt
    T_ref2=217.8-T_ref
    es_over_Tkappa_Tref = 1.7743E-6
    const=12.992
    res = es_over_Tkappa_Tref * np.exp(const * (T - T_ref)/(T+T_ref2))
    return res


def temperature(theta, p):
    '''
    Input:   Potential temperature of dry air (K) numpy array
             Pressure (Pa) numpy array
    Output : Temperature (K) numpy array
    '''
    T = theta*(p/tc.p_ref_um)**tc.kappa
    return T

def potential_temperature(T, p):
    '''
    Input:   Temperature (K) numpy array
             Pressure (Pa) numpy array
    Output : Potential temperature of dry air (K) numpy array
    '''
    theta=T*(tc.p_ref_um/p)**tc.kappa
    return theta

def moist_potential_temperature(T, p, m):
    '''
     Input:   Temperature (K) numpy array
              pressure (Pa) numpy array
              m mixing ratio(kg/kg) numpy array
     Output : Potential temperature of moist air (K) numpy array
    '''
    theta=T*(tc.p_ref_um/p)**(tc.kappa*(1-tc.kappa_v * m))
    return theta

def q_to_mix(q):
    '''
     Input  : Specific humidity (kg/kg)
     Output : Mixing Ratio(kg/kg)
    '''
    qc=np.clip(q,1e-18,0.999)
    m = qc / (1-qc)
    return m

def mix_to_q(m):
    '''
     Input  : Mixing Ratio(kg/kg)
     Output : Specific humidity (kg/kg)
    '''
    q = m / (1+m)
    return q

def q_p_to_e(q, p):
    '''
     Input  : Specific humidity (kg/kg)
            : pressure (Pa)
     Output : Vapour pressure (Pa)
    '''
    e = np.clip(q * p / (q * (1-tc.epsilon) + tc.epsilon ),1.0E-18, None)
    return e

def e_p_to_q(e, p):
    '''
     Input  : Vapour pressure (Pa)
            : Pressure (Pa)
     Output : Specific humidity (kg/kg)
    '''
    q = tc.epsilon * e / (p - (1-tc.epsilon) * e)
    np.clip(q,0,0.999,out = q)
    return q

def T_LCL_TD(T, TD):
    '''
    From Bolton 1980
     Input :  Temperature (K) numpy array
              Dew point (K) numpy array
     Output : temperature at lifting condensation level (K)

    '''
    T_ref = 56.0
    const = 800.0
    res = 1 / (1 / (TD-T_ref)+np.log(T/TD) / const) + T_ref
    return res

def T_LCL_e(T, e):
    '''
    From Bolton 1980
     Input :  Temperature (K) numpy array
              Vapour pressure (Pa) numpy array
     Output : temperature at lifting condensation level (K)
    '''
    T_ref = 55.0
    const = 2840.0
    C2 = -0.199829814012
    C3 = 3.5
    res = const / (C3 * np.log(T) - np.log(e) + C2) + T_ref
    return res

def T_LCL_RH(T, RH):
    '''
    From Bolton 1980
     Input :  Temperature (K) numpy array
              Relative humidity (%)
     Output : temperature at lifting condensation level (K)
    '''
    T_ref = 55.0
    const = 2840.0
    res = 1 / (1 / (T-T_ref) - np.log(RH/100) / const) + T_ref
    return res

def latheat(T, sublim=0, Model=0, focwil_T=[]) :
    '''
     Input :  Temperature (K) numpy array
     Output : Latent heat of condensation(K) numpy array
     Options :sublim = 1 return Latent heat of sublimation
              Model = 1 use UM fixed values
              focwil_T = nonempty array or single element
                         use linear ramp in ice fraction from
                         focwil_T to freezing.
    '''

#    T=np.array(T)
    if Model == 1 :
        el0 = tc.cvap_water
        elm = tc.cfus_water

        el=np.copy(T)
        el[:]=el0
        if (sublim == 1) or (np.size(focwil_T) == 1) :

            ix = np.where(T < tc.freeze_pt)[0]
#            print ix
            if np.size(ix) > 0 :
                TC=T[ix]-tc.freeze_pt
                if np.size(focwil_T) == 1 :
                    focwil=TC/focwil_T
                    focwil[np.where(focwil > 1)] = 1
                    el[ix]=el[ix]+elm*focwil
                else :
                    el[ix]=el[ix]+elm
    else :
# From Pruppacher and Klett
        el0 = 2.5e6
        p1 = 0.167e0
        pg = 3.67e-4
        lm0 = 333584.0
        lm1 = 2029.97
        lm2 = -10.4638
        el = el0*((tc.freeze_pt/T)**(p1+pg*T))

        if (sublim == 1) or (np.size(focwil_T) == 1) :
            ix = np.where(T < tc.freeze_pt)[0]
            if np.size(ix) > 0 :
                TC=T[ix]-tc.freeze_pt
                elm=lm0+lm1*TC+lm2*TC*TC
                if np.size(focwil_T) == 1 :
                    focwil=TC/focwil_T
                    focwil[np.where(focwil > 1)] = 1
                    el[ix]=el[ix]+elm*focwil
                else :
                    el[ix]=el[ix]+elm

    return el

def dewpoint(T, p, q) :
    '''
     Input:   Temperature (K) numpy array
              pressure (Pa) numpy array
              q specific humidity (kg/kg) numpy array
     Output : Dew-point temperature (K) numpy array
    '''
#    T=np.array(T)
#    q=np.array(q)
#    p=np.array(p)

    rv=tc.gas_const_air/tc.epsilon
    TD=np.copy(T)

#  calculate vapour pressure, and from that the dewpoint in kelvins.
    v_pres = q * p/( tc.epsilon + q)

#    print v_pres

    v_pres[np.where(v_pres <= 0.0)] = 1e-10

    TD = inv_esat(v_pres)

    i = np.where(TD > T)[0]
    if np.size(i) > 0 :
        TD[i] = T[i]
    return TD

def qsat(T, p) :
    '''
     Input:   Temperature (K) numpy array
              pressure (Pa) numpy array
     Output : saturation specific humidity (kg/kg) over water numpy array
    '''
    es = esat(T)
    fsubw = 1.0 + 1.0E-8 * p * (4.5 + 6.0E-4 * (T - tc.freeze_pt) * (T - tc.freeze_pt) )
    es = es * fsubw
    qs = e_p_to_q(es, p)
    return qs

def dqsatbydT(T, p) :
    alpha = tc.epsilon * tc.cvap_water * qsat(T, p) / \
            (tc.gas_const_air * T * T)
    return alpha

def equiv_potential_temperature(T, p, q):
    '''
    From Bolton 1980
     Input:   Temperature (K) numpy array
              pressure (Pa) numpy array
              q specific humidity (kg/kg) numpy array
     Output : Fast estimate of equivalent potential temperature (K) numpy array

    '''
    C1 = 3.376E3
    C2 = 2.54
    C3 = 0.81
    e = q_p_to_e(q, p)
    m = q_to_mix(q)
    T_LCL = T_LCL_e(T, e)
    theta= moist_potential_temperature(T , p, m)

    theta_e = theta * \
      np.exp((C1/T_LCL-C2) * m * (1 + C3 * m) )
    return theta_e

def equiv_potential_temperature_accurate(T, p, q) :
    '''
    From Bolton 1980
     Input:   Temperature (K) numpy array
              pressure (Pa) numpy array
              q specific humidity (kg/kg) numpy array
     Output : Accurate equivalent potential temperature (K) numpy array

    '''
    C1 = 3.036E3
    C2 = 1.78
    C3 = 0.448
    e = q_p_to_e(q, p)
    m = q_to_mix(q)
    T_LCL = T_LCL_e(T, e)
    theta_DL = T * ( tc.p_ref_um/(p - e) )**tc.kappa * (T / T_LCL)**(tc.kappa_v*m)

    theta_e = theta_DL * \
      np.exp((C1/T_LCL-C2) * m * (1 + C3 * m) )

    return theta_e

def wet_bulb_potential_temperature(T, p, q):
    '''
     From Davies-Jones 2007
     Input:   Temperature (K) numpy array
              pressure (Pa) numpy array
              q specific humidity (kg/kg) numpy array
     Output : Wet-bulb potential temperature (K) numpy array
    '''
    A = 2675.0
    T_ref=tc.freeze_pt
    b = 243.04
    T_ref2 = b-T_ref      # Bolton uses 243.5
    es_Tref = 610.94      # Bolton uses 611.2
                          # a
    const = 17.625        # Bolton uses 17.67
    Tref3 = 45.114 + T_ref
    Tref4 = 43.380 + T_ref
    C1 = -51.489
    C2 = 0.6069
    C3 = -0.01005

    th_E = equiv_potential_temperature_accurate(T, p, q)
    ir1 = np.where( th_E <= 257  )
    ir2 = np.where((257  <  th_E) & (th_E < 377) )
    ir3 = np.where((377  <= th_E) & (th_E < 674) )
    ir4 = np.where( 674  <= th_E )
    th_W = np.copy(th_E)

    if np.size(ir1) !=  0:
        Ars = A * q_to_mix(qsat(th_E[ir1], p[ir1]))
        dlnesdT = const * b / (th_E[ir1] + T_ref2)**2
        th_W[ir1] = th_E[ir1] -  Ars/(1+Ars*dlnesdT)
    if np.size(ir2) !=  0:
        th_W[ir2] = Tref3 + C1 * (T_ref/th_E[ir2])**tc.rk
    if np.size(ir3) !=  0:
        th_W[ir3] = Tref4 + C1 * (T_ref/th_E[ir3])**tc.rk \
          + C2 * (th_E[ir3]/T_ref)**tc.rk \
          + C3 * (th_E[ir3]/T_ref)**(2*tc.rk)
    if np.size(ir4) !=  0:
        th_W[ir4] = np.nan
    return th_W

def wet_bulb_temperature(T, p, q):
    '''
     From Davies-Jones 2007
     Input:   Temperature (K) numpy array
              pressure (Pa) numpy array
              q specific humidity (kg/kg) numpy array
     Output : Wet-bulb temperature (K) numpy array
    '''
    T_ref=tc.freeze_pt
    A = 2675.0
    const = 17.625        # Bolton uses 17.67
    b = 243.04
    T_ref2 = b-T_ref # Bolton uses 243.5
    pi = (p/tc.p_ref_um)**tc.kappa

    pi2 = pi * pi

    D_pi=1/(0.1859 * (p/tc.p_ref_um)+0.6512)

    k1_pi = 137.81 * pi -38.5 * pi2 - 53.737

    k2_pi = 56.831 * pi -4.392 * pi2 - 0.384

#    print D_pi, k1_pi, k2_pi

    th_E = equiv_potential_temperature_accurate(T, p, q)
    TE=th_E/pi
    inv_TE = (tc.freeze_pt/TE)**tc.rk

    ir1 = np.where(  inv_TE > D_pi )
    ir2 = np.where( (1 <= inv_TE) & (inv_TE <= D_pi) )
    ir3 = np.where( (0.4 <= inv_TE) & (inv_TE < 1) )
    ir4 = np.where(  inv_TE < 0.4 )
    TW = np.copy(TE)
    if np.size(ir1) !=  0:
        Ars = A * q_to_mix(qsat(TE[ir1], p))
        dlnesdT = const * b / (TE[ir1] + T_ref2)**2
#        print '1:', TE, Ars, dlnesdT
        TW[ir1] = TE[ir1] -  Ars/(1+Ars*dlnesdT)
    if np.size(ir2) !=  0:
#        print '2:', TE, inv_TE
        TW[ir2] = tc.freeze_pt + k1_pi[ir2] - k2_pi[ir2] * inv_TE[ir2]
    if np.size(ir3) !=  0:
#        print '3:', TE, inv_TE
        TW[ir3] = tc.freeze_pt + (k1_pi[ir3] - 1.21) - (k2_pi[ir3] - 1.21) * inv_TE[ir3]
    if np.size(ir4) !=  0:
#        print '4:', TE, inv_TE
        TW[ir4] = tc.freeze_pt + (k1_pi[ir4] - 2.66) - (k2_pi[ir4] - 1.21) * inv_TE[ir4] \
          +0.58 / inv_TE[ir4]
    return TW
