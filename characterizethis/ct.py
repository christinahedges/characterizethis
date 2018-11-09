import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import G, m_p, k_B
import logging
import os
import stat
import datetime
from glob import glob
from . import PACKAGEDIR

log = logging.getLogger()

small = 3*u.earthRad

class CTFailure(Exception):
    pass

def check_data_on_import():
    fname = glob("{}/data/planets.csv".format(PACKAGEDIR))
    if len(fname) == 0:
        retrieve_online_data()
        fname = glob("{}/data/planets.csv".format(PACKAGEDIR))
    st = os.stat(fname[0])
    mtime = st.st_mtime
    # If database is out of date, get it again.
    if (datetime.datetime.now() - datetime.datetime.fromtimestamp(mtime) > datetime.timedelta(days=7)):
        log.warning('Database out of date. Redownloading...')
        retrieve_online_data()

def _fill_data(df, guess_masses=False):
    # Fill missing EqT
    nan = ~np.isfinite(df.pl_eqt)
    sep = np.asarray(df.pl_orbsmax)*u.AU
    rstar = (np.asarray(df.st_rad)*u.solRad).to(u.AU)
    temp = np.asarray(df.st_teff)*u.K
    df.loc[nan, ['pl_eqt']] = (temp[nan]*np.sqrt(rstar[nan]/(2*sep[nan])))

    # Fill in missing trandep
    nan = ~np.isfinite(df.pl_trandep)
    trandep = (np.asarray(df.pl_radj*u.jupiterRad.to(u.solRad))/np.asarray(df.st_rad))**2
    df.loc[nan, ['pl_trandep']] = trandep[nan]

    df['pl_density'] = (np.asarray(df.pl_bmassj)*u.jupiterMass.to(u.g))/((4/3) * np.pi * (np.asarray(df.pl_radj)*u.jupiterRad.to(u.cm))**3)

    # Fill missing Mass values with Weiss and Marcy 2014 values
    if guess_masses:
        nan = ~np.isfinite(df.pl_bmassj)
        higherrs = (df.pl_bmassjerr1/df.pl_bmassj) > 0.1
        low = (np.nan_to_num(np.asarray(df.pl_radj)*u.jupiterRad)).to(u.earthRad)  > 1.5*u.earthRad
        high = (np.nan_to_num(np.asarray(df.pl_radj)*u.jupiterRad)).to(u.earthRad)  < 4.*u.earthRad
        recalculate = np.all([low,high, np.any([nan, higherrs], axis=0)], axis=0)
        rade = (np.asarray(df.loc[recalculate,'pl_radj'])*u.jupiterRad).to(u.earthRad).value
        df.loc[recalculate, 'pl_bmassj'] = (((2.69* rade)**0.93)*u.earthMass).to(u.jupiterMass).value

    df['pl_rade'] = df.pl_radj * u.jupiterRad.to(u.earthRad)
    df['pl_bmasse'] = df.pl_bmassj * u.jupiterMass.to(u.earthMass)


    # Assume a fully hydrogen atmosphere that has 5 scale heights and is completely opaque in the WFC3 bandpass

    mu = np.zeros(len(df)) + 2
    mu[df.pl_density > 5] = 32
    mu[df.pl_rade < 1.3] = 32
    df['mu'] = mu
    g = G * (np.asarray(df.pl_bmassj)*u.jupiterMass)/(np.asarray(df.pl_radj)*u.jupiterRad)**2
    g = g.to(u.m/u.second**2)
    H = ((k_B*np.asarray(df.pl_eqt)*u.K)/(mu * m_p*g)).to(u.km)
    df['H'] = H

    # Find the change in transit depth due to the atmosphere
    delta = ((H*5) + ((np.asarray(df.pl_radj)*u.jupiterRad).to(u.km)))**2/((np.asarray(df.st_rad)*u.solRad).to(u.km))**2
    delta = delta.value - (((np.asarray(df.pl_radj)*u.jupiterRad).to(u.km))**2/((np.asarray(df.st_rad)*u.solRad).to(u.km))**2)
    df['delta'] = delta

    #One second scan and a 50 pixel spectrum
    exptime = 1
    scansize = 50

    star_fl = (5.5/0.15)*10.**(-0.4*(df.st_h-15))
    #star_fl[star_fl>33000] = 33000
    fl = df.delta*star_fl
    fl *= scansize * exptime

    df['snr'] = fl**0.5
    return df


def retrieve_online_data(guess_masses=False):
    NEXSCI_API = 'http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI'
    try:
        planets = pd.read_csv(NEXSCI_API + '?table=planets&select=pl_hostname,pl_letter,pl_disc,ra,dec,pl_trandep,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_tranflag,pl_trandur,pl_pnum,pl_k2flag,pl_kepflag,pl_facility,pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,pl_orblper,st_mass,st_masserr1,st_masserr2,st_rad,st_raderr1,st_raderr2,st_teff,st_tefferr1,st_tefferr2,st_optmag,st_j,st_h', comment='#')
        composite = pd.read_csv(NEXSCI_API + '?table=compositepars&select=fpl_hostname,fpl_letter,fpl_smax,fpl_smaxerr1,fpl_smaxerr2,fpl_radj,fpl_radjerr1,fpl_radjerr2,fpl_bmassj,fpl_bmassjerr1,fpl_bmassjerr2,fpl_eqt,fpl_orbper,fpl_orbpererr1,fpl_orbpererr2,fpl_eccen,fpl_eccenerr1,fpl_eccenerr2,', comment='#')
        composite.columns = ['pl_hostname','pl_letter','pl_orbsmax','pl_orbsmaxerr1','pl_orbsmaxerr2','pl_radj','pl_radjerr1','pl_radjerr2','pl_bmassj','pl_bmassjerr1','pl_bmassjerr2','pl_eqt','pl_orbper','pl_orbpererr1','pl_orbpererr2', 'pl_eccen','pl_eccenerr1','pl_eccenerr2']
    except:
        raise CTFailure("Couldn't obtain data from NEXSCI. Do you have an internet connection?")
    df = pd.merge(left=planets, right=composite, how='left', left_on=['pl_hostname', 'pl_letter'],
         right_on = ['pl_hostname', 'pl_letter'])

    df = _fill_data(df, guess_masses=guess_masses)
    df[df.pl_tranflag==1].to_csv("{}/data/planets.csv".format(PACKAGEDIR), index=False)
    return df


check_data_on_import()

def get_data():
    '''Obtain pandas dataframe of all exoplanet data
    '''
    return pd.read_csv("{}/data/planets.csv".format(PACKAGEDIR))

def plot_hist(ax=None, cap=3, keplerk2=False):
#    with plt.style.use('ggplot'):
    if ax is None:
        fig, ax = plt.subplots()

    df = pd.read_csv("{}/data/planets.csv".format(PACKAGEDIR))
    if keplerk2:
        k2 = df[(df.pl_k2flag==1)&(df.pl_facility=='K2')].reset_index(drop=True)
        kepler = df[(df.pl_kepflag==1)&(df.pl_facility=='Kepler')].reset_index(drop=True)
        list = [kepler, k2]
        labels = ['Kepler', 'K2']
    else:
        list = [df]
        labels = ['Confirmed Planets']

    for idx, li, lab in zip(range(len(list)), list, labels):
        fl = li.snr
        h = plt.hist((fl[np.isfinite(fl)]), np.arange(0, np.nanmax(li.snr), 0.5), density=True, alpha=0.7, label=lab)
        # Annotations
        ok = ((np.nan_to_num(np.asarray(li.pl_radj)) * u.jupiterRad).to(u.earthRad) < small) & np.isfinite(li.snr)
        i = 0
        df1 = li[ok].sort_values('snr', ascending=False)

        for i, n, l, x, y in zip(range(len(df)), df1.pl_hostname, df1.pl_letter, df1.snr, df1.pl_orbper):
            if i>= cap:
                break
            ann = ax.annotate("{}{}".format(n, l),
                              xy=(x, 0.+i*0.05), xycoords='data',
                              xytext=(x, 0.1+i*0.1), textcoords='data',
                              size=10, va="center", ha="center",
                              bbox=dict(boxstyle="round4", fc="C{}".format(idx), alpha=0.5),
                              arrowprops=dict(arrowstyle="simple",
                                              connectionstyle="arc3, rad=-{}".format(0),
                                              fc="C{}".format(idx)),
                              )



    ax.set_xlabel('SNR for 5H opaque atmosphere (HST WFC3, 1s exposure)', fontsize=13)
    ax.set_ylabel('Normalized Frequency', fontsize=13)
    ax.set_title('Observability of Exoplanet Atmospheres', fontsize=13)
    ax.legend(fontsize=12)
    #plt.savefig('{}/figures/K2observability_annotated.png'.format(PACKAGEDIR), dpi=300, bbox_inches='tight')
    return ax

def top(n=10, keplerk2=False, radius_limit=None):
    '''Return the top n exoplanets to characterize.
    '''
    df = pd.read_csv("{}/data/planets.csv".format(PACKAGEDIR))
    ok = np.ones(len(df), dtype=bool)
    if keplerk2:
        k2 = df[(df.pl_k2flag==1)&(df.pl_facility=='K2')].reset_index(drop=True)
        kepler = df[(df.pl_kepflag==1)&(df.pl_facility=='Kepler')].reset_index(drop=True)
        both =kepler.append(k2).reset_index(drop=True)
        ok = np.ones(len(both), dtype=bool)
        if radius_limit is not None:
            ok &= (np.asarray(both.pl_radj)*u.jupiterRad).to(u.earthRad) < radius_limit
        top = both[ok][['pl_hostname','pl_letter', 'st_h','pl_eqt','pl_rade','pl_bmasse','pl_density','pl_orbper','pl_trandep','delta','snr','pl_facility','pl_disc']].sort_values('snr', ascending=False)[0:n].reset_index(drop=True)

    else:
        ok = np.ones(len(df), dtype=bool)
        if radius_limit is not None:
            ok &= (np.asarray(df.pl_radj)*u.jupiterRad).to(u.earthRad) < radius_limit
        top = df[ok][['pl_hostname','pl_letter', 'ra', 'dec', 'st_h','pl_eqt','pl_rade','pl_bmasse','pl_density','pl_orbper','pl_trandep','delta','snr','pl_facility','pl_disc']].sort_values('snr', ascending=False)[0:n].reset_index(drop=True)

    top['delta'] *= 1e6
    top['pl_trandep'] *= 1e6
    return top
