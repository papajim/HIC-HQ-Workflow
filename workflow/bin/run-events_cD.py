#!/usr/bin/env python3
"""
run full space-time HQ evolution in heavy-ion collisions (seperated in several stages):
    -- initial condition (trento + HQ_sample)
    -- hydro (2+1D VishNew)
    -- HQ transport (Langevin)
    -- HQ hardonization (fragmentation + recombination)
    -- light hadron particalization (osu-sampler)
    -- hadronic afterburner (UrQMD)
"""

import itertools 
import sys, os
import subprocess
import numpy as np
import math
import h5py
import shutil
from scipy.interpolate import interp1d

#sys.path.insert(1, 'lib/python3.5/site-packages')
import freestream
import frzout

share=os.environ.get('XDG_DATA_HOME')

def run_cmd(*args, **kwargs):
    print(*args, flush=True)
    subprocess.check_call(list(itertools.chain.from_iterable(a.split() for a in args)), **kwargs)


def read_text_file(fileName):
    with open(fileName, 'r') as f:
        result = [l.split() for l in f if not l.startswith('#')]

    ID, charge, mass, px, py, pz, y, eta, ipT, wt = (
        np.array(col, dtype=dtype) for (col, dtype) in 
        zip(zip(*result), (2*[int]+8*[float]))
    )
    return ID, charge, px, py, pz, y, eta, ipT


def read_oscar_file(fileName):
    try:
        with open(fileName, 'r') as f:
            result = [l.replace('D','E').split() for l in f if len(l.split()) == 19]

        NUM, ID, px, py, pz, p0, mass, rx, ry, rz, r0, temp, c_vx, c_vy, c_vz, ed, ip0, ipT, wt = (
            np.array(col, dtype=dtype) for (col, dtype) in 
            zip(zip(*result), (2*[int]+17*[float]))
        )
    except ValueError as e:
        handle = str(e).split()[-1]
        with open(fileName, 'r') as f:
            result = [l.replace('D', 'E').replace(handle, ' 0 ').split() for l in f if len(l.split()) == 19]
            NUM, ID, px, py, pz, p0, mass, rx, ry, rz, r0, temp, c_vx, c_vy, c_vz, ed, ip0, ipT, wt = (
                np.array(col, dtype=dtype) for (col, dtype) in 
                zip(zip(*result), (2*[int]+17*[float]))
            )

    return ID, px, py, pz, p0, ipT


def save_fs_history(ic, event_size, grid_step, tau_fs, xi, grid_max, steps=5, coarse=False):
    f = h5py.File('FreeStream.h5', 'w')
    dxy = grid_step*(coarse or 1)
    ls = math.ceil(event_size/dxy)
    n = 2*ls + 1
    NX, NY = ic.shape
    # roll ic by index 1 to match hydro
    ix = np.roll(np.roll(ic, shift=-1, axis=0), shift=-1, axis=1)
    tau0 = tau_fs * xi
    taus = np.linspace(tau0, tau_fs, steps)
    dtau = taus[1] - taus[0]
    gp = f.create_group('Event')
    gp.attrs.create('XL', [-ls])
    gp.attrs.create('XH', [ls])
    gp.attrs.create('YL', [-ls])
    gp.attrs.create('YH', [ls])
    gp.attrs.create('Tau0', [tau0])
    gp.attrs.create('dTau', [dtau])
    gp.attrs.create('DX', [dxy])
    gp.attrs.create('DY', [dxy])
    gp.attrs.create('NTau', [steps])
    gp.attrs.create('OutputViscousFlag', [1])

    for itau, tau in enumerate(taus):
        print(tau)
        frame = gp.create_group('Frame_{:04d}'.format(itau))
        fs = freestream.FreeStreamer(ic, grid_max, tau)
        for fmt, data, arglist in [
            ('e', fs.energy_density, [()]),
            ('V{}', fs.flow_velocity, [(1,), (2,)]),
            ('Pi{}{}', fs.shear_tensor, [(0, 0), (0, 1), (0, 2), (1,1), (1,2), (2,2)])
        ]:
            for a in arglist:
                X = data(*a).T  # to get the correct x-y with vishnew?? (need to check this)
                if fmt == 'V{}':
                    X = X/data(0).T
                if coarse:
                    X = X[::coarse, ::coarse]
                diff = X.shape[0] - n
                start = int(abs(diff)/2)

                if diff > 0:
                    # original grid is larger -> cut out middle square
                    s = slice(start, start + n)
                    X = X[s, s]
                elif diff < 0:
                    # original grid is smaller -> create new array and place original grid in middle
                    Xn = np.zeros((n, n))
                    s = slice(start, start + X.shape[0])
                    Xn[s, s] = X
                    X = Xn

                if fmt == 'V{}':
                    Comp = {1:'x', 2:'y'}
                    frame.create_dataset(fmt.format(Comp[a[0]]), data=X)
                if fmt == 'e':
                    frame.create_dataset(fmt.format(*a), data=X)
                    frame.create_dataset('P', data=X/3.)
                    frame.create_dataset('BulkPi', data=X*0.)
                    prefactor =  1.0/15.62687/5.068**3
                    frame.create_dataset('Temp', data=(X*prefactor)**0.25)
                    s = (X + frame['P'].value)/(frame['Temp'].value + 1e-14)
                    frame.create_dataset('s', data=s)
                if fmt == 'Pi{}{}':
                    frame.create_dataset(fmt.format(*a), data=X)
        pi33 = -(frame['Pi00'].value + frame['Pi11'].value + frame['Pi22'].value)
        frame.create_dataset('Pi33', data=pi33)
        pi3z = np.zeros_like(pi33)
        frame.create_dataset('Pi03', data=pi3z)
        frame.create_dataset('Pi13', data=pi3z)
        frame.create_dataset('Pi23', data=pi3z)
    f.close()



def run_initial(collision_sys, nevents, grid_step, grid_max, config):
    proj = collision_sys[:2]
    targ = collision_sys[2:4]
    run_cmd(
        'trento {} {}'.format(proj, targ), str(nevents),
        '--grid-step {} --grid-max {}'.format(grid_step, grid_max),
        '--output initial.hdf5',
        config.get('trento_args', '')
    )

def run_hydro(tau_fs, dtau, grid_step, Nhalf, config):
    run_cmd(
        'vishnew initialuread=1 iein=0',
        't0={} dt={} dxy={} nls={}'.format(tau_fs, dtau, grid_step, Nhalf),

    )


def run_HQsample():
    run_cmd('HQ_sample HQ_sample.conf')


def run_qhat(args):
    run_cmd('qhat_pQCD', str(args))
    args_list = args.split()
    if args_list[0] == '--mass':
        mass = float(args_list[1])
    elif args_list[2] == '--mass':
        mass = float(args_list[3])

    if np.abs(mass - 1.3) < 0.2:
        flavor = 'charm'
    elif np.abs(mass - 4.2) < 0.2:
        flavor = 'bottom'


    gridE = 101
    gridT = 31
    qhat_Qq2Qq = h5py.File('qhat_Qq2Qq.hdf5', 'r')['Qhat-tab']
    qhat_Qg2Qg = h5py.File('qhat_Qg2Qg.hdf5', 'r')['Qhat-tab']
    rate_Qq2Qq = h5py.File('rQq2Qq.hdf5', 'r')['Rates-tab']
    rate_Qg2Qg = h5py.File('rQg2Qg.hdf5', 'r')['Rates-tab']

    GeV_to_fmInv = 5.068
    fmInv_to_GeV = 0.1973

    E = np.linspace(mass*1.01, 140., gridE)
    temp = np.linspace(0.15, 0.75, gridT)

    kperp = qhat_Qq2Qq[2,:,:] + qhat_Qg2Qg[2,:,:]
    qhat_over_T3 = 2 * kperp / temp**3

    tempM, EM = np.meshgrid(temp, E)

    res = []
    for i in range(len(kperp)):
        for j in range(len(kperp[i])):
            dum = np.array([tempM[i,j], EM[i,j], qhat_over_T3[i,j]])
            res.append(dum)

    res = np.array(res)
    np.savetxt('gamma-table_{}.dat'.format(flavor), res, header = 'temp energy qhat_over_T3', fmt='%10.6f')




def run_diffusion(args):
    os.environ['ftn10'] = '%s/dNg_over_dt_cD6.dat'%share
    os.environ['ftn20'] = 'HQ_AAcY.dat'
    os.environ['ftn30'] = 'initial_HQ.dat'
    run_cmd('diffusion', str(args))

def run_fragPLUSrecomb():
    os.environ['ftn20'] = 'Dmeson_AAcY.dat'
    child1 = 'cat HQ_AAcY.dat'
    p1 = subprocess.Popen(child1.split(), stdout=subprocess.PIPE)
    p2 = subprocess.Popen('fragPLUSrecomb', stdin=p1.stdout)
    p1.stdout.close()
    output = p2.communicate()[0]


def participant_plane_angle(ed, grid_max):
    grid_size = ed.shape
    x_init = np.linspace(-grid_max, grid_max, grid_size[0])
    y_init = np.linspace(-grid_max, grid_max, grid_size[1])
    rx0, ry0 = np.meshgrid(x_init, y_init)
    x0 = np.sum(rx0 * ed) / np.sum(ed)
    y0 = np.sum(ry0 * ed) / np.sum(ed)
    rx = rx0 - x0
    ry = ry0 - y0

    r = np.sqrt(rx**2 + ry**2)
    phi = np.arctan2(ry, rx)
    aver_cos2 = np.sum(r**2 * np.cos(2*phi) * ed) / np.sum(ed)
    aver_sin2 = np.sum(r**2 * np.sin(2*phi) * ed) / np.sum(ed)
    psi = 0.5 * np.arctan2(aver_sin2, aver_cos2)

    astring = 'psi = {}'.format(str(psi))
    f = open('pp_angle.dat', 'w')
    f.write(astring)
    f.close()
    return psi

        
def run_afterburner(ievent):
    """
    particalization + UrQMD afterburner
    input: sampler/surface.dat
    output: urqmd_inputFile (urqmd/urqmd_initial.dat)
    return: nsamples (number of oversampled urqmd events)
    """
    surfaceFile = 'surface.dat'
    eventinfo_File = 'initial/{}.info.dat'.format(ievent)

    # now particalize the hypersurface
    if os.stat(surfaceFile).st_size == 0:
        print('empty hypersurface')
        return 0

    finfo = open(eventinfo_File, 'r')
    finfo.seek(0)
    for line in finfo:
        inputline = line.split()
        if len(inputline) > 1 and inputline[0] == 'mult':
            initial_mult = float(inputline[2])
        else:
            continue
    finfo.close()

    nsamples = min(max(int(2*1e5/initial_mult), 2), 100)
    run_cmd('sampler oversamples={}'.format(nsamples))

    # a non-empty hypersurface can still emit zero particles. If no particles is produced, the output 
    # will contain the three-line oscar header and nothing else. In this case, throw this event
    with open('oscar.dat', 'rb') as f:
        if next(itertools.islice(f, 3, None), None) is None:
            print('no particle emitted')
            return 0

    run_cmd('./afterburner {} urqmd_final.dat'.format(nsamples))
    return nsamples

def calculate_Dmeson_Raa(spectraFile, ID_, px_, py_, initial_pT_):
    print('now analyze particle: ', np.unique(ID_))

    pT_AA, dsigmadpT2_AA = np.loadtxt(spectraFile, unpack=True)
    dsigmadpT_AA = pT_AA * dsigmadpT2_AA
    sigma_AA = dsigmadpT_AA.sum() * (pT_AA[1] - pT_AA[0])
    dsigmadpT_AA = dsigmadpT_AA / sigma_AA
    dfdpT = interp1d(pT_AA, dsigmadpT_AA)

    final_pT = np.sqrt(px_**2 + py_**2)

    Raa_result = []
    pT_weight = dfdpT(initial_pT_)
    dNdpT, pT_bins = np.histogram(final_pT, bins=100, range=(0, 100.0), weights = pT_weight)
    Raa_result.append(pT_bins[:-1])
    Raa_result.append(dNdpT)


    ### ------------ some other stuff I'd like to output ------------------
    ## in terms of rotated initial-reaction plane
    ## in reaction plane (px_rec, py_rec), in event-plane (px_eve, py_eve)
    ## px_rec = pT * cos(phi), py_rec = pT * sin(phi)
    ## px_eve = pT * cos(phi + theta), py_eve = pT * sin(phi + theta)
    ## py_eve**2 - px_eve**2 = (py**2 - px**2) * cos(2*theta) + 2*px*py*sin(2*theta)

    if os.path.isfile('pp_angle.dat'):
        angle = open('pp_angle.dat', 'r').readline().split('=')[-1]
        angle = float(angle)
    else:
        angle = 0
        print('No pp_angle.dat found')

    px_eve = px_ * np.cos(angle) - py_ * np.sin(angle)
    py_eve = px_ * np.sin(angle) + py_ * np.cos(angle)

    dNdpx, px_bins = np.histogram(px_eve, bins=100, range=(-100.0, 100.0), weights = pT_weight)
    Raa_result.append(dNdpx)

    dNdpx, px_bins = np.histogram(np.abs(px_eve), bins=100, range=(0, 100.0), weights = pT_weight)
    Raa_result.append(dNdpx)

    dNdpx, px_bins = np.histogram(px_eve**2, bins=100, range=(0, 100.0), weights = pT_weight)
    Raa_result.append(dNdpx)

    dNdpy, px_bins = np.histogram(py_eve, bins=100, range=(-100.0, 100.0), weights = pT_weight)
    Raa_result.append(dNdpy)

    dNdpy, px_bins = np.histogram(np.abs(py_eve), bins=100, range=(0, 100.0), weights = pT_weight)
    Raa_result.append(dNdpy)

    dNdpy, px_bins = np.histogram(py_eve**2, bins=100, range=(0, 100.0), weights = pT_weight)
    Raa_result.append(dNdpy)
    ### ------------------ end of outputing the px, py distribution ------------------

    ID_labels = np.unique(ID_)
    for label in ID_labels:
        initial_pT_label = initial_pT_[ID_ == label]
        final_pT_label = final_pT[ID_ == label]
        if (len(initial_pT_label) > 0):
            pT_weight = dfdpT(initial_pT_label)
            dNdpT_, pT_bins = np.histogram(final_pT_label, bins=100, range=(0, 100.0), weights=pT_weight)
            Raa_result.append(dNdpT_)

    return np.array(Raa_result).T



def calculate_Dmeson_v2_EP(spectraFile, ID_, px_, py_, initial_pT_):
    pT_AA, dsigmadpT2_AA = np.loadtxt(spectraFile, unpack=True)
    dsigmadpT_AA = pT_AA * dsigmadpT2_AA 
    sigma_AA = dsigmadpT_AA.sum() * (pT_AA[1] - pT_AA[0])
    dsigmadpT_AA = dsigmadpT_AA / sigma_AA
    dfdpT = interp1d(pT_AA, dsigmadpT_AA)

    final_pT = np.sqrt(px_**2 + py_**2)
    pT_weight = dfdpT(initial_pT_)

    if os.path.isfile('pp_angle.dat'):
        angle = open('pp_angle.dat', 'r').readline().split('=')[-1]
        angle = float(angle)
    else:
        angle = 0
        print('pp_angle.dat not found.')

    final_v2 = ((py_**2 - px_**2) * np.cos(2*angle) + 2 * px_*py_*np.sin(2*angle)) / (final_pT**2)
    final_v2_with_weight = final_v2 * pT_weight

    v2_result = []
    pT_bins = np.linspace(0.0, 100.0, 101)
    v2_result.append(pT_bins[:-1])

    dum_v2 = []
    dum_weight = []
    for i in range(len(pT_bins)-1):
        cut = (final_pT > pT_bins[i]) & (final_pT < pT_bins[i+1])
        v2_sum = final_v2_with_weight[cut].sum()
        weight_sum = pT_weight[cut].sum()
        try:
            dum_v2.append(v2_sum/weight_sum)
        except ZeroDivisionError:
            dum_v2.append(0)
        dum_weight.append(weight_sum)

    v2_result.append(np.array(dum_v2))
    v2_result.append(np.array(dum_weight))

    return np.array(v2_result).T



def calculate_Dmeson_v2_cumulant(spectraFile, ID_, px_, py_, phi_, initial_pT_):
    pT_AA, dsigmadpT2_AA = np.loadtxt(spectraFile, unpack=True)
    dsigmadpT_AA = pT_AA * dsigmadpT2_AA
    sigma_AA = dsigmadpT_AA.sum() * (pT_AA[1] - pT_AA[0])
    dsigmadpT_AA = dsigmadpT_AA / sigma_AA
    dfdpT = interp1d(pT_AA, dsigmadpT_AA)

    final_pT = np.sqrt(px_**2 + py_**2)
    pT_weight = dfdpT(initial_pT_)

    if os.path.isfile('pp_angle.dat'):
        angle = open('pp_angle.dat', 'r').readline().split('=')[-1]
        angle = float(angle)
    else:
        angle = 0
        print('pp_angle.dat not found!')

    final_v2 = ((py_**2 - px_**2) * np.cos(2*angle) + 2*px_*py_*np.sin(2*angle) )/(final_pT**2)
    final_v2_with_weight = final_v2 * pT_weight

    v2_result = []
    pT_bins = np.linspace(0.0, 100.0, 101)
    v2_result.append(pT_bins[:-1])

    dum_q1 = []
    dum_q2 = []
    dum_q3 = []
    dum_q4 = []
    dum_mD = []

    for i in range(len(pT_bins) -1):
        cut = (final_pT >= pT_bins[i]) & (final_pT < pT_bins[i+1])
        phi_Dmeson = phi_[cut]
        pT_weight_Dmeson = pT_weight[cut]
        q1 = (np.exp(1j*1*phi_Dmeson) * pT_weight_Dmeson).sum()
        q2 = (np.exp(1j*2*phi_Dmeson) * pT_weight_Dmeson).sum()
        q3 = (np.exp(1j*3*phi_Dmeson) * pT_weight_Dmeson).sum()
        q4 = (np.exp(1j*4*phi_Dmeson) * pT_weight_Dmeson).sum()
        mD = pT_weight_Dmeson.sum()

        dum_q1.append(q1)
        dum_q2.append(q2)
        dum_q3.append(q3)
        dum_q4.append(q4)
        dum_mD.append(mD)

    v2_result.append(np.array(dum_q1))
    v2_result.append(np.array(dum_q2))
    v2_result.append(np.array(dum_q3))
    v2_result.append(np.array(dum_q4))
    v2_result.append(np.array(dum_mD))

    return np.array(v2_result).T

def calculate_Dmeson_dNdpTdphi(spectraFile, px_, py_, phi_, initial_pT_):
    pT_AA, dsigmadpT2_AA = np.loadtxt(spectraFile, unpack=True)
    dsigmadpT_AA = pT_AA * dsigmadpT2_AA
    sigma_AA = dsigmadpT_AA.sum() * (pT_AA[1] - pT_AA[0])
    dsigmadpT_AA = dsigmadpT_AA / sigma_AA
    dfdpT = interp1d(pT_AA, dsigmadpT_AA)
    
    pT_ = np.sqrt(px_**2 + py_**2)
    pT_weight = dfdpT(initial_pT_)

    H, xbins, ybins = np.histogram2d(pT_, phi_, bins=(100, 50), range=[[0, 100], [-np.pi, np.pi]], weights=pT_weight)

    return (H, xbins, ybins)



def calculate_Dmeson_dNdpTdy(spectraFile, px_, py_, y_, initial_pT_):
    pT_AA, dsigmadpT2_AA = np.loadtxt(spectraFile, unpack=True)
    dsigmadpT_AA = pT_AA * dsigmadpT2_AA
    sigma_AA = dsigmadpT_AA.sum() * (pT_AA[1] - pT_AA[0])
    dsigmadpT_AA = dsigmadpT_AA / sigma_AA
    dfdpT = interp1d(pT_AA, dsigmadpT_AA)

    pT_ = np.sqrt(px_**2 + py_**2)
    pT_weight = dfdpT(initial_pT_)
    H, xbins, ybins = np.histogram2d(pT_, y_, bins=(100, 60), range=[[0, 100], [-3, 3]], weights=pT_weight)

    return (H, xbins, ybins)


def calculate_beforeUrQMD(spectraFile, DmesonFile, resultFile, grpName, ycut, status='a'):
    ID, px, py, pz, p0, ipT = read_oscar_file(DmesonFile)
    abs_ID = np.abs(ID)
    y = 0.5 * np.log((p0 + pz)/(p0 - pz))
    phi = np.arctan2(py, px)

    #------------ Dmeson midrapidity ------------------------
    fres = h5py.File(resultFile, status)
    group_Dmeson = fres.create_group(grpName)
    group_Dmeson.create_dataset('pID', data=np.unique(ID))

    midrapidity = (np.fabs(y) < ycut)
    ID_Dmeson = abs_ID[midrapidity]
    px_Dmeson = px[midrapidity]
    py_Dmeson = py[midrapidity]
    phi_Dmeson = phi[midrapidity]
    ipT_Dmeson = ipT[midrapidity]

    Raa_result =calculate_Dmeson_Raa(spectraFile,ID_Dmeson,px_Dmeson, py_Dmeson, ipT_Dmeson)
    group_Dmeson.create_dataset('multi_{}_y_lt_{}'.format(grpName, ycut), data = Raa_result)  # for checking purpose
    
    def write_rapidity(ycut):
        midrapidity = (np.fabs(y) < ycut)
        ID_Dmeson = abs_ID[midrapidity]
        px_Dmeson = px[midrapidity]
        py_Dmeson = py[midrapidity]
        phi_Dmeson = phi[midrapidity]
        ipT_Dmeson = ipT[midrapidity]

        v2_result =calculate_Dmeson_v2_EP(spectraFile, ID_Dmeson, px_Dmeson, py_Dmeson, ipT_Dmeson)
        group_Dmeson.create_dataset('v2_{}_y_lt_{}_EP'.format(grpName, ycut), data = v2_result)

    for idum in [0.5, 0.6, 0.7, 0.8, 1.0, 2.4]:
        write_rapidity(idum)

    (Raa_result, bin1, bin2) = calculate_Dmeson_dNdpTdphi(spectraFile, px_Dmeson, py_Dmeson, phi_Dmeson, ipT_Dmeson)
    group_Dmeson.create_dataset('multi_{}_dNdpTdphi_y_lt_{}'.format(grpName, ycut), data = Raa_result)
    

    # ----------- Dmeson dNdpTdy
    (Raa_result, bin1, bin2) = calculate_Dmeson_dNdpTdy(spectraFile, px, py, y, ipT)
    ds_multi = group_Dmeson.create_dataset('multi_{}_dNdpTdy'.format(grpName), data = Raa_result)
    ds_multi.attrs.create('pT_min', bin1[0])
    ds_multi.attrs.create('pT_max', bin1[-1])
    ds_multi.attrs.create('dpT', bin1[1] - bin1[0])
    ds_multi.attrs.create('y_min', bin2[0])
    ds_multi.attrs.create('y_max', bin2[-1])
    ds_multi.attrs.create('dy', bin2[1] - bin2[0])
 
    fres.close()

def calculate_afterUrQMD(spectraFile, urqmd_outputFile, resultFile, grpName, ycut, status='a'):
    urqmd_outputFile = 'urqmd_final.dat'
    fres = h5py.File(resultFile, status)

    group_Dmeson = fres.create_group('afterUrQMD/Dmeson')

    ID, charge, px, py, pz, y, eta, ipT = read_text_file(urqmd_outputFile)
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    abs_ID = np.abs(ID)
    abs_eta = np.fabs(eta)
    Dmeson_ID = ((abs_ID==411) | (abs_ID==421) | (abs_ID==10411) | (abs_ID==10421))
    light_ID = ((abs_ID!=411) & (abs_ID!=421) & (abs_ID!=10411) & (abs_ID!=10421))
    charged = ((charge != 0) & (light_ID))


    ##--------- D meson ---------------------------------------------
    group_Dmeson.attrs.create('NDmeson', np.count_nonzero(Dmeson_ID))

    def write_rapidity(ycut):
        midrapidity = (np.fabs(y) < ycut)
        dum_ID = abs_ID[Dmeson_ID & midrapidity]
        dum_px = px[Dmeson_ID & midrapidity]
        dum_py = py[Dmeson_ID & midrapidity]
        dum_ipT = ipT[Dmeson_ID & midrapidity]
        dum_phi = phi[Dmeson_ID & midrapidity]

        Raa_result = calculate_Dmeson_Raa(spectraFile, dum_ID, dum_px, dum_py, dum_ipT)
        group_Dmeson.create_dataset('multi_Dmeson_y_lt_%s'%ycut, data=Raa_result)
        v2_result = calculate_Dmeson_v2_EP(spectraFile, dum_ID, dum_px, dum_py, dum_ipT)
        group_Dmeson.create_dataset('v2_Dmeson_y_lt_%s_EP'%ycut, data=v2_result)
        v2_result2 = calculate_Dmeson_v2_cumulant(spectraFile, dum_ID, dum_px, dum_py, dum_phi, dum_ipT)
        group_Dmeson.create_dataset('vn_Dmeson_y_lt_%s_cumulant'%ycut, data=v2_result2)
        Raa_result, bin1, bin2 = calculate_Dmeson_dNdpTdphi(spectraFile, dum_px, dum_py, dum_phi, dum_ipT)
        group_Dmeson.create_dataset('multi_Dmeson_dNdpTdphi_y_lt_%s'%ycut, data=Raa_result)

    write_rapidity(0.5)
    write_rapidity(0.8)
    write_rapidity(1.0)

    ####-- Dmeson dNdpTdy -----
    px_Dmeson = px[Dmeson_ID]
    py_Dmeson = py[Dmeson_ID]
    y_Dmeson = y[Dmeson_ID]
    phi_Dmeson = phi[Dmeson_ID]
    eta_Dmeson = eta[Dmeson_ID]
    ipT_Dmeson = ipT[Dmeson_ID]
    (Raa_result, bin1, bin2) = calculate_Dmeson_dNdpTdy(spectraFile, px_Dmeson, py_Dmeson, y_Dmeson, ipT_Dmeson)
    ds_multi = group_Dmeson.create_dataset('multi_Dmeson_dNdpTdy', data = Raa_result)
    ds_multi.attrs.create('pT_min', bin1[0])
    ds_multi.attrs.create('pT_max', bin1[-1])
    ds_multi.attrs.create('dpT', bin1[1] - bin1[0])
    ds_multi.attrs.create('y_min', bin2[0])
    ds_multi.attrs.create('y_max', bin2[-1])
    ds_multi.attrs.create('dy', bin2[1] - bin2[0])
 

    ##------------------ light hadron ---------------------------------
    group_light = fres.create_group('afterUrQMD/light')
    ##---------------------------------------------------------------
    # dNch_deta
    nsamples = fres['initial'].attrs.get('nsamples')
    H, bins = np.histogram(eta[charged], range=(-5, 5), bins=100)
    dNch_deta = H / nsamples / (bins[1] - bins[0])
    ds_dNchdeta = group_light.create_dataset('dNch_deta', data=dNch_deta)
    ds_dNchdeta.attrs.create('etamin', -5)
    ds_dNchdeta.attrs.create('etamax', 5)
    ds_dNchdeta.attrs.create('deta', 0.1)
    ds_dNchdeta.attrs.create('Nch_eta_lt_0d5', np.count_nonzero((abs_eta<0.5) & charged)/nsamples)
    ds_dNchdeta.attrs.create('Nch_eta_lt_0d8', np.count_nonzero((abs_eta<0.8) & charged)/nsamples)

    # dNdy, mean-pT
    species = [('pion', 211), ('kaon', 321), ('proton', 2212)]
    for name, i in species:
        cut = (abs_ID == i) & (np.fabs(y) < 0.5)
        N = np.count_nonzero(cut)
        ds_dNchdeta.attrs.create('dNdy-%s'%name, 1.*N/nsamples)
        if N==0:
            mean_pT = 0
        else:
            mean_pT = pT[cut].mean()

        ds_dNchdeta.attrs.create('mean-pT-%s'%name, mean_pT)

    ## differential flow cumulants
    pT_bins = np.linspace(0.2, 5, 25)
    diff_Q1 = []
    diff_Q2 = []
    diff_Q3 = []
    diff_Q4 = []
    diff_M = []
    for i in range(len(pT_bins) -1):
        diff_cut = (charged & (abs_eta < 0.8) & (pT > pT_bins[i]) & (pT < pT_bins[i+1]))
        q1 = np.exp(1j*1*phi[diff_cut]).sum()
        q2 = np.exp(1j*2*phi[diff_cut]).sum()
        q3 = np.exp(1j*3*phi[diff_cut]).sum()
        q4 = np.exp(1j*4*phi[diff_cut]).sum()
        m_ = np.count_nonzero(phi[diff_cut])
        diff_Q1.append(q1)
        diff_Q2.append(q2)
        diff_Q3.append(q3)
        diff_Q4.append(q4)
        diff_M.append(m_)

    ds_flow = group_light.create_dataset('differential_flow', data = [pT_bins[:-1], diff_Q1, diff_Q2, diff_Q3, diff_Q4, diff_M])

    ## integrated flow cumulants
    phi_ALICE = phi[charged & (abs_eta < 0.8) & (pT > 0.2) & (pT < 5.0)]
    flow_Qn_ALICE = [np.exp(1j*n*phi_ALICE).sum() for n in range(1, 7)]

    phi_ALICE_a = phi[charged & (eta < 0.8) & (eta>0.0) & (pT>0.2) & (pT<5.0)]
    flow_Qa_ALICE = [np.exp(1j*n*phi_ALICE_a).sum() for n in range(1, 7)]

    phi_ALICE_b = phi[charged & (eta<0.0) & (eta>-0.8) & (pT>0.2) & (pT<5.0)]
    flow_Qb_ALICE = [np.exp(1j*n*phi_ALICE_b).sum() for n in range(1, 7)]

    ds_flow2 = group_light.create_dataset('integrated_flow', data=[range(1, 7), flow_Qn_ALICE, flow_Qa_ALICE, flow_Qb_ALICE])
    ds_flow2.attrs.create('integrated_M', np.count_nonzero(phi_ALICE))
    ds_flow2.attrs.create('integrated_Ma', np.count_nonzero(phi_ALICE_a))
    ds_flow2.attrs.create('integrated_Mb', np.count_nonzero(phi_ALICE_b))


    ## light hadron distribution
    H_light, bin1, bin2 = np.histogram2d(pT[charged & (abs_eta<0.8)], phi[charged & (abs_eta<0.8)], bins=(25, 50), range=[[0, 5], [-np.pi, np.pi]])
    ds_dNdphi = group_light.create_dataset('dNdpTdphi', data=H_light)
    ds_dNdphi.attrs.create('pT_min', bin1[0])
    ds_dNdphi.attrs.create('pT_max', bin1[-1])
    ds_dNdphi.attrs.create('dpT', bin1[1]-bin1[0])
    ds_dNdphi.attrs.create('phi_min', bin2[0])
    ds_dNdphi.attrs.create('phi_max', bin2[-1])
    ds_dNdphi.attrs.create('dphi', bin2[1]-bin2[0])

    fres.close()


def parseConfig(configFile):
    config = {}
    with open(configFile, 'r') as f:
        config = dict(
            (i.strip() for i in l.split('=', maxsplit=1))
            for l in f if not l.startswith('#')
        )

    print('read in config successfully :)')
    return config




def main():
    collision_sys = 'PbPb5020'
    spectraFile = '%s/spectra/LHC5020-AA2ccbar.dat'%share

    # ==== parse the config file ============================================
    if len(sys.argv) == 3:
        config = parseConfig(sys.argv[1])
        jobID = sys.argv[2]
    else:
        config = {}
        jobID = 0


    # ====== set up grid size variables ======================================
    grid_step = 0.1
    grid_max = 15.05
    dtau = 0.25 * grid_step
    Nhalf = int(grid_max/grid_step)
    
    tau_fs = float(config.get('tau_fs'))
    xi_fs = float(config.get('xi_fs'))
    nevents = int(config.get('nevents'))

    # ========== initial condition ============================================
    proj = collision_sys[:2]
    targ = collision_sys[2:4]
        
    run_cmd(
        'trento {} {}'.format(proj, targ), str(nevents),
        '--grid-step {} --grid-max {}'.format(grid_step, grid_max),
        '--output {}'.format('initial.hdf5'),
        config.get('trento_args', '')
    )

    run_qhat(config.get('qhat_args'))
    # set up sampler HRG object 
    Tswitch = float(config.get('Tswitch'))
    hrg = frzout.HRG(Tswitch, species = 'urqmd', res_width=True)
    eswitch = hrg.energy_density()


    finitial = h5py.File('initial.hdf5', 'r')

    for (ievent, dset) in enumerate(finitial.values()):
        resultFile = 'result_{}-{}.hdf5'.format(jobID, ievent)
        fresult = h5py.File(resultFile, 'w')
        print('# event: ', ievent)
        ic = [dset['matter_density'].value, dset['Ncoll_density'].value]
        event_gp = fresult.create_group('initial')
        event_gp.attrs.create('initial_entropy', grid_step**2 * ic[0].sum())
        event_gp.attrs.create('N_coll', grid_step**2 * ic[1].sum())
        for (k, v) in list(finitial['event_{}'.format(ievent)].attrs.items()):
            event_gp.attrs.create(k, v)

        # =============== Freestreaming =========================================== 
        save_fs_history(ic[0], event_size=grid_max, grid_step=grid_step,
                        tau_fs=tau_fs, xi=xi_fs, steps=5, grid_max=grid_max, coarse=2)
        fs = freestream.FreeStreamer(ic[0], grid_max, tau_fs)
        e = fs.energy_density()
        e_above = e[e> eswitch].sum()
        event_gp.attrs.create('multi_factor', e.sum()/e_above if e_above > 0 else 1)
        e.tofile('ed.dat')

        # calculate the participant plane angle
        participant_plane_angle(e, int(grid_max))

        for i in [1, 2]:
            fs.flow_velocity(i).tofile('u{}.dat'.format(i))
        for ij in [(1,1), (1,2),(2,2)]:
            fs.shear_tensor(*ij).tofile('pi{}{}.dat'.format(*ij))

        # ============== vishnew hydro ===========================================
        run_cmd(
            'vishnew initialuread=1 iein=0',
            't0={} dt={} dxy={} nls={}'.format(tau_fs, dtau, grid_step, Nhalf),
            config.get('hydro_args', '')
        )

        # ============= frzout sampler =========================================
        surface_data = np.fromfile('surface.dat', dtype='f8').reshape(-1, 16)
        if surface_data.size == 0:
            print("empty event")
            continue
        print('surface_data.size: ', surface_data.size)

        surface = frzout.Surface(**dict(
            zip(['x', 'sigma', 'v'], np.hsplit(surface_data, [3, 6, 8])),
            pi=dict(zip(['xx', 'xy', 'yy'], surface_data.T[11:14])),
            Pi=surface_data.T[15]),
            ymax=3.)

        minsamples, maxsamples = 10, 100
        minparts = 30000
        nparts = 0   # for tracking total number of sampeld particles

        # sample soft particles and write to file
        with open('particle_in.dat', 'w') as f:
            nsamples = 0
            while nsamples < maxsamples + 1:
                parts = frzout.sample(surface, hrg)
                if parts.size == 0:
                    continue
                else:
                    nsamples += 1
                    nparts += parts.size
                    print("#", parts.size, file=f)
                    for p in parts:
                        print(p['ID'], *itertools.chain(p['x'], p['p']), file=f)

                    if nparts >= minparts and nsamples >= minsamples:
                        break

        event_gp.attrs.create('nsamples', nsamples, dtype=np.int)

        # =============== HQ initial position sampling ===========================
        initial_TAA = ic[1]
        np.savetxt('initial_Ncoll_density.dat', initial_TAA)
        HQ_sample_conf = {'IC_file': 'initial_Ncoll_density.dat',\
                          'XY_file': 'initial_HQ.dat', \
                          'IC_Nx_max': initial_TAA.shape[0], \
                          'IC_Ny_max': initial_TAA.shape[1], \
                          'IC_dx': grid_step, \
                          'IC_dy': grid_step, \
                          'IC_tau0': 0, \
                          'N_sample': 60000, \
                          'N_scale': 0.05, \
                          'scale_flag': 0}

        ftmp = open('HQ_sample.conf', 'w')
        for (key, value) in zip(HQ_sample_conf.keys(), HQ_sample_conf.values()):
            inputline = ' = '.join([str(key), str(value)]) + '\n'
            ftmp.write(inputline)
        ftmp.close()

        run_cmd('HQ_sample HQ_sample.conf')

        # ================ HQ evolution (pre-equilibirum stages) =================
        os.environ['ftn00'] = 'FreeStream.h5'
        os.environ['ftn10'] = '%s/dNg_over_dt_cD6.dat'%share
        print(os.environ['ftn10'])
        os.environ['ftn20'] = 'HQ_AAcY_preQ.dat'
        os.environ['ftn30'] = 'initial_HQ.dat'
        run_cmd('diffusion hq_input=3.0 initt={}'.format(tau_fs*xi_fs),
                config.get('diffusion_args', '')
        )

        # ================ HQ evolution (in medium evolution) ====================
        os.environ['ftn00'] = 'JetData.h5'
        os.environ['ftn10'] = '%s/dNg_over_dt_cD6.dat'%share
        os.environ['ftn20'] = 'HQ_AAcY.dat'
        os.environ['ftn30'] = 'HQ_AAcY_preQ.dat'
        run_cmd('diffusion hq_input=4.0 initt={}'.format(tau_fs),
                config.get('diffusion_args', '')
        )

    
        # ============== Heavy quark hardonization ==============================
        os.environ['ftn20'] = 'Dmeson_AAcY.dat'
        child1 = 'cat HQ_AAcY.dat'
        p1 = subprocess.Popen(child1.split(), stdout=subprocess.PIPE)
        p2 = subprocess.Popen('fragPLUSrecomb', stdin = p1.stdout)
        p1.stdout.close()
        output = p2.communicate()[0]

        # ============ Heavy + soft UrQMD =================================
        run_cmd('afterburner {} urqmd_final.dat particle_in.dat Dmeson_AAcY.dat'.format(nsamples))

        # =========== processing data ====================================
        calculate_beforeUrQMD(spectraFile, 'Dmeson_AAcY.dat', resultFile, 'beforeUrQMD/Dmeson', 1.0, 'a')
        calculate_beforeUrQMD(spectraFile, 'HQ_AAcY.dat', resultFile, 'beforeUrQMD/HQ', 1.0, 'a')
        calculate_beforeUrQMD(spectraFile, 'HQ_AAcY_preQ.dat', resultFile, 'beforeUrQMD/HQ_preQ', 1.0, 'a')
        if nsamples != 0:
            calculate_afterUrQMD(spectraFile, 'urqmd_final.dat', resultFile, 'afterUrQMD/Dmeson', 1.0, 'a')
        
        shutil.move('urqmd_final.dat', 'urqmd_final{}-{}.dat'.format(jobID, ievent))
        shutil.move('Dmeson_AAcY.dat', 'Dmeson_AAcY{}-{}.dat'.format(jobID, ievent))
        shutil.move('HQ_AAcY.dat', 'HQ_AAcY{}-{}.dat'.format(jobID, ievent))
        shutil.move('HQ_AAcY_preQ.dat', 'HQ_AAcY_preQ{}-{}.dat'.format(jobID, ievent))
    
    #=== after everything, save initial profile (depends on how large the size if, I may choose to forward this step)
    shutil.move('initial.hdf5', 'initial_{}.hdf5'.format(jobID))

if __name__ == '__main__':
    main()
