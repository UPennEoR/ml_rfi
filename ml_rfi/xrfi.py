# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""Module for performing RFI identification and excision."""

from __future__ import print_function, division, absolute_import
import numpy as np
import os
from pyuvdata import UVData
from pyuvdata import UVCal
from pyuvdata import UVFlag
from . import utils as qm_utils
from pyuvdata import utils as uvutils
from .version import hera_qm_version_str
from .metrics_io import process_ex_ants
import warnings
import copy
import collections
from six.moves import range


#############################################################################
# Utility functions
#############################################################################

def flag_xants(uv, xants, inplace=True):
    """Flag visibilities containing specified antennas.
    Parameters
    ----------
    uv : UVData or UVCal or UVFlag
        Object containing data to be flagged. Should be a UVData, UVCal, or
        UVFlag object.
    xants : list of ints
        List of antenna numbers to completely flag.
    inplace : bool, optional
        If True, apply flags to the uv object. If False, return a UVFlag object
        with only xants flaged. Default is True.
    Returns
    -------
    uvo : UVData or UVCal or UVFlag
        If inplace is True, uvo is a reference to the input uv object, but with
        the flags specified in xants flagged. If inplace is False, uvo is a new
        UVFlag object with only xants flaged.
    Raises
    ------
    ValueError:
        If uv is not a UVData, UVCal, UVFlag, or subclassed object, a ValueError
        is raised. If a UVFlag of a "waterfall" type is passed in, a ValueError
        is also raised. If a UVFlag object that is not in "flag" mode is passed
        in, a ValueError is raised.
    """
    # check that we got an appropriate object
    if not issubclass(uv.__class__, (UVData, UVCal, UVFlag)):
        raise ValueError('First argument to flag_xants must be a UVData, UVCal, '
                         ' or UVFlag object.')
    if isinstance(uv, UVFlag) and uv.type == 'waterfall':
        raise ValueError('Cannot flag antennas on UVFlag obejct of type "waterfall".')

    if not inplace:
        if isinstance(uv, UVFlag):
            uvo = uv.copy()
            uvo.to_flag()
        else:
            uvo = UVFlag(uv, mode='flag')
    else:
        uvo = uv

    if isinstance(uvo, UVFlag) and uvo.mode != 'flag':
        raise ValueError('Cannot flag antennas on UVFlag obejct in mode ' + uvo.mode)

    if not isinstance(xants, collections.Iterable):
        xants = [xants]
    if issubclass(uvo.__class__, UVData) or (isinstance(uvo, UVFlag) and uvo.type == 'baseline'):
        all_ants = np.unique(np.append(uvo.ant_1_array, uvo.ant_2_array))
        for ant in all_ants:
            for xant in xants:
                blts = uvo.antpair2ind(ant, xant)
                uvo.flag_array[blts, :, :, :] = True
                blts = uvo.antpair2ind(xant, ant)
                uvo.flag_array[blts, :, :, :] = True
    elif issubclass(uvo.__class__, UVCal) or (isinstance(uvo, UVFlag) and uvo.type == 'antenna'):
        for xant in xants:
            ai = np.where(uvo.ant_array == xant)[0]
            uvo.flag_array[ai, :, :, :, :] = True

    if not inplace:
        return uvo


def resolve_xrfi_path(xrfi_path, fname):
    """Determine xrfi_path based on given directory or default to dirname of given file.
    Parameters
    ----------
    xrfi_path : str
        Directory to which to write xrfi outputs.
    fname : str
        Filename to determine backup directory if xrfi_path == ''.
    Returns
    -------
    dirname : str
        If xrfi_path is not '', dirname is xrfi_path. Otherwise it returns the
        directory of the file.
    """
    if (xrfi_path != '') and (os.path.exists(xrfi_path)):
        dirname = xrfi_path
    else:
        dirname = os.path.dirname(os.path.abspath(fname))
    return dirname


def _check_convolve_dims(data, Kt, Kf):
    """Check the kernel sizes to be used in various convolution-like operations.
    If the kernel sizes are too big, replace them with the largest allowable size
    and issue a warning to the user.
    Parameters
    ----------
    data : array
        2D array that will undergo convolution-like operations.
    Kt : int
        Integer representing box dimension in time to apply statistic.
    Kf : int
        Integer representing box dimension in frequency to apply statistic.
    Returns
    -------
    Kt : int
        Input Kt or data.shape[0] if Kt is larger than first dim of arr.
    Kf : int
        Input Kf or data.shape[1] if Kf is larger than first dim of arr.
    Raises
    ------
    ValueError:
        If the number of dimensions of the arr array is not 2, a ValueError is raised.
    """
    if data.ndim != 2:
        raise ValueError('Input to filter must be 2D array.')
    if Kt > data.shape[0]:
        warnings.warn("Kt value {0:d} is larger than the data of dimension {1:d}; "
                      "using the size of the data for the kernel size".format(Kt, data.shape[0]))
        Kt = data.shape[0]
    if Kf > data.shape[1]:
        warnings.warn("Kf value {0:d} is larger than the data of dimension {1:d}; "
                      "using the size of the data for the kernel size".format(Kf, data.shape[1]))
        Kf = data.shape[1]
    return Kt, Kf


def robust_divide(num, den):
    """Prevent division by zero.
    This function will compute division between two array-like objects by setting
    values to infinity when the denominator is small for the given data type. This
    avoids floating point exception warnings that may hide genuine problems
    in the data.
    Parameters
    ----------
    num : array
        The numerator.
    den : array
        The denominator.
    Returns
    -------
    out : array
        The result of dividing num / den. Elements where b is small (or zero) are set
        to infinity.
    """
    thresh = np.finfo(den.dtype).eps
    out = np.true_divide(num, den, where=(np.abs(den) > thresh))
    out = np.where(np.abs(den) > thresh, out, np.inf)
    return out


#############################################################################
# Functions for preprocessing data prior to RFI flagging
#############################################################################

def medmin(data, flags=None):
    """Calculate the median minus minimum statistic of array.
    Note
    ----
    The statistic first computes the minimum value of the array along the
    first axis (the time axis, if the array is passed in as (time, frequency,
    so that a single spectrum is returned). The median of these values is
    computed, multiplied by 2, and then the minimum value is subtracted off.
    The goal is to get a proxy for the "noise" in the 2d array.
    Parameters
    ----------
    data : array
        2D data array of the shape (time,frequency).
    flags : array, optional
        2D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.
    Returns
    -------
    medmin : array
        The result of the medmin statistic.
    """
    _ = _check_convolve_dims(data, 1, 1)  # Just check data dims
    mn = np.min(data, axis=0)
    return 2 * np.median(mn) - np.min(mn)


def medminfilt(data, flags=None, Kt=8, Kf=8):
    """Filter an array on scales of Kt,Kf indexes with medmin.
    Parameters
    ----------
    data : array
        2D data array of the shape (time, frequency).
    flags : array, optional
        2D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.
    Kt : int, optional
        An integer representing box dimension in time to apply statistic. Default
        is 8 pixels.
    Kf : int, optional
        An integer representing box dimension in frequency to apply statistic.
        Default is 8 pixels.
    Returns
    -------
    d_sm : array
        The filtered array with the same shape as input array.
    """
    Kt, Kf = _check_convolve_dims(data, Kt, Kf)
    d_sm = np.empty_like(data)
    for ind1 in range(data.shape[0]):
        for ind2 in range(data.shape[1]):
            i0, j0 = max(0, ind1 - Kt), max(0, ind2 - Kf)
            i1, j1 = min(data.shape[0], ind1 + Kt), min(data.shape[1], ind2 + Kf)
            d_sm[ind1, ind2] = medmin(data[i0:i1, j0:j1])
    return d_sm


def detrend_deriv(data, flags=None, dt=True, df=True):
    """Detrend array by taking the derivative in either time, frequency, or both.
    Note
    ----
    When taking the derivative of both, the derivative in frequency is performed
    first, then in time.
    Parameters
    ----------
    data : array
        2D data array of the shape (time,frequency).
    flags : array, optional
        2D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.
    dt : bool, optional
        The derivative across time bins. Default is True.
    df : bool, optional
        The derivative across frequency bins. Default is True.
    Returns
    -------
    out : array
        A detrended array with same shape as input array.
    """
    _ = _check_convolve_dims(data, 1, 1)  # Just check data dims
    if not (dt or df):
        raise ValueError("dt and df cannot both be False when calling detrend_deriv")
    if df:
        # take gradient along frequency
        d_df = np.gradient(data, axis=1)
    else:
        d_df = data
    if dt:
        # take gradient along time
        d_dtdf = np.gradient(d_df, axis=0)
    else:
        d_dtdf = d_df

    d2 = np.abs(d_dtdf)**2
    # model sig as separable function of 2 axes
    sig_f = np.median(d2, axis=0)
    sig_f.shape = (1, -1)
    sig_t = np.median(d2, axis=1)
    sig_t.shape = (-1, 1)
    sig = np.sqrt(sig_f * sig_t / np.median(sig_t))
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_dtdf, sig)
    return out


def detrend_medminfilt(data, flags=None, Kt=8, Kf=8):
    """Detrend array using medminfilt statistic. See medminfilt.
    Parameters
    ----------
    data : array
        2D data array of the shape (time, frequency) to detrend.
    flags : array, optional
        2D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.
    Kt : int, optional
        An integer representing box dimension in time to apply statistic. Default
        is 8 pixels.
    Kf : int, optional
        An integer representing box dimension in frequency to apply statistic.
        Default is 8 pixels.
    Returns
    -------
    out : array
        An array of outlier significance metric.
    """
    _ = _check_convolve_dims(data, 1, 1)  # Just check data dimensions
    d_sm = medminfilt(np.abs(data), Kt=(2 * Kt + 1), Kf=(2 * Kf + 1))
    d_rs = data - d_sm
    d_sq = np.abs(d_rs)**2
    # puts minmed on same scale as average
    sig = np.sqrt(medminfilt(d_sq, Kt=(2 * Kt + 1), Kf=(2 * Kf + 1))) * (np.sqrt(Kt**2 + Kf**2) / .64)
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_rs, sig)
    return out


def detrend_medfilt(data, flags=None, Kt=8, Kf=8):
    """Detrend array using a median filter.
    Parameters
    ----------
    data : array
        2D data array to detrend.
    flags : array, optional
        2D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.
    Kt : int, optional
        The box size in time (first) dimension to apply medfilt over. Default is
        8 pixels.
    Kf : int, optional
        The box size in frequency (second) dimension to apply medfilt over. Default
        is 8 pixels.
    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as d.
    """
    # Delay import so scipy is not required for any use of hera_qm
    from scipy.signal import medfilt2d

    Kt, Kf = _check_convolve_dims(data, Kt, Kf)
    data = np.concatenate([data[Kt - 1::-1], data, data[:-Kt - 1:-1]], axis=0)
    data = np.concatenate([data[:, Kf - 1::-1], data, data[:, :-Kf - 1:-1]], axis=1)
    if np.iscomplexobj(data):
        d_sm_r = medfilt2d(data.real, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
        d_sm_i = medfilt2d(data.imag, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
        d_sm = d_sm_r + 1j * d_sm_i
    else:
        d_sm = medfilt2d(data, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
    d_rs = data - d_sm
    d_sq = np.abs(d_rs)**2
    # puts median on same scale as average
    sig = np.sqrt(medfilt2d(d_sq, kernel_size=(2 * Kt + 1, 2 * Kf + 1)) / .456)
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_rs, sig)
    return out[Kt:-Kt, Kf:-Kf]


def detrend_meanfilt(data, flags=None, Kt=8, Kf=8):
    """Detrend array using a mean filter.
    Parameters
    ----------
    data : array
        2D data array to detrend.
    flags : array, optional
        2D flag array to be interpretted as mask for d.
    Kt : int, optional
        The box size in time (first) dimension to apply medfilt over. Default is
        8 pixels.
    Kf : int, optional
        The box size in frequency (second) dimension to apply medfilt over.
        Default is 8 pixels.
    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as d.
    """
    # Delay import so astropy is not required for any use of hera_qm
    # Using astropy instead of scipy for treatement of Nan: http://docs.astropy.org/en/stable/convolution/
    from astropy.convolution import convolve

    Kt, Kf = _check_convolve_dims(data, Kt, Kf)
    kernel = np.ones((2 * Kt + 1, 2 * Kf + 1))
    # do a mirror extend, like in scipy's convolve, which astropy doesn't support
    data = np.concatenate([data[Kt - 1::-1], data, data[:-Kt - 1:-1]], axis=0)
    data = np.concatenate([data[:, Kf - 1::-1], data, data[:, :-Kf - 1:-1]], axis=1)
    if flags is not None:
        flags = np.concatenate([flags[Kt - 1::-1], flags, flags[:-Kt - 1:-1]], axis=0)
        flags = np.concatenate([flags[:, Kf - 1::-1], flags, flags[:, :-Kf - 1:-1]], axis=1)
    d_sm = convolve(data, kernel, mask=flags, boundary='extend')
    d_rs = data - d_sm
    d_sq = np.abs(d_rs)**2
    sig = np.sqrt(convolve(d_sq, kernel, mask=flags))
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_rs, sig)
    return out[Kt:-Kt, Kf:-Kf]


# Update algorithm_dict whenever new metric algorithm is created.
algorithm_dict = {'medmin': medmin, 'medminfilt': medminfilt, 'detrend_deriv': detrend_deriv,
                  'detrend_medminfilt': detrend_medminfilt, 'detrend_medfilt': detrend_medfilt,
                  'detrend_meanfilt': detrend_meanfilt}

#############################################################################
# RFI flagging algorithms
#############################################################################


def watershed_flag(uvf_m, uvf_f, nsig_p=2., nsig_f=None, nsig_t=None, avg_method='quadmean',
                   inplace=True):
    """Expand a set of flags using a watershed algorithm.
    This function uses a UVFlag object in 'metric' mode (i.e. how many sigma the data
    point is from the center) and a set of flags to grow the flags using defined
    thresholds.
    Parameters
    ----------
    uvf_m : UVFlag object
        A UVFlag object in 'metric' mode.
    uvf_f : UVFlag object
        A UVFlag object in 'flag' mode.
    nsig_p : float, optional
        The Number of sigma above which to flag pixels which are near previously
        flagged pixels. Default is 2.0.
    nsig_f : float, optional
        The Number of sigma above which to flag channels which are near fully
        flagged frequency channels. Bypassed if None (Default).
    nsig_t : float, optional
        Number of sigma above which to flag integrations which are near fully
        flagged integrations. Bypassed if None (Default)
    avg_method : {"mean", "absmean", "quadmean"}, optional
        Method to average metric data for frequency and time watershedding.
        Default is "quadmean".
    inplace : bool, optional
        If True, update uvf_f. If False, create a new flag object. Default is True.
    Returns
    -------
    uvf : UVFlag object
       A UVFlag object in 'flag' mode with flags after watershed.
    Raises
    ------
    ValueError:
        If uvf_m is not in "metric" mode, if uvf_f is not in "flag" mode, if
        uvf_m and uvf_f do not have the same shape, or if uvf_m has an unknown
        type, then a ValueError is raised.
    """
    # Check inputs
    if (not isinstance(uvf_m, UVFlag)) or (uvf_m.mode != 'metric'):
        raise ValueError('uvf_m must be UVFlag instance with mode == "metric."')
    if (not isinstance(uvf_f, UVFlag)) or (uvf_f.mode != 'flag'):
        raise ValueError('uvf_f must be UVFlag instance with mode == "flag."')
    if uvf_m.metric_array.shape != uvf_f.flag_array.shape:
        raise ValueError('uvf_m and uvf_f must have data of same shape. Shapes '
                         'are: ' + str(uvf_m.metric_array.shape) + ' and '
                         + str(uvf_f.flag_array.shape))
    # Handle in place
    if inplace:
        uvf = uvf_f
    else:
        uvf = copy.deepcopy(uvf_f)

    # Convenience
    farr = uvf.flag_array
    marr = uvf_m.metric_array
    warr = uvf_m.weights_array

    if uvf_m.type == 'baseline':
        # Pixel watershed
        # TODO: bypass pixel-based if none
        for bl in np.unique(uvf.baseline_array):
            ind = np.where(uvf.baseline_array == bl)[0]
            for pi in range(uvf.polarization_array.size):
                farr[ind, 0, :, pi] += _ws_flag_waterfall(marr[ind, 0, :, pi],
                                                          farr[ind, 0, :, pi], nsig_p)
        if nsig_f is not None:
            # Channel watershed
            tempd = uvutils.collapse(marr, avg_method, axis=(0, 1, 3), weights=warr)
            tempf = np.all(farr, axis=(0, 1, 3))
            farr[:, :, :, :] += _ws_flag_waterfall(tempd, tempf, nsig_f).reshape(1, 1, -1, 1)
        if nsig_t is not None:
            # Time watershed
            ts = np.unique(uvf.time_array)
            tempd = np.zeros(ts.size)
            tempf = np.zeros(ts.size, dtype=np.bool)
            for ti, time in enumerate(ts):
                tempd[ti] = uvutils.collapse(marr[uvf.time_array == time, 0, :, :], avg_method,
                                             weights=warr[uvf.time_array == time, 0, :, :])
                tempf[ti] = np.all(farr[uvf.time_array == time, 0, :, :])
            tempf = _ws_flag_waterfall(tempd, tempf, nsig_t)
            for ti, time in enumerate(ts):
                farr[uvf.time_array == time, :, :, :] += tempf[ti]
    elif uvf_m.type == 'antenna':
        # Pixel watershed
        for ai in range(uvf.ant_array.size):
            for pi in range(uvf.polarization_array.size):
                farr[ai, 0, :, :, pi] += _ws_flag_waterfall(marr[ai, 0, :, :, pi].T,
                                                            farr[ai, 0, :, :, pi].T, nsig_p).T
        if nsig_f is not None:
            # Channel watershed
            tempd = uvutils.collapse(marr, avg_method, axis=(0, 1, 3, 4), weights=warr)
            tempf = np.all(farr, axis=(0, 1, 3, 4))
            farr[:, :, :, :, :] += _ws_flag_waterfall(tempd, tempf, nsig_f).reshape(1, 1, -1, 1, 1)
        if nsig_t is not None:
            # Time watershed
            tempd = uvutils.collapse(marr, avg_method, axis=(0, 1, 2, 4), weights=warr)
            tempf = np.all(farr, axis=(0, 1, 2, 4))
            farr[:, :, :, :, :] += _ws_flag_waterfall(tempd, tempf, nsig_t).reshape(1, 1, 1, -1, 1)
    elif uvf_m.type == 'waterfall':
        # Pixel watershed
        for pi in range(uvf.polarization_array.size):
            farr[:, :, pi] += _ws_flag_waterfall(marr[:, :, pi], farr[:, :, pi], nsig_p)
        if nsig_f is not None:
            # Channel watershed
            tempd = uvutils.collapse(marr, avg_method, axis=(0, 2), weights=warr)
            tempf = np.all(farr, axis=(0, 2))
            farr[:, :, :] += _ws_flag_waterfall(tempd, tempf, nsig_f).reshape(1, -1, 1)
        if nsig_t is not None:
            # Time watershed
            tempd = uvutils.collapse(marr, avg_method, axis=(1, 2), weights=warr)
            tempf = np.all(farr, axis=(1, 2))
            farr[:, :, :] += _ws_flag_waterfall(tempd, tempf, nsig_t).reshape(-1, 1, 1)
    else:
        raise ValueError('Unknown UVFlag type: ' + uvf_m.type)
    return uvf


def _ws_flag_waterfall(data, fin, nsig=2.):
    """Perform the watershed algorithm on 1D or 2D arrays of metric and input flags.
    This is a helper function for watershed_flag, but not usually called
    by end users.
    Parameters
    ----------
    data : array
        A 2D or 1D array. Should be in units of standard deviations.
    fin : array
        The input (boolean) flags used as the seed of the watershed. Same size as d.
    nsig : float, optional
        The number of sigma to flag above for points near flagged points. Default is 2.
    Returns
    -------
    fout : array
        A boolean array matching size of d and fin, with watershedded flags.
    Raises
    ------
    ValueError:
        If the shapes of data and fin do not match, or if the number of dimensions is not
        equal to 1 or 2, a ValueError is raised.
    """
    if data.shape != fin.shape:
        raise ValueError('data and fin must match in shape. Shapes are: ' + str(data.shape)
                         + ' and ' + str(fin.shape))
    fout = copy.deepcopy(fin)
    # There may be an elegant way to combine these... for the future.
    if data.ndim == 1:
        prevn = 0
        foutx = np.where(fout)[0]
        while foutx.size != prevn:
            prevn = foutx.size
            for dx in [-1, 1]:
                xp = (foutx + dx).clip(0, fout.size - 1)
                ind = np.where(data[xp] > nsig)[0]  # if our metric > sig
                fout[xp[ind]] = 1
                foutx = np.where(fout)[0]
    elif data.ndim == 2:
        prevx, prevy = 0, 0
        foutx, fouty = np.where(fout)
        while foutx.size != prevx and fouty.size != prevy:
            prevx, prevy = foutx.size, fouty.size
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                xp, yp = ((foutx + dx).clip(0, fout.shape[0] - 1),
                          (fouty + dy).clip(0, fout.shape[1] - 1))
                ind = np.where(data[xp, yp] > nsig)[0]  # if our metric > sig
                fout[xp[ind], yp[ind]] = 1
                foutx, fouty = np.where(fout)
    else:
        raise ValueError('Data must be 1D or 2D.')
    return fout


def xrfi_waterfall(data, flags=None, Kt=8, Kf=8, nsig_init=6., nsig_adj=2.,
                   algorithm='detrend_medfilt'):
    """Compute metrics, flag, and then watershed on a single waterfall.
    Parameters
    ----------
    data : array
        2D data array (Ntimes, Nfreqs) to use in flagging.
    flags : array, optional
        2D flag array to be interpretted as mask for data. Ignored for many algorithms
        for calculating metrics of "outlierness" (e.g. detrend_medfilt) but always
        ORed with the intial flags from the metrics before the watershed is applied.
    Kt : int, optional
        The box size in time (first) dimension to apply medfilt over. Default is
        8 pixels.
    Kf : int, optional
        The box size in frequency (second) dimension to apply medfilt over. Default
        is 8 pixels.
    nsig_init : float, optional
        The number of sigma in the metric above which to flag pixels. Default is 6.
    nsig_adj : float, optional
        The number of sigma to flag above for points near flagged points. Default is 2.
    algorithm : str
        The metric algorithm name. Must be defined in algorithm_dict.
    Returns
    -------
    new_flags : array
        A final boolean array of flags matching the size of data.
    """
    try:
        alg_func = algorithm_dict[algorithm]
    except KeyError:
        raise KeyError('Algorithm not found in list of available functions.')
    metrics = alg_func(data, flags=flags, Kt=Kt, Kf=Kf)
    init_flags = (metrics >= nsig_init)
    if flags is not None:
        init_flags |= flags
    new_flags = _ws_flag_waterfall(metrics, init_flags, nsig=nsig_adj)
    return new_flags


def flag(uvf_m, nsig_p=6., nsig_f=None, nsig_t=None, avg_method='quadmean'):
    """Create a set of flags based on a "metric" type UVFlag object.
    Parameters
    ----------
    uvf_m : UVFlag object
        A UVFlag object in 'metric' mode (i.e., number of sigma data is from middle).
    nsig_p : float, optional
        The number of sigma above which to flag pixels. Default is 6.0. Bypassed
        if None.
    nsig_f : float, optional
        The number of sigma above which to flag channels. Bypassed if None (Default).
    nsig_t : float, optional
        The number of sigma above which to flag integrations. Bypassed if None (Default).
    avg_method : {"mean", "absmean", "quadmean"}, optional
        Method to average metric data for frequency and time flagging. Default is
        "quadmean".
    Returns
    -------
    uvf_f : UVFlag object
        A UVFlag object in 'flag' mode with flags determined from uvf_m.
    Raises
    ------
    ValueError:
        If uvf_m is not a UVFlag object in metric mode, or if the type of uvf_m
        is not recognized, a ValueError is raised.
    """
    # Check input
    if (not isinstance(uvf_m, UVFlag)) or (uvf_m.mode != 'metric'):
        raise ValueError('uvf_m must be UVFlag instance with mode == "metric."')

    # initialize
    uvf_f = copy.deepcopy(uvf_m)
    uvf_f.to_flag()

    # Pixel flagging
    if nsig_p is not None:
        uvf_f.flag_array[uvf_m.metric_array >= nsig_p] = True

    if uvf_m.type == 'baseline':
        if nsig_f is not None:
            # Channel flagging
            data = uvutils.collapse(uvf_m.metric_array, avg_method, axis=(0, 1, 3),
                                    weights=uvf_m.weights_array)
            indf = np.where(data >= nsig_f)[0]
            uvf_f.flag_array[:, :, indf, :] = True
        if nsig_t is not None:
            # Time flagging
            ts = np.unique(uvf_m.time_array)
            data = np.zeros(ts.size)
            for ti, time in enumerate(ts):
                data[ti] = uvutils.collapse(uvf_m.metric_array[uvf_m.time_array == time, 0, :, :],
                                            avg_method,
                                            weights=uvf_m.weights_array[uvf_m.time_array == time, 0, :, :])
            indf = np.where(data >= nsig_t)[0]
            for time in ts[indf]:
                uvf_f.flag_array[uvf_f.time_array == time, :, :, :] = True
    elif uvf_m.type == 'antenna':
        if nsig_f is not None:
            # Channel flag
            data = uvutils.collapse(uvf_m.metric_array, avg_method, axis=(0, 1, 3, 4),
                                    weights=uvf_m.weights_array)
            indf = np.where(data >= nsig_f)[0]
            uvf_f.flag_array[:, :, indf, :, :] = True
        if nsig_t is not None:
            # Time watershed
            data = uvutils.collapse(uvf_m.metric_array, avg_method, axis=(0, 1, 2, 4),
                                    weights=uvf_m.weights_array)
            indt = np.where(data >= nsig_t)[0]
            uvf_f.flag_array[:, :, :, indt, :] = True
    elif uvf_m.type == 'waterfall':
        if nsig_f is not None:
            # Channel flag
            data = uvutils.collapse(uvf_m.metric_array, avg_method, axis=(0, 2),
                                    weights=uvf_m.weights_array)
            indf = np.where(data >= nsig_f)[0]
            uvf_f.flag_array[:, indf, :] = True
        if nsig_t is not None:
            # Time watershed
            data = uvutils.collapse(uvf_m.metric_array, avg_method, axis=(1, 2),
                                    weights=uvf_m.weights_array)
            indt = np.where(data >= nsig_t)[0]
            uvf_f.flag_array[indt, :, :] = True
    else:
        raise ValueError('Unknown UVFlag type: ' + uvf_m.type)
    return uvf_f


def flag_apply(uvf, uv, keep_existing=True, force_pol=False, history='',
               return_net_flags=False):
    """Apply flags from UVFlag or list of UVFlag objects to UVData or UVCal.
    Parameters
    ----------
    uvf : UVFlag or str or list
        A UVFlag object, path to UVFlag file, or list of these. These must be in
        'flag' mode, and either match the uv argument, or be a waterfall that can
        be made to match it.
    uv : UVData or UVCal
        A UVData or UVCal object to which to apply flags.
    keep_existing : bool, optional
        If True, add flags to existing flags in uv. If False, replace existing
        flags in uv. Default is True.
    force_pol : bool, optional
        If True, will use 1 pol to broadcast to any other pol. If False, will
        require polarizations to match. Default is False.
    history : str, optional
        The history string to be added to uv.history. Default is empty string.
    return_net_flags : bool, optional
        If True, return a UVFlag object with net flags applied. If False, do not
        return net flags. Default is False.
    Returns
    -------
    net_flags : UVFlag object
        If return_net_flags is True, returns UVFlag object with net flags.
    Raises
    ------
    ValueError:
        If uv is not a UVData or UVCal object, if uvf is not a string or
        UVFlag object, or if the UVFlag objects in uvf are not in "flag" mode,
        a ValueError is raised.
    """
    if issubclass(uv.__class__, UVData):
        expected_type = 'baseline'
    elif issubclass(uv.__class__, UVCal):
        expected_type = 'antenna'
    else:
        raise ValueError('Flags can only be applied to UVData or UVCal objects.')
    if not isinstance(uvf, (list, tuple, np.ndarray)):
        uvf = [uvf]
    net_flags = UVFlag(uv, mode='flag', copy_flags=keep_existing, history=history)
    for uvf_i in uvf:
        if isinstance(uvf_i, str):
            uvf_i = UVFlag(uvf_i)  # Read file
        elif not isinstance(uvf_i, UVFlag):
            raise ValueError('Input to apply_flag must be UVFlag or path to UVFlag file.')
        if uvf_i.mode != 'flag':
            raise ValueError('UVFlag objects must be in mode "flag" to apply to data.')
        if uvf_i.type == 'waterfall':
            uvf_i = uvf_i.copy()  # don't change the input object
            if expected_type == 'baseline':
                uvf_i.to_baseline(uv, force_pol=force_pol)
            else:
                uvf_i.to_antenna(uv, force_pol=force_pol)
        # Use built-in or function
        net_flags |= uvf_i
    uv.flag_array += net_flags.flag_array
    uv.history += 'FLAGGING HISTORY: ' + history + ' END OF FLAGGING HISTORY.'

    if return_net_flags:
        return net_flags


#############################################################################
# Higher level functions that loop through data to calculate metrics
#############################################################################

def calculate_metric(uv, algorithm, cal_mode='gain', **kwargs):
    """Make a UVFlag object of mode 'metric' from a UVData or UVCal object.
    Parameters
    ----------
    uv : UVData or UVCal
        A UVData or UVCal object to calculate metrics on.
    algorithm : str
        The metric algorithm name. Must be defined in algorithm_dict.
    cal_mode : {"gain", "chisq", "tot_chisq"}, optional
        The mode to calculate metric if uv is a UVCal object. The options use
        the gain_array, quality_array, and total_quality_array attributes,
        respectively. Default is "gain".
    **kwargs : dict
        A dictionary of Keyword arguments that are passed to algorithm.
    Returns
    -------
    uvf : UVFlag object
        A UVFlag of mode 'metric' corresponding to the uv object.
    Raises
    ------
    ValueError:
        If uv is not a UVData or UVCal object, or if "cal_mode" is not in the list
        above, then a ValueError is raised.
    KeyError:
        If "algorithm" is not in the list of known algorithm options, a KeyError
        is raised.
    """
    if not issubclass(uv.__class__, (UVData, UVCal)):
        raise ValueError('uv must be a UVData or UVCal object.')
    try:
        alg_func = algorithm_dict[algorithm]
    except KeyError:
        raise KeyError('Algorithm not found in list of available functions.')
    uvf = UVFlag(uv)
    if issubclass(uv.__class__, UVData):
        uvf.weights_array = uv.nsample_array * np.logical_not(uv.flag_array).astype(np.float)
    else:
        uvf.weights_array = np.logical_not(uv.flag_array).astype(np.float)
    if issubclass(uv.__class__, UVData):
        for key, data in uv.antpairpol_iter():
            ind1, ind2, pol = uv._key2inds(key)
            for ind, ipol in zip((ind1, ind2), pol):
                if len(ind) == 0:
                    continue
                flags = uv.flag_array[ind, 0, :, ipol]
                uvf.metric_array[ind, 0, :, ipol] = alg_func(np.abs(data), flags=flags, **kwargs)
    elif issubclass(uv.__class__, UVCal):
        if cal_mode == 'tot_chisq':
            uvf.to_waterfall()
            for pi in range(uv.Njones):
                data = np.abs(uv.total_quality_array[0, :, :, pi].T)
                flags = np.all(uv.flag_array[:, 0, :, :, pi], axis=0).T
                uvf.metric_array[:, :, pi] = alg_func(data, flags=flags, **kwargs)
        else:
            for ai in range(uv.Nants_data):
                for pi in range(uv.Njones):
                    # Note transposes are due to freq, time dimensions rather than the
                    # expected time, freq
                    flags = uv.flag_array[ai, 0, :, :, pi].T
                    if cal_mode == 'gain':
                        data = np.abs(uv.gain_array[ai, 0, :, :, pi].T)
                    elif cal_mode == 'chisq':
                        data = np.abs(uv.quality_array[ai, 0, :, :, pi].T)
                    else:
                        raise ValueError('When calculating metric for UVCal object, '
                                         'cal_mode must be "gain", "chisq", or "tot_chisq".')
                    uvf.metric_array[ai, 0, :, :, pi] = alg_func(data, flags=flags, **kwargs).T
    return uvf


#############################################################################
# "Pipelines" -- these routines define the flagging strategy for some data
#   Note: "current" pipes should have simple names, but when replaced,
#         they should stick around with more descriptive names.
#############################################################################

def xrfi_h1c_pipe(uv, Kt=8, Kf=8, sig_init=6., sig_adj=2., px_threshold=0.2,
                  freq_threshold=0.5, time_threshold=0.05, return_summary=False,
                  cal_mode='gain'):
    """Run the xrfi excision pipeline we used for H1C.
    This pipeline uses the detrending and watershed algorithms above.
    Parameters
    ----------
    uv : UVData or UVCal
        The UVData or UVCal object to flag.
    Kt : int, optional
        The time size for detrending box. Default is 8.
    Kf : int, optional
        The frequency size for detrending box. Default is 8.
    sig_init : float, optional
        The initial sigma to flag. Default is 6.0.
    sig_adj : float, optional
        The number of sigma to flag adjacent to flagged data. Default is 2.0.
    px_threshold : float, optional
        The fraction of flags required to trigger a broadcast across baselines
        for a given (time, frequency) pixel. Default is 0.2.
    freq_threshold : float, optional
        The fraction of channels required to trigger broadcast across frequency
        (for a single time). Default is 0.5.
    time_threshold : float, optional
        The fraction of times required to trigger broadcast across
        time (for a single frequency). Default is 0.05.
    return_summary : bool, optional
        If True, return a UVFlag object with the fraction of baselines/antennas
        that were flagged in the initial flag/watershed (before broadcasting).
        Default is False.
    cal_mode : {"gain", "chisq", "tot_chisq"}, optional
        The mode to calculate metric if uv is a UVCal object. The options use
        the gain_array, quality_array, and total_quality_array attributes,
        respectively. Default is "gain".
    Returns
    -------
    uvf_f : UVFlag object
        A UVFlag object of the "initial flags" (initial flag + watershed).
    uvf_wf : UVFlag object
        A UVFlag object of waterfall type after thresholding in time/freq.
    uvf_w : UVFlag object
        If return_summary is True, a UVFlag object with fraction of flags in uvf_f.
    """
    uvf = calculate_metric(uv, 'detrend_medfilt', Kt=Kt, Kf=Kf, cal_mode=cal_mode)
    uvf_f = flag(uvf, nsig_p=sig_init, nsig_f=None, nsig_t=None)
    uvf_f = watershed_flag(uvf, uvf_f, nsig_p=sig_adj, nsig_f=None, nsig_t=None)
    uvf_w = copy.deepcopy(uvf_f)
    uvf_w.to_waterfall()
    # I realize the naming convention has flipped, which results in nsig_f=time_threshold.
    # time_threshold is defined as fraction of time flagged to flag a given channel.
    # nsig_f is defined as significance required to flag a channel.
    uvf_wf = flag(uvf_w, nsig_p=px_threshold, nsig_f=time_threshold,
                  nsig_t=freq_threshold)

    if return_summary:
        return uvf_f, uvf_wf, uvf_w
    else:
        return uvf_f, uvf_wf


def xrfi_pipe(uv, alg='detrend_medfilt', Kt=8, Kf=8, xants=[], cal_mode='gain',
              sig_init=6.0, sig_adj=2.0):
    """Run the xrfi excision pipeline used for H1C IDR2.2.
    This pipeline uses the detrending and watershed algorithms above.
    Parameters
    ----------
    uv : UVData or UVCal
        A UVData or UVCal object on which to calculate the metric.
    alg : str, optional
        The algorithm for calculating the metric. Default is "detrend_medfilt".
    Kt : int, optional
        The size of kernel in time dimension for detrending in the xrfi algorithm.
        Default is 8.
    Kf : int, optional
        The size of kernel in frequency dimension for detrending in the xrfi
        algorithm. Default is 8.
    xants : list, optional
        A list of antennas to flag. Default is an empty list.
    cal_mode : {"gain", "chisq", "tot_chisq"}, optional
        The mode to calculate metric if uv is a UVCal object. The options use
        the gain_array, quality_array, and total_quality_array attributes,
        respectively. Default is "gain".
    sig_init : float, optional
        The starting number of sigmas to flag on. Default is 6.0.
    sig_adj : float, optional
        The number of sigmas to flag on for data adjacent to a flag. Default is 2.0.
    Returns
    -------
    uvf_m : UVFlag object
        A UVFlag object with metric after collapsing to waterfall and to single pol.
        The weights array is set to ones.
    uvf_fws : UVFlag object
        A UVFlag object with flags after watershed.
    """
    flag_xants(uv, xants)
    uvf_m = calculate_metric(uv, alg, Kt=Kt, Kf=Kf, cal_mode=cal_mode)
    uvf_m.to_waterfall(keep_pol=False)
    # This next line resets the weights to 1 (with data) or 0 (no data) to equally
    # combine with the other metrics.
    uvf_m.weights_array = uvf_m.weights_array.astype(np.bool).astype(np.float)
    alg_func = algorithm_dict[alg]
    # Pass the z-scores through the filter again to get a zero-centered, width-of-one distribution.
    uvf_m.metric_array[:, :, 0] = alg_func(uvf_m.metric_array[:, :, 0],
                                           flags=~(uvf_m.weights_array[:, :, 0].astype(np.bool)),
                                           Kt=Kt, Kf=Kf)
    # Flag and watershed on each data product individually.
    # That is, on each complete file (e.g. calibration gains), not on individual
    # antennas/baselines. We don't broadcast until the very end.
    uvf_f = flag(uvf_m, nsig_p=sig_init)
    uvf_fws = watershed_flag(uvf_m, uvf_f, nsig_p=sig_adj, inplace=False)
    return uvf_m, uvf_fws

#############################################################################
# Wrappers -- Interact with input and output files
#   Note: "current" wrappers should have simple names, but when replaced,
#         they should stick around with more descriptive names.
#############################################################################


def xrfi_run(ocalfits_file, acalfits_file, model_file, data_file, history,
             init_metrics_ext='init_xrfi_metrics.h5', init_flags_ext='init_flags.h5',
             final_metrics_ext='final_xrfi_metrics.h5', final_flags_ext='final_flags.h5',
             xrfi_path='', kt_size=8, kf_size=8, sig_init=5.0, sig_adj=2.0,
             freq_threshold=0.35, time_threshold=0.5, ex_ants=None, metrics_file=None,
             cal_ext='flagged_abs', clobber=False):
    """Run the xrfi excision pipeline used for H1C IDR2.2.
    This pipeline uses the detrending and watershed algorithms above.
    Parameters
    ----------
    ocalfits_file : str
        The omnical calfits file to use to flag on gains and chisquared values.
    acalfits_file : str
        The abscal calfits file to use to flag on gains and chisquared values.
    model_file : str
        THe model visibility file to flag on.
    data_file : str
        The raw visibility data file to flag.
    history : str
        The history string to include in files.
    init_metrics_ext : str, optional
        The extension to be appended to input file name for the initial metric object.
        Default is "init_xrfi_metrics.h5".
    init_flags_ext : str, optional
        The extension to be appended to input file name for the initial flag object.
        Default is "init_flags.h5".
    final_metrics_ext : str, optional
        The extension to be appended to input file name for final metric object.
        Default is "final_xrfi_metrics.h5".
    final_flags_ext : str, optional
        The extension to be appended to input file name for final flag object.
        Default is "final_flags.h5".
    xrfi_path : str, optional
        Path to save xrfi files to. Default is the same directory as data_file.
    kt_size : int, optional
        The size of kernel in time dimension for detrend in xrfi algorithm.
        Default is 8.
    kf_size : int, optional
        Size of kernel in frequency dimension for detrend in xrfi algorithm.
        Default is 8.
    sig_init : float, optional
        The starting number of sigmas to flag on. Default is 5.0.
    sig_adj : float, optional
        The number of sigmas to flag on for data adjacent to a flag.
        Default is 2.0.
    freq_threshold : float, optional
        The fraction of times required to trigger a broadcast across times
        (at a single freq). Default is 0.35.
    time_threshold : float, optional
        The fraction of channels required to trigger broadcast across frequency
        (at a single time). Default is 0.5.
    ex_ants : str, optional
        A comma-separated list of antennas to exclude. Flags of visibilities formed
        with these antennas will be set to True. Default is None (i.e., no antennas
        will be excluded).
    metrics_file : str, optional
        Metrics file that contains a list of excluded antennas. Flags of visibilities
        formed with these antennas will be set to True. Default is None (i.e.,
        no antennas will be excluded).
    cal_ext : str, optional
        The extension to replace penultimate extension in a calfits file for output
        calibration including flags. Defaults is "flagged_abs". For example, an
        input_cal of "foo.goo.calfits" would result in "foo.flagged_abs.calfits".
    clobber : bool, optional
        If True, overwrite existing files. Default is False.
    Returns
    -------
    None
    """
    history = 'Flagging command: "' + history + '", Using ' + hera_qm_version_str
    dirname = resolve_xrfi_path(xrfi_path, data_file)
    xants = process_ex_ants(ex_ants=ex_ants, metrics_file=metrics_file)

    # Initial run on cal data products
    # Calculate metric on abscal data
    uvc_a = UVCal()
    uvc_a.read_calfits(acalfits_file)
    uvf_apriori = UVFlag(uvc_a, mode='flag', copy_flags=True)
    uvf_ag, uvf_agf = xrfi_pipe(uvc_a, alg='detrend_medfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj)
    uvf_ax, uvf_axf = xrfi_pipe(uvc_a, alg='detrend_medfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj)

    # Calculate metric on omnical data
    uvc_o = UVCal()
    uvc_o.read_calfits(ocalfits_file)
    flag_apply(uvf_apriori, uvc_o, keep_existing=True)
    uvf_og, uvf_ogf = xrfi_pipe(uvc_o, alg='detrend_medfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj)
    uvf_ox, uvf_oxf = xrfi_pipe(uvc_o, alg='detrend_medfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj)

    # Calculate metric on model vis
    uv_v = UVData()
    uv_v.read(model_file)
    uvf_v, uvf_vf = xrfi_pipe(uv_v, alg='detrend_medfilt', xants=[], Kt=kt_size, Kf=kf_size,
                              sig_init=sig_init, sig_adj=sig_adj)

    # Combine the metrics together
    uvf_metrics = uvf_v.combine_metrics([uvf_og, uvf_ox, uvf_ag, uvf_ax],
                                        method='quadmean', inplace=False)
    alg_func = algorithm_dict['detrend_medfilt']
    uvf_metrics.metric_array[:, :, 0] = alg_func(uvf_metrics.metric_array[:, :, 0],
                                                 flags=~uvf_metrics.weights_array[:, :, 0].astype(np.bool),
                                                 Kt=kt_size, Kf=kf_size)

    # Flag on combined metrics
    uvf_f = flag(uvf_metrics, nsig_p=sig_init)
    uvf_fws = watershed_flag(uvf_metrics, uvf_f, nsig_p=sig_adj, inplace=False)
    # OR everything together for initial flags
    uvf_apriori.to_waterfall(method='and', keep_pol=False)
    uvf_init = uvf_fws | uvf_ogf | uvf_oxf | uvf_agf | uvf_axf | uvf_vf | uvf_apriori

    # Write out initial (combined) metrics and flags
    basename = qm_utils.strip_extension(os.path.basename(data_file))
    outfile = '.'.join([basename, init_metrics_ext])
    outpath = os.path.join(dirname, outfile)
    uvf_metrics.write(outpath, clobber=clobber)
    outfile = '.'.join([basename, init_flags_ext])
    outpath = os.path.join(dirname, outfile)
    uvf_init.write(outpath, clobber=clobber)

    # Second round -- use init flags to mask and recalculate everything
    # Read in data file
    uv_d = UVData()
    uv_d.read(data_file)
    for uv in [uvc_o, uvc_a, uv_v, uv_d]:
        flag_apply(uvf_init, uv, keep_existing=True, force_pol=True)

    # Do next round of metrics
    # Change to meanfilt because it can mask flagged pixels
    # Calculate metric on abscal data
    uvf_ag2, uvf_agf2 = xrfi_pipe(uvc_a, alg='detrend_meanfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj)
    uvf_ax2, uvf_axf2 = xrfi_pipe(uvc_a, alg='detrend_meanfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj)

    # Calculate metric on omnical data
    uvf_og2, uvf_ogf2 = xrfi_pipe(uvc_o, alg='detrend_meanfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj)
    uvf_ox2, uvf_oxf2 = xrfi_pipe(uvc_o, alg='detrend_meanfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj)

    # Calculate metric on model vis
    uvf_v2, uvf_vf2 = xrfi_pipe(uv_v, alg='detrend_meanfilt', xants=[], Kt=kt_size, Kf=kf_size,
                                sig_init=sig_init, sig_adj=sig_adj)

    # Calculate metric on data file
    uvf_d2, uvf_df2 = xrfi_pipe(uv_d, alg='detrend_meanfilt', xants=[], Kt=kt_size, Kf=kf_size,
                                sig_init=sig_init, sig_adj=sig_adj)

    # Combine the metrics together
    uvf_metrics2 = uvf_d2.combine_metrics([uvf_og2, uvf_ox2, uvf_ag2, uvf_ax2, uvf_v2, uvf_d2],
                                          method='quadmean', inplace=False)
    alg_func = algorithm_dict['detrend_meanfilt']
    uvf_metrics2.metric_array[:, :, 0] = alg_func(uvf_metrics2.metric_array[:, :, 0],
                                                  flags=uvf_init.flag_array[:, :, 0],
                                                  Kt=kt_size, Kf=kf_size)

    # Flag on combined metrics
    uvf_f2 = flag(uvf_metrics2, nsig_p=sig_init)
    uvf_fws2 = watershed_flag(uvf_metrics2, uvf_f2, nsig_p=sig_adj, inplace=False)
    uvf_combined2 = (uvf_fws2 | uvf_ogf2 | uvf_oxf2 | uvf_agf2 | uvf_axf2
                     | uvf_vf2 | uvf_df2 | uvf_init)

    # Threshold
    uvf_temp = uvf_combined2.copy()
    uvf_temp.to_metric(convert_wgts=True)
    uvf_final = flag(uvf_temp, nsig_p=1.0, nsig_f=freq_threshold, nsig_t=time_threshold)

    # Write out final metrics and flags
    outfile = '.'.join([basename, final_metrics_ext])
    outpath = os.path.join(dirname, outfile)
    uvf_metrics2.write(outpath, clobber=clobber)
    outfile = '.'.join([basename, final_flags_ext])
    outpath = os.path.join(dirname, outfile)
    uvf_final.write(outpath, clobber=clobber)

    # Save calfits with new flags
    flag_apply(uvf_final, uvc_a, force_pol=True, history=history)
    basename = qm_utils.strip_extension(os.path.basename(acalfits_file))
    basename = qm_utils.strip_extension(basename)  # Also get rid of .abs
    outfile = '.'.join([basename, cal_ext, 'calfits'])
    outpath = os.path.join(dirname, outfile)
    uvc_a.write_calfits(outpath, clobber=clobber)


def xrfi_h1c_run(indata, history, infile_format='miriad', extension='flags.h5',
                 summary=False, summary_ext='flag_summary.h5', xrfi_path='',
                 model_file=None, model_file_format='uvfits',
                 calfits_file=None, kt_size=8, kf_size=8, sig_init=6.0, sig_adj=2.0,
                 px_threshold=0.2, freq_threshold=0.5, time_threshold=0.05,
                 ex_ants=None, metrics_file=None, filename=None):
    """Run the RFI-flagging algorithm from H1C and store results in npz files.
    This function runs on a single data file, and optionally calibration files, and
    writes the results to npz files.
    Notes
    -----
    This function will take in a UVData object or data file and optionally a cal file and
    model visibility file, and run an RFI-flagging algorithm to identify contaminated
    observations. Each set of flagging will be stored, as well as compressed versions.
    Parameters
    ----------
    indata : UVData or str
        A UVData object or data file on which to run RFI flagging.
    history : str
        The history string to include in files.
    infile_format : str, optional
        The file format for input files. Not currently used because we use pyuvdata's
        generic read function, but will be implemented for partial io.
    extension : str, optional
        The extension to be appended to input file name. Default is "flags.h5".
    summary : bool, optional
        If True, compute a summary of RFI flags and store in a .h5 file.
        Default is False.
    summary_ext : str, optional
        The extension for the summary file. Default is "flag_summary.h5".
    xrfi_path : str, optional
        The path to save flag files to. Default is the same directory as input file.
    model_file : str, optional
        The model visibility file to flag on. This step is skipped if not specified.
    model_file_format : str, optional
        The file format for input model file. Not currently used because we use
        pyuvdata's generic read function, but will be implemented for partial io.
    calfits_file : str, optional
        The calfits file to use to flag on gains and/or chisquared values.
    kt_size : int, optional
        The size of the kernel in time dimension for detrend in xrfi algorithm.
        Default is 8.
    kf_size : int, optional
        The size of the kernel in frequency dimension for detrend in xrfi algorithm.
        Default is 8.
    sig_init : float, optional
        The starting number of sigmas to flag on. Default is 6.0.
    sig_adj : float, optional
        The number of sigmas to flag on for data adjacent to a flag. Default is 2.0.
    px_threshold : float, optional
        The fraction of flags required to trigger a broadcast across baselines for
        a given (time, frequency) pixel. Default is 0.2.
    freq_threshold : float, optional
        The fraction of channels required to trigger a broadcast across frequency
        (for a single time). Default is 0.5.
    time_threshold : float, optional
        The fraction of times required to trigger a broadcast across time
        (for a single frequency). Default is 0.05.
    ex_ants : str, optional
        A comma-separated list of antennas to exclude. Flags of visibilities
        formed with these antennas will be set to True.
    metrics_file : str, optional
        A metrics file that contains a list of excluded antennas. Flags of
        visibilities formed with these antennas will be set to True.
    filename : str, optional
        The file for which to flag RFI (only one file allowed).
    Returns
    -------
    None
    Raises
    ------
    AssertionError:
        If "indata", "model_file", and "calfits_file" are all not provided,
        or if the filename for the input UVData object is not provided,
        an AssertionError is raised.
    ValueError:
        If filename is not a string, or if there is a mis-match in the time
        or frequency axes between the UVData object and the model visibility/
        calibration solution file, a ValueError is raised.
    """
    if indata is None:
        if (model_file is None) and (calfits_file is None):
            raise AssertionError('Must provide at least one of: indata, '
                                 'model_file, or calfits_file.')
        warnings.warn('indata is None, not flagging on any data visibilities.')
    elif issubclass(indata.__class__, UVData):
        uvd = indata
        if filename is None:
            raise AssertionError('Please provide a filename to go with UVData object. '
                                 'The filename is used in conjunction with "extension" '
                                 'to determine the output filename.')
        else:
            if not isinstance(filename, str):
                raise ValueError('filename must be string path to file.')
    else:
        if filename is None or filename == '':
            filename = indata
        elif not isinstance(filename, str):
            raise ValueError('filename must be string path to file.')
        uvd = UVData()
        uvd.read(filename)

    # append to history
    history = 'Flagging command: "' + history + '", Using ' + hera_qm_version_str

    # Flag on data
    if indata is not None:
        # Flag visibilities corresponding to specified antennas
        xants = process_ex_ants(ex_ants=ex_ants, metrics_file=metrics_file)
        flag_xants(uvd, xants)
        uvf_f, uvf_wf, uvf_w = xrfi_h1c_pipe(uvd, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                             sig_adj=sig_adj, px_threshold=px_threshold,
                                             freq_threshold=freq_threshold, time_threshold=time_threshold,
                                             return_summary=True)
        dirname = resolve_xrfi_path(xrfi_path, filename)
        basename = qm_utils.strip_extension(os.path.basename(filename))
        # Save watersheded flags
        outfile = '.'.join([basename, extension])
        outpath = os.path.join(dirname, outfile)
        uvf_f.history += history
        uvf_f.write(outpath, clobber=True)
        # Save thresholded waterfall
        outfile = '.'.join([basename, 'waterfall', extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath, clobber=True)
        if summary:
            sum_file = '.'.join([basename, summary_ext])
            sum_path = os.path.join(dirname, sum_file)
            uvf_w.history += history
            uvf_w.write(sum_path, clobber=True)

    # Flag on model visibilities
    if model_file is not None:
        uvm = UVData()
        uvm.read(model_file)
        if indata is not None:
            if not (np.allclose(np.unique(uvd.time_array), np.unique(uvm.time_array),
                                atol=1e-5, rtol=0)
                    and np.allclose(uvd.freq_array, uvm.freq_array, atol=1., rtol=0)):
                raise ValueError('Time and frequency axes of model vis file must match'
                                 'the data file.')
        uvf_f, uvf_wf = xrfi_h1c_pipe(uvm, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                      sig_adj=sig_adj, px_threshold=px_threshold,
                                      freq_threshold=freq_threshold, time_threshold=time_threshold)
        dirname = resolve_xrfi_path(xrfi_path, model_file)
        # Only save thresholded waterfall
        basename = qm_utils.strip_extension(os.path.basename(model_file))
        outfile = '.'.join([basename, extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath, clobber=True)

    # Flag on gain solutions and chisquared values
    if calfits_file is not None:
        uvc = UVCal()
        uvc.read_calfits(calfits_file)
        if indata is not None:
            if not (np.allclose(np.unique(uvd.time_array), np.unique(uvc.time_array),
                                atol=1e-5, rtol=0)
                    and np.allclose(uvd.freq_array, uvc.freq_array, atol=1., rtol=0)):
                raise ValueError('Time and frequency axes of calfits file must match'
                                 'the data file.')
        # By default, runs on gains
        uvf_f, uvf_wf = xrfi_h1c_pipe(uvd, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                      sig_adj=sig_adj, px_threshold=px_threshold,
                                      freq_threshold=freq_threshold, time_threshold=time_threshold)
        dirname = resolve_xrfi_path(xrfi_path, calfits_file)
        basename = qm_utils.strip_extension(os.path.basename(calfits_file))
        outfile = '.'.join([basename, 'g', extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath, clobber=True)
        # repeat for chisquared
        uvf_f, uvf_wf = xrfi_h1c_pipe(uvd, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                      sig_adj=sig_adj, px_threshold=px_threshold,
                                      freq_threshold=freq_threshold, time_threshold=time_threshold,
                                      cal_mode='chisq')
        outfile = '.'.join([basename, 'x', extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath)

    return


def xrfi_h1c_apply(filename, history, infile_format='miriad', xrfi_path='',
                   outfile_format='miriad', extension='R', overwrite=False,
                   flag_file=None, waterfalls=None, output_uvflag=True,
                   output_uvflag_ext='flags.h5'):
    """Apply flags in the fashion of H1C.
    Read in a flag array and optionally several waterfall flags, and insert into
    a data file.
    Parameters
    ----------
    filename : str
        Data file in which update flag array.
    history : str
        The history string to include in files.
    infile_format : str, optional
        File format for input files. Not currently used because we use pyuvdata's
        generic read function, but will be implemented for partial io.
    xrfi_path : str, optional
        The path to save output to. Default is same directory as input file.
    outfile_format : {"miriad", "uvfits", "uvh5"}, optional
        The file format for output files. Default is "miriad".
    extension : str, optional
        The extension to be appended to input file name. Default is "R".
    overwrite : bool, optional
        If True, overwrite the output file if it already exists. Default is False.
    flag_file : str, optional
        The path to the npz file containing full flag array to insert into data file.
    waterfalls, optional
        A list or comma separated string of npz file names containing waterfalls
        of flags to broadcast to full flag array and union with flag array in flag_file.
    output_uvflag : bool, optional
        If True, save a UVFlag file with the final flag array. The flag array will
        be identical to what is stored in the data.
    output_uvflag_ext : str, optional
        The extension to be appended to input file name. Default is "flags.h5".
    Returns
    -------
    None
    Raises
    ------
    AssertionError:
        If no input filename is provided, an AssertionError is raised.
    ValueError:
        If outfile_format is not valid, or if the target output file exists and
        overwrite is False, a ValueError is raised.
    """
    # make sure we were given files to process
    if len(filename) == 0:
        raise AssertionError('Please provide a visibility file')
    if isinstance(filename, (list, np.ndarray, tuple)) and len(filename) > 1:
        raise AssertionError('xrfi_apply currently only takes a single data file.')
    if isinstance(filename, (list, np.ndarray, tuple)):
        filename = filename[0]
    uvd = UVData()
    uvd.read(filename)

    full_list = []
    # Read in flag file
    if flag_file is not None:
        full_list += [flag_file]

    # Read in waterfalls
    if waterfalls is not None:
        if not isinstance(waterfalls, list):
            # Assume comma separated list
            waterfalls = waterfalls.split(',')
        full_list += waterfalls

    uvf = flag_apply(full_list, uvd, force_pol=True, return_net_flags=True)

    # save output when we're done
    dirname = resolve_xrfi_path(xrfi_path, filename)
    basename = qm_utils.strip_extension(os.path.basename(filename))
    outfile = '.'.join([basename, extension])
    outpath = os.path.join(dirname, outfile)
    extension_dict = {'miriad': '.uv', 'uvfits': '.uvfits', 'uvh5': '.uvh5'}
    try:
        outpath += extension_dict[outfile_format]
    except KeyError:
        raise ValueError('Unrecognized output file format ' + str(outfile_format))
    if outfile_format == 'miriad':
        uvd.write_miriad(outpath, clobber=overwrite)
    elif outfile_format == 'uvfits':
        if os.path.exists(outpath) and not overwrite:
            raise ValueError('File ' + outpath + ' exists: skipping')
        uvd.write_uvfits(outpath, force_phase=True, spoof_nonessential=True)
    elif outfile_format == 'uvh5':
        uvd.write_uvh5(outpath, clobber=overwrite)
    if output_uvflag:
        # Save uvflag with the final flag array and relevant metadata
        outfile = '.'.join([basename, extension, output_uvflag_ext])
        outpath = os.path.join(dirname, outfile)
uvf.write(outpath)
