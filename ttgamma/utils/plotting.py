import hist
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import itertools

def SetRangeHist(histogram, axisName, lower_bound=None, upper_bound=None):
    edges = list(histogram.axes[axisName].edges)
    i_min = edges.index(lower_bound)
    i_max = edges.index(upper_bound)

    return histogram[{axisName:slice(i_min,i_max)}]

# Taken from Yi-Mu: https://gist.github.com/yimuchen/a5e200c001ef4ea01681a7dd8fe89162#file-integrate_and_rebin-py-L158
def RebinHist(h, **kwargs):
    """
    Rebinning a scikit-hist histogram. 2 types of values can be accepted as the
    argument values:
    - Derivatives of the `hist.rebin` argument. In this case we directly use the
    UHI facilities to perform the rebinning.
    - A new axis object where all the bin edges lands on the old bin edges of the
    given histogram. In this case a custom intergration loop is performed to
    extract the rebinning. Beware that this methods is very slow, as it requires
    a loop generation of all possible UHI values after the rebinning, so be sure
    that rebinning is performed as the final step of the histogram reduction. See
    `_rebin_single_scikit` for more information regarding this method.
    """
    h = h.copy()
    for var, var_val in kwargs.items():
        if isinstance(var_val, hist.rebin):
            h = h[{var: var_val}]
        else:
            h = _rebin_single_scikit(h, var, var_val)
    return h


def __check_scikit_axis_compat(axis1, axis2):
    """
    Checking that axis 2 is rebin-compatible with axis 1. This checks that:
    1. The two histogram share the same name.
    2. The edges of the second axis all land on the edges of the first axis.
    If the two axis are compatible the function will return an array of the bin
    index of the axis 1 that the bin edges of axis 2 falls on.
    """
    assert axis1.name == axis2.name, \
    'Naming of the axis is required to match'
    # Getting the new bin edges index for the old bin edges
    try:
        return [
        np.argwhere(axis1.edges == new_edge)[0][0] for new_edge in axis2.edges
        ]
    except IndexError as err:
        raise ValueError(f"Bin edges of the axis {axis2} is incompatible with {axis1}")


def _get_all_indices(axis):
    """
    Getting all possible (integer) bin index values given a scikit-hep histogram.
    The special indices of hist.underflow and hist.overflow will be included if the
    axis in questions has those traits.
    """
    idxs = list(range(len(axis)))
    if axis.traits.underflow:  # Extension to include the under/overflow bins
        idxs.insert(0, hist.underflow)
        if axis.traits.overflow:
            idxs.append(hist.overflow)
    return idxs


def _rebin_single_scikit(h, old_axis, new_axis):
    """
    Rebinning a single axis of a scikit-hist histogram. This includes the following
    routines:
    - Generating a new scikit hep instance that perserves axis ordering with the
    exception of the rebinned axis (in place) replacement.
    - Setting up the integration ranges required to calculate the bin values of the
    new histogram.
    - Looping over the UHI values of the new histogram and performing the a
    summation over the specified range on the old histogram to fill in the new
    values.
    As here we have variable number of axis each with variable number of bins, this
    method will require the use of more old fashioned python looping, which can be
    very slow for large dimensional histograms with many bins for each axis. So be
    sure to make rebinning be the final step in histogram reduction.
    """

    #assert isinstance(h, hist.NamedHist), "Can only process named histograms"

    # Additional type casing
    if type(old_axis) == str:
        return _rebin_single_scikit(h, h.axes[old_axis], new_axis)
    axis_name = old_axis.name

    ## Creating the new histogram instance with identical axis ordering.
    all_axes = list(h.axes)
    all_axes[all_axes.index(old_axis)] = new_axis
    h_rebinned = hist.NamedHist(*all_axes, storage=h._storage_type())

    # Getting the all possible bin indices for all axes in the old histogram
    bin_idx_dict = {ax.name: _get_all_indices(ax) for ax in h.axes}

    # Getting the new bin edges index for the old bin edges
    new_bin_edge_idx = __check_scikit_axis_compat(old_axis, new_axis)
    if new_axis.traits.underflow:  # Adding additional underflow/overflow
        new_bin_edge_idx.insert(0, bin_idx_dict[axis_name][0])
        if new_axis.traits.overflow:
            new_bin_edge_idx.append(bin_idx_dict[axis_name][-1])

    # Generating a the int range pair. Additional parsing will be required for the
    # under/overflow bins

    def make_slice(index):

        start = new_bin_edge_idx[index]
        stop = new_bin_edge_idx[index + 1]
        if start == hist.underflow:
            start = -1
        if stop == hist.overflow:
            stop = len(old_axis)
        return slice(int(start), int(stop))

    new_axis_idx = _get_all_indices(new_axis)
    new_int_slice = [make_slice(i) for i in range(len(new_axis_idx))]
    assert len(new_axis_idx) == len(new_bin_edge_idx) - 1

    new_idx_dict = bin_idx_dict.copy()
    new_idx_dict[axis_name] = new_axis_idx
    bin_idx_dict[axis_name] = new_int_slice

    name_list = list(bin_idx_dict.keys())
    new_idx = [x for x in itertools.product(*[x for x in new_idx_dict.values()])]
    old_int = [x for x in itertools.product(*[x for x in bin_idx_dict.values()])]

    for o, n in zip(old_int, new_idx):
        n_uhi = {name: n[name_idx] for name_idx, name in enumerate(name_list)}
        o_uhi = {name: o[name_idx] for name_idx, name in enumerate(name_list)}
        # Single variable histogram, with just the axis of interest
        h_rebinned[n_uhi] = integrate_hist_scikit(h, **o_uhi)

    return h_rebinned

def integrate_hist_scikit(h, **kwargs):
    """
    Given a scikit-hist histogram object return a reduced histogram with specified
    axes integrated out.
    For scikit-hist histograms, the integration should be formed in 3 steps:
    - slicing the histogram to contain only the range of interest
    - Setting overflow values to 0 (excluding the values from future calculations)
    - Summing over the axes of interest.
    The latter 2 steps will only be carried out if the var_slice doesn't uniquely
    identify a singular bin in the histogram axis
    """
    # Reduction in parallel.
    r = h[kwargs]
    for var, var_slice in kwargs.items():
        # In the case that histogram has been reduced to singular value simple return
        #if not isinstance(r, hist.NamedHist):
        #    return r
        if var in [x.name for x in r.axes]:
            ax = h.axes[var]
            
            get_underflow = var_slice.start == None or var_slice.start == -1
            get_overflow = var_slice.stop == None or var_slice.stop == len(ax)
            if not get_underflow and ax.traits.underflow:
                r[{var: hist.underflow}] = np.zeros_like(r[{var: hist.underflow}])
            if not get_overflow and ax.traits.overflow:
                r[{var: hist.overflow}] = np.zeros_like(r[{var: hist.overflow}])
                
            # Sum over all remaining elements on axis
            r = r[{var: sum}]
    return r

# Thanks for this, Nick and Andrzej ;)
def GroupBy(h, oldname, newname, grouping):
    hnew = hist.Hist(
        hist.axis.StrCategory(grouping, name=newname),
        *(ax for ax in h.axes if ax.name != oldname),
        storage=h._storage_type,
    )
    for i, indices in enumerate(grouping.values()):
        hnew.view(flow=True)[i] = h[{oldname: indices}][{oldname: sum}].view(flow=True)

    return hnew

def plotWithRatio(
    h,
    hData,
    overlay,
    stacked=True,
    density=False,
    lumi=35.9,
    label="CMS Preliminary",
    colors=None,
    ratioRange=[0.5, 1.5],
    xRange=None,
    yRange=None,
    logY=False,
    extraText=None,
    leg="upper right",
    binwnorm=None
):

    # make a nice ratio plot
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    if not hData is None:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
        )
        fig.subplots_adjust(hspace=0.07)
    else:
        fig, ax = plt.subplots(
            1, 1, figsize=(7, 7)
        )  # , gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

    # Here is an example of setting up a color cycler to color the various fill patches
    # http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=6
    from cycler import cycler

    if not colors is None:
        if invertStack:
            _n = len(h.identifiers(overlay)) - 1
            colors = colors[_n::-1]
        ax.set_prop_cycle(cycler(color=colors))

    h.plot(
        overlay=overlay,
        ax=ax,
        stack=stacked,
        histtype='fill',
        binwnorm=binwnorm,
        edgecolor='black',
        linewidth=1,
    )
    
    if binwnorm:
        
        mcStatUp = np.append((h[{overlay:sum}].values() + np.sqrt(h[{overlay:sum}].variances()))/np.diff(hData.axes[0].edges),[0])
        mcStatDo = np.append((h[{overlay:sum}].values() - np.sqrt(h[{overlay:sum}].variances()))/np.diff(hData.axes[0].edges),[0])
    
        uncertainty_band = ax.fill_between(
            hData.axes[0].edges,
            mcStatUp,
            mcStatDo,
            step='post',
            hatch='///',
            facecolor='none',
            edgecolor='gray',
            linewidth=0,
    )
    else:
        
        mcStatUp = np.append(h[{overlay:sum}].values() + np.sqrt(h[{overlay:sum}].variances()),[0])
        mcStatDo = np.append(h[{overlay:sum}].values() - np.sqrt(h[{overlay:sum}].variances()),[0])
    
        uncertainty_band = ax.fill_between(
            hData.axes[0].edges,
            mcStatUp,
            mcStatDo,
            step='post',
            hatch='///',
            facecolor='none',
            edgecolor='gray',
            linewidth=0,
    )

    if not hData is None:
        
        if binwnorm:
            ax.errorbar(x=hData.axes[0].centers,
                        y=hData.values()/np.diff(hData.axes[0].edges),
                        yerr=np.sqrt(hData.values())/np.diff(hData.axes[0].edges),
                        color='black',
                        marker='.',
                        markersize=10,
                        linewidth=0,
                        elinewidth=0.5,
                        label="Data",
            )
        else:
            ax.errorbar(x=hData.axes[0].centers,
                        y=hData.values(),
                        yerr=np.sqrt(hData.values()),
                        color='black',
                        marker='.',
                        markersize=10,
                        linewidth=0,
                        elinewidth=1,
                        label="Data",
            )
        
    if not binwnorm is None:
        ax.set_ylabel(f"<Events/{binwnorm}>")
        if "[" in ax.get_xlabel():
            units = ax.get_xlabel().split("[")[-1].split("]")[0]
            ax.set_ylabel(f"<Events / {binwnorm} {units}>")
    else:
        ax.set_ylabel('Events')

    ax.autoscale(axis="x", tight=True)
    ax.set_ylim(0, None)

    ax.set_xlabel(None)

    if leg == "right":
        leg_anchor = (1.0, 1.0)
        leg_loc = "upper left"
    elif leg == "upper right":
        leg_anchor = (1.0, 1.0)
        leg_loc = "upper right"
    elif leg == "upper left":
        leg_anchor = (0.0, 1.0)
        leg_loc = "upper left"

    if not leg is None:
        ax.legend(bbox_to_anchor=leg_anchor, loc=leg_loc)
        
    ratio_mcStatUp = np.append(1 + np.sqrt(h[{overlay:sum}].variances())/h[{overlay:sum}].values(),[0])
    ratio_mcStatDo = np.append(1 - np.sqrt(h[{overlay:sum}].variances())/h[{overlay:sum}].values(),[0])
    
        
    ratio_uncertainty_band = rax.fill_between(
        hData.axes[0].edges,
        ratio_mcStatUp,
        ratio_mcStatDo,
        step='post',
        color='lightgray',
    )
        
    if not hData is None:
        
        hist_1_values, hist_2_values = hData.values(), h[{overlay:sum}].values()
    
        ratios = hist_1_values / hist_2_values
        ratio_uncert = hist.intervals.ratio_uncertainty(
            num=hist_1_values,
            denom=hist_2_values,
            uncertainty_type="poisson",
            
        )
        # ratio: plot the ratios using Matplotlib errorbar or bar
        hist.plot.plot_ratio_array(
            hData, ratios, ratio_uncert, ax=rax, uncert_draw_type='line',
        );

        rax.set_ylim(ratioRange[0], ratioRange[1])

    if logY:
        ax.set_yscale("log")
        ax.set_ylim(1, ax.get_ylim()[1] * 5)

    if not xRange is None:
        ax.set_xlim(xRange[0], xRange[1])
    if not yRange is None:
        ax.set_ylim(yRange[0], yRange[1])

    CMS = plt.text(
        0.0,
        1.0,
        r"$\bf{CMS}$ Preliminary",
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )

    if not extraText is None:

        extraLabel = plt.text(
            0.02,
            0.99,
            extraText,
            fontsize=16,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

    lumi = plt.text(
        1.0,
        1.0,
        r"%.1f fb$^{-1}$ (13 TeV)" % (lumi),
        fontsize=16,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    
