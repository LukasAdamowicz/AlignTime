"""
Methods for aligning time-series data with different time stamps.

Lukas Adamowicz
GNU GPL 3.0
May 28, 2019
"""
from numpy import mean, diff, searchsorted, argmax, abs as nabs, zeros, arange, ndarray, timedelta64
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy import fftpack
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, date2num, drange
from matplotlib.widgets import SpanSelector, Cursor
from pandas import to_timedelta, to_datetime


class AlignTime:
    def __init__(self, filter=True, filt_ord=4, filt_cut=5, datetime=False):
        """
        Method for aligning time-stamps of different time series data

        Parameters
        ----------
        filter : {bool, array_like}, optional
            Filter the input data. Either bool for both data series, or an array_like of bools if only one series of
            data should be filtered.
        filt_ord : {int, array_like}, optional
            Filter order for the filtering process. Either a single int for both data series, or an array_like of 2
            ints, with orders for the first and second time series. Default is 4. Ignored if filter is False.
        filt_cut : {float, array_like}, optional
            Low-pass filter cutoffs for the filtering process. Either a signal float for both data series, or an
            array_like of 2 floats, with orders for the first and second time series. Default is 5Hz. Ignored if filter
            is False.
        datetime : bool, optional
            Whether or not datetime units are provided for the the time. Default is False, in which case seconds
            are expected
        """
        # assign values as appropriate
        if isinstance(filter, (tuple, list, ndarray)):
            self._filt1, self._filt2 = filter
        else:
            self._filt1, self._filt2 = filter, filter

        if isinstance(filt_ord, (tuple, list, ndarray)):
            self._ord1, self._ord2 = filt_ord
        else:
            self._ord1, self._ord2 = filt_ord, filt_ord

        if isinstance(filt_cut, (tuple, list, ndarray)):
            self._cut1, self._cut2 = filt_cut
        else:
            self._cut1, self._cut2 = filt_cut, filt_cut

        self.datetime = datetime

        # line for plotting the aligned signals
        self.line = None

    def fit(self, time1, data1, time2, data2, dt1=None, dt2=None, xlim1=None, xlim2=None, xnear1=None, xnear2=None):
        """
        Align the two time series

        Parameters
        ----------
        time1 : numpy.ndarray
            (N1, ) array of time-stamps for the first series of data. Will be resampled to match the sampling rate of
            the second data series if necessary during the alignment process.
        data1 : numpy.ndarray
            (N1, M1) array of the first series of data
        time2 : numpy.ndarray
            (N2, ) array of time-stamps for the second series of data. Does not have to be the same sampling rate as
            the first series of data. The first series of data will be resampled to match that of the second series
        data2 : numpy.ndarray
            (N2, M2) array of the second series of data
        dt1 : {None, float}, optional
            Sampling time for the first series time stamps, if necessary. Default is None, which will be ignored and
            the mean difference in time-stamps for the first 100 samples of the provided time stamps will be used.
        dt2 : {None, float}, optional
            Sampling time for the second series time stamps, if necessary. Default is None, which will be ignored and
            the mean difference in time-stamps for the first 100 samples of the provided time stamps will be used.
        xlim1 : array_like, optional
            X-limits for plotting series 1 data. Useful if you know approximately where the events to time sync are
            located in the series 1 data.
        xlim2 : array_like, optional
            X-limits for plotting series 2 data. Useful if you now approximately where the events to time sync are
            located in the series 2 data.
        xnear1 : float, optional
            X-value where to search near in the first signal. Will be marked on the graph with a vertical line.
        xnear2 : float, optional
            X-value where to search near in the second signal. Will be marked on the graph with a vertical line.

        Returns
        -------
        dt_1_2 : float
            The time difference between series 2 and series 1, calculated by subtracting the aligned time 2 - time 1

        Attributes
        ----------
        t1_0 : float
            Aligned time int time1 detected by the convolution of the two signals in th regions chosen.
        t2_0 : float
            Aligned time in time2 detected by the convolution of the two signals in the regions chosen.
        """
        # assign the times
        self._t1 = time1
        self._t2 = time2

        # assign the raw data
        self._rx1 = data1
        self._rx2 = data2

        # compute the sampling times
        if dt1 is None:
            if self.datetime:
                self._dt1 = mean(diff(self._t1[:100])) / timedelta64(1, 's')
            else:
                self._dt1 = mean(diff(self._t1[:100]))
        else:
            self._dt1 = dt1
        if dt2 is None:
            if self.datetime:
                self._dt2 = mean(diff(self._t2[:100])) / timedelta64(1, 's')
            else:
                self._dt2 = mean(diff(self._t2[:100]))
        else:
            self._dt2 = dt2

        # filter the data
        if self._filt1:
            fc1 = butter(self._ord1, 2 * self._cut1 * self._dt1, btype='low')
            self._x1 = filtfilt(fc1[0], fc1[1], self._rx1)
        else:
            self._x1 = self._rx1

        if self._filt2:
            fc2 = butter(self._ord2, 2 * self._cut2 * self._dt2, btype='low')
            self._x2 = filtfilt(fc2[0], fc2[1], self._rx2)
        else:
            self._x2 = self._rx2
            
        # plot the data
        self._f, (self._ax1, self._ax2) = plt.subplots(2, figsize=(20, 10))

        if self._filt1:
            self._ax1.plot(self._t1, self._rx1, color='C0', linewidth=1.5, label='Raw', alpha=0.7)
        if self._filt2:
            self._ax2.plot(self._t2, self._rx2, color='C0', linewidth=1.5, label='Raw', alpha=0.7)
        self._ax1.plot(self._t1, self._x1, color='C1', linewidth=2, label='Filtered')
        self._ax2.plot(self._t2, self._x2, color='C1', linewidth=2, label='Filtered')

        if xlim1 is not None:
            self._ax1.set_xlim(xlim1)
        if xlim2 is not None:
            self._ax2.set_xlim(xlim2)

        if xnear1 is not None:
            self._ax1.axvline(xnear1, color='k', linewidth=3, alpha=0.5)
        if xnear2 is not None:
            self._ax2.axvline(xnear2, color='k', linewidth=3, alpha=0.5)

        self._ax1.legend(loc='best')
        self._ax2.legend(loc='best')

        # create the cursor and span selectors for the first axis
        self._ax1.set_title('Navigate and select the region to check: ')  # let user know what they are doing
        self._cursor1 = Cursor(self._ax1, color='C0', useblit=True, linewidth=1)
        self._span1 = SpanSelector(self._ax1, self._on_select1, 'horizontal', useblit=True,
                                   rectprops=dict(alpha=0.5, facecolor='red'), button=1, span_stays=True)

        self._ax2.set_title('Navigate and select region to match: ')
        self._cursor2 = Cursor(self._ax2, color='C0', useblit=True, linewidth=1)
        self._span2 = SpanSelector(self._ax2, self._on_select2, 'horizontal', useblit=True,
                                   rectprops=dict(alpha=0.5, facecolor='red'), button=1, span_stays=True)

        self._cursor2.set_active(False)
        self._span2.set_visible(False)

        self._f.tight_layout()

        plt.show(block=True)

        return self.t_diff

    def _on_select1(self, xmin, xmax):
        self._ax1.set_title(None)
        if self.datetime:
            t1 = date2num(self._t1)
        else:
            t1 = self._t1

        start, stop = searchsorted(t1, (xmin, xmax))

        f = interp1d(t1[start - 10:stop + 10], self._x1[start - 10:stop + 10], kind='cubic')
        if self.datetime:
            self._ta = drange(self._t1[start], self._t1[stop], to_timedelta(self._dt2, unit='s'))
        else:
            self._t1 = arange(self._t1[start], self._t1[stop], self._dt2)

        self._a = f(self._ta)

        self._cursor2.set_active(True)
        self._span2.set_visible(True)

    def _on_select2(self, xmin, xmax):
        self._ax2.set_title(None)
        if self.datetime:
            t2 = date2num(self._t2)
        else:
            t2 = self._t2
        start, stop = searchsorted(t2, (xmin, xmax))

        self._b = zeros(self._a.shape)
        self._tb = t2[stop - self._a.size:stop]

        self._b[-(stop - start):] = self._x2[start:stop]

        A = fftpack.fft(self._a)
        B = fftpack.fft(self._b)

        Ar = -A.conjugate()

        self._ind_shift = argmax(nabs(fftpack.ifft(Ar * B)))  # convolve the arrays and find the peak

        self.t1_0 = self._ta[0]  # find the time value in the first signal
        self.t2_0 = self._tb[self._ind_shift]  # find the time value in the second signal

        t_pl = self._ta + (self.t2_0 - self.t1_0)
        x_pl = self._a

        if self.line is not None:
            self.line.set_data([], [])
        self.line, = self._ax2.plot(t_pl, x_pl, color='C2', label='Aligned')

        if self.datetime:
            self.t1_0 = to_datetime(num2date(self.t1_0)).tz_localize(None)
            self.t2_0 = to_datetime(num2date(self.t2_0)).tz_localize(None)

        # time difference between the signals
        self.t_diff = self.t2_0 - self.t1_0


