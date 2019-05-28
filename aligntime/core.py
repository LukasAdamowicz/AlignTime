"""
Methods for aligning time-series data with different time stamps.

Lukas Adamowicz
GNU GPL 3.0
May 28, 2019
"""
from numpy import mean, diff, searchsorted, argmax, abs as nabs, zeros, arange, ndarray
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy import fftpack
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Cursor


class AlignTime:
    def __init__(self, filter=True, filt_ord=4, filt_cut=5):
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
        """
        # assign values as appropriate
        if isinstance(filter, (tuple, list, ndarray)):
            self._filt1, self._filt2 = filter
        else:
            self._filt1, self.filt_2 = filter, filter

        if isinstance(filt_ord, (tuple, list, ndarray)):
            self._ord1, self._ord2 = filt_ord
        else:
            self._ord1, self._ord2 = filt_ord, filt_ord

        if isinstance(filt_cut, (tuple, list, ndarray)):
            self._cut1, self._cut2 = filt_cut
        else:
            self._cut1, self._cut2 = filt_cut, filt_cut

        # line for plotting the aligned signals
        self.line = None

    def fit(self, time1, data1, time2, data2, dt1=None, dt2=None):
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
            self._dt1 = mean(diff(self._t1[:100]))
        else:
            self._dt1 = dt1
        if dt2 is None:
            self._dt2 = mean(diff(self._t2[:100]))
        else:
            self._dt2 = dt2

        # filter the data
        if filter:
            b1, a1 = butter(self._ord1, 2 * self._cut1 * self._dt1, btype='low')
            b2, a2 = butter(self._ord2, 2 * self._cut2 * self._dt2, btype='low')

            self._x1 = filtfilt(b1, a1, self._rx1)
            self._x2 = filtfilt(b2, a2, self._rx2)
        else:
            self._x1 = self._rx1
            self._x2 = self._rx2

        # plot the data
        self._f, (self._ax1, self._ax2) = plt.subplots(2, figsize=(20, 10))

        if filter:
            self._ax1.plot(self._t1, self._rx1, color='C0', linewidth=1.5, label='Raw', alpha=0.7)
            self._ax2.plot(self._t2, self._rx2, color='C0', linewidth=1.5, label='Raw', alpha=0.7)
        self._ax1.plot(self._t1, self._x1, color='C1', linewidth=2, label='Filtered')
        self._ax2.plot(self._t2, self._x2, color='C1', linewidth=2, label='Filtered')

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

        self._f.tight_layout()

        plt.show(block=True)

        return self.t_diff

    def _on_select1(self, xmin, xmax):
        self._ax1.set_title(None)
        start, stop = searchsorted(self._t1, (xmin, xmax))

        f = interp1d(self._t1[start - 10:stop + 10], self._x1[start - 10:stop + 10], kind='cubic')
        self._ta = arange(self._t1[start], self._t1[stop], self._dt2)

        self._a = f(self._ta)

    def _on_select2(self, xmin, xmax):
        self._ax2.set_title(None)
        start, stop = searchsorted(self._t2, (xmin, xmax))

        self._b = zeros(self._a.shape)
        self._tb = self._t2[stop - self._a.size:stop]

        self._b[-(stop - start):] = self._x2[start:stop]

        A = fftpack.fft(self._a)
        B = fftpack.fft(self._b)

        Ar = -A.conjugate()

        self._ind_shift = argmax(nabs(fftpack.ifft(Ar * B)))  # convolve the arrays and find the peak

        self.t1_0 = self._ta[0]  # find the time value in the first signal
        self.t2_0 = self._tb[self._ind_shift]  # find the time value in the second signal

        # time difference between the signals
        self.t_diff = self.t2_0 - self.t1_0

        t_pl = self._ta + (self.t2_0 - self.t1_0)
        x_pl = self._a

        if self.line is not None:
            self.line.set_data([], [])
        self.line, = self._ax2.plot(t_pl, x_pl, color='C2', label='Aligned')


