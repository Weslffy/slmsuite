"""
Datastructures, methods, and calibrations for an SLM monitored by a camera.
"""

import os
import time
import cv2
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, ndimage
from scipy.spatial import Delaunay, Voronoi, delaunay_plot_2d
from tqdm.auto import tqdm
import warnings

_logger = logging.getLogger(__name__)

from slmsuite import __version__
from slmsuite.hardware import _Picklable
from slmsuite.holography import analysis
from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import Hologram, SpotHologram, CompressedSpotHologram
from slmsuite.holography.toolbox import imprint, format_2vectors, format_vectors, smallest_distance, fit_3pt, convert_vector
from slmsuite.holography.toolbox.phase import blaze, _zernike_indices_parse, zernike, zernike_sum, binary, ZERNIKE_NAMES
from slmsuite.holography.analysis import image_remove_blaze, image_remove_vortices, image_reduce_wraps
from slmsuite.holography.analysis.files import load_h5, save_h5, generate_path, latest_path
from slmsuite.holography.analysis.fitfunctions import cos, _sinc2d_nomod
from slmsuite.holography.analysis.fitfunctions import _sinc2d_centered_taylor as sinc2d_centered
from slmsuite.misc.math import INTEGER_TYPES, REAL_TYPES

from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.slms.simulated import SimulatedSLM

class CameraSLM(_Picklable):
    """
    Base class for an SLM with camera feedback.

    Attributes
    ----------
    cam : ~slmsuite.hardware.cameras.camera.Camera
        Instance of :class:`~slmsuite.hardware.cameras.camera.Camera`
        which interfaces with a camera. This camera is
        used to provide closed-loop feedback to an SLM for calibration and holography.
    slm : ~slmsuite.hardware.slms.slm.SLM
        Instance of :class:`~slmsuite.hardware.slms.slm.SLM`
        which interfaces with a phase display.
    name : str
        Stores ``cam.name + '-' + slm.name``.
    mag : float
        Magnification of the camera relative to an experiment plane. For instance,
        ``mag = 10`` could refer to the use of a 10x objective (with appropriate
        imaging lensing) between the experiment plane and the camera.
        In this case, the images apparent on the camera are 10x larger than the true
        objects at the experiment plane.
    """
    _pickle = ["name", "cam", "slm", "mag"]
    _pickle_data = []

    def __init__(self, cam, slm, mag=1):
        """
        Initialize an SLM linked to a camera, with given magnification between the
        camera and experiment planes.

        Parameters
        ----------
        cam : ~slmsuite.hardware.cameras.camera.Camera
            Instance of :class:`~slmsuite.hardware.cameras.camera.Camera`
            which interfaces with a camera. This camera is
            used to provide closed-loop feedback to an SLM for calibration and holography.
        slm : ~slmsuite.hardware.slms.slm.SLM
            Instance of :class:`~slmsuite.hardware.slms.slm.SLM`
            which interfaces with a phase display.
        mag : float
            Magnification of the camera relative to an experiment plane. For instance,
            ``mag = 10`` could refer to the use of a 10x objective (with appropriate
            imaging lensing) between the experiment plane and the camera.
            In this case, the images apparent on the camera are ten times larger than
            the true objects at the experiment plane.

            Note
            ~~~~
            This magnification is currently isotropic. In the future, anisotropy between
            the camera and experiment planes could be implemented.
        """
        if not hasattr(cam, "get_image"):
            raise ValueError(f"Expected Camera to be passed as cam. Found {type(cam)}")
        self.cam = cam

        if not hasattr(slm, "set_phase"):
            raise ValueError(f"Expected SLM to be passed as slm. Found {type(slm)}")
        self.slm = slm

        self.name = self.cam.name + "-" + self.slm.name

        self.mag = float(mag)

        self.calibrations = {}

    def plot(
            self,
            phase=None,
            image=None,
            slm_limits=None,
            cam_limits=None,
            title="",
            axs=None,
            cbar=True,
            **kwargs
        ):
        """
        Plots the provided phase and image for the child hardware on a pair of subplot axes.

        Parameters
        ----------
        phase : ndarray OR None
            Phase to be plotted.
            If ``None``, grabs the last written :attr:`phase` from the SLM.

            Important
            ---------
            Writes this ``phase`` to the SLM if ``image`` is ``None``.
        image : ndarray OR None
            Image to be plotted. If ``None``, grabs an image from the camera.
        slm_limits, cam_limits : None OR float OR [[float, float], [float, float]]
            Scales the limits by a given factor or uses the passed limits directly.
        title : str
            Super title for the axes.
        ax : (matplotlib.pyplot.axis, matplotlib.pyplot.axis) OR None
            Axes to plot upon.
        cbar : bool
            Also plot a colorbar.
        **kwargs
            Passed to :meth:`set_phase()`

        Returns
        -------
        (matplotlib.pyplot.axis, matplotlib.pyplot.axis)
            Axes of the plotted phase and image.
        """
        if image is None and phase is not None and np.shape(phase) == self.slm.shape:
            self.slm.set_phase(phase, **kwargs)

        if len(plt.get_fignums()) > 0:
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(20,8))

        if axs is None:
            axs = (fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2))

        self.slm.plot(phase=phase, limits=slm_limits, title="", ax=axs[0], cbar=cbar)
        self.cam.plot(image=image, limits=cam_limits, title="", ax=axs[1], cbar=cbar)

        fig.suptitle(title)
        plt.tight_layout()

        return axs


class NearfieldSLM(CameraSLM):
    """
    **(NotImplemented)** Class for an SLM which is not nearly in the Fourier domain of a camera.

    Parameters
    ----------
    mag : number OR None
        Magnification between the plane where the SLM image is created
        and the camera sensor plane.
    """

    def __init__(self, cam, slm, mag=None):
        """See :meth:`CameraSLM.__init__`."""
        super().__init__(cam, slm)
        self.mag = mag


def _blaze_offset(grid, vector, offset=0):
    return blaze(grid=grid, vector=vector) + offset


class FourierSLM(CameraSLM):
    r"""
    Class for an SLM and camera separated by a Fourier transform.
    This class includes methods for system calibration.

    Attributes
    ----------
    calibrations : dict
        "fourier" : dict
            The affine transformation that maps between
            the k-space of the SLM (kxy) and the pixel-space of the camera (ij).

            See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate()`.

            This data is critical for much of :mod:`slmsuite`'s functionality.
        "wavefront" : dict
            Raw data for correcting aberrations in the optical system (``phase``) and
            measuring the optical amplitude distribution incident on the SLM (``amp``).

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate_zernike()`
            and
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate_superpixel()`
            Usable data for the superpixel implementation is produced by running
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_superpixel_process()`.

            This data is critical for crisp holography.
        "pixel" : dict
            Raw data for measuring the crosstalk and :math:`V_\pi` of sections of the
            SLM via measurements on the diffractive orders of binary gratings.

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.pixel_calibrate()`.
            Usable data is produced by running
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.pixel_calibration_process()`.

            **This data is currently unused; exploring
            computationally-efficient ways to apply the crosstalk without oversampling.**
        "settle" : dict
            Raw data for determining the temporal system response of the SLM.

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.settle_calibrate()`.
            Usable data is produced by running
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.settle_calibration_process()`.

            This data informs the user's choice of `settle_time_s`, the time to wait to
            acquire data after a pattern is displayed. This is, of course, a tradeoff
            between measurement speed and measurement precision.
    """
    _pickle = ["name", "cam", "slm", "mag"]
    _pickle_data = ["calibrations"]

    def __init__(self, *args, **kwargs):
        r"""See :meth:`CameraSLM.__init__`."""
        super().__init__(*args, **kwargs)

        # Size of the calibration point window relative to the spot radius.
        self._wavefront_calibration_window_multiplier = 4

    def simulate(self):
        """
        Clones the hardware-based experiment into a simulation.

        Note
        ~~~~
        Since simulation mode needs the Fourier relationship between the SLM and
        camera, the :class:`~slmsuite.hardware.cameraslms.FourierSLM` should be
        Fourier-calibrated prior to cloning for simulation.

        Returns
        -------
        FourierSLM
            A :class:`~slmsuite.hardware.cameraslms.FourierSLM` object with simulated
            hardware.
        """
        # Make sure we have a Fourier calibration.
        if not "fourier" in self.calibrations:
            raise ValueError("Cannot simulate() a FourierSLM without a Fourier calibration.")

        # Make a simulated SLM
        slm_sim = SimulatedSLM(
            self.slm.shape[::-1],
            source=self.slm.source,
            bitdepth=self.slm.bitdepth,
            name=self.slm.name+"_sim",
            wav_um=self.slm.wav_um,
            wav_design_um=self.slm.wav_design_um,
            pitch_um=self.slm.pitch_um,
        )

        # Make a simulated camera using the current Fourier calibration
        cam_sim = SimulatedCamera(
            slm_sim,
            resolution=self.cam.shape[::-1],
            M=copy.copy(self.calibrations["fourier"]["M"]),
            b=copy.copy(self.calibrations["fourier"]["b"]),
            bitdepth=self.cam.bitdepth,
            averaging=self.cam.averaging,
            hdr=self.cam.hdr,
            pitch_um=self.cam.pitch_um,
            name=self.cam.name+"_sim"
        )
        cam_sim.transform = copy.copy(self.cam.transform)

        #Combine the two and pass FourierSLM attributes from hardware
        fs_sim = FourierSLM(cam_sim, slm_sim)
        fs_sim.calibrations = copy.deepcopy(self.calibrations)
        fs_sim._wavefront_calibration_window_multiplier = self._wavefront_calibration_window_multiplier

        return fs_sim

    @staticmethod
    def load(file_path : str):
        """
        Creates a simulation of a system from a file.

        Returns
        -------
        FourierSLM
            A :class:`~slmsuite.hardware.cameraslms.FourierSLM` object with simulated
            hardware.
        """
        # Read in the file.
        data = load_h5(file_path)

        # Check to see if it has the information we need.
        if not "__meta__" in data:
            raise ValueError(
                f"Cannot interpret file {file_path} without field '__meta__'. "
            )
        if not "cam" in data["__meta__"]:
            raise ValueError(
                f"Cannot interpret file {file_path} without metadata field 'cam'. "
            )
        cam_data = data["__meta__"]["cam"]
        if not "slm" in data["__meta__"]:
            raise ValueError(
                f"Cannot interpret file {file_path} without metadata field 'slm'. "
            )
        slm_data = data["__meta__"]["slm"]

        # Create the SLM and Camera objects.
        slm = SimulatedSLM(
            resolution=np.flip(slm_data["shape"]),
            pitch_um=slm_data["pitch_um"],
        )
        cam = SimulatedCamera(
            slm=slm,
            resolution=np.flip(cam_data["shape"]),
            bitdepth=cam_data["bitdepth"],
            pitch_um=cam_data["pitch_um"],
            name=cam_data["name"],
        )

        fs = FourierSLM(cam, slm, mag=data["__meta__"]["mag"])
        fs.name = data["__meta__"]["name"]

        return fs

    ### Calibration Helpers ###

    def name_calibration(self, calibration_type):
        """
        Creates ``"{self.name}-{calibration_type}-calibration"``.

        Parameters
        ----------
        calibration_type : str
            The type of calibration to save. See :attr:`calibrations` for supported
            options.

        Returns
        -------
        name : str
            The generated name.
        """
        return f"{self.name}-{calibration_type}-calibration"

    def write_calibration(self, calibration_type, path, name):
        "Backwards-compatibility alias for :meth:`save_calibration()`."
        warnings.warn(
            "The backwards-compatible alias FourierSLM.write_calibration will be depreciated "
            "in favor of FourierSLM.save_calibration in a future release."
        )
        self.save_calibration(calibration_type, path, name)

    def save_calibration(self, calibration_type, path=".", name=None):
        """
        to a file like ``"path/name_id.h5"``.

        Parameters
        ----------
        calibration_type : str
            The type of calibration to save. See :attr:`calibrations` for supported
            options. Works for any key of :attr:`calibrations`.
        path : str
            Path to directory to save in. Default is current directory.
        name : str OR None
            Name of the save file. If ``None``, will use :meth:`name_calibration`.

        Returns
        -------
        str
            The file path that the calibration was saved to.
        """
        if not calibration_type in self.calibrations:
            raise ValueError(
                f"Could not find calibration '{calibration_type}' in calibrations. Options:\n"
                + str(list(self.calibrations.keys()))
            )

        if name is None:
            name = self.name_calibration(calibration_type)
        file_path = generate_path(path, name, extension="h5")
        save_h5(file_path, self.calibrations[calibration_type])

        return file_path

    def read_calibration(self, calibration_type, file_path=None):
        "Backwards-compatibility alias for :meth:`load_calibration()`."
        warnings.warn(
            "The backwards-compatible alias FourierSLM.read_calibration will be depreciated "
            "in favor of FourierSLM.load_calibration in a future release."
        )
        self.load_calibration(calibration_type, file_path)

    def load_calibration(self, calibration_type, file_path=None):
        """
        from a file.

        Parameters
        ----------
        calibration_type : str
            The type of calibration to save. See :attr:`calibrations` for supported
            options.
        file_path : str OR None
            Full path to the calibration file. If ``None``, will
            search the current directory for a file with a name like
            the one returned by :meth:`name_calibration`.

        Returns
        -------
        str
            The file path that the calibration was loaded from.

        Raises
        ------
        FileNotFoundError
            If a file is not found.
        """
        if file_path is None:
            path = os.path.abspath(".")

            if len(calibration_type) > 4 and calibration_type[-3:] == ".h5":
                file_path = calibration_type
                split = file_path.split("-")
                if len(split) > 3 and "calibration_" in split[-1]:
                    calibration_type = split[-2]
                else:
                    raise ValueError(
                        f"Could not parse calibration type from '{file_path}'."
                    )
            else:
                name = self.name_calibration(calibration_type)
                file_path = latest_path(path, name, extension="h5")

            if file_path is None:
                raise FileNotFoundError(
                    "Unable to find a calibration file like\n{}"
                    "".format(os.path.join(path, name))
                )

        self.calibrations[calibration_type] = cal = load_h5(file_path)
        cal_ver = "an unknown version" if not "__version__" in cal else cal["__version__"]

        if cal_ver != __version__:
            warnings.warn(
                f"You are using slmsuite {__version__}, but the calibration "
                f"in '{file_path}' was created in {cal_ver}."
            )

        return file_path

    def _get_calibration_metadata(self):
        return self.pickle(attributes=False, metadata=True)      # Pickle without heavy data.

    ### Settle Time Calibration ###

    def settle_calibrate(
        self, vector=(.005, .005), size=None, times=None, settle_time_s=1
    ):
        """
        Approximates the :math:`1/e` settle time of the SLM.
        This is done by successively removing and applying a blaze to the SLM,
        measuring the intensity at the first order spot versus time delay.

        **(This feature is experimental.)**

        Parameters
        ----------
        vector : array_like
            Point to measure settle time at via a simple blaze in the ``"kxy"`` basis.
        size : int
            Size in pixels of the integration region in the ``"ij"`` basis.
            If ``None``, sets to sixteen times the approximate size of a diffraction-limited spot.
        times : array_like OR None OR int
            List of times to sweep over in search of the :math:`1/e` settle time.
            If ``None``, defaults to 21 points over one second.
            If an integer, defaults to that given number of points over one second.
        settle_time_s : float OR None
            Time between measurements to allow the SLM to re-settle. If ``None``, uses the
            current default in the SLM.
        """
        # Parse vector.
        point = self.kxyslm_to_ijcam(vector)
        blaze = toolbox.phase.blaze(grid=self.slm, vector=vector)

        # Parse size.
        if size is None:
            size = 16 * toolbox.convert_radius(
                self.slm.get_spot_radius_kxy(),
                to_units="ij",
                hardware=self
            )
        size = int(size)

        # Parse times.
        if times is None:
            times = 21
        if np.isscalar(times):
            times = np.linspace(0, 1, int(times), endpoint=True)
        times = np.ravel(times)

        # Parse settle_time_s.
        if settle_time_s is None:
            settle_time_s = self.slm.settle_time_s
        settle_time_s = float(settle_time_s)

        results = []

        verbose = True
        iterations = times
        if verbose:
            iterations = tqdm(times)

        # Collect data
        for t in iterations:
            self.cam.flush()

            # Reset the pattern and wait for it to settle
            self.slm.set_phase(None, settle=False, phase_correct=False)
            time.sleep(settle_time_s)

            # Turn on the pattern and wait for time t
            self.slm.set_phase(blaze, settle=False, phase_correct=False)
            time.sleep(t)

            image = self.cam.get_image()
            results.append(analysis.take(image, point, size, centered=True, integrate=True))

        self.calibrations["settle"] = {
            "times" : times,
            "data" : np.array(results)
        }
        self.calibrations["settle"].update(self._get_calibration_metadata())

        self.settle_calibration_process(plot=False)

        return self.calibrations["settle"]

    def settle_calibration_process(self, plot=True):
        """
        Fits an exponential to the measured data to
        approximate the :math:`1/e` settle time of the SLM.

        Parameters
        ----------
        plot : bool
            Whether to show a debug plot with the exponential fit.

        Returns
        -------
        dict
            The settle time and communication time measured.
        """
        times = self.calibrations["settle"]["times"]
        results = self.calibrations["settle"]["data"]

        if plot:
            plt.plot(times, np.squeeze(results), "k.")
            plt.ylabel("Signal [a.u.]")
            plt.xlabel("Time [sec]")
            plt.show()

        # Function to interpolate
        def exponential_jump(x, x0, a, b, c):
            return (c - a*np.exp(-(x-x0) / b)) * np.heaviside(x - x0, 0)

        guess = (np.max(times)/2, np.max(results), np.max(times), np.max(results))

        # Fit the date with the function
        params, _ = optimize.curve_fit(
            exponential_jump,
            times,
            results,
            p0=guess,
            maxfev=10000
        )
        x0, a, b, c = params
        print(params)

        relax_time = b
        com_time = x0
        settle_time = com_time + relax_time*4

        # Evaluate the fitting function in the interval
        x_interp = np.linspace(min(times), max(times), 100)
        g_interp = exponential_jump(x_interp, *guess)
        y_interp = exponential_jump(x_interp, *params)

        if plot:
            title = (
                f"Communication time: {int((1e3*com_time))} ms\n"
                f"$1/e$ Relaxation time: {int((1e3*relax_time))} ms\n"
                f"Suggested $1/e^4$ Settle time: {int((1e3*settle_time))} ms"
            )
            # plt.plot(x_interp, g_interp, "--", linewidth=1, color='g', alpha=.5, label='interpolation')
            plt.plot(x_interp, y_interp, "--", linewidth=2, color='red', label='interpolation')
            plt.plot(times, results, "k.", markersize=7, label='capta')
            plt.xlabel("Time [sec]")
            plt.ylabel("Signal [a.u.]")
            plt.title(title)
            plt.show()

        # Update dictionary with results. FUTURE: Return error bars?
        processed = {
            "settle_time" : settle_time,
            "relax_time" : relax_time,
            "communication_time" : com_time
        }
        self.calibrations["settle"].update(processed)

        return processed

    ### Pixel Crosstalk and Vpi Calibration ###

    def pixel_calibrate(
        self,
        levels=2,
        periods=2,
        orders=3,
        window=None,
        field_period=10,
    ):
        r"""
        Measure the pixel crosstalk and phase response of the SLM.

        **(This feature is experimental.)**

        Physical SLMs do not produce perfectly sharp and discrete blocks of a desired
        phase at each pixel. Rather, the realized phase might deviate from the desired
        phase (error) and be blurred between pixels (crosstalk).

        We adopt a literature approach to calibrating both phenomena by `measuring the
        system response of binary gratings <https://doi.org/10.1364/OE.20.022334>`_.
        In the future, we intend to fit the measured data to `an upgraded asymmetric
        model of phase crosstalk <https://doi.org/10.1364/OE.27.025046>`_, and then
        apply the model to beam propagation during holographic optimization. A better
        understanding of the system error can lead to holograms that take this error
        into account.

        Note that this algorithm does not operate at the level of individual pixels, but
        rather on aggregate statistics over a region of pixels.
        Right now, this calibration is done for one region (which defaults to the full
        SLM). In the future, we might want to calibrate many regions across the SLM to
        measure `spatially varying phase response <https://doi.org/10.1364/OE.21.016086>`_

        Note
        ~~~~
        A Fourier calibration must be loaded.

        Caution
        ~~~~~~~
        Data must be acquired without wavefront calibration applied.
        If the uncalibrated SLM produces too defocussed of a spot,
        then this measurement may not be ideal. On the flip side, a
        too-focussed spot might increase error by integration over fewer camera pixels.

        Parameters
        ----------
        levels : int OR array_like of int
            Which bitlevels to test, out of the :math:`2^B` levels available for a
            :math:`B`-bit SLM. Note that runtime scales with :math:`\mathcal{O}(B^2)`.
            If an integer is passed, the integer is rounded to the next largest power of
            two, and this number of bitlevels are sampled.
        periods : int OR array_like of int
            List of periods (in pixels) of the binary gratings that we will apply.
            Must be even integers.
            If a single ``int`` is provided, then a list containing the given number of
            periods is chosen, based upon the field of view of the camera.
        orders : int OR array_like of int
            Orders (..., -1st, 0th, 1st, ...) of the binary gratings to measure data at.
            If scalar is provided, measures orders between -nth and nth order, inclusive.
        window
            If not ``None``, the pixel calibration is only done over the region of the SLM
            defined by ``window``.
            Passed to :meth:`~slmsuite.holography.toolbox.window_slice()`.
            See :meth:`~slmsuite.holography.toolbox.window_slice()` for various options.
        field_period : int
            If ``window`` is not ``None``, then the field is deflected away in an
            orthogonal direction with a grating of the given period.
        """
        # Parse levels by forcing range and datatype.
        if np.isscalar(levels):
            if levels < 1:
                levels = 1
            levels = 2 ** (np.ceil(np.log2(levels)))

            if levels > self.slm.bitresolution:
                warnings.warn("Requested more levels than available. Rounding down.")
                levels = self.slm.bitresolution

            levels = np.arange(levels) * (self.slm.bitresolution / levels)
        levels = np.mod(levels, self.slm.bitresolution).astype(self.slm.display.dtype)
        N = len(levels)

        # Parse periods by forcing integer.
        if np.isscalar(periods):
            raise NotImplementedError("TODO")

        periods = np.array(periods).astype(int)
        periods = 2 * (periods // 2)
        P = len(periods)

        if len(np.unique(periods)) != len(periods):
            raise RuntimeError(f"Repeated periods in {periods}")

        if np.any(periods <= 0):
            raise ValueError("period should not be negative.")

        # Parse orders by forcing integer.
        if np.isscalar(orders):
            orders = int(orders)
            orders = np.arange(-orders, orders+1)
        orders = orders.astype(int)
        M = len(orders)

        if not 1 in orders:
            raise ValueError("1st order must be included.")

        # Figure out our shape.
        shape = (2, P, N, N, M)
        data = np.zeros(shape)

        # Make all of the x-pointing vectors, then all of the y-pointing vectors.
        vectors_freq = np.zeros((2, 2*P))
        vectors_freq[0, :P] = vectors_freq[1, P:] = np.reciprocal(periods.astype(float))
        vectors_kxy = toolbox.convert_vector(
            vectors_freq,
            from_units="freq",
            to_units="norm",
            hardware=self
        )

        # Make the y-pointing field vector, then the x-pointing field vector.
        field_freq = np.zeros((2, 2))
        field_freq[0, 0] = field_freq[1, 1] = 1 / float(field_period)
        field_kxy = toolbox.convert_vector(
            field_freq,
            from_units="freq",
            to_units="norm",
            hardware=self
        )
        field_values = np.array([self.slm.bitresolution / 2, 0]).astype(self.slm.display.dtype)
        field_hi, field_lo = field_values

        field_ij = toolbox.convert_vector(
            field_freq,
            from_units="freq",
            to_units="ij",
            hardware=self
        )

        # Figure out where the orders will appear on the camera.
        vectors_ij = self.kxyslm_to_ijcam(vectors_kxy)
        center = self.kxyslm_to_ijcam((0,0))

        dorder = vectors_ij - center
        dfield = field_ij - center
        order_ij = []

        for i in range(2*P):
            order_ij.append(center + orders * dorder[:, [i]])

        integration_size = int(np.ceil(np.min([
            np.min(np.max(dorder, axis=1)),
            np.min(np.max(dfield, axis=1))
        ])))

        # FUTURE: Warn the user if any order is outside the field of view.
        if False:
            warnings.warn("FUTURE")

        # if True: iterations = tqdm(range(P*(N-1)*N))
        if True: iterations = tqdm(range(2*P*N*N))

        # Big sweep.
        for i in [0,1]:                                         # Direction
            prange = np.arange(P) + i*P
            for j in range(P):                                  # Period
                for k in range(N):                              # Upper triangular gray level selection.
                    for l in range(N):                          # Periodic normalization when equal.
                    # for l in range(k, N):                       # Periodic normalization when equal.
                        if window is None:
                            phase = binary(
                                self.slm,
                                vector=vectors_kxy[:, prange[j]],
                                a=levels[k],
                                b=levels[l]
                            )
                        else:
                            # In windowed mode, blaze the field away from the 0th order,
                            # in the direction perpendicular to the target.
                            phase = binary(
                                grid=self.slm,
                                vector=field_kxy[:, i],
                                a=field_hi,
                                b=field_lo
                            )
                            toolbox.imprint(
                                phase,
                                window=window,
                                function=binary,
                                grid=self.slm,
                                vector=vectors_kxy[:, prange[j]],
                                a=levels[k],
                                b=levels[l]
                            )

                        # We're writing integers, so this goes directly to the SLM,
                        # bypassing phase2gray.
                        self.slm.set_phase(phase, phase_correct=False, settle=True)

                        data[i,j,k,l,:] = analysis.take(    # = data[i,j,l,k,:]
                            images=self.cam.get_image(),
                            vectors=order_ij[prange[j]],
                            size=integration_size,
                            integrate=True,
                        ).astype(float)

                        if True: iterations.update()

        if True: iterations.close()

        # Assemble the return dictionary.
        self.calibrations["pixel"] = {
            "levels" : levels,
            "periods" : periods,
            "orders" : orders,
            "data": data
        }
        self.calibrations["pixel"].update(self._get_calibration_metadata())

        # Process by default because we currently don't have any arguments.
        # self.pixel_calibration_process()

        return self.calibrations["pixel"]

    def pixel_calibration_process(self):
        """
        Currently, this method only displays debug plots of the measurements.
        In the future, the measurements will be fit in a way that can be applied to
        propagation.
        """
        cal = self.calibrations["pixel"]
        periods = cal["periods"]
        orders = cal["orders"]
        levels = cal["levels"]
        data = cal["data"]

        first_order = np.arange(len(orders))[orders == 1][0]

        rolled = data.copy()

        # rolled /= rolled[:,:,:,:,[first_order]]

        # for i in range(1, len(levels)):
        #     rolled[:,:,[i],:,:] = np.roll(rolled[:,:,[i],:,:], -i, axis=3)

        for i, direction in enumerate(["x"]): #, "y"]):
            for j, period in enumerate(periods[[0]]):
                for o, order in enumerate(orders):
                    plt.imshow(rolled[i,j,:,:,o], vmin=0)
                    plt.title(f"{period}-pixel, ${direction}$ grating; measuring order {order}")
                    # plt.clim(0,1)
                    plt.show()

    @staticmethod
    def pixel_kernel(x, a1_pix=.1, a2_pix=.1, n1=1, n2=1):
        r"""
        Blurring kernel

        .. math:: K(x) =    \left\{
                                \begin{array}{ll}
                                    \exp\left(-\left|\frac{x}{\alpha_1}\right|^{n_1}\right), & x < 0, \\
                                    \exp\left(-\left|\frac{x}{\alpha_2}\right|^{n_2}\right), & x \ge 0.
                                \end{array}
                            \right.
        """
        kernel = np.where(
            x >= 0,
            np.exp(-np.power(np.abs(x) / a1_pix, n1)),
            np.exp(-np.power(np.abs(x) / a2_pix, n2)),
        )
        kernel[len(kernel) // 2] = 1
        kernel /= np.sum(kernel)

        return kernel

    def _pixel_calibrate_simulate(self, period=16, supersample=16, **kwargs):
        N = int(period * supersample)

        x = np.linspace(-supersample, supersample, N)
        x -= np.mean(x)

        y = np.zeros_like(x)
        y[x < 0] = 1

        plt.plot(x, y)

        x2 = np.linspace(-supersample, supersample, N-1)
        x2 -= np.mean(x2)
        K = FourierSLM.pixel_kernel(x2, **kwargs)

        y = ndimage.convolve1d(y, K, mode="wrap")

        plt.plot(x, y)

        plt.show()

        kx = np.arange(float(N))
        kx -= np.mean(kx)

        Y = np.fft.fftshift(np.fft.fft(np.exp(1j * np.pi * y)))

        plt.plot(kx, np.square(np.abs(Y)))
        plt.xlim(-10, 10)
        plt.show()

    ### Fourier Calibration ###

    def fourier_calibrate(
        self,
        array_shape=10,
        array_pitch=10,
        array_center=None,
        plot=False,
        autofocus=False,
        autoexposure=False,
        **kwargs,
    ):
        """
        Project and fit a SLM computational Fourier space ``"knm"`` grid onto
        camera pixel space ``"ij"`` for affine fitting.
        An array produced by
        :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
        is projected for analysis by
        :meth:`~slmsuite.holography.analysis.blob_array_detect()`.
        These arguments are in ``"knm"`` space because:

        - The ``"ij"`` space has not yet been calibrated.
        - The ``"kxy"`` space can lead to non-integer ``array_pitch`` in
          ``"knm"``-space. This is not ideal (see Tip).

        Tip
        ~~~
        For best results, ``array_pitch`` should be integer data. Otherwise non-uniform
        rounding to the SLM's computational :math:`k`-space (``"knm"``-space) can result
        in non-uniform pitch and a bad fit. The user is warned if non-integer data is given.

        Parameters
        ----------
        array_shape, array_pitch
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**
        array_center
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**  ``array_center`` is not passed directly, and is
            processed as being relative to the center of ``"knm"`` space, the position
            of the 0th order.
        plot : bool OR int
            Enables debug plots:

            - 0 is no plots,
            - 1 is only the final fit plot, unless there is an error,
            - 2 is all plots.
        autofocus : bool OR dict
            Whether or not to autofocus the camera.
            If a dictionary is passed, autofocus is performed,
            and the dictionary is passed to
            :meth:`~slmsuite.hardware.cameras.camera.Camera.autofocus()`.
        autoexposure : bool OR dict
            Whether or not to automatically set the camera exposure.
            If a dictionary is passed, autoexposure is performed,
            and the dictionary is passed to
            :meth:`~slmsuite.hardware.cameras.camera.Camera.autoexposure()`.
        **kwargs : dict
            Passed to :meth:`.fourier_grid_project()`, which passes them to
            :meth:`~slmsuite.holography.algorithms.SpotHologram.optimize()`.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["fourier"]`
        """
        # Parse variables
        if isinstance(array_shape, REAL_TYPES):
            array_shape = [int(array_shape), int(array_shape)]
        if isinstance(array_pitch, REAL_TYPES):
            array_pitch = [array_pitch, array_pitch]
        if np.any(np.array(array_pitch) <= 0):
            raise ValueError("array_pitch must be positive.")

        # Make and project a GS hologram across a normal grid of kvecs
        try:
            hologram = self.fourier_grid_project(
                array_shape=array_shape, array_pitch=array_pitch, array_center=array_center, **kwargs
            )
        except Exception as e:
            warnings.warn(
                "fourier_calibrate failed during array holography. Try the following:\n"
                "- Reducing the array_pitch or array_shape,\n"
                "- Checking SLM parameters."
            )
            raise e

        # The rounding of the values might cause the center to shift from the desired
        # value. To compensate for this, we find the true written center.
        # The first two points are ignored for balance against the parity check omission
        # of the last two points.
        array_center = np.mean(hologram.spot_kxy_rounded[:, 2:], axis=1)

        if plot > 1:
            hologram.plot_farfield()
            hologram.plot_nearfield()

        self.cam.flush()

        # Optional step -- autofocus and autoexpose the spots
        if autofocus or isinstance(autofocus, dict):
            # Pre-expose
            if autoexposure or isinstance(autoexposure, dict):
                if isinstance(autoexposure, dict):
                    self.cam.autoexposure(**autoexposure)
                else:
                    self.cam.autoexposure()

            # Focus
            if isinstance(autofocus, dict):
                self.cam.autofocus(plot=plot, **autofocus)
            else:
                self.cam.autofocus(plot=plot)

        # Post-expose
        if autoexposure or isinstance(autoexposure, dict):
            if isinstance(autoexposure, dict):
                self.cam.autoexposure(**autoexposure)
            else:
                self.cam.autoexposure()

        img = self.cam.get_image()

        # Get orientation of projected array
        try:
            orientation = analysis.blob_array_detect(img, array_shape, plot=plot)
        except Exception as e:
            warnings.warn("fourier_calibrate failed during array detection and fitting.")
            raise e

        a = format_2vectors(array_center)
        M = np.array(orientation["M"])
        b = format_2vectors(orientation["b"])

        # blob_array_detect returns the calibration from ij to the space of the array, so
        # as a last step we must convert from the array to (centered) knm space, and then
        # one step further to kxy space. This is done by a simple scaling.
        scaling = (
            self.slm.pitch
            * np.flip(np.squeeze(hologram.shape))
            / np.squeeze(array_pitch)
        )

        M = np.array([
            [M[0, 0] * scaling[0], M[0, 1] * scaling[1]],
            [M[1, 0] * scaling[0], M[1, 1] * scaling[1]],
        ])

        self.calibrations["fourier"] = {
            "M": M,
            "b": b,
            "a": a
        }
        self.calibrations["fourier"].update(self._get_calibration_metadata())

        return self.calibrations["fourier"]

    ### Fourier Calibration Helpers ###

    def fourier_grid_project(self, array_shape=10, array_pitch=10, array_center=None, **kwargs):
        """
        Projects a Fourier space grid ``"knm"`` onto pixel space ``"ij"``.
        The chosen computational :math:`k`-space ``"knm"`` uses a computational shape generated by
        :meth:`~slmsuite.holography.algorithms.SpotHologram.get_padded_shape()`
        corresponding to the smallest square shape with power-of-two sidelength that is
        larger than the SLM's shape.

        Parameters
        ----------
        array_shape, array_pitch
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**
        array_center
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**  ``array_center`` is not passed directly, and is
            processed as being relative to the center of ``"knm"`` space, the position
            of the 0th order.
        **kwargs
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.optimize()`.

        Returns
        -------
        ~slmsuite.holography.algorithms.SpotHologram
            Optimized hologram.
        """
        # Check that the pitch is an integer.
        if not np.all(np.isclose(array_pitch, np.rint(array_pitch))):
            warnings.warn("array_pitch is non-integer")

        # Make the spot array
        shape = SpotHologram.get_padded_shape(self, padding_order=1, square_padding=True)
        hologram = SpotHologram.make_rectangular_array(
            shape,
            array_shape=array_shape,
            array_pitch=array_pitch,
            array_center=None
            if array_center is None
            else (
                format_2vectors(array_center) +
                format_2vectors((shape[1] / 2.0, shape[0] / 2.0))
            ),
            basis="knm",
            orientation_check=True,
            cameraslm=self,
        )

        # Default optimize settings.
        if "maxiter" not in kwargs:
            kwargs["maxiter"] = 10

        # Warn the user in case they mistyped a default argument or something.
        for key in kwargs.keys():
            if key not in [
                "method", "maxiter", "verbose", "callback", "feedback",
                "stat_groups", "name", "fixed_phase", "raw_stats", "blur_ij",
            ]:
                warnings.warn(
                    f"Unexpected argument '{key}' passed to fourier_grid_project(). "
                    "This may be ignored."
                )

        # Optimize and project the hologram
        hologram.optimize(**kwargs)

        self.slm.set_phase(hologram.get_phase(), settle=True)

        return hologram

    def fourier_calibrate_analytic(self, M, b):
        """
        Sets the Fourier calibration to a user-selected affine transformation.

        See :meth:`fourier_calibration_build()` to generate this transformation from a
        known or measured focal length.

        Parameters
        ----------
        numpy.ndarray
            Affine matrix :math:`M`. Shape ``(2, 2)``.
        numpy.ndarray
            Affine vector :math:`b`. Shape ``(1, 2)``.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["fourier"]`
        """
        # Parse arguments.
        M = np.squeeze(M)
        if np.any(M.shape != (2,2)):
            raise ValueError("Expected a 2x2 matrix for M.")
        a = format_2vectors([0,0])
        b = format_2vectors(b)

        self.calibrations["fourier"] = {
            "M": M,
            "b": b,
            "a": a
        }
        self.calibrations["fourier"].update(self._get_calibration_metadata())

        # Set the camera's virtual calibration if it is not already set.
        if hasattr(self.cam, "set_affine") and not hasattr(self.cam, "M"):
            self.cam.set_affine(M, b)

        return self.calibrations["fourier"]

    def fourier_calibration_build(
            self,
            f_eff,
            units="norm",
            theta=0,
            shear_angle=0,
            offset=None,
        ):
        """
        META: This docstring will be overwritten by ``SimulatedCamera.build_affine``'s
        after this class.
        """
        if offset is None:
            offset = np.flip(self.cam.shape) / 2
        return SimulatedCamera._build_affine(
            f_eff,
            units=units,
            theta=theta,
            shear_angle=shear_angle,
            offset=offset,
            cam_pitch_um=self.cam.pitch_um,
            wav_um=self.slm.wav_um,
        )

    ### Fourier Calibration User Results ###

    def _kxyslm_to_ijcam_depth(self, kxy_depth):
        """Helper function for handling depth conversion."""
        f_eff = np.mean(self.get_effective_focal_length("norm"))
        if self.cam.pitch_um is None:
            cam_pitch_um = np.nan
        else:
            cam_pitch_um = np.mean(self.cam.pitch_um)
        return kxy_depth * (self.slm.wav_um * f_eff * f_eff / cam_pitch_um)

    def _ijcam_to_kxyslm_depth(self, ij_depth):
        """Helper function for handling depth conversion."""
        f_eff = np.mean(self.get_effective_focal_length("norm"))
        if self.cam.pitch_um is None:
            cam_pitch_um = np.nan
        else:
            cam_pitch_um = np.mean(self.cam.pitch_um)
        return ij_depth * (cam_pitch_um / (self.slm.wav_um * f_eff * f_eff))

    def kxyslm_to_ijcam(self, kxy):
        r"""
        Converts SLM Fourier space (``"kxy"``) to camera pixel space (``"ij"``).
        For blaze vectors :math:`\vec{x}` and camera pixel indices :math:`\vec{y}`, computes:

        .. math:: \vec{y} = M \cdot (\vec{x} - \vec{a}) + \vec{b}

        where :math:`M`, :math:`\vec{b}`, and :math:`\vec{a}` are stored in

        If the vectors are three-dimensional, the third depth dimension is treated according to:

        .. math:: y_z = \frac{f_\text{eff}^2}{\pi}x_z

        where :math:`y_z` is the normalized depth of the spot relative to the focal plane and
        :math:`x_z` is equivalent to focal power, equivalent to the
        the quadratic term of a simple thin :meth:`~slmsuite.holography.toolbox.phase.lens()`.
        The constant of proportionality makes use of the normalized effective focal length
        :math:`f_\text{eff}` of the imaging system between the SLM and camera.
        This information is encoded in the Fourier calibration, and revealed by
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.get_effective_focal_length()`.

        Parameters
        ----------
        kxy : array_like
            Vector or array of vectors to convert. Can be 2D or 3D.
            Cleaned with :meth:`~slmsuite.holography.toolbox.format_vectors()`.

        Returns
        -------
        ij : numpy.ndarray
            Vector or array of vectors in camera spatial coordinates. Can be 2D or 3D.

        Raises
        ------
        RuntimeError
            If the fourier plane calibration does not exist.
        """
        if not "fourier" in self.calibrations:
            raise RuntimeError("Fourier calibration must exist to be used.")

        self._check_fourier_calibration_stale()

        kxy = format_vectors(kxy, handle_dimension="pass")

        # Apply the xy transformation.
        ij = np.matmul(
            self.calibrations["fourier"]["M"],
            kxy[:2, :] - self.calibrations["fourier"]["a"]
        ) + self.calibrations["fourier"]["b"]

        # Handle z if needed.
        if kxy.shape[0] == 3:
            return np.vstack((ij, self._kxyslm_to_ijcam_depth(kxy[[2], :])))
        else:
            return ij

    def ijcam_to_kxyslm(self, ij):
        r"""
        Converts camera pixel space (``"ij"``) to SLM Fourier space (``"kxy"``).
        For camera pixel indices :math:`\vec{y}` and blaze vectors :math:`\vec{x}`, computes:

        .. math:: \vec{x} = M^{-1} \cdot (\vec{y} - \vec{b}) + \vec{a}

        where :math:`M`, :math:`\vec{b}`, and :math:`\vec{a}` are stored in
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["fourier"]`.

        Important
        ~~~~~~~~~

        If the vectors are three-dimensional, the third depth dimension is treated according to:

        .. math:: x_z = \frac{1}{f} = \frac{1}{f_\text{eff}^2}\frac{\Delta_{xy} y_z}{\lambda}

        where :math:`x_z`, equivalent to normalized focal power, is the focal term
        needed to focus a spot at :math:`y_z` pixel depth.
        Here, :math:`\frac{\Delta_{xy} y_z}{\lambda}` is the same depth in normalized units.
        Importantly, this is depth relative to the plane of the camera, which might
        differ from the relative depth in an experimental plane.
        Focal power is equivalent to the
        the quadratic term of a simple thin :meth:`~slmsuite.holography.toolbox.phase.lens()`.
        The constant of proportionality makes use of the normalized effective focal length
        :math:`f_\text{eff}` of the imaging system between the SLM and camera.
        This information is encoded in the Fourier calibration, and revealed by
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.get_effective_focal_length()`.

        Parameters
        ----------
        ij : array_like
            Vector or array of vectors to convert. Can be 2D or 3D.
            Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.

        Returns
        -------
        kxy : numpy.ndarray
            Vector or array of vectors in slm angular coordinates. Can be 2D or 3D.

        Raises
        ------
        RuntimeError
            If the fourier plane calibration does not exist.
        """
        if not "fourier" in self.calibrations:
            raise RuntimeError("Fourier calibration must exist to be used.")

        self._check_fourier_calibration_stale()

        ij = format_vectors(ij, handle_dimension="pass")

        # Apply the xy transformation.
        kxy = np.matmul(
            np.linalg.inv(self.calibrations["fourier"]["M"]),
            ij[:2, :] - self.calibrations["fourier"]["b"]
        ) + self.calibrations["fourier"]["a"]

        # Handle z if needed.
        if ij.shape[0] == 3:
            return np.vstack((kxy, self._ijcam_to_kxyslm_depth(ij[[2], :])))
        else:
            return kxy

    def _check_fourier_calibration_stale(self):
        """
        Checks if the wavefront calibration is newer than the Fourier calibration.

        Warns if this is true. Does nothing if either calibration is not present or
        if another error occurs.
        """
        try:
            if "wavefront_superpixel" in self.calibrations and "fourier" in self.calibrations:
                if (
                    self.calibrations["wavefront_superpixel"]["__timestamp__"] >
                    self.calibrations["fourier"]["__timestamp__"]
                ):
                    warnings.warn(
                        f"The wavefront calibration is newer "
                        f"({self.calibrations['wavefront_superpixel']['__time__']}) "
                        f"than the Fourier calibration "
                        f"({self.calibrations['fourier']['__time__']}). "
                        "The Fourier calibration may be stale."
                    )
        except:
            pass

    def get_farfield_spot_size(self, slm_size=None, basis="kxy"):
        """
        Calculates the size of a spot produced by blazed patch of size ``slm_size`` on the SLM.
        If this patch is the size of the SLM, then we will find in the farfield (camera)
        domain, the size of a diffraction-limited spot for a fully-illuminated surface.
        As the ``slm_size`` of the patch on the SLM decreases, the diffraction limited
        spot size in the farfield domain will of course increase. This calculation
        is accomplished using the calibration produced by
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate()`
        and stored in
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["fourier"]`.

        Parameters
        ----------
        slm_size : (float, float) OR int OR float OR None
            Size of patch on the SLM in normalized units.
            A scalar is interpreted as the width and height of a square.
            If ``None``, defaults to the normalized SLM size.
        basis : {"kxy", "ij"}
            Basis of the returned size;
            ``"kxy"`` for SLM :math:`k`-space, ``"ij"`` for camera size.

        Returns
        -------
        (float, float)
            Size in x and y of the spot in the desired ``basis``.

        Raises
        ------
        ValueError
            If the basis argument was malformed.
        """
        # Default to effective SLM aperture size (based on amplitude profile if measured)
        if slm_size is None:
            psf_kxy = self.slm.get_spot_radius_kxy()
            slm_size = (1 / psf_kxy, 1 / psf_kxy)
        # Float input -> square region
        elif isinstance(slm_size, REAL_TYPES):
            slm_size = (slm_size, slm_size)

        if basis == "kxy":
            return (1 / slm_size[0], 1 / slm_size[1])
        elif basis == "ij":
            M = self.calibrations["fourier"]["M"]
            # Compensate for spot rotation s.t. spot size is along camera axes
            size_kxy = np.linalg.inv(M / np.sqrt(np.abs(np.linalg.det(M)))) @ np.array(
                (1 / slm_size[0], 1 / slm_size[1])
            )
            # return np.abs(self.kxyslm_to_ijcam([0, 0]) - self.kxyslm_to_ijcam(size_kxy)).flatten()
            return np.abs(self.kxyslm_to_ijcam([0, 0]) - self.kxyslm_to_ijcam(size_kxy))
        else:
            raise ValueError('Unrecognized basis "{}".'.format(basis))

    def get_effective_focal_length(self, units="norm"):
        """
        Uses the Fourier calibration to estimate the scalar effective focal length of the
        optical train separating the Fourier-domain SLM from the camera.
        This currently assumes an isotropic imaging train without cylindrical optics.

        Tip
        ~~~
        This effective focal length between the SLM and camera is potentially different
        from the effective focal length between the SLM and experiment.

        Parameters
        ----------
        units : str {"ij", "norm", "m", "cm", "mm", "um", "nm"}
            Units for the focal length.

            -  ``"ij"``
                Focal length in units of camera pixels.

            -  ``"norm"``
                Normalized focal length in wavelengths.

            -  ``"m"``, ``"cm"``, ``"mm"``, ``"um"``, ``"nm"``
                Focal length in metric units.

        Returns
        -------
        f_eff : float
            Effective focal length.
        """
        if not "fourier" in self.calibrations:
            raise RuntimeError("Fourier calibration must exist to be used.")

        # Gather f_eff in pix/rad.
        f_eff = np.sqrt(np.abs(np.linalg.det(self.calibrations["fourier"]["M"])))

        # Gather other conversions.
        if units != "ij" and self.cam.pitch_um is None:
            warnings.warn(f"cam.pitch_um must be set to use units '{units}'")
            return np.nan

        # Convert.
        if units == "ij":
            pass
        elif units == "norm":
            f_eff *= np.array(self.cam.pitch_um) / self.slm.wav_um
        elif units in toolbox.LENGTH_FACTORS.keys():
            f_eff *= np.array(self.cam.pitch_um) / toolbox.LENGTH_FACTORS[units]
        else:
            raise ValueError(f"Unit '{units}' not recognized as a length.")

        return f_eff

    ### Wavefront Calibration ###

    def wavefront_calibrate(
        self,
        *args,
        method=None,
        **kwargs,
    ):
        """
        Backwards-compatible method to switch between
        the superpixel :meth:`wavefront_calibrate_superpixel`
        and Zernike :meth:`wavefront_calibrate_zernike`
        implementations of wavefront calibration.

        Important
        ~~~~~~~~~
        Wavefront calibration will generally shift spot centers slightly, making a
        previous Fourier calibration "stale". It is recommended to perform Fourier
        calibration after wavefront calibration.
        """
        if method is None:
            method = "superpixel"

        if method == "superpixel":
            if "interference_point" in kwargs:
                warnings.warn(
                    "The 'interference_point' argument is deprecated. "
                    "Use 'calibration_points' instead."
                )
                kwargs["calibration_points"] = kwargs.pop("interference_point")

            if "calibration_point" in kwargs:
                warnings.warn(
                    "The 'calibration_point' argument is deprecated. "
                    "Use 'calibration_points' instead."
                )
                kwargs["calibration_points"] = kwargs.pop("calibration_point")

            return self.wavefront_calibrate_superpixel(*args, **kwargs)
        elif method == "zernike":
            return self.wavefront_calibrate_zernike(*args, **kwargs)
        else:
            raise ValueError(f"Wavefront calibration method '{method}' not recognized.")

    ### Zernike Wavefront Calibration ###

    def wavefront_calibrate_zernike(
        self,
        calibration_points=None,
        zernike_indices=9,
        perturbation=1,
        callback=None,
        metric=None,
        global_correction=False,
        optimize_focus=True,
        optimize_position=True,
        optimize_weights=True,
        plot=0,
    ):
        r"""
        Perform wavefront calibration by iteratively scanning and subtracting Zernike
        coefficients.

        Parameters
        ----------
        calibration_points : (float, float) OR numpy.ndarray OR float OR None
            Position(s) in the camera domain where interference occurs.
            A passed array should be a standard ``(D, N)`` matrix,
            where ``D`` is the dimension of the Zernike space and ``N`` is the number of points.
            If ``int``, fills the camera field of view with roughly this number of calibration
            points.
            If ``None``, defaults to 100 points, unless a calibration is already saved
            in :attr:`calibrations` under the ``"wavefront_zernike"`` key, in which case
            the ``"corrected_spots"`` from the calibration are used as the baseline.
            This allows the user to iterate on previous calibrations. Note that
            ``zernike_indices`` is also overwritten in this case.

            Important
            ~~~~~~~~~
            These coordinates must be in the ``"zernike"`` basis. Use
            :meth:`~slmsuite.holography.toolbox.convert_vector()` to convert between 2 or
            3 dimensional coordinates to their Zernike counterparts.
        zernike_indices : int OR list of int OR None
            Which Zernike polynomials to calibrate against, defined by ANSI indices. Of shape ``(D,)``.

            Tip
            ~~~
            Use :meth:`~slmsuite.holography.toolbox.phase.zernike_convert_index()`
            to convert to ANSI from various other common indexing conventions.

            Important
            ~~~~~~~~~
            If ``None`` is passed, the assumed Zernike basis depends on the
            dimensionality of the provided spots:

            -   If ``D == 2``, then the basis is assumed to be ``[2,1]``
                corresponding to the :math:`x = Z_2 = Z_1^1`
                and :math:`y = Z_1 = Z_1^{-1}` tilt terms.

            -   If ``D == 3``, then the basis is assumed to be ``[2,1,4]``
                corresponding to the previous, with the addition of the
                :math:`Z_4 = Z_2^0` focus term.

            -   If ``D > 3``, then the basis is assumed to be ``[2,1,4,3,5,6...,D]``.
                The piston term (Zernike index 0) is ignored as this constant phase is
                not relevant.
        perturbation : list of float OR float OR None
            Perturbation in radians to iteratively multiply with each of the
            :math:`\pm 1`-normalized Zernike terms.
            If ``float``, tests 11 points in a range of plus to minus this value in radians.
            Defaults to a range of :math:`\pm 1` radians.
            If ``0`` or ``None``, the starting spots are projected and the function returns before optimizing.
        callback : None OR function
            Measure the system to determine the level of aberration. Expected to return
            a list of floats of length ``N`` corresponding to the chosen metric evaluated
            on all the spots. The optimizer will *minimize* the figure of merit.
            This data is fit using a parabola, and the x-offset of the
            parabola is interpreted as the minimum aberration.
        metric : None OR function
            If ``callback`` is ``None``, then the camera is used to measure the system.
            This parameter allows the user to impart a custom figure of merit upon the
            measured camera data. ``metric`` is required to accept a stack of ``N`` images
            consisting of the regions about each of the ``N`` target spots. It is expected to
            return a list of length ``N`` corresponding to the chosen metric evaluated
            on all the images.
            If ``None``, :meth:`._wavefront_calibrate_zernike_default_metric()`
            is used, which is just a wrapper for
            :meth:`~slmsuite.holography.analysis.image_areas()`, a measurement of spot
            size. The optimizer will *minimize* the figure of merit.
        global_correction : bool
            If ``True``, the optimized Zernike coefficients are meaned and applied to the entire SLM.
            This can be useful for the first step of calibration to remove large global aberration terms
            while avoiding noise and uncertainty on individual spots.
            When `optimize_position` is `True`, the `fit_affine` flag of
            :meth:`~slmsuite.holography.algorithms.SpotHologram.refine_offset()` is used to extract the global shift.
        optimize_focus : bool
            If ``False``, does not optimize the focus term (ansi index 4). Useful in
            cases where the ``callback`` method is insensitive to :meth:`z`-translation
            (e.g. Stark effect of an atom) or in cases where the :meth:`z` axis should
            be unchanged.
        optimize_position : bool
            If ``False``, does not optimize the position terms (ansi indices 1 and 2).
        optimize_weights : bool OR int
            If ``True``,  optimizes the WGS weights of the hologram one time
            at the beginning of the calibration. Defaults to 10 iterations.
            If integer, then uses this number as the number of iterations to optimize the weights.
            Must be at least 1.
        plot : int or bool
            Whether to provide visual feedback, options are:

            - ``-1`` : No plots or tqdm prints.
            - ``0``, ``False`` : No plots, but tqdm prints.
            - ``1``, ``True`` : Plots on fits and essentials.
            - ``2`` : Plots on everything.

        Returns
        -------
        dict
            The contents of
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["wavefront"]`.

        Raises
        ------
        RuntimeError
            If the Fourier plane calibration does not exist.
        ValueError
            If various points are out of range.
        """
        # Helper function to sweep the amplitude of a Zernike over a pattern.
        def sweep_term(sweep, term, pattern, callback, desc=None):
            result = None
            sweep = np.ravel(sweep)
            N = len(sweep)
            M = None

            iterable = list(enumerate(sweep))
            if plot >= 0:
                iterable = tqdm(iterable, desc=desc, position=0, leave=False)

            for i, x in iterable:
                phase = pattern + x * term
                self.slm.set_phase(phase, settle=True, phase_correct=False)
                this_result = np.array(callback())

                if result is None:
                    M = len(this_result)    # Number of points to measure at.
                    result = np.full((N, M), np.nan, dtype=this_result.dtype)

                if len(this_result) != M:
                    raise RuntimeError()
                else:
                    result[i, :] = this_result

            return result

        # Helper function to fit a parabola to the result of the sweep.
        def fit_term(sweep, result, term, status):
            ddy = np.diff(result, n=2, axis=0)
            a0 = .5 * np.mean(ddy, axis=0) / np.square(np.mean(np.diff(sweep)))
            if True or np.mean(a0) >= 0:    # Determine whether the system has a + or - x^2 term. For now, we force +.
                c0 = np.min(result, axis=0)
                x0 = sweep[np.argmin(result, axis=0)]
            else:
                c0 = np.max(result, axis=0)
                x0 = sweep[np.argmax(result, axis=0)]

            def parabola(x, x0, a, c):
                return c + a * np.square(x - x0)

            g = np.zeros(result.shape[1])
            x = np.zeros(result.shape[1])
            dx = np.zeros(result.shape[1])

            for i in range(result.shape[1]):
                guess = (x0[i], a0[i], c0[i])
                try:
                    popt, pcov = optimize.curve_fit(
                        parabola,
                        sweep,
                        result[:, i],
                        ftol=1e-5,
                        p0=guess,
                        bounds=(
                            [-np.inf, 0, -np.inf],
                            [np.inf, np.inf, np.inf]
                        )
                    )
                    perr = np.sqrt(np.diag(pcov))   # Single sigma error, which can be multiplied later.
                except Exception as e:
                    popt = guess
                    perr = np.zeros_like(guess)

                g[i] = guess[0]
                x[i] = popt[0]
                dx[i] = perr[0]

            x = np.clip(x, np.min(sweep), np.max(sweep))
            railed = np.sum(np.logical_or(x == np.min(sweep), x == np.max(sweep))) / float(len(x))

            if plot > 0:
                result_plot = result - np.min(result, axis=0, keepdims=True)
                result_plot = result_plot / np.max(result_plot, axis=0, keepdims=True)
                plt.imshow(
                    result_plot,
                    interpolation="none",
                    extent=[-.5, result.shape[1]-.5, np.max(sweep), np.min(sweep)]
                )
                cbar = plt.colorbar()
                plt.scatter(
                    np.arange(result.shape[1]),
                    g,
                    c="r",
                    marker='x',
                    alpha=.25,
                )
                plt.errorbar(
                    np.arange(result.shape[1]),
                    x,
                    yerr=dx,
                    c="r",
                    marker='.',
                    linestyle='none'
                )
                plt.gca().set_aspect("auto")
                plt.title("Zernike $Z_{" + str(term) + "}$")
                plt.xlabel("Calibration Point [#]")
                plt.ylabel("Perturbation [rad]")
                plt.xlim(-.5, result.shape[1]-.5)
                plt.ylim(np.max(sweep), np.min(sweep))
                cbar.ax.set_ylabel("Figure of Merit [norm]") #, rotation=270)
                plt.show()

            return x, dx, railed

        # Parse calibration_points and zernike_indices
        calibration_points_ij = None
        metric_stats = []
        position_stats = []
        weights = None
        spot_integration_width_ij = None

        if calibration_points is None:
            if "wavefront_zernike" in self.calibrations:
                dat = self.calibrations["wavefront_zernike"]
                calibration_points = np.copy(dat["corrected_spots"])
                calibration_points_ij = np.copy(dat["calibration_points_ij"])
                spot_integration_width_ij = np.copy(dat["spot_integration_width_ij"])

                if zernike_indices is None:
                    zernike_indices = np.copy(dat["zernike_indices"])
                else:
                    if np.isscalar(zernike_indices) and zernike_indices < calibration_points.shape[0]:
                            zernike_indices = calibration_points.shape[0]

                    zernike_indices = _zernike_indices_parse(
                        zernike_indices,
                        calibration_points.shape[0],
                        smaller_okay=True
                    )

                    stored_zi = np.copy(dat["zernike_indices"])

                    if len(zernike_indices) >= len(stored_zi):
                        if np.all(zernike_indices[:len(stored_zi)] == stored_zi):
                            pass # Extend zernike indices.
                        else:
                            raise ValueError(
                                f"Requested indices {zernike_indices} "
                                f"is not compatible with stored indices {stored_zi}."
                            )
                    else:
                        raise ValueError(
                            f"Requested indices {zernike_indices} "
                            f"is not compatible with stored indices {stored_zi}."
                        )

                if "metric_stats" in dat:
                    metric_stats = list(copy.copy(dat["metric_stats"]))
                else:
                    metric_stats = []

                if "position_stats" in dat:
                    position_stats = list(copy.copy(dat["position_stats"]))
                else:
                    position_stats = []

                if "weights" in dat:
                    weights = dat["weights"]
                else:
                    weights = None
            else:
                calibration_points = 100

        if np.isscalar(calibration_points):
            pitch = np.sqrt(np.prod(self.cam.shape) / calibration_points)
            calibration_points = self.wavefront_calibration_points(pitch, plot=True)
            # wavefront_calibration_points returns "ij"; convert to "zernike" basis.
            calibration_points = convert_vector(
                calibration_points, from_units="ij", to_units="zernike", hardware=self
            )

        calibration_points = format_vectors(np.copy(calibration_points), handle_dimension="pass")
        zernike_indices = _zernike_indices_parse(zernike_indices, calibration_points.shape[0], smaller_okay=True)
        dp = len(zernike_indices) - calibration_points.shape[0]
        if dp:  # Pad with zeros if the points don't have certain terms.
            calibration_points = np.pad(calibration_points, ((0,dp), (0,0)))

        initial_points = calibration_points.copy()

        # Build hologram
        if calibration_points.shape[1] > 1:
            hologram = CompressedSpotHologram(
                spot_vectors=calibration_points,
                basis=zernike_indices,
                cameraslm=self,
            )

            if not (weights is None):
                hologram.set_weights(weights)

            if calibration_points_ij is None:
                calibration_points_ij = hologram.spot_ij
            else:
                hologram.spot_ij = calibration_points_ij
        else:
            hologram = None
            if calibration_points_ij is None:
                calibration_points_ij = convert_vector(
                    calibration_points,
                    from_units="zernike",
                    to_units="ij",
                    hardware=self,
                )

        if calibration_points.shape[1] > 1:
            max_window_size = smallest_distance(calibration_points_ij)
        else:
            max_window_size = np.min(self.cam.shape)
        max_spot_integration_width_ij = int(2 * np.ceil(np.min((.5*max_window_size, 51)) / 2) + 1)
        if spot_integration_width_ij is None:
            spot_integration_width_ij = max_spot_integration_width_ij
        else:
            spot_integration_width_ij = min(int(spot_integration_width_ij), max_spot_integration_width_ij)
        if hologram is not None:
            hologram.spot_integration_width_ij = spot_integration_width_ij

        # Parse callback.
        if callback is None:
            def default_callback():
                # self.cam.flush()
                img = self.cam.get_image()

                images = analysis.take(img, calibration_points_ij, spot_integration_width_ij, clip=True).astype(float)
                images = analysis.image_remove_field(images)
                images[np.isnan(images)] = 0
                images = images.astype(float) / np.sum(images)        # Remove laser noise

                if metric is None:
                    return FourierSLM._wavefront_calibrate_zernike_default_metric(images)
                else:
                    return metric(images)

            callback = default_callback

        # Tick function.
        def tick():
            if hologram is None:
                pattern = zernike_sum(
                    self.slm,
                    zernike_indices,
                    calibration_points,
                    use_mask=False
                )
            else:
                # Reoptimize the hologram at each step.
                hologram.spot_zernike = calibration_points

                hologram.optimize(
                    "GS",
                    maxiter=3,
                    verbose=0,
                    # raw_stats=True,
                )
                pattern = hologram.get_phase()

            return pattern

        # Parse perturbation
        no_perturbation = (
            perturbation is None or
            (np.isscalar(perturbation) and perturbation <= 0) or
            (not np.isscalar(perturbation) and len(perturbation) == 0)
        )
        if perturbation is None:
            perturbation = 1

        if hologram is not None:
            hologram.optimize(
                "GS", maxiter=3, verbose=0,
                # raw_stats=True,
                stat_groups=["computational_spot",],
            )

        if optimize_weights and hologram is not None:
            if isinstance(optimize_weights, bool):
                maxiter = 10
            else:
                maxiter = int(optimize_weights)
                if maxiter < 1:
                    raise ValueError("optimize_weights must be True, False, or a positive integer.")

            hologram.optimize(
                "WGS-Kim",
                feedback="experimental_spot",
                maxiter=maxiter,
                verbose=True,
                name="optimize_weights",
                stat_groups=["computational_spot", "experimental_spot",],
            )
            if "wavefront_zernike" in self.calibrations:
                self.calibrations["wavefront_zernike"]["weights"] = hologram.get_weights()

        # If no perturbation, just project the initial spots and return.
        if no_perturbation:
            self.slm.set_phase(tick(), settle=True, phase_correct=False)
            # self.slm.set_phase(hologram.get_phase(), settle=True, phase_correct=False)

            self.cam.flush()
            img = self.cam.get_image()

            if plot and hologram is not None:
                take = analysis.take(
                    img,
                    hologram.spot_ij,
                    hologram.spot_integration_width_ij if hasattr(hologram, 'spot_integration_width_ij') else spot_integration_width_ij,
                    centered=True,
                    integrate=False,
                )
                max = np.max(take)

                if max >= self.cam.bitresolution-1:
                    warnings.warn("Image is overexposed.")
                elif max > .5*self.cam.bitresolution:
                    warnings.warn(
                        f"Image might become overexposed during optimization ({max}/{self.cam.bitresolution-1})."
                    )

                self.cam.plot(img, title="Zernike Calibration Status")

                if plot >= 2:

                    plt.figure(figsize=(12, 12))
                    # plt.imshow(tiled)
                    analysis.take_plot(take, separate_axes=False)
                    plt.title("Zernike Calibration Status (Zoom)")
                    plt.show()

            return hologram

        # Parse perturbation, maybe returning if perturbation is negative.
        if np.isscalar(perturbation):
            perturbation = np.linspace(-perturbation, perturbation, 11, endpoint=True)
        else:
            perturbation = np.ravel(perturbation)

        # Refine hologram position.
        if optimize_position:
            self.slm.set_phase(tick())
            hologram.refine_offset(img=None, basis="kxy", force_affine=global_correction, plot=plot)

        # Calibration loop.
        result = None
        self.cam.flush()
        for j, i in enumerate(zernike_indices):
            # Ignore the piston and tilt terms, maybe also the focus too.
            if i in [0, 2, 1] or (i == 4 and not optimize_focus):
                continue

            # Generate hologram and record current stats.
            pattern = tick()
            self.slm.set_phase(pattern, settle=True, phase_correct=False)
            metric_stats.append(callback())

            # Determine which Zernike polynomial we are testing.
            term = zernike(self.slm, i, use_mask=False)

            # Test the polynomial. This returns a (N, S) array,
            # where N is the number of spots and S is the number of sweep points.
            result = sweep_term(perturbation, term, pattern, callback, f"Z_{i}")

            # Analyze the results by fitting each to a parabola.
            correction, correction_error, railed = fit_term(perturbation, result, i, calibration_points[j, :])

            # Apply the correction to the spots (globally if desired).
            if global_correction:
                correction = np.mean(correction)
            calibration_points[j, :] += correction

        # Record final stats.
        pattern = tick()
        self.slm.set_phase(pattern, settle=True, phase_correct=False)
        metric_stats.append(callback())
        # position_stats.append(calibration_points)

        self.calibrations["wavefront_zernike"] = {
            "initial_points": initial_points,
            "zernike_indices": zernike_indices,
            "corrected_spots": calibration_points,
            "last_result": result,
            "calibration_points_ij" : calibration_points_ij,
            "spot_integration_width_ij" : spot_integration_width_ij,
            "metric_stats" : metric_stats,
            # "position_stats" : position_stats,
            "weights" : hologram.get_weights(),
        }
        self.calibrations["wavefront_zernike"].update(self._get_calibration_metadata())

        # return hologram

        del hologram

        return self.calibrations["wavefront_zernike"]

    def _wavefront_calibrate_zernike_plot_raw(self, calibration_points=None, index=0):
        dat = self.calibrations["wavefront_zernike"]

        if calibration_points is None:
            calibration_points = np.copy(dat["corrected_spots"])
        calibration_points_ij = np.copy(dat["calibration_points_ij"])
        zernike_indices = np.copy(dat["zernike_indices"])

        aberration = calibration_points[index, :]

        lim = np.max(np.abs(aberration))

        plt.scatter(
            calibration_points_ij[0, :],
            calibration_points_ij[1, :],
            c=aberration,
            cmap="seismic"
        )
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("Aberration Correction [rad]") #, rotation=270)
        plt.clim(-lim, lim)
        plt.title(f"Zernike $Z_{zernike_indices[index]}$")

    @staticmethod
    def _wavefront_calibrate_zernike_default_metric(images):
        """
        Calculates the spot areas of all the spots in the stack of ``images``.
        Spot area (determinant of the variances) is here a metric of spot aberration,
        where a spot with smaller and tighter area is better.
        """
        variances = analysis.image_variances(images)
        return analysis.image_areas(variances)

    def wavefront_calibrate_zernike_smooth(
        self,
        smoothing=0.25,
        smoothing_xy=0.25,
        smoothing_z=None,
        plot=False,
    ):
        """
        For a 2D array of Zernike-corrected spots, produces a smoothed version of the
        spot coordinates in aberration space by averaging the coordinates of neighbors.
        This is useful for noise reduction.

        Parameters
        ----------
        smoothing : float
            Smoothing factor for higher order terms.
            This weights the original spot coordinates with the average of the
            neighbors. Should be between 0 and 1. Zero retains the original coordinates.
            One fully replaces it with the neighbor average.
        smoothing_xy : float
            Behaves similarly to ``smoothing`` for tip tilt terms. Instead of averaging the full
            coordinate (which would result in all spots eventually converging), the
            error between the current and expected xy position is averaged. The expected
            xy position from an affine Fourier calibration does not account for barrel
            and pincushion distortion, shifts from higher order Zernike terms, or other effects.
            This correction can help mitigate those issues.
        smoothing_z : float OR None
            Not yet implemented. Would behave similarly to ``smoothing_xy`` for the
            focus term. If ``None``, focus would be treated the same as the higher order
            terms.
        plot : bool
            Whether to enable debug plots.
        """
        # Parse inputs.
        if smoothing < 0 or smoothing > 1:
            raise ValueError("Smoothing factor must be between 0 and 1.")
        if smoothing_xy < 0 or smoothing_xy > 1:
            raise ValueError("Smoothing factor must be between 0 and 1.")

        # Build triangulation.
        indices = self.calibrations["wavefront_zernike"]["zernike_indices"]
        I = np.arange(len(indices))
        to_smooth = I[indices > 2]
        x_smooth = I[indices == 2]
        y_smooth = I[indices == 1]
        if smoothing_z is not None:
            raise RuntimeError("Zernike z-smoothing not yet implemented.")

            # if smoothing_z < 0 or smoothing_z > 1:
            #     raise ValueError("Smoothing factor must be between 0 and 1.")
            # z_smooth = I[indices == 4]

        vectors = self.calibrations["wavefront_zernike"]["corrected_spots"]
        final = np.zeros_like(vectors)

        points_ij = self.calibrations["wavefront_zernike"]["calibration_points_ij"]
        base_xy = convert_vector(
            points_ij,
            from_units="ij",
            to_units="zernike",
            hardware=self,
        )

        # Build triangulation (cache in future?).
        points = points_ij[:2, :].T
        tri = Delaunay(points)

        edges = np.array([(i, j) for t in tri.simplices for i, j in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        lens = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        max_len = 1.5 * np.median(lens)

        simplices = np.array([
            t for t in tri.simplices
            if all(np.linalg.norm(points[[t[i]]]-points[[t[j]]]) <= max_len
                for i, j in [(0,1),(1,2),(2,0)])
        ])

        # Average spot coordinates.
        if plot:
            plt.scatter(*points_ij[:2], c="r", zorder=10)

        for i in range(points_ij.shape[1]):
            neighbors = set()

            for simplex in simplices:
                if i in simplex:
                    neighbors.update(simplex)

            neighbors.discard(i)

            if plot:
                for n in neighbors:
                    plt.plot(
                        [points_ij[0, n], points_ij[0, i]],
                        [points_ij[1, n], points_ij[1, i]],
                        c="k",
                        linewidth=1,
                    )

            if len(neighbors) == 0:
                # No neighbors: preserve original values without attenuation.
                final[x_smooth, i] = vectors[x_smooth, i]
                final[y_smooth, i] = vectors[y_smooth, i]
                final[to_smooth, i] = vectors[to_smooth, i]
            else:
                # Handle XY terms.
                final[x_smooth, i] = (1-smoothing_xy) * (vectors[x_smooth, i] - base_xy[0, i]) + base_xy[0, i]
                final[y_smooth, i] = (1-smoothing_xy) * (vectors[y_smooth, i] - base_xy[1, i]) + base_xy[1, i]

                for n in neighbors:
                    final[x_smooth, i] += smoothing_xy * (vectors[x_smooth, n] - base_xy[0, n]) / len(neighbors)
                    final[y_smooth, i] += smoothing_xy * (vectors[y_smooth, n] - base_xy[1, n]) / len(neighbors)

                # Handle higher order terms.
                final[to_smooth, i] = (1-smoothing) * vectors[to_smooth, i]

                for n in neighbors:
                    final[to_smooth, i] += smoothing * vectors[to_smooth, n] / len(neighbors)

        if plot:
            plt.gca().invert_yaxis()
            plt.title("Nearest Neighbor Smoothing")

        return final

    def _wavefront_calibrate_zernike_apply(
        self,
        vector,
        from_units="norm",
    ):
        raise NotImplementedError("Expected as a part of 0.5.0")

        if from_units == "knm":
            warnings.warn(
                "'knm' requires shape information, which here defaults to the SLM shape. "
                "This may give unexpected results."
            )

        pass

    def _wavefront_calibrate_zernike(
        self,
        calibration_points=None,
        zernike_indices=9,
        perturbation=(-np.pi, np.pi),
        global_correction=False,
        optimize_focus=True,
        optimize_position=True,
        optimize_weights=True,
    ):
        r"""
        Interactive Jupyter widget for manual Zernike coefficient adjustment.

        Provides a live camera view with sliders for each Zernike term,
        allowing the user to manually tune coefficients and confirm optimal values.
        The original :meth:`wavefront_calibrate_zernike` is not modified.

        Parameters
        ----------
        calibration_points : numpy.ndarray OR float OR None
            Same semantics as :meth:`wavefront_calibrate_zernike`.
        zernike_indices : int OR list of int OR None
            Same semantics as :meth:`wavefront_calibrate_zernike`.
        perturbation : (float, float) OR float
            Slider range in radians. A tuple ``(lower, upper)`` sets the slider
            bounds directly. A scalar ``p`` is shorthand for ``(-|p|, |p|)``.
        global_correction : bool
            Initial mode for the interactive Global / Per-Spot toggle.
            If ``True``, the slider initially adjusts all spots uniformly
            (and ``refine_offset`` is run with ``force_affine=True`` during the
            optional position refinement).
            If ``False``, the slider initially edits only the selected spot.
            The user may switch between modes interactively in the widget at
            any time without restarting the calibration.
        optimize_focus : bool
            If ``False``, hides the focus term (ANSI index 4) from the UI.
        optimize_position : bool
            If ``True``, runs :meth:`refine_offset` before launching the widget.
        optimize_weights : bool OR int
            If ``True``, optimizes WGS weights before launching the widget.

        Returns
        -------
        :class:`_ZernikeCalibrationWidget`
            The interactive widget. Results are stored into
            ``self.calibrations["wavefront_zernike"]`` when the user clicks
            *Save*.
        """
        try:
            from ipywidgets import Image as _  # noqa: F401
        except ImportError:
            raise ImportError("ipywidgets must be installed to use the interactive calibrator.")

        # --- Parse perturbation bounds ---
        if np.isscalar(perturbation):
            perturbation_bounds = (-abs(float(perturbation)), abs(float(perturbation)))
        else:
            perturbation_bounds = (float(perturbation[0]), float(perturbation[1]))

        # --- Parse calibration_points and zernike_indices (same as wavefront_calibrate_zernike) ---
        calibration_points_ij = None
        weights = None
        spot_integration_width_ij = None

        if calibration_points is None:
            if "wavefront_zernike" in self.calibrations:
                dat = self.calibrations["wavefront_zernike"]
                calibration_points = np.copy(dat["corrected_spots"])
                calibration_points_ij = np.copy(dat["calibration_points_ij"])
                spot_integration_width_ij = np.copy(dat["spot_integration_width_ij"])

                if zernike_indices is None:
                    zernike_indices = np.copy(dat["zernike_indices"])
                else:
                    if np.isscalar(zernike_indices) and zernike_indices < calibration_points.shape[0]:
                        zernike_indices = calibration_points.shape[0]

                    zernike_indices = _zernike_indices_parse(
                        zernike_indices,
                        calibration_points.shape[0],
                        smaller_okay=True,
                    )

                    stored_zi = np.copy(dat["zernike_indices"])
                    if len(zernike_indices) >= len(stored_zi):
                        if not np.all(zernike_indices[:len(stored_zi)] == stored_zi):
                            raise ValueError(
                                f"Requested indices {zernike_indices} "
                                f"is not compatible with stored indices {stored_zi}."
                            )
                    else:
                        raise ValueError(
                            f"Requested indices {zernike_indices} "
                            f"is not compatible with stored indices {stored_zi}."
                        )

                if "weights" in dat:
                    weights = dat["weights"]
            else:
                calibration_points = 100

        if np.isscalar(calibration_points):
            pitch = np.sqrt(np.prod(self.cam.shape) / calibration_points)
            calibration_points = self.wavefront_calibration_points(pitch, plot=True)
            calibration_points = convert_vector(
                calibration_points, from_units="ij", to_units="zernike", hardware=self
            )

        calibration_points = format_vectors(np.copy(calibration_points), handle_dimension="pass")
        zernike_indices = _zernike_indices_parse(
            zernike_indices, calibration_points.shape[0], smaller_okay=True
        )
        dp = len(zernike_indices) - calibration_points.shape[0]
        if dp:
            calibration_points = np.pad(calibration_points, ((0, dp), (0, 0)))

        # --- Build hologram ---
        if calibration_points.shape[1] > 1:
            hologram = CompressedSpotHologram(
                spot_vectors=calibration_points,
                basis=zernike_indices,
                cameraslm=self,
            )
            if weights is not None:
                hologram.set_weights(weights)
            if calibration_points_ij is None:
                calibration_points_ij = hologram.spot_ij
            else:
                hologram.spot_ij = calibration_points_ij
        else:
            hologram = None
            if calibration_points_ij is None:
                calibration_points_ij = convert_vector(
                    calibration_points,
                    from_units="zernike",
                    to_units="ij",
                    hardware=self,
                )

        if calibration_points.shape[1] > 1:
            max_window_size = smallest_distance(calibration_points_ij)
        else:
            max_window_size = np.min(self.cam.shape)
        max_siw = int(2 * np.ceil(np.min((.5 * max_window_size, 51)) / 2) + 1)
        if spot_integration_width_ij is None:
            spot_integration_width_ij = max_siw
        else:
            spot_integration_width_ij = min(int(spot_integration_width_ij), max_siw)
        if hologram is not None:
            hologram.spot_integration_width_ij = spot_integration_width_ij

        # --- Initial hologram optimisation ---
        if hologram is not None:
            hologram.optimize("GS", maxiter=3, verbose=0,
                              stat_groups=["computational_spot"])

        if optimize_weights and hologram is not None:
            maxiter = 10 if isinstance(optimize_weights, bool) else int(optimize_weights)
            if maxiter < 1:
                raise ValueError("optimize_weights must be True, False, or a positive integer.")
            hologram.optimize(
                "WGS-Kim",
                feedback="experimental_spot",
                maxiter=maxiter,
                verbose=True,
                name="optimize_weights",
                stat_groups=["computational_spot", "experimental_spot"],
            )

        if optimize_position and hologram is not None:
            def _tick_tmp():
                hologram.spot_zernike = calibration_points
                hologram.optimize("GS", maxiter=3, verbose=0)
                return hologram.get_phase()
            self.slm.set_phase(_tick_tmp())
            hologram.refine_offset(img=None, basis="kxy", force_affine=global_correction, plot=False)

        # --- Create and display interactive widget ---
        widget = _ZernikeCalibrationWidget(
            fs=self,
            calibration_points=calibration_points,
            zernike_indices=zernike_indices,
            calibration_points_ij=calibration_points_ij,
            spot_integration_width_ij=spot_integration_width_ij,
            hologram=hologram,
            perturbation_bounds=perturbation_bounds,
            global_correction=global_correction,
            optimize_focus=optimize_focus,
        )

        return widget

    ### Superpixel Wavefront Calibration ###

    def wavefront_calibrate_superpixel(
        self,
        calibration_points=None,
        superpixel_size=50,
        reference_superpixels=None,
        exclude_superpixels=(0, 0),
        test_index=None,
        field_point=(0,0),
        field_point_units="kxy",
        phase_steps=1,
        fresh_calibration=True,
        measure_background=False,
        corrected_amplitude=False,
        plot=0,
    ):
        """
        Perform wavefront calibration by
        `iteratively interfering superpixel patches on the SLM
        <https://doi.org/10.1038/nphoton.2010.85>`_.
        This procedure measures the wavefront phase and amplitude.

        Interference occurs at a given ``calibration_points`` in the camera's imaging plane.
        It is at each point where the computed correction is ideal; the further away
        from each point, the less ideal the correction is.
        Correction at many points over the plane permits a better understanding of the
        aberration and greater possibility of compensation.

        Sets :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["wavefront"]`.
        Run :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_process`
        afterwards to produce the usable calibration which can be written to the SLM.

        Note
        ~~~~
        A Fourier calibration must be loaded.

        Tip
        ~~~
        If *only amplitude calibration* is desired,
        use ``phase_steps=None`` to omit the more time-consuming phase calibration.

        Tip
        ~~~
        Use ``phase_steps=1`` for faster calibration. This fits the phase fringes of an
        image rather than scanning the fringes over a single camera pixel over many
        ``phase_steps``. This is usually optimal except in cases with excessive noise.

        Parameters
        ----------
        calibration_points : (float, float) OR numpy.ndarray OR None
            Position(s) in the camera domain where interference occurs.
            For multiple positions, this must be of shape ``(2, N)``.
            This is naturally in the ``"ij"`` basis.
            If ``None``, densely fills the camera field of view with calibration points.
        superpixel_size : int
            The width and height in pixels of each SLM superpixel.
            If this is not a devisor of both dimensions in the SLM's :attr:`shape`,
            then superpixels at the edge of the SLM may be cropped and give undefined results.
            Currently, superpixels are forced to be square, and this value must be a scalar.
        reference_superpixels : (int,int) OR numpy.ndarray of int OR None
            The coordinate(s) of the superpixel(s) to reference from.
            For multiple positions, this must be of shape ``(2, N)``.
            Defaults to the center of the SLM if ``None``. If multiple calibration
            points are requested when ``None``, then the references are clustered at the center.
        exclude_superpixels : (int, int) OR numpy.ndarray OR None
            If in ``(nx, ny)`` form, optionally exclude superpixels from the margin,
            That is, the ``nx`` superpixels are omitted from the left and right sides
            of the SLM, with the same for ``ny``. As power is
            typically concentrated in the center of the SLM, this function is useful for
            excluding points that are known to be blocked (e.g. with an iris or other
            pupil), or for quickly testing calibration at the most relevant points.
            Otherwise, if exclude_superpixels is an image with the same dimension as the
            superpixeled SLM, this image is interpreted as a denylist.
            Defaults to ``None``, where no superpixels are excluded.
        test_index : int OR None
            If ``int``, then tests the scheduled calibration corresponding to this index.
            Defaults to ``None``, which runs the full wavefront calibration instead of
            testing at only one index.

            Note
            ~~~~
            Scheduling is relative to the reference superpixel(s), and a change to the
            reference may also shift the positions of the test superpixels for the given index.
        field_point : (float, float)
            Position in the camera domain where the field (pixels not included in superpixels)
            is blazed toward in order to reduce light in the camera's field. The suggested
            approach is to set this outside the field of view of the camera and make
            sure that other diffraction orders are far from the ``calibration_points``.
            Defaults to no blaze (``(0,0)`` in ``"kxy"`` units).
        field_point_units : str
            A unit compatible with
            :meth:`~slmsuite.holography.toolbox.convert_vector()`.
            Defaults to ``"kxy"``.

            Tip
            ~~~
            Setting one coordinate of ``field_point`` to zero is suggested
            to minimize higher order diffraction.
        phase_steps : int OR None
            The number of interference phases to measure.
            If ``phase_steps`` is 1 (the default), does a one-shot fit on the interference
            pattern to improve speed.

            Tip
            ~~~
            If ``phase_steps`` is ``None``, phase is not measured and only amplitude is measured.
        fresh_calibration : bool
            If ``True``, the calibration is performed without an existing calibration
            (any old global calibration is wiped from the :class:`SLM` and :class:`CameraSLM`).
            If ``False``, the calibration is performed on top of any existing
            calibration. This is useful to determine the quality of a previous
            calibration, as a new calibration should yield zero phase correction needed
            if the previous was perfect. The old calibration will be stored in the
            :attr:`calibrations` under ``"previous_phase_correction"``,
            so keep in mind that this (uncompressed) image will take up significantly
            more memory when saved.
        measure_background : bool
            Whether to measure the background at each point.
        corrected_amplitude : bool
            If ``False``, the power in the uncorrected target beam
            is used as the source of amplitude measurement.
            If ``True``, adds another step to measure the power of the corrected target beam.
        plot : int or bool
            Whether to provide visual feedback, options are:

            - ``-1`` : No plots or tqdm prints.
            - ``0``, ``False`` : No plots, but tqdm prints.
            - ``1``, ``True`` : Plots on fits and essentials.
            - ``2`` : Plots on everything.
            - ``3`` : ``test_index`` not ``None`` only: returns image frames
              to make a movie from the phase measurement (not for general use).

        Returns
        -------
        dict
            The contents of
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["wavefront"]`.

        Raises
        ------
        RuntimeError
            If the Fourier plane calibration does not exist.
        ValueError
            If various points are out of range.
        """
        # Parse the superpixel size and derived quantities.
        superpixel_size = int(superpixel_size)

        slm_supershape = tuple(
            np.ceil(np.array(self.slm.shape) / superpixel_size).astype(int)
        )
        num_superpixels = slm_supershape[0] * slm_supershape[1]

        # Next, we get the size of the window necessary to measure a spot
        # produced by the given superpixel size.
        interference_window = self.wavefront_calibration_superpixel_window(superpixel_size).ravel()
        interference_size = interference_window / self._wavefront_calibration_window_multiplier

        interference_window = (interference_window // 2) * 2 + 1
        interference_size = (interference_size // 2) * 2 + 1

        # Now that we have the supershape, we label each of the pixels with an index.
        # It's sometimes useful to map that index to the xy coordinates of the pixel,
        # hence this function and its inverse.
        def index2coord(index):
            return format_2vectors(
                np.stack((index % slm_supershape[1], index // slm_supershape[1]), axis=0)
            )
        def coord2index(coord):
            coord = np.array(coord)
            return coord[1,:] * slm_supershape[1] + coord[0,:]

        # It's also useful to make an image showing the given indices.
        def index2image(index):
            image = np.zeros(slm_supershape)
            image.ravel()[index] = True
            return image

        # Parse exclude_superpixels
        exclude_superpixels = np.array(exclude_superpixels)

        if exclude_superpixels.shape == slm_supershape:
            exclude_superpixels = exclude_superpixels != 0
        elif exclude_superpixels.size == 2:
            exclude_margin = exclude_superpixels.astype(int)

            # Make the image based on margin values
            exclude_superpixels = np.zeros(slm_supershape)
            exclude_superpixels[:, :exclude_margin[0]] = True
            exclude_superpixels[:, slm_supershape[1]-exclude_margin[0]:] = True
            exclude_superpixels[:exclude_margin[1], :] = True
            exclude_superpixels[slm_supershape[0]-exclude_margin[1]:, :] = True
        else:
            raise ValueError("Did not recognize type for exclude_superpixels")

        num_active_superpixels = int(np.sum(np.logical_not(exclude_superpixels)))

        # Parse calibration_points.
        if calibration_points is None:
            # If None, then use the built-in generator.
            calibration_points = self.wavefront_calibration_points(
                1.5*np.max(interference_window),
                np.max(interference_window),
                field_point,
                field_point_units,
                plot=plot
            )

        # TODO: warn if matrix is transposed.
        calibration_points = np.rint(format_2vectors(calibration_points)).astype(int)
        num_points = calibration_points.shape[1]

        # Clean the base and field points.
        base_point = np.rint(self.kxyslm_to_ijcam([0, 0])).astype(int)

        if field_point_units != "ij":
            field_blaze = toolbox.convert_vector(
                format_2vectors(field_point),
                from_units=field_point_units,
                to_units="kxy",
                hardware=self.slm
            )

            field_point = self.kxyslm_to_ijcam(field_blaze)
        else:
            field_blaze = toolbox.convert_vector(
                field_point,
                from_units="ij",
                to_units="kxy",
                hardware=self
            )

        field_point = np.rint(format_2vectors(field_point)).astype(int)

        # Use the Fourier calibration to help find points/sizes in the imaging plane.
        if not "fourier" in self.calibrations:
            raise RuntimeError("Fourier calibration must be done before wavefront calibration.")
        calibration_blazes = self.ijcam_to_kxyslm(calibration_points)
        reference_blazes = calibration_blazes.copy()

        # Set the reference superpixels to be centered on the SLM if undefined.
        if reference_superpixels is None:
            all_superpixels = np.arange(num_superpixels)
            all_superpixels_coords = index2coord(all_superpixels)

            distance = np.sum(np.square(all_superpixels_coords - format_2vectors(slm_supershape[::-1])/2), axis=0)
            I = np.argsort(distance)

            reference_superpixels = I[:num_points]
        else:
            reference_superpixels = np.rint(format_2vectors(reference_superpixels)).astype(int)
            reference_superpixels = coord2index(reference_superpixels)

        # Error check the reference superpixels.
        reference_superpixels_coords = index2coord(reference_superpixels)
        reference_superpixels_image = index2image(reference_superpixels)
        if (np.any(np.logical_and(reference_superpixels_image, exclude_superpixels))):
            raise ValueError("reference_superpixels out of range of calibration.")

        # Now we have to solve the challenge of when to measure each target-reference pair.
        num_measurements = num_active_superpixels + ((2*num_points - 2) if phase_steps is not None else 0)

        index_image = np.reshape(np.arange(num_superpixels, dtype=int), slm_supershape)
        active_superpixels = index_image[np.logical_not(exclude_superpixels)].ravel()

        # The base schedule cycles through all indices apart from the base reference index.
        scheduling = np.zeros((num_points, num_measurements), dtype=int)

        scheduling[:, :(num_active_superpixels-1)] = np.mod(
            np.repeat(np.arange(num_active_superpixels-1, dtype=int)[np.newaxis, :] + 1, num_points, axis=0) +
            np.repeat(reference_superpixels[:, np.newaxis], num_active_superpixels-1, axis=1),
            num_active_superpixels
        )

        # Account for some superpixels being excluded.
        scheduling = active_superpixels[scheduling]
        scheduling[:, (num_active_superpixels-1):] = -1

        # Remove conflicts where other calibration pairs are targeting another
        # reference superpixel. Only do this when we are measuring relative phase
        # (if phase_steps is None, then we never write reference superpixels).
        if phase_steps is not None:
            for i in range(num_points):     # Future: Make more efficient.
                # For each calibration point, determine where the reference index is being overwritten.
                reference_index = reference_superpixels[i]

                conflicts = scheduling == reference_index
                conflict_indices = np.array(np.where(conflicts))

                for j in range(int(np.sum(conflicts))):
                    # For each time that the index is overwritten,
                    # reassign the target to empty space.
                    c_index = conflict_indices[:, j]

                    # This is the overwritten target index.
                    displaced_index = scheduling[i, c_index[1]]
                    scheduling[i, c_index[1]] = -1

                    # Find a point in empty space to resettle the index, if it is not already unused.
                    # This algorithm is currently quite slow. Consider speeding?
                    if displaced_index != -1:
                        for k in range(num_active_superpixels-1, num_measurements+1):
                            if k == num_measurements:
                                raise RuntimeError(
                                    "Some unexpected error happened in calibration scheduling."
                                )
                            elif (
                                scheduling[i, k] == -1
                                and not np.any(scheduling[:, k] == reference_index)
                                and not np.any(scheduling[:, k] == displaced_index)
                            ):
                                scheduling[i, k] = displaced_index
                                break

        # Cleanup the scheduling.
        empty_schedules = np.all(scheduling == -1, axis=0)
        scheduling = scheduling[:, np.logical_not(empty_schedules)]
        num_measurements = scheduling.shape[1]

        # Error check whether we expect to be able to see fringes.
        # max_dist_slmpix = np.max(np.hstack((
        #     reference_superpixels - exclude_superpixels,
        #     slm_supershape[::-1, [0]] - exclude_superpixels - reference_superpixels
        # )), axis=1)
        # max_r_slmpix = np.sqrt(np.sum(np.square(max_dist_slmpix)))
        # fringe_period = np.mean(interference_size) / max_r_slmpix / 2

        # if fringe_period < 2:
        #     warnings.warn(
        #         "Non-resolvable interference fringe period "
        #         "for the given SLM calibration extent. "
        #         "Either exclude more of the SLM or magnify the field on the camera."
        #     )

        # Error-check if we're measuring multiple sites at once.
        if num_points > 1:
            calibration_distance = smallest_distance(calibration_points, "euclidean")
            if np.max(interference_window) > calibration_distance:
                message = (
                    "Requested calibration points are too close together. "
                    "The minimum distance {} pix is smaller than the window size {} pix."
                    .format(calibration_distance, interference_window)
                )
                if test_index is None:
                    raise ValueError(message)
                else:
                    warnings.warn(message + " This message will error if running the full calibration.")

        # Error check interference point proximity to the 0th order.
        dorder = field_point - base_point
        order_distance = np.inf
        for order in range(-5, 5):
            order_distance_this = smallest_distance(
                np.hstack((
                    calibration_points,     # +1st calibration order
                    base_point + order * dorder,
                )),
                "euclidean"
            )
            if order_distance_this < order_distance:
                order_distance = order_distance_this

        if np.mean(interference_window) > order_distance:
            warnings.warn(
                "The requested calibration point(s) are close to the expected positions of "
                "the field diffractive orders. Consider moving calibration regions further away."
            )

        # Check proximity to -1th orders.
        calibration_reflections = 2 * base_point - calibration_points
        reflection_distance = smallest_distance(
            np.hstack((
                calibration_points,         # +1st calibration order
                calibration_reflections,    # -1st calibration order
            )),
            "euclidean"
        )

        if np.mean(interference_window)/2 > reflection_distance:
            warnings.warn(
                "The requested calibration points are close to the expected positions of "
                "the -1th orders of calibration points. Consider shifting the calibration regions "
                "relative to the 0th order. Alternatively, use the avoid_mirrors= parameter "
                "of wavefront_calibration_points"
            )

        # Save the current calibration in case we are just testing (test_index != None)
        amplitude = self.slm._get_source_amplitude()
        phase = self.slm._get_source_phase()

        # If we're starting fresh, remove the old calibration such that this does not
        # muddle things. If we're only testing, the stored data above will be reinstated.
        if fresh_calibration:
            self.slm.source.pop("amplitude", "First calibration.")
            self.slm.source.pop("phase", "First calibration.")
            self.slm.source.pop("r2", "First calibration.")

        # Parse phase_steps
        if phase_steps is not None:
            if not np.isclose(phase_steps, int(phase_steps)):
                raise ValueError(f"Expected integer phase_steps. Received {phase_steps}.")
            phase_steps = int(phase_steps)
            if phase_steps <= 0:
                raise ValueError(f"Expected positive phase_steps. Received {phase_steps}.")

        # Interpret the plot command.
        return_movie = plot == 3 and test_index is not None
        if return_movie:
            plot = 1
            if phase_steps is None or phase_steps == 1:
                raise ValueError(
                    "cameraslms.py: Must have phase_steps > 1 to produce a movie."
                )
        verbose = plot >= 0
        plot_fits = plot >= 1
        plot_everything = plot >= 2

        # Build the calibration dict.
        calibration_dict = {
            "__version__" : __version__,
            "__time__" : time.time(),
            "calibration_points" : calibration_points,
            "superpixel_size" : superpixel_size,
            "slm_supershape" : slm_supershape,
            "reference_superpixels" : reference_superpixels,
            "phase_steps" : phase_steps,
            "interference_size" : interference_size,
            "interference_window" : interference_window,
            "previous_phase_correction": (
                None if "phase" not in self.slm.source else np.copy(self.slm.source["phase"])
            ),
            "scheduling" : scheduling,
        }

        keys = [
            "power",
            "normalization",
            "background",
            "phase",
            "kx",
            "ky",
            "amp_fit",
            "contrast_fit",
            "r2_fit",
        ]

        for key in keys:
            calibration_dict.update(
                {key: np.full((num_points,) + slm_supershape, np.nan, dtype=np.float32)}
            )

        def superpixels(
                schedule=None,
                reference_phase=None,
                target_phase=None,
                reference_blaze=reference_blazes,
                target_blaze=calibration_blazes,
                phase_baselines=None,
                plot=False
            ):
            """
            Helper function for making superpixel phase masks.

            Parameters
            ----------
            schedule : list of int
                Defines which superpixels to source targets from.
            reference_phase, target_phase : float OR None
                Phase of reference/target superpixel; not rendered if None.
            reference_blaze, target_blaze : (float, float)
                Blaze vector(s) for the given superpixel.
            """
            matrix = blaze(self.slm, field_blaze)

            if reference_phase is not None:
                for i in range(num_points):
                    if schedule is None or schedule[i] != -1:
                        imprint(
                            matrix,
                            np.array([
                                reference_superpixels_coords[0, i], 1,
                                reference_superpixels_coords[1, i], 1
                            ]) * superpixel_size,
                            _blaze_offset,
                            self.slm,
                            # shift=True,
                            vector=reference_blaze[:, [i]],
                            offset=reference_phase  # This is usually zero when not None.
                        )

            if target_phase is not None and schedule is not None:
                target_coords = index2coord(schedule)
                for i in range(num_points):
                    if schedule[i] != -1:
                        phase_baseline = 0 if phase_baselines is None else phase_baselines[i]
                        imprint(
                            matrix,
                            np.array([
                                target_coords[0, i], 1,
                                target_coords[1, i], 1
                            ]) * superpixel_size,
                            _blaze_offset,
                            self.slm,
                            # shift=True,
                            vector=target_blaze[:, [i]],
                            offset=phase_baseline + (target_phase if np.isscalar(target_phase) else target_phase[i])
                        )

            self.slm.set_phase(matrix, settle=True)
            self.cam.flush()
            if plot:
                plt.figure(figsize=(20, 20))
                self.slm.plot()
            return self.cam.get_image()

        def fit_phase(phases, intensities, plot_fits=False):
            """
            Fits a sine function to the intensity vs phase, and extracts best phase and amplitude
            that give constructive interference.
            If fit fails return 0 on all values.

            Parameters
            ----------
            phases : numpy.ndarray
                Phase measurements.
            intensities : numpy.ndarray
                Intensity measurements.
            plot_fits : bool
                Whether to plot fit results.

            Returns
            -------
            best_phase : float
                b
            amp : float
                a
            r2 : float
                R^2 of fit
            contrast : float
                a / (a + c)
            """
            guess = [
                phases[np.argmax(intensities)],
                np.max(intensities) - np.min(intensities),
                np.min(intensities),
            ]

            try:
                popt, _ = optimize.curve_fit(cos, phases, intensities, p0=guess)
            except BaseException:
                warnings.warn("Curve fitting failed; nulling response from this superpixel.")
                return 0, 0, 0, 0

            # Extract phase and amplitude from fit.
            best_phase = popt[0]
            amp = popt[1]
            contrast = popt[1] / (popt[1] + popt[2])

            # Residual and total sum of squares, producing the R^2 metric.
            ss_res = np.sum((intensities - cos(phases, *popt)) ** 2)
            ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            if plot_fits:
                plt.scatter(phases / np.pi, intensities, color="k", label="Data")

                phases_fine = np.linspace(0, 2 * np.pi, 100)

                plt.plot(phases_fine / np.pi, cos(phases_fine, *popt), "k-", label="Fit")
                plt.plot(phases_fine / np.pi, cos(phases_fine, *guess), "k--", label="Guess")
                plt.plot(best_phase / np.pi, popt[1] + popt[2], "xr", label="Phase")

                plt.legend(loc="best")
                plt.title("Interference ($R^2$={:.3f})".format(r2))
                plt.grid()
                plt.xlim([0, 2])
                plt.xlabel(r"$\phi$ $[\pi]$")
                plt.ylabel("Signal")

                plt.show()

            return best_phase, amp, r2, contrast

        def fit_phase_image(img, dsuperpixel, plot_fits=True):
            """
            Fits a modulated 2D sinc function to an image, and extracts best phase and
            amplitude that give constructive interference.
            If fit fails return 0 on all values.

            Parameters
            ----------
            img : numpy.ndarray
                2D image centered on the interference point.
            dsuperpixel : ndarray
                Integer distance (dx,dy) between superpixels.

            Returns
            -------
            best_phase : float
                b
            amp : float
                a
            r2 : float
                R^2 of fit
            contrast : float
                a / (a + c)
            """
            # Future: Cache this outside to avoid repeating memory allocation.
            xy = np.meshgrid(
                *[
                    np.arange(-(img.shape[1 - a] - 1) / 2, +(img.shape[1 - a] - 1) / 2 + 0.5)
                    for a in range(2)
                ]
            )
            xyr = [l.ravel() for l in xy]

            # Process dsuperpixel by rotating it according to the Fourier calibration.
            M = self.calibrations["fourier"]["M"]
            M_norm = M / np.sqrt(np.abs(np.linalg.det(M)))
            dsuperpixel = np.squeeze(np.matmul(M_norm, format_2vectors(dsuperpixel)))

            # Make the guess and bounds.
            d = float(np.amin(img))
            c = 0
            a = float(np.amax(img)) - c
            R = float(np.mean(img.shape)) / 4
            # theta = np.arctan2(M[1, 0],  -M[0, 0])

            guess = [
                R, a, 0, c, d,
                8 * np.pi * dsuperpixel[0] / img.shape[1],
                8 * np.pi * dsuperpixel[1] / img.shape[0]
            ]
            dk = 8 * np.pi * np.max(slm_supershape) / np.min(img.shape)
            lb = [
                .9*R, 0, -4*np.pi, 0, 0,
                guess[5]-dk,
                guess[6]-dk
            ]
            ub = [
                1.1*R, 2*a, 4*np.pi, a, a,
                guess[5]+dk,
                guess[6]+dk
            ]

            # # Restrict sinc2d to be centered (as expected).
            # def sinc2d_local(xy, R, a=1, b=0, c=0, d=0, kx=1, ky=1, theta=0):
            #     # When centered, rotation can be applied to xy, kxy
            #     c = np.cos(theta)
            #     s = np.sin(theta)
            #     rotation = np.array([[c, -s], [s, c]])
            #     kxy = rotation @ np.array([kx, ky])

            #     # If raveled (for optimization)
            #     xy = np.array(xy)
            #     if len(np.array(xy).shape) < 3:
            #         xy_rot = rotation @ xy
            #     # But otherwise not raveled
            #     else:
            #         xy_rot = np.array([rotation @ xy[:, :, i] for i in range(xy.shape[-1])])
            #         xy_rot = np.transpose(xy_rot, (1, 2, 0))

            #     return sinc2d(xy_rot, 0, 0, R, a, b, c, d, kxy[0], kxy[1])

            # Determine the guess phase byt overlapping shifted guesses with the image.
            differences = []
            N = 20
            phases = np.arange(N) * 2 * np.pi / N

            for phase in phases:
                guess[2] = phase
                differences.append(np.sum(np.square(img - sinc2d_centered(xy, *guess))))

            guess[2] = phases[int(np.min(np.argmin(differences)))]

            # Try the fit!
            try:
                popt, _ = optimize.curve_fit(
                    sinc2d_centered,
                    xyr,
                    img.ravel().astype(float),
                    p0=guess,
                    bounds=(lb, ub), #, maxfev=20
                    # method="dogbox",
                    # jac=sinc2d_centered_jacobian
                )
            except BaseException:
                return [np.nan, np.nan, 0, np.nan]

            # Extract phase and amplitude from fit.
            best_phase = popt[2]
            amp = np.abs(popt[1])
            contrast = np.abs(popt[1] / (np.abs(popt[1]) + np.abs(popt[3])))

            # Remove the sinc term when doing the rsquared.
            popt_nomod = np.copy(popt)
            popt_nomod[3] += popt_nomod[1] / 2
            popt_nomod[1] = 0
            img0 = img - sinc2d_centered(xy, *popt_nomod)
            fit0 = sinc2d_centered(xy, *popt) - sinc2d_centered(xy, *popt_nomod)

            # Residual and total sum of squares, producing the R^2 metric.
            ss_res = np.sum((img0 - fit0) ** 2)
            ss_tot = np.sum((img0 - np.mean(img0)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            final = (np.mod(-best_phase, 2*np.pi), amp, r2, contrast)

            # Plot the image, guess, and fit, if desired.
            if plot_fits:
                _, axs = plt.subplots(1, 3, figsize=(20,10))

                axs[0].imshow(img)
                axs[1].imshow(sinc2d_centered(xy, *guess))
                axs[2].imshow(sinc2d_centered(xy, *popt))

                for index, title in enumerate(["Image", "Guess", "Fit"]):
                    axs[index].set_title(title)

                plt.show()

            return final

        def plot_labeled(schedule, img, phase=None, plot=False, title="", plot_zoom=False, focus=None):
            if plot_everything or plot:
                def plot_labeled_rects(ax, points, labels, colors, wh, hh):
                    for point, label, color in zip(points, labels, colors):
                        rect = plt.Rectangle(
                            (float(point[0] - wh/2), float(point[1] - hh/2)),
                            float(wh), float(hh),
                            ec=color, fc="none"
                        )
                        ax.add_patch(rect)
                        ax.annotate(
                            label, (point[0], point[1]),
                            c=color, size="x-small", ha="center", va="center"
                        )

                if return_movie:
                    fig, axs = plt.subplots(1, 3, figsize=(16, 4), facecolor="white")
                else:
                    fig, axs = plt.subplots(1, 3, figsize=(16,4))

                # Plot phase on the first axis.
                if phase is None:
                    phase = self.slm.phase
                axs[0].imshow(
                    np.mod(phase, 2*np.pi),
                    cmap=plt.get_cmap("twilight"),
                    interpolation="none",
                )

                points = []
                labels = []
                colors = []
                center_offset = np.array([superpixel_size/2, superpixel_size/2])

                for i in range(num_points):
                    if schedule is None or schedule[i] != -1:
                        if focus is None:
                            focus = i
                        points.append(reference_superpixels_coords[:, i] * superpixel_size + center_offset)
                        if schedule is not None: points.append(index2coord(schedule[i]).ravel() * superpixel_size + center_offset)
                        if num_points > 1:
                            labels.append("{}".format(i))
                            if schedule is not None: labels.append("{}".format(i))
                        else:
                            labels.append("Reference\nSuperpixel")
                            if schedule is not None: labels.append("Test\nSuperpixel")
                        c1 = (1 if i == focus else .5, .2, 0)
                        colors.append(c1)
                        c2 = (1 if i == focus else .5, 0, .2)
                        if schedule is not None: colors.append(c2)

                plot_labeled_rects(axs[0], points, labels, colors, superpixel_size, superpixel_size)

                # FUTURE: fix for multiple
                # if plot_zoom:
                #     for a in [0, 1]:
                #         ref = reference_superpixels[a] * superpixel_size
                #         test = test_superpixel[a] * superpixel_size

                #         lim = [min(ref, test) - .5 * superpixel_size, max(ref, test) + 1.5 * superpixel_size]

                #         if a:
                #             axs[0].set_ylim([lim[1], lim[0]])
                #         else:
                #             axs[0].set_xlim(lim)

                if img is not None:
                    im = axs[1].imshow(np.log10(img + .1))
                    im.set_clim(0, np.log10(self.cam.bitresolution))

                dpoint = field_point - base_point

                # Assemble points and labels.
                points = [(base_point + N * dpoint).ravel() for N in range(-2, 3)]
                labels = ["-2nd", "-1st", "0th", "1st", "2nd"]
                colors = ["b"] * 5

                focus_point = None

                for i in range(num_points):
                    if schedule is None or schedule[i] != -1:
                        points.append(calibration_points[:, i])
                        if num_points > 1:
                            labels.append("{}".format(i))
                        else:
                            labels.append("Calibration\nPoint")
                        c = (1 if i == focus else .5, 0, 0)
                        colors.append(c)
                        if i == focus:
                            focus_point = calibration_points[:, i]

                # Plot points and labels.
                wh = int(interference_window[0])
                hh = int(interference_window[1])

                plot_labeled_rects(axs[1], points, labels, colors, wh, hh)

                if img is not None:
                    im = axs[2].imshow(np.log10(img + .1))
                    im.set_clim(0, np.log10(self.cam.bitresolution))

                    if self.cam.bitdepth > 10:
                        step = 2
                    else:
                        step = 1

                    bitres_list = np.power(2, np.arange(0, self.cam.bitdepth+1, step), dtype=int)

                    cbar = fig.colorbar(im, ax=axs[2])
                    cbar.ax.set_yticks(np.log10(bitres_list))
                    cbar.ax.set_yticklabels(bitres_list)

                point = focus_point

                axs[2].scatter([point[0]], [point[1]], 5, "r", "*")
                axs[2].set_xlim(point[0] - wh/2, point[0] + wh/2)
                axs[2].set_ylim(point[1] + hh/2, point[1] - hh/2)

                # Axes coloring and colorbar.
                for spine in ["top", "bottom", "right", "left"]:
                    axs[2].spines[spine].set_color("r")
                    axs[2].spines[spine].set_linewidth(1.5)

                axs[0].set_title("SLM Phase")
                axs[1].set_title("Camera Result")
                axs[2].set_title(title)

                if plot_zoom and return_movie:
                    fig.tight_layout()
                    fig.canvas.draw()

                    try:
                        try:
                            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                            image_from_plot = image_from_plot.reshape(
                                fig.canvas.get_width_height()[::-1] + (3,)
                            )
                        except:
                            image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                            image_from_plot = image_from_plot.reshape(
                                fig.canvas.get_width_height()[::-1] + (4,)
                            )[:,:,:3]
                    except:
                        warnings.warn(
                            "Failed to convert figure to image for wavefront_calibrate movie. "
                            "Returning a blank image instead."
                        )
                        image_from_plot = np.zeros(
                            fig.canvas.get_width_height()[::-1] + (3,),
                            dtype=np.uint8
                        )

                    plt.close()

                    return image_from_plot
                else:
                    plt.show()

        def take_interference_regions(img, integrate=True):
            """Helper function for grabbing the data at the calibration points."""
            return analysis.take(
                img,
                calibration_points,
                interference_window, # / (2 if integrate else 1),
                clip=True,
                integrate=integrate
            )

        def find_centers(img, fit=True):
            """Helper function for finding the center of images around the calibration points."""
            imgs = take_interference_regions(img, integrate=False)  # N x W x H
            centers = analysis.image_positions(imgs)                # 2 x N

            a = np.max(imgs, axis=(1,2))
            R = np.mean(imgs.shape[1:])/4

            guess = np.transpose(
                np.vstack((
                    centers,
                    np.full_like(a, R),
                    a,
                    np.full_like(a, 0),
                ))
            )

            result = analysis.image_fit(imgs, function=_sinc2d_nomod, guess=guess) #, plot=True)

            centers = result[:, 1:3].T

            # if not fit:
            return centers + calibration_points
            # else:
            #     return centers + calibration_points, amps_fit

        def measure(schedule, plot=False):
            # self.cam.flush()

            # Step 0: Measure the background.
            if measure_background:
                back_image = superpixels(schedule, None, None)
                plot_labeled(schedule, back_image, plot=plot, title="Background")
                back = take_interference_regions(back_image)
            else:
                back = [np.nan] * num_points

            # Step 0.5: Measure the power in the reference mode.
            norm_image = superpixels(schedule, 0, None)
            plot_labeled(schedule, norm_image, plot=plot, title="Reference Diffraction")
            norm = take_interference_regions(norm_image)

            # Step 1: Check the target mode, and return if we don't need to correct.
            position_image = superpixels(schedule, None, 0)
            plot_labeled(schedule, position_image, plot=plot, title="Base Target Diffraction")
            if phase_steps is None and not corrected_amplitude:
                pwr = take_interference_regions(position_image)
                return {
                    "power": pwr,
                    "normalization": norm,
                    "background": back,
                    "phase": [np.nan] * num_points,
                    "kx": [np.nan] * num_points,
                    "ky": [np.nan] * num_points,
                    "amp_fit": [np.nan] * num_points,
                    "contrast_fit": [np.nan] * num_points,
                    "r2_fit": [np.nan] * num_points,
                }

            # Step 1.25: Add a blaze to the target mode so that it overlaps with reference mode.
            found_centers = find_centers(position_image)
            blaze_differences = self.ijcam_to_kxyslm(found_centers) - calibration_blazes
            target_blaze_fixed = calibration_blazes - blaze_differences

            # Step 1.5: Measure the power...
            if corrected_amplitude:      # ...in the corrected target mode.
                fixed_image = superpixels(schedule, None, 0, target_blaze=target_blaze_fixed)
                plot_labeled(schedule, fixed_image, plot=plot, title="Corrected Target Diffraction")
                pwr = take_interference_regions(fixed_image)
            else:                       # ...in the uncorrected target mode.
                pwr = take_interference_regions(position_image)

            # Step 1.75: Stop here if we don't need to measure the phase (only save powers).
            if phase_steps is None:
                return {
                    "power": pwr,
                    "normalization": norm,
                    "background": back,
                    "phase": [np.nan] * num_points,
                    "kx": -blaze_differences[0, :],
                    "ky": -blaze_differences[1, :],
                    "amp_fit": [np.nan] * num_points,
                    "contrast_fit": [np.nan] * num_points,
                    "r2_fit": [np.nan] * num_points,
                }

            results = []
            first_index = np.where(schedule != -1)[0][0]

            # target_coords = index2coord(schedule)
            # phase_baselines = np.sum(
            #     2 * np.pi * target_blaze_fixed *
            #     (target_coords - reference_superpixels_coords) *
            #     superpixel_size * self.slm.pitch[:, np.newaxis],
            #     axis=0,
            # )
            phase_baselines = None

            # Step 2: Measure interference and find relative phase. Future: vectorize.
            if phase_steps == 1:
                # Step 2.1: Gather a single image.
                result_img = superpixels(schedule, 0, 0, target_blaze=target_blaze_fixed, phase_baselines=phase_baselines)
                cropped_img = take_interference_regions(result_img, integrate=False)

                # Step 2.2: Fit the data and return.
                coord_difference = index2coord(schedule) - index2coord(reference_superpixels)

                results = [
                    (
                        fit_phase_image(
                            cropped_img[i],
                            coord_difference[:,i],
                            plot_fits=plot and i == first_index
                        )
                        if schedule[i] != -1 else
                        [np.nan] * 4
                    )
                    for i in range(num_points)
                ]
            else:
                # Gather multiple images at different phase offsets.
                phases = np.linspace(0, 2 * np.pi, phase_steps, endpoint=False)
                iresults = []  # list for recording the intensity of the reference point

                # Determine whether to use a progress bar.
                if verbose:
                    description = "phase_measurement"
                    prange = tqdm(phases, position=0, leave=False, desc=description)
                else:
                    prange = phases

                if return_movie:
                    frames = []

                # Step 2.1: Measure phases
                for phase in prange:
                    interference_image = superpixels(schedule, 0, phase, target_blaze=target_blaze_fixed, phase_baselines=phase_baselines)
                    iresults.append(
                        [
                            interference_image[calibration_points[1, i], calibration_points[0, i]]
                            for i in range(num_points)
                        ]
                    )

                    if return_movie:
                        frames.append(
                            plot_labeled(
                                schedule,
                                interference_image,
                                plot=plot,
                                title=r"Phase = ${:1.2f}\pi$".format(phase / np.pi),
                                plot_zoom=True,
                            )
                        )

                iresults = np.array(iresults)

                # Step 2.2: Fit to sine and return.
                for i in range(num_points):
                    results.append(fit_phase(phases, iresults[:, i], plot and i == first_index))

            results = np.array(results)

            phase_fit =     results[:, 0]
            amp_fit =       results[:, 1]
            r2_fit =        results[:, 2]
            contrast_fit =  results[:, 3]

            # Step 2.5: maybe plot a picture of the correct phase.
            if plot:
                interference_image = superpixels(schedule, 0, phase_fit, target_blaze=target_blaze_fixed, phase_baselines=phase_baselines)
                plot_labeled(schedule, interference_image, plot=plot, title="Best Interference")

            # Step 3: Return the result.
            if return_movie:
                return frames

            return {
                "power": pwr,
                "normalization": norm,
                "background": back,
                "phase": phase_fit,
                "kx": -blaze_differences[0, :],
                "ky": -blaze_differences[1, :],
                "amp_fit": amp_fit,
                "contrast_fit": contrast_fit,
                "r2_fit": r2_fit,
            }

        # Correct exposure and position of the reference mode(s).
        # self.cam.flush()
        base_image = superpixels(None, 0, None)
        plot_labeled(None, base_image, plot=plot_everything, title="Base Reference Diffraction")
        found_centers = find_centers(base_image)

        # Correct the original blaze using the measured result.
        reference_blaze_differences = self.ijcam_to_kxyslm(found_centers) - reference_blazes
        np.subtract(reference_blazes, reference_blaze_differences, out=reference_blazes)

        if plot_fits:
            fixed_image = superpixels(None, 0, None)
            plot_labeled(None, fixed_image, plot=plot_everything, title="Corrected Reference Diffraction")

        # If we just want to debug/test one region, then do so.
        if test_index is not None:
            result = measure(scheduling[:, test_index], plot=plot_fits)

            # Reset the phase and amplitude of the SLM to the stored data.
            self.slm.source["amplitude"] = amplitude
            self.slm.source["phase"] = phase

            return result

        measurements = range(num_measurements)
        if plot > -1:
            measurements = tqdm(measurements, position=1, leave=True, desc="calibration")

        # Proceed with all of the superpixels.
        for n in measurements:
            schedule = scheduling[:, n]

            # Measure!
            measurement = measure(schedule)

            # Update dictionary.
            coords = index2coord(schedule)
            for i in range(num_points):
                if schedule[i] != -1:
                    for key in measurement.keys():
                        result = measurement[key]
                        if np.size(result) > 1:
                            result = result[i]
                        elif not np.isscalar(result):
                            result = np.squeeze(result)

                        calibration_dict[key][i, coords[1, i], coords[0, i]] = result

        self.calibrations["wavefront_superpixel"] = calibration_dict
        self.calibrations["wavefront_superpixel"].update(self._get_calibration_metadata())

        return calibration_dict

    ### Wavefront Calibration Helpers ###

    def wavefront_calibration_points(
        self,
        pitch,
        field_exclusion=None,
        field_point=(0,0),
        field_point_units="kxy",
        avoid_points=None,
        avoid_mirrors=True,
        avoid_nyquist=True,
        plot=False,
    ):
        """
        Generates a grid of points to perform wavefront calibration at.

        Parameters
        ----------
        pitch : float OR (float, float)
            The grid of points must have pitch greater than this value.
        field_exclusion : float OR None
            Remove all points within ``field_exclusion`` of a ``field_point``.
            Set to zero if no removal is desired.
            If ``None``, defaults to ``pitch``.
        field_point : (float, float)
            Position in the camera domain where the field (pixels not included in superpixels)
            is blazed toward in order to reduce light in the camera's field. The suggested
            approach is to set this outside the field of view of the camera and make
            sure that other diffraction orders are far from the ``calibration_points``.
            Defaults to no blaze (``(0,0)`` in ``"kxy"`` units).
        field_point_units : str
            A unit compatible with
            :meth:`~slmsuite.holography.toolbox.convert_vector()`.
            Defaults to ``"kxy"``.

            Tip
            ~~~
            Setting one coordinate of ``field_point`` to zero is suggested
            to minimize higher order diffraction.
        avoid_points : numpy.ndarray
            Additional points to avoid in the same manner as avoiding the ``field_point``
            and diffractive orders (with the same radius ``field_exclusion``).
            This can, for instance, omit the points outside the camera's field of view,
            points around known stray reflections, or unusual topology.
        avoid_mirrors : bool
            When a 1st order calibration beam is sourced from a
            weak superpixel in the SLM domain, the -1st order of a different
            calibration beam can act as a strong noise source if
            it is sourced from a strong central superpixel.
            If ``True``, this flag aligns the -1st orders to be inbetween
            the 1st orders of the grid of calibration points.
        avoid_nyquist : bool
            If ``True``, omits points that are outside the first Nyquist zone.

        Returns
        -------
        numpy.ndarray
            List of points of shape ``(2, N)`` to calibrate at in the ``"ij"`` basis.

        Raises
        ------
        AssertionError
            If the fourier plane calibration does not exist.
        """
        # Parse field_point.
        field_point = toolbox.convert_vector(
            format_2vectors(field_point),
            from_units=field_point_units,
            to_units="ij",
            hardware=self
        )
        field_point = np.rint(format_2vectors(field_point)).astype(int)

        # Parse field_exclusion.
        if field_exclusion is None:
            field_exclusion = pitch
        if not np.isscalar(field_exclusion):
            field_exclusion = np.mean(field_exclusion)

        # Gather other information.
        zeroth_order = np.rint(self.kxyslm_to_ijcam([0, 0])).astype(int)

        # Generate the initial grid.
        plane = format_2vectors(self.cam.shape[::-1])
        grid = np.ceil(plane / pitch - .5)
        spacing = np.floor(plane / (grid + (.5 if avoid_mirrors else 0))).astype(int)
        if avoid_mirrors:
            base_point = spacing * (np.remainder(zeroth_order / spacing - .5, 1) + .25)
        else:
            base_point = spacing / 2

        # In ij coordinates.
        calibration_points = fit_3pt(
            base_point,
            (spacing[0,0], 0),
            (0, spacing[1,0]),
            np.squeeze(grid).astype(int),
            x1=None,
            x2=None
        )

        if avoid_nyquist:
            calibration_points_knm = convert_vector(
                calibration_points,
                from_units="ij",
                to_units="knm",
                hardware=self,
                shape=[1,1]
            )

            outside_first_nyquist_zone = (
                (calibration_points_knm[0] < 0) +
                (calibration_points_knm[1] < 0) +
                (calibration_points_knm[0] > 1) +
                (calibration_points_knm[1] > 1)
            ) > 0
            calibration_points = np.delete(calibration_points, outside_first_nyquist_zone, axis=1)

        # Sort by proximity to the center, avoiding the 0th order.
        distance = np.sum(np.square(calibration_points - zeroth_order), axis=0)
        I = np.argsort(distance)
        calibration_points = calibration_points[:, I]

        # Prune points within field_exclusion from a given order (-2, ..., 2).
        dorder = field_point - zeroth_order
        order_points = np.hstack([zeroth_order + dorder * i for i in range(-2, 3)])

        if avoid_points is None:
            avoid_points = order_points
        else:
            avoid_points = np.hstack((format_2vectors(avoid_points), order_points))

        for i in range(avoid_points.shape[1]):
            point = avoid_points[:, [i]]
            distance = np.sum(np.square(calibration_points - point), axis=0)
            calibration_points = np.delete(
                calibration_points,
                distance < field_exclusion*field_exclusion,
                axis=1
            )

            # Plot bad points.
            if plot: plt.scatter(point[0], point[1], c="r")

        if plot:
            # Points
            plt.scatter(
                calibration_points[0,:],
                calibration_points[1,:],
                c=np.arange(calibration_points.shape[1]),
                cmap="Blues"
            )

            # Mirrors
            plt.scatter(
                2*zeroth_order[0,0] - calibration_points[0,:],
                2*zeroth_order[1,0] - calibration_points[1,:],
                c=np.arange(calibration_points.shape[1]),
                marker=".",
                cmap="Reds"
            )

            # Future: Plot SLM FoV?

            plt.xlim([0, self.cam.shape[1]])
            plt.ylim([self.cam.shape[0], 0])
            plt.show()

        return calibration_points

    def wavefront_calibration_superpixel_window(self, superpixel_size):
        """
        Returns the window size for the interference regions.
        This is inversely proportional to the size of the superpixel because the
        superpixel and interference zone are separated by a Fourier transform.
        The computation works by estimating the spot size of the interference beams and
        then enlarging by a stored multiplier
        :attr:`_wavefront_calibration_window_multiplier`
        which defaults to 4.

        Parameters
        ----------
        superpixel_size : int
            The size of the superpixel on the SLM.
        """
        interference_size = np.rint(np.array(
            self.get_farfield_spot_size(
                superpixel_size * self.slm.pitch,
                basis="ij"
            )
        )).astype(int)

        return self._wavefront_calibration_window_multiplier * interference_size

    def wavefront_calibration_superpixel_process(
        self,
        index=0,
        smooth=True,
        r2_threshold=0.9,
        remove_vortices=False,
        remove_blaze=True,
        remove_background=True,
        apply=True,
        plot=False
    ):
        """
        Processes :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations` ``["wavefront"]``
        into the desired phase correction and amplitude measurement. Applies these
        parameters to the respective variables in the SLM if ``apply`` is ``True``.

        Parameters
        ----------
        index : int
            The calibration point index to process, in the case of a multi-point calibration.
            In the future, this should include the option to request an "ij" position,
            then the return will automatically interpolate between the Zernike results
            of the local calibration points.
        smooth : bool OR int
            Whether to blur the correction data to avoid aliasing.
            If ``int``, uses this as the number of smoothing iterations.
            Defaults to 16 if ``True``.
        r2_threshold : float
            Threshold for a "good fit". Proxy for whether a datapoint should be used or
            ignored in the final data, depending upon the rsquared value of the fit.
            Should be within [0, 1].
        remove_vortices : bool
            A wavefront correct should be smooth when using smooth optics (lenses).
            However, incorrect phase wrapping can lead to phase vortices surrounding the
            exceptional points at the ends of improperly chosen branches.
            This is unphysical. If `True`, these exceptional points are eliminated
            halfway through the phase smoothing process. If `smooth=False`, this is
            ignored. The nature of vortex removal might add a global blaze to the
            pattern, so it is recommended to also set ``remove_blaze=True``.
        remove_blaze : bool
            If ``True``, removes the global blaze from the phase correction, as defined
            by the average blaze weighted by the measured power.
        remove_background : bool
            If the experimental background was not measured, this flag estimates the
            interference region's background by looking at the noisefloor of the
            measured power distribution. If the noisefloor is flat enough, the
            power is shifted to have a minimum at zero.
        apply : bool
            Whether to apply the processed calibration to the associated SLM.
            Otherwise, this function only returns and maybe
            plots these results. Defaults to ``True``.
        plot : bool
            Whether to enable debug plots.

        Returns
        -------
        dict
            The updated source dictionary containing the processed source amplitude and phase.
        """
        # Step 0: Initialize helper variables and functions.
        if "wavefront_superpixel" in self.calibrations:
            data = self.calibrations["wavefront_superpixel"]
        elif "wavefront" in self.calibrations:
            data = self.calibrations["wavefront"]
        else:
            raise RuntimeError("Could not find wavefront calibration.")

        if len(data) == 0:
            raise RuntimeError("No raw wavefront data to process. Either load data or calibrate.")

        if not "__version__" in data:
            data["__version__"] = "0.0.1"

        if data["__version__"] == "0.0.1":
            return self._wavefront_calibration_superpixel_process_r001(
                data,
                smooth=smooth,
                r2_threshold=r2_threshold,
                remove_vortices=remove_vortices,
                remove_blaze=remove_blaze,
                remove_background=remove_background,
                apply=apply,
                plot=plot
            )
        else:
            # For now, make a 0.0.1 calibration dict based on a single index.
            slm_supershape = data["slm_supershape"]

            def index2coord(index):
                return format_2vectors(
                    np.stack((index % slm_supershape[1], index // slm_supershape[1]), axis=0)
                )

            reference_superpixel = index2coord(data["reference_superpixels"][index]).ravel()

            correction_dict = {
                "NX": slm_supershape[1],
                "NY": slm_supershape[0],
                "nxref": reference_superpixel[0],
                "nyref": reference_superpixel[1],
                "superpixel_size": data["superpixel_size"],
                "interference_point": data["calibration_points"][:, index],
                "interference_size": data["interference_size"],
            }

            keys = [
                "power",
                "normalization",
                "background",
                "phase",
                "kx",
                "ky",
                "amp_fit",
                "contrast_fit",
                "r2_fit",
            ]

            for key in keys:
                correction_dict.update({key: data[key][index]})

            return self._wavefront_calibration_superpixel_process_r001(
                correction_dict,
                smooth=smooth,
                r2_threshold=r2_threshold,
                remove_vortices=remove_vortices,
                remove_blaze=remove_blaze,
                remove_background=remove_background,
                apply=apply,
                plot=plot,
            )

    def _wavefront_calibration_superpixel_process_r001(
            self,
            data,
            smooth=True,
            r2_threshold=0.9,
            remove_vortices=False,
            remove_blaze=True,
            remove_background=True,
            apply=True,
            plot=False,
        ):
        """
        Old wavefront calibration processing for release 0.0.1.
        See docstring for :meth:`wavefront_calibration_superpixel_process`.

        Returns
        -------
        dict
            The updated source dictionary containing the processed source amplitude and phase.
        """
        # Parse smooth.
        if smooth is True:
            smooth = 16
        smooth = int(smooth)
        if smooth < 0:
            raise ValueError("Smoothing iterations must be a non-negative integer.")

        # Parse r2_threshold.
        r2_threshold = float(r2_threshold)

        # Step 0: Initialize helper variables and functions.
        if len(data) == 0:
            raise RuntimeError("No raw wavefront data to process. Either load data or calibrate.")

        NX = data["NX"]
        NY = data["NY"]
        nxref = data["nxref"]
        nyref = data["nyref"]

        def average_neighbors(matrix):
            n = 0
            result = 0
            for xy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                x =  nxref + xy[0]
                y =  nyref + xy[1]

                if x >= 0 and x < NX and y >= 0 and y < NY:
                    result += matrix[y, x]
                    n += 1

            matrix[nyref, nxref] = result / n

        size_blur_k = 3

        # Step 1: Process R^2
        superpixel_size = int(data["superpixel_size"])
        w = superpixel_size * NX
        h = superpixel_size * NY

        r2 = np.copy(data["r2_fit"])
        r2[nyref, nxref] = 1
        r2s = r2

        r2s_large = cv2.resize(r2s, (w, h), interpolation=cv2.INTER_NEAREST)
        r2s_large = r2s_large[: self.slm.shape[0], : self.slm.shape[1]]

        # Step 2: Process the measured amplitude
        # Fix the reference pixel by averaging the 8 surrounding pixels
        pwr = np.copy(data["power"])
        pwr[pwr == np.inf] = np.amax(pwr)
        average_neighbors(pwr)
        if smooth:
            pwr = cv2.GaussianBlur(pwr, (size_blur_k, size_blur_k), 0)

        norm = np.copy(data["normalization"])
        average_neighbors(norm)
        if smooth:
            norm = cv2.GaussianBlur(norm, (size_blur_k, size_blur_k), 0)

        back = np.copy(data["background"])
        back[np.isnan(back)] = 0
        average_neighbors(back)
        if smooth:
            back = cv2.GaussianBlur(back, (size_blur_k, size_blur_k), 0)

        if remove_background:
            is_noise = r2s < r2_threshold
            if np.all(back == 0) and np.sum(is_noise) > 0:
                # Check the area defined as noise.
                pwr_below_r2 = pwr[is_noise]
                pwr_below_r2[np.isnan(pwr_below_r2)] = np.nanmin(pwr_below_r2)
                pwr_below_r2[np.isnan(pwr_below_r2)] = 0

                # If the median is within 0.5 std of the minimum, then we assume the
                # minimum is close to the noise floor.
                pwr_min = np.min(pwr_below_r2)
                norm_ave = np.nanmean(norm)
                norm_min = np.nanmin(norm)
                if (np.median(pwr_below_r2) - pwr_min) / np.nanstd(pwr) < .5 and pwr_min < norm_min:
                    warnings.warn(
                        f"remove_background is enabled and a noise floor was detected; "
                        f"removing this background ({pwr_min/norm_ave}% of the average normalization)."
                    )
                    back[:] = pwr_min

        pwr -= back
        norm -= back

        # Normalize and resize
        pwr_norm = np.divide(pwr, norm)

        pwr_norm[np.isnan(pwr_norm)] = 0
        pwr_norm[~np.isfinite(pwr_norm)] = 0
        pwr_norm[pwr_norm < 0] = 0

        pwr_large = cv2.resize(pwr_norm, (w, h), interpolation=cv2.INTER_CUBIC)
        pwr_large = pwr_large[: self.slm.shape[0], : self.slm.shape[1]]

        pwr_large[np.isnan(pwr_large)] = 0
        pwr_large[~np.isfinite(pwr_large)] = 0
        pwr_large[pwr_large < 0] = 0

        if smooth:
            size_blur = 4 * int(superpixel_size) + 1
            pwr_large = cv2.GaussianBlur(pwr_large, (size_blur, size_blur), 0)

        amp_large = np.sqrt(pwr_large)
        amp_large /= np.nanmax(amp_large)

        # Step 3: Process the wavefront
        # Load data.
        kx = np.copy(data["kx"])
        ky = np.copy(data["ky"])
        offset = np.copy(data["phase"])

        # Handle nans.
        kx[np.isnan(kx)] = 0
        ky[np.isnan(ky)] = 0
        offset[np.isnan(offset)] = 0
        r2[np.isnan(r2)] = 0

        # Fix a change in how data is aquired pre-0.3.0.
        # if phase_shift_pre_030:
        #     X = np.arange(float(NX)); X -= np.mean(X)
        #     Y = np.arange(float(NY)); Y -= np.mean(Y)
        #     grid_x, grid_y = np.meshgrid(X, Y)
        #     (dx, dy) = (
        #         2 * np.pi * superpixel_size * self.slm.pitch[0],
        #         2 * np.pi * superpixel_size * self.slm.pitch[1],
        #     )
        #     offset += dx * kx * grid_x + dy * ky * grid_y

        # Fill in the reference pixel with surrounding data.
        real = np.cos(offset)
        imag = np.sin(offset)

        average_neighbors(real)
        average_neighbors(imag)

        average_neighbors(kx)
        average_neighbors(ky)

        offset = np.arctan2(imag, real) + np.pi

        # Apply the R^2 threshold.
        kx[r2s < r2_threshold] = 0
        ky[r2s < r2_threshold] = 0
        offset[r2s < r2_threshold] = 0
        phase_maybe = np.zeros_like(offset)
        pathing = 0 * r2s - 100

        # Step 3.1: Infer phase for superpixels which do satisfy the R^2 threshold.
        # For each row...
        # Go forward and then back along each row.
        for nx in list(range(NX)) + list(range(NX - 1, -1, -1)):
            for ny in range(NY):
                if r2s[ny, nx] >= r2_threshold:
                    # Superpixels exceeding the threshold need no correction.
                    pass
                else:
                    # Otherwise, do a majority-vote with adjacent superpixels.
                    kx2 = []
                    ky2 = []
                    offset2 = []
                    source = []

                    (dx0, dy0) = (
                        2 * np.pi * (nx-nxref) * superpixel_size * self.slm.pitch[0],
                        2 * np.pi * (ny-nyref) * superpixel_size * self.slm.pitch[1],
                    )

                    # Loop through the adjacent superpixels (including diagonals).
                    for ax, ay in [
                        (1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1),
                        # (1, -1),
                        # (-1, -1),
                        # (1, 1),
                        # (-1, 1),
                    ]:
                        (tx, ty) = (nx + ax, ny + ay)
                        # (dx, dy) = (
                        #     2 * np.pi * ax * superpixel_size * self.slm.pitch[0],
                        #     2 * np.pi * ay * superpixel_size * self.slm.pitch[1],
                        # )

                        # Make sure our adjacent pixel under test is within range and above threshold.
                        if (
                            tx >= 0
                            and tx < NX
                            and ty >= 0
                            and ty < NY
                            and (
                                r2s[ty, tx] >= r2_threshold
                                or pathing[ty, tx] == ny
                                or (abs(pathing[ty, tx] - ny) == 1 and ax != 0)
                            )
                        ):
                            kx3 = kx[ty, tx]
                            ky3 = ky[ty, tx]

                            kx2.append(kx3)
                            ky2.append(ky3)
                            offset2.append(offset[ty, tx] + (dx0 * kx3 + dy0 * ky3))
                            source.append((ax, ay))

                    # Do a majority vote (within std) for the phase.
                    if len(kx2) > 0:
                        kx[ny, nx] = np.mean(kx2)
                        ky[ny, nx] = np.mean(ky2)

                        minstd = np.inf
                        for phi in range(4):
                            shift = phi * np.pi / 2
                            offset3 = np.mod(np.array(offset2) + shift, 2 * np.pi)

                            if minstd > np.std(offset3):
                                minstd = np.std(offset3)
                                offset[ny, nx] = np.mod(np.mean(offset3) - shift, 2 * np.pi)

                        offset[ny, nx] -=  dx0 * kx[ny, nx] + dy0 * ky[ny, nx]
                        pathing[ny, nx] = ny

        # Step 3.2: Make the SLM-sized correction using the compressed data from each superpixel.
        phase = np.zeros(self.slm.shape)
        for nx in range(NX):
            for ny in range(NY):
                imprint(
                    phase,
                    np.array([nx, 1, ny, 1]) * superpixel_size,
                    _blaze_offset,
                    self.slm,
                    # shift=True,
                    vector=(kx[ny, nx], ky[ny, nx]),
                    offset=offset[ny, nx],
                )

        # Step 3.3: Iterative smoothing helps to preserve slopes while avoiding superpixel boundaries.
        # Consider, for instance, a fine blaze which smooths flat.
        if smooth:
            for i in tqdm(range(smooth), desc="smooth"):
                real = np.cos(phase)
                imag = np.sin(phase)

                # Blur the phase to smooth it out
                size_blur = 2 * int(superpixel_size / 4) + 1
                real = cv2.GaussianBlur(real, (size_blur, size_blur), 0)
                imag = cv2.GaussianBlur(imag, (size_blur, size_blur), 0)

                phase = np.arctan2(imag, real) + np.pi

                # If selected, remove vortices halfway through the smoothing.
                if remove_vortices and i == smooth//2:
                    phase = image_remove_vortices(phase)
        else:
            real = np.cos(phase)
            imag = np.sin(phase)
            phase = np.arctan2(imag, real) + np.pi

        # Step 3.4: Pattern cleanup.
        if remove_blaze:
            phase = image_remove_blaze(phase, mask=pwr_large)

        # Shift the final phase to minimize the effect of phase wrapping
        # (only matters when projecting patterns with small dynamic range).
        phase = image_reduce_wraps(phase, mask=pwr_large)

        # Add the old phase correction if it's there.
        if (
            "previous_phase_correction" in data and
            data["previous_phase_correction"] is not None
        ):
            phase += data["previous_phase_correction"]

        # Step 4: Data export.
        # Build the final dict.
        wavefront_calibration = {
            "phase": phase,
            "amplitude": amp_large,
            "r2": r2s_large,
            "r2_threshold": r2_threshold,
        }

        # Step 4.1: Load the correction to the SLM
        if apply:
            self.slm.source.update(wavefront_calibration)

        # Plot the result
        if plot:
            self.slm.plot_source(source=wavefront_calibration)

        return wavefront_calibration

    def _wavefront_calibration_superpixel_plot_raw(self, index=0, r2_threshold=0, phase_detail=True):
        """
        Plots raw data from the superpixel-style wavefront calibration. Specifically,
        plots:

        - The location of the point in the camera plane,
        - The measured source phase at each superpixel,
        - The measured source power at each superpixel,
        - The rsquared of the fit at each superpixel.

        Parameters
        ----------
        index : int OR None:
            For multi-point calibrations, the index of the point to plot data for.
            If ``None``, displays a single plot with the location of all indices.
        r2_threshold : float
            Ignores points with fit quality below this threshold.
        phase_detail : bool
            If ``True``, plots the derivatives of the phase instead of the power and rsquared.
        """
        plt.figure(figsize=(16, 8))

        data = self.calibrations["wavefront_superpixel"]

        if index is None:
            coords = data["calibration_points"]

            plt.subplot(1, 4, 1)
            plt.scatter(coords[0,:], coords[1,:], c="r")
            for i in range(coords.shape[1]):
                plt.annotate(str(i), (coords[0, i], coords[1, i]))
            plt.title("Calibration Points")
            plt.xlabel("Camera $x$ [pix]")
            plt.ylabel("Camera $y$ [pix]")
            plt.xlim([0, self.cam.shape[1]])
            plt.ylim([self.cam.shape[0], 0])
            plt.gca().set_aspect(1)

            return

        # Grab all the data
        coord = data["calibration_points"][:, index].copy()
        phase = data["phase"][index, :, :].copy()
        kx = data["kx"][index, :, :].copy()
        ky = data["ky"][index, :, :].copy()
        power = data["power"][index, :, :] / data["normalization"][index, :, :]
        amp = np.sqrt(power)
        r2 = data["r2_fit"][index, :, :].copy()

        # Threshold the data
        below_thresh = r2 < r2_threshold
        phase[below_thresh] = np.nan
        kx[below_thresh] = np.nan
        ky[below_thresh] = np.nan
        amp[below_thresh] = np.nan

        kscale = np.max([np.nanmax(np.abs(kx)), np.nanmax(np.abs(ky))])

        plt.subplot(1, 4, 1)
        plt.scatter(coord[0], coord[1], c="r")
        plt.annotate(str(index), (coord[0], coord[1]))
        plt.title("Calibration Point {}".format(index))
        plt.xlabel("Camera $x$ [pix]")
        plt.ylabel("Camera $y$ [pix]")
        plt.xlim([0, self.cam.shape[1]])
        plt.ylim([self.cam.shape[0], 0])
        plt.gca().set_aspect(1)

        plt.subplot(1, 4, 2)
        plt.imshow(
            phase,
            clim=(0,2*np.pi),
            cmap=plt.get_cmap("twilight"),
            interpolation="none",
        )
        plt.title(r"Phase Correction $\phi$")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 4, 3)
        if phase_detail:
            plt.imshow(
                kx,
                clim=(-kscale, kscale),
                cmap=plt.get_cmap("twilight"),
                interpolation="none",
            )
            plt.title(r"$k_x \propto \partial\phi/\partial x$")
        else:
            plt.imshow(power)
            plt.title("Measured Beam Power")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 4, 4)
        if phase_detail:
            plt.imshow(
                ky,
                clim=(-kscale, kscale),
                cmap=plt.get_cmap("twilight"),
                interpolation="none",
            )
            plt.title(r"$k_y \propto \partial\phi/\partial y$")
        else:
            plt.imshow(r2, clim=(0,1))
        # plt.contour(r2, [r2_threshold], c="r")
            plt.title("$R^2$")
        plt.xticks([])
        plt.yticks([])

        plt.show()


FourierSLM.fourier_calibration_build.__doc__ = SimulatedCamera.build_affine.__doc__


class _ZernikeCalibrationWidget:
    """
    Interactive Jupyter widget for manual Zernike coefficient adjustment.

    Created by :meth:`FourierSLM._wavefront_calibrate_zernike`.  Provides a
    live camera view with spot markers, per-term sliders, zoom controls,
    a hot-swappable Global / Per-Spot mode toggle, and click-to-select
    spot support (when the optional :mod:`ipyevents` package is installed).

    Notes
    -----
    Rendering is performed via :mod:`cv2` (resize + draw + PNG encode) for
    efficiency; the displayed image is rendered at fixed pixel dimensions so
    that mouse coordinates from :mod:`ipyevents` map directly into camera
    pixels without any CSS-stretch ambiguity. The widget additionally consults
    ``boundingRectWidth/Height`` from the click event as a defensive fallback
    in case the image is ever scaled by the host theme/layout.
    """

    # Natural display sizes for the embedded camera images. The 3:1 height
    # ratio between the full and zoomed views is enforced here.
    _FULL_DISPLAY_HEIGHT = 540
    _FULL_DISPLAY_MAX_WIDTH = 960
    _ZOOM_DISPLAY_SIZE = _FULL_DISPLAY_HEIGHT // 3  # 3:1 height ratio (180px)
    # Both panels share an explicit pixel height so the right-panel flex
    # spacer has a definite parent height to grow into; without this,
    # ``flex-grow`` on the spacer was a no-op because the parent VBox
    # collapsed to ``height: auto`` (= max-content). The +40 buffer covers
    # the zoom slider (~32px) plus a few pixels of margin.
    _PANEL_HEIGHT = _FULL_DISPLAY_HEIGHT + 40

    def __init__(
        self,
        fs,
        calibration_points,
        zernike_indices,
        calibration_points_ij,
        spot_integration_width_ij,
        hologram,
        perturbation_bounds,
        global_correction,
        optimize_focus,
    ):
        from ipywidgets import (
            Box,
            Dropdown,
            FloatSlider,
            FloatText,
            BoundedIntText,
            Button,
            Image,
            HTML,
            Output,
            HBox,
            VBox,
            Layout,
            ToggleButtons,
        )
        from IPython.display import display

        # ---- store references ------------------------------------------------
        self.fs = fs
        self.calibration_points = calibration_points
        self.initial_points = calibration_points.copy()
        self.zernike_indices = zernike_indices
        self.calibration_points_ij = calibration_points_ij
        self.spot_integration_width_ij = spot_integration_width_ij
        self.hologram = hologram
        self.perturbation_bounds = perturbation_bounds
        # ``global_correction`` is now the *initial* mode for the toggle.
        self.global_correction = bool(global_correction)
        self.optimize_focus = optimize_focus

        self._syncing = False
        self._done = False
        self._last_img = None
        self._full_crop_offset = (0, 0)
        self._full_display_scale = 1.0
        self._full_natural_size = (0, 0)

        N = calibration_points.shape[1]

        # Reusable layout helpers.
        #
        # ``_INPUT_DESC_WIDTH`` fixes the description label column for *all*
        # right-panel inputs so their input fields share a common left edge.
        # The slider row below the coefficient input pads its left side by
        # the same amount, putting the slider track flush with the inputs.
        _INPUT_DESC_WIDTH = "120px"
        _input_style = {"description_width": _INPUT_DESC_WIDTH}
        _flex_row = Layout(
            display="flex", flex_flow="row wrap",
            align_items="center", width="100%",
        )
        _flex_row_top = Layout(
            display="flex", flex_flow="row nowrap",
            align_items="stretch", width="100%",
            overflow_x="hidden",
        )
        _full_field = Layout(width="100%")

        # ---- determine adjustable terms --------------------------------------
        self._adjustable = []
        for j, idx in enumerate(zernike_indices):
            if idx in (0, 1, 2):
                continue
            if idx == 4 and not optimize_focus:
                continue
            self._adjustable.append((j, int(idx)))

        if len(self._adjustable) == 0:
            raise ValueError("No adjustable Zernike terms for the given settings.")

        self._current_j, self._current_idx = self._adjustable[0]
        self._current_spot = 0

        # ---- build term dropdown ---------------------------------------------
        term_options = []
        for j, idx in self._adjustable:
            name = ZERNIKE_NAMES[idx] if idx < len(ZERNIKE_NAMES) else f"index {idx}"
            term_options.append((f"Z_{idx}: {name}", (j, idx)))

        # ---- determine initial slider value & range --------------------------
        init_val = self._read_value()
        lo, hi = perturbation_bounds
        lo = min(lo, init_val - 0.1)
        hi = max(hi, init_val + 0.1)
        step = (hi - lo) / 2000.0

        # ---- top-bar widgets -------------------------------------------------
        self._reset_btn = Button(
            description="Reset",
            button_style="danger",
            layout=Layout(width="80px", margin="0 4px 0 0"),
        )
        self._save_btn = Button(
            description="Save",
            button_style="success",
            layout=Layout(width="80px", margin="0 4px 0 0"),
        )
        self._status = HTML(
            value=self._format_status("Ready", "green"),
            layout=Layout(width="auto"),
        )

        # ---- image widgets ---------------------------------------------------
        # Fixed pixel sizes ensure that `offsetX/Y` from ipyevents click events
        # map 1:1 to PNG natural pixels (no CSS stretching). On narrow viewports
        # the layout simply scrolls, but the click mapping remains correct.
        self._full_image = Image(
            format="png",
            layout=Layout(
                width=f"{self._FULL_DISPLAY_MAX_WIDTH}px",
                height=f"{self._FULL_DISPLAY_HEIGHT}px",
                max_width="100%",
                object_fit="contain",
            ),
        )
        self._zoom_image = Image(
            format="png",
            layout=Layout(
                width=f"{self._ZOOM_DISPLAY_SIZE}px",
                height=f"{self._ZOOM_DISPLAY_SIZE}px",
            ),
        )

        # ---- right-panel control widgets ------------------------------------
        max_half = spot_integration_width_ij // 2
        self._spot_radius = BoundedIntText(
            value=max_half,
            min=1,
            max=max_half,
            description="Radius [pixel]:",
            style=_input_style,
            layout=_full_field,
        )

        self._spot_input = BoundedIntText(
            value=0,
            min=0,
            max=max(N - 1, 0),
            description="Spot:",
            style=_input_style,
            layout=_full_field,
        )

        # Mode toggle. ipywidgets collapses the description column when
        # ``description=""`` is set (regardless of ``description_width``),
        # so any attempt to align the buttons with the input fields'
        # left edge via an empty description is futile. We accept the
        # natural left-flush layout for the toggle.
        #
        # ``max_width: 100%`` prevents the natural button row from poking
        # past the panel's content area on themes whose button padding is
        # generous (which would otherwise trigger a horizontal scrollbar).
        self._mode_toggle = ToggleButtons(
            options=[("Global Mode", True), ("Local Mode", False)],
            value=self.global_correction,
            style={"button_width": "auto"},
            layout=Layout(width="auto", max_width="100%"),
        )

        self._term_dropdown = Dropdown(
            options=term_options,
            description="Term:",
            style=_input_style,
            layout=_full_field,
        )

        self._coeff_text = FloatText(
            value=init_val,
            description="Coefficient [rad]:",
            step=0.01,
            style=_input_style,
            layout=_full_field,
        )

        # Slider with no inline readout. The numeric bounds are shown as
        # explicit (compact) labels on either side; the editable number is in
        # ``_coeff_text`` above.
        self._coeff_slider = FloatSlider(
            value=init_val,
            min=lo,
            max=hi,
            step=step,
            readout=False,
            continuous_update=False,
            layout=Layout(
                flex="1 1 0%", min_width="60px", margin="0 4px",
            ),
        )
        self._slider_min_label = HTML(
            value=self._format_bound(lo),
            layout=Layout(width="auto", flex="0 0 auto"),
        )
        self._slider_max_label = HTML(
            value=self._format_bound(hi),
            layout=Layout(width="auto", flex="0 0 auto"),
        )
        # Stash the shared description width so layout assembly below can pad
        # the slider row by the matching amount.
        self._input_desc_width = _INPUT_DESC_WIDTH

        # Full-image zoom slider (sits below the full image).
        # * ``max_width`` keeps it from stretching to the full left-panel width.
        # * ``margin: auto 0 0 0`` pins the slider to the *bottom* of the left
        #   panel so the visible content of the left and right panels share a
        #   common bottom edge (the right panel uses a flex spacer for the
        #   same purpose).
        self._full_zoom = FloatSlider(
            value=1.0, min=1.0, max=8.0, step=0.25,
            description="Zoom:",
            readout=True,
            readout_format=".1f",
            continuous_update=False,
            layout=Layout(
                width="auto", max_width="360px",
                margin="auto 0 0 0",
            ),
        )

        self._output = Output()

        # ---- wire callbacks --------------------------------------------------
        self._term_dropdown.observe(self._on_term_change, names="value")
        self._coeff_slider.observe(self._on_slider_change, names="value")
        self._coeff_text.observe(self._on_text_change, names="value")
        self._spot_input.observe(self._on_spot_change, names="value")
        self._mode_toggle.observe(self._on_mode_change, names="value")
        self._full_zoom.observe(self._on_full_zoom_change, names="value")
        self._spot_radius.observe(self._on_spot_radius_change, names="value")
        self._reset_btn.on_click(self._on_reset)
        self._save_btn.on_click(self._on_save)

        # ---- optional click-to-select via ipyevents --------------------------
        self._click_event = None
        self._click_supported = False
        try:
            from ipyevents import Event
            self._click_event = Event(
                source=self._full_image,
                watched_events=["click"],
                prevent_default_action=True,
            )
            self._click_event.on_dom_event(self._on_image_click)
            self._click_supported = True
        except ImportError:
            msg = (
                "Tip: install 'ipyevents' to enable click-to-select "
                "functionality on the camera image."
            )
            warnings.warn(msg, stacklevel=2)
            _logger.info(msg)

        # ---- assemble layout -------------------------------------------------
        # Top bar: [Reset][Save]   ............   [Status]
        top_left = HBox(
            [self._reset_btn, self._save_btn],
            layout=Layout(width="auto"),
        )
        top_right = HBox(
            [self._status],
            layout=Layout(
                flex="1 1 auto", justify_content="flex-end",
                align_items="center",
            ),
        )
        top_bar = HBox(
            [top_left, top_right],
            layout=Layout(
                width="100%", justify_content="space-between",
                align_items="center", margin="0 0 8px 0",
            ),
        )

        # Coefficient slider row.
        #
        # Originally we added a left padding of ``_input_desc_width`` to align
        # the slider with the input fields above. That intermittently
        # overflowed the right panel and triggered a horizontal scrollbar
        # (``width: 100% + padding`` is content-box safe only if the inherited
        # ``box-sizing: border-box`` is honoured -- which depends on the
        # ipywidgets/theme version).
        #
        # Using an explicit invisible spacer widget instead makes the row's
        # geometry unambiguous: ``width: 100%`` literally means "fill parent
        # content area", and the spacer simply consumes the first
        # ``_input_desc_width`` pixels. No box-sizing assumptions required.
        slider_row_spacer = HTML(
            value="",
            layout=Layout(
                width=self._input_desc_width,
                flex="0 0 auto",
            ),
        )
        coeff_row = HBox(
            [
                slider_row_spacer,
                self._slider_min_label,
                self._coeff_slider,
                self._slider_max_label,
            ],
            layout=Layout(width="100%", align_items="center"),
        )

        # Vertical spacer used to push the calibration-control group to the
        # bottom of the right panel. ``flex: 1`` (== ``1 1 0%``) gives it a
        # zero basis with grow=1 so it absorbs all leftover vertical space.
        # The flex-grow only takes effect when the parent VBox has a definite
        # height -- that is provided by ``height=_PANEL_HEIGHT`` on the panel
        # below.
        right_spacer = Box(
            layout=Layout(flex="1 1 0%", min_height="8px"),
        )

        # Right panel: two logical groups separated by ``right_spacer``.
        #   Top group (Spot-related):
        #     - zoom_image
        #     - Radius
        #     - Spot
        #   Bottom group (calibration-related):
        #     - Mode toggle (Global / Local)
        #     - Term dropdown
        #     - Coefficient text
        #     - Coefficient slider row
        #
        # Layout notes:
        # * ``height=_PANEL_HEIGHT`` gives the VBox a definite main-axis size
        #   so the flex spacer can actually grow (``height: 100%`` was
        #   unreliable because the parent HBox is auto-height).
        # * ``box_sizing="border-box"`` so ``min_width`` and any width assigned
        #   by the parent flex include the 12px padding -- otherwise some
        #   themes treat this as content-box, the padding adds 24px to the
        #   visual size, and the panel pokes past its allotted space.
        # * Symmetric ``padding="0 12px 0 12px"`` gives the input widgets a
        #   12px buffer on each side so their borders never reach the panel
        #   edge.
        # * ``overflow_x="hidden"`` is required for PyCharm's notebook viewer:
        #   PyCharm applies ``overflow-x: auto`` to the underlying
        #   ``.widget-vbox`` element, which surfaces a horizontal scrollbar
        #   whenever a child has even a sub-pixel overflow (a common
        #   side-effect of the way themes round flex-distributed widths).
        #   We've eliminated *real* overflow above; this just suppresses the
        #   stray scrollbar PyCharm would otherwise show. ``overflow_y`` is
        #   left at the default so any vertical overflow (e.g. unusually
        #   tall theme buttons) is still visible.
        right_panel = VBox(
            [
                # --- top (Spot) group ---
                self._zoom_image,
                self._spot_radius,
                self._spot_input,
                # --- elastic spacer ---
                right_spacer,
                # --- bottom (calibration) group ---
                self._mode_toggle,
                self._term_dropdown,
                self._coeff_text,
                coeff_row,
            ],
            layout=Layout(
                flex="1 1 0%",
                min_width="320px",
                height=f"{self._PANEL_HEIGHT}px",
                padding="0 12px 0 12px",
                box_sizing="border-box",
                overflow_x="hidden",
            ),
        )

        # Left panel: full image + zoom slider. The same explicit height as
        # the right panel ensures both panels' bottom edges are aligned;
        # ``margin-top: auto`` on the zoom slider (set above) then pushes the
        # slider to that shared bottom edge.
        left_panel = VBox(
            [self._full_image, self._full_zoom],
            layout=Layout(
                flex="3 1 0%",
                min_width="0",
                height=f"{self._PANEL_HEIGHT}px",
            ),
        )

        body = HBox([left_panel, right_panel], layout=_flex_row_top)

        # Outer width caps the whole widget at ~2/3 of the cell width.
        # ``max_width`` keeps it well-behaved on narrow notebook themes.
        self._layout = VBox(
            [top_bar, body, self._output],
            layout=Layout(
                width="66%", max_width="100%",
                overflow="hidden",
            ),
        )
        display(self._layout)

        # ---- initial render --------------------------------------------------
        self._apply_and_render()
        if self._click_supported:
            with self._output:
                print("Tip: click on the camera image to select a spot.")

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_status(message, color):
        """Return formatted HTML for the status indicator."""
        return (
            f"<span style='color:{color}; font-weight:bold;'>"
            f"&#x25CF; {message}</span>"
        )

    @staticmethod
    def _format_bound(value):
        """Return formatted HTML for a slider min/max label.

        Uses a compact two-decimal representation so the labels remain narrow
        and do not crowd the slider track inside the right panel.
        """
        return (
            "<span style='font-family:monospace; font-size:0.85em; "
            "white-space:nowrap;'>"
            f"{value:+.2f}</span>"
        )

    def _set_status(self, message, color="green"):
        """Update the status indicator (also forces an immediate kernel flush)."""
        self._status.value = self._format_status(message, color)

    def _handle_error(self, context, exc):
        """Centralised error handler: log, surface to status, and dump traceback."""
        _logger.exception("[%s] %s", context, exc)
        self._set_status(f"Error: {context}", color="red")
        with self._output:
            print(f"[{context}] {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Value helpers
    # ------------------------------------------------------------------

    def _read_value(self):
        """Return the displayed coefficient for the active term/spot."""
        j = self._current_j
        if self.global_correction:
            return float(np.mean(self.calibration_points[j, :]))
        return float(self.calibration_points[j, self._current_spot])

    # ------------------------------------------------------------------
    # SLM / camera pipeline
    # ------------------------------------------------------------------

    def _tick(self):
        """Generate the SLM phase pattern from current calibration_points."""
        if self.hologram is None:
            return zernike_sum(
                self.fs.slm,
                self.zernike_indices,
                self.calibration_points,
                use_mask=False,
            )
        self.hologram.spot_zernike = self.calibration_points
        self.hologram.optimize("GS", maxiter=3, verbose=0)
        return self.hologram.get_phase()

    # ------------------------------------------------------------------
    # Rendering helpers (cv2-based)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_uint8(img):
        """Normalise an arbitrary 2-D array to uint8 in [0, 255]."""
        arr = np.clip(np.asarray(img, dtype=np.float32), 0.0, None)
        mx = float(arr.max()) if arr.size else 0.0
        if mx > 0.0:
            arr = (arr * (255.0 / mx))
        return arr.astype(np.uint8, copy=False)

    @staticmethod
    def _encode_png(arr):
        """Encode a uint8 ndarray (gray or BGR) into PNG bytes via cv2."""
        success, buf = cv2.imencode(".png", arr)
        if not success:
            raise RuntimeError("cv2.imencode failed to produce PNG bytes.")
        return buf.tobytes()

    def _compute_full_target_size(self, src_w, src_h):
        """Return the (w, h) target size for the rendered full image PNG."""
        if src_w <= 0 or src_h <= 0:
            return (1, 1)
        target_h = self._FULL_DISPLAY_HEIGHT
        target_w = int(round(target_h * (src_w / src_h)))
        if target_w > self._FULL_DISPLAY_MAX_WIDTH:
            target_w = self._FULL_DISPLAY_MAX_WIDTH
            target_h = int(round(target_w * (src_h / src_w)))
        return (max(1, target_w), max(1, target_h))

    def _render_full(self, img):
        """
        Render the full camera image with spot markers and zoom crop.

        Exceptions are *not* swallowed here so that callers can decide whether
        to keep showing an "Error" status or fall back to a recovery path.
        """
        if img is None:
            return
        h, w = img.shape[:2]
        zoom_level = float(self._full_zoom.value)
        selected = self._current_spot

        if zoom_level > 1.0:
            ci = float(self.calibration_points_ij[0, selected])
            cj = float(self.calibration_points_ij[1, selected])
            crop_w = max(int(w / zoom_level), 32)
            crop_h = max(int(h / zoom_level), 32)
            x0 = int(np.clip(ci - crop_w / 2, 0, max(0, w - crop_w)))
            y0 = int(np.clip(cj - crop_h / 2, 0, max(0, h - crop_h)))
            cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]
        else:
            cropped = img
            x0, y0 = 0, 0

        src_h, src_w = cropped.shape[:2]
        target_w, target_h = self._compute_full_target_size(src_w, src_h)
        scale = target_h / max(src_h, 1)

        self._full_crop_offset = (x0, y0)
        self._full_display_scale = scale
        self._full_natural_size = (target_w, target_h)

        gray = self._normalise_uint8(cropped)
        # INTER_AREA for clean downscale, INTER_NEAREST when upscaling so the
        # raw camera pixels remain sharp/discernible.
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
        resized = cv2.resize(gray, (target_w, target_h), interpolation=interp)
        bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        r = max(4, int(round(6 * scale)))
        n_spots = self.calibration_points_ij.shape[1]
        for k in range(n_spots):
            sx = int(round((float(self.calibration_points_ij[0, k]) - x0) * scale))
            sy = int(round((float(self.calibration_points_ij[1, k]) - y0) * scale))
            if not (0 <= sx < target_w and 0 <= sy < target_h):
                continue
            if k == selected:
                color = (50, 50, 255)  # red (BGR)
                thickness = 2
            else:
                color = (50, 220, 50)  # green (BGR)
                thickness = 1
            cv2.circle(bgr, (sx, sy), r, color, thickness=thickness)

        self._full_image.value = self._encode_png(bgr)
        # Keep widget size locked to PNG natural size so click coords are
        # unambiguous (no CSS stretching).
        self._full_image.layout.width = f"{target_w}px"
        self._full_image.layout.height = f"{target_h}px"

    def _render_zoom(self, img=None):
        """
        Refresh only the zoomed spot view.

        Exceptions propagate to the caller so that error status messages are
        not stomped on by subsequent unconditional updates.
        """
        if img is None:
            img = self._last_img
        if img is None:
            return
        spot = self._current_spot
        radius = int(self._spot_radius.value)
        size = 2 * radius + 1
        spot_imgs = analysis.take(
            img,
            self.calibration_points_ij[:, spot:spot + 1],
            size,
            clip=True,
        )
        crop = np.asarray(spot_imgs[0])
        gray = self._normalise_uint8(crop)
        display_px = self._ZOOM_DISPLAY_SIZE
        resized = cv2.resize(
            gray, (display_px, display_px),
            interpolation=cv2.INTER_NEAREST,
        )
        self._zoom_image.value = self._encode_png(resized)

    def _apply_and_render(self):
        """Apply *calibration_points* to the SLM and refresh both views."""
        if self._done:
            return
        try:
            self._set_status("Computing phase...", color="orange")
            pattern = self._tick()

            self._set_status("Updating SLM...", color="orange")
            self.fs.slm.set_phase(pattern, settle=True, phase_correct=False)

            self._set_status("Capturing...", color="orange")
            self.fs.cam.flush()
            img = self.fs.cam.get_image()

            self._set_status("Rendering...", color="orange")
            self._last_img = img
            self._render_full(img)
            self._render_zoom(img)

            self._set_status("Ready", color="green")
        except Exception as exc:
            self._handle_error("apply_and_render", exc)

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def _on_term_change(self, change):
        try:
            self._current_j, self._current_idx = change["new"]
            self._load_slider_for_current()
        except Exception as exc:
            self._handle_error("term_change", exc)

    def _on_spot_change(self, change):
        try:
            self._current_spot = int(change["new"])
            # Always reload slider for the new selection. In Per-Spot mode this
            # exposes that spot's stored value; in Global mode the displayed
            # value (mean across spots) is unchanged but we still refresh
            # markers/zoom view.
            self._load_slider_for_current()
            self._render_full(self._last_img)
            self._render_zoom()
        except Exception as exc:
            self._handle_error("spot_change", exc)

    def _on_mode_change(self, change):
        """
        Hot-swap between Global and Per-Spot modes.

        The underlying ``calibration_points`` array is *not* destructively
        rewritten: per-spot values are preserved when toggling, and only the
        slider's "what-you-see / what-the-slider-applies-to" semantics change.
        Subsequent slider/text edits will:

        * Global mode  -> overwrite *all* spots with the new value.
        * Per-Spot mode -> overwrite only the currently selected spot.
        """
        new_mode = bool(change["new"])
        if new_mode == self.global_correction:
            return
        self.global_correction = new_mode
        # Sync slider to the value that this mode "shows" for the current term.
        self._load_slider_for_current()
        self._set_status(
            f"Switched to {'Global' if new_mode else 'Local'} mode",
            color="blue",
        )

    def _load_slider_for_current(self):
        """Sync the slider/text to the stored value for the active term/spot."""
        try:
            val = self._read_value()
            lo, hi = self.perturbation_bounds
            lo = min(lo, val - 0.1)
            hi = max(hi, val + 0.1)
            self._syncing = True
            # Order matters: set min/max bracket first to avoid a transient
            # "value out of bounds" exception when the new value lies outside
            # the previous range.
            if lo <= self._coeff_slider.min:
                self._coeff_slider.min = lo
                self._coeff_slider.max = hi
            else:
                self._coeff_slider.max = hi
                self._coeff_slider.min = lo
            self._coeff_slider.value = val
            self._coeff_text.value = val
            self._slider_min_label.value = self._format_bound(lo)
            self._slider_max_label.value = self._format_bound(hi)
            self._syncing = False
        except Exception as exc:
            self._syncing = False
            self._handle_error("load_slider", exc)

    def _on_slider_change(self, change):
        if self._syncing:
            return
        try:
            self._syncing = True
            self._coeff_text.value = change["new"]
            self._syncing = False
            self._set_coefficient(change["new"])
        except Exception as exc:
            self._syncing = False
            self._handle_error("slider_change", exc)

    def _on_text_change(self, change):
        if self._syncing:
            return
        try:
            val = float(change["new"])
            self._syncing = True
            if val < self._coeff_slider.min:
                self._coeff_slider.min = val
                self._slider_min_label.value = self._format_bound(val)
            if val > self._coeff_slider.max:
                self._coeff_slider.max = val
                self._slider_max_label.value = self._format_bound(val)
            self._coeff_slider.value = val
            self._syncing = False
            self._set_coefficient(val)
        except Exception as exc:
            self._syncing = False
            self._handle_error("text_change", exc)

    def _set_coefficient(self, value):
        """Write the coefficient and trigger hardware update."""
        j = self._current_j
        value = float(value)
        if self.global_correction:
            self.calibration_points[j, :] = value
        else:
            self.calibration_points[j, self._current_spot] = value
        self._apply_and_render()

    def _on_full_zoom_change(self, _change):
        if self._last_img is None:
            return
        try:
            self._render_full(self._last_img)
        except Exception as exc:
            self._handle_error("full_zoom_change", exc)

    def _on_spot_radius_change(self, _change):
        try:
            self._render_zoom()
        except Exception as exc:
            self._handle_error("spot_radius_change", exc)

    def _on_image_click(self, event):
        """
        Map a click on the rendered full-image PNG back to a camera pixel and
        select the nearest calibration spot.

        The click handler is robust to two distinct coordinate systems:

        1. The PNG natural pixel size we used at render time (stored in
           ``_full_natural_size`` / ``_full_display_scale``).
        2. The CSS-rendered size of the ``<img>`` element if the host theme or
           container ever stretches it (e.g. ``width: 100%`` parent). When the
           ipyevents payload includes ``boundingRectWidth/Height`` we use those
           to convert CSS pixels back to natural PNG pixels first.
        """
        try:
            offset_x = event.get("offsetX", event.get("relativeX"))
            offset_y = event.get("offsetY", event.get("relativeY"))
            if offset_x is None or offset_y is None:
                return

            nat_w, nat_h = self._full_natural_size
            if nat_w <= 0 or nat_h <= 0:
                return

            rect_w = event.get("boundingRectWidth")
            rect_h = event.get("boundingRectHeight")

            try:
                rw = float(rect_w) if rect_w is not None else 0.0
                rh = float(rect_h) if rect_h is not None else 0.0
            except (TypeError, ValueError):
                rw = rh = 0.0

            if rw > 0.0 and rh > 0.0:
                # CSS-rendered size differs from natural: rescale.
                nat_x = float(offset_x) * (nat_w / rw)
                nat_y = float(offset_y) * (nat_h / rh)
            else:
                # Image not stretched (or info unavailable): use directly.
                nat_x = float(offset_x)
                nat_y = float(offset_y)

            ox, oy = self._full_crop_offset
            scale = self._full_display_scale or 1.0
            cam_x = nat_x / scale + ox
            cam_y = nat_y / scale + oy

            dists = (
                (self.calibration_points_ij[0] - cam_x) ** 2
                + (self.calibration_points_ij[1] - cam_y) ** 2
            )
            nearest = int(np.argmin(dists))
            # Triggers _on_spot_change which refreshes the views.
            self._spot_input.value = nearest
        except Exception as exc:
            self._handle_error("image_click", exc)

    # ------------------------------------------------------------------
    # Reset / Save
    # ------------------------------------------------------------------

    def _on_reset(self, _=None):
        """Set all adjustable Zernike coefficients to zero."""
        try:
            for j, _idx in self._adjustable:
                self.calibration_points[j, :] = 0.0
            self._load_slider_for_current()
            self._apply_and_render()
            with self._output:
                self._output.clear_output(wait=True)
                print("All adjustable Zernike coefficients reset to zero.")
        except Exception as exc:
            self._handle_error("reset", exc)

    def _on_save(self, _=None):
        """Persist current coefficients and disable the widget."""
        try:
            self._set_status("Saving...", color="orange")
            pattern = self._tick()
            self.fs.slm.set_phase(pattern, settle=True, phase_correct=False)

            self.fs.calibrations["wavefront_zernike"] = {
                "initial_points": self.initial_points,
                "zernike_indices": self.zernike_indices,
                "corrected_spots": self.calibration_points.copy(),
                "calibration_points_ij": self.calibration_points_ij,
                "spot_integration_width_ij": self.spot_integration_width_ij,
                "weights": (
                    self.hologram.get_weights() if self.hologram is not None else None
                ),
            }
            self.fs.calibrations["wavefront_zernike"].update(
                self.fs._get_calibration_metadata()
            )

            self._done = True
            self._set_status("Saved", color="blue")

            for w in (
                self._coeff_slider,
                self._coeff_text,
                self._term_dropdown,
                self._spot_input,
                self._spot_radius,
                self._full_zoom,
                self._reset_btn,
                self._save_btn,
                self._mode_toggle,
            ):
                w.disabled = True

            with self._output:
                self._output.clear_output(wait=True)
                print(
                    "Calibration saved to fs.calibrations['wavefront_zernike']. "
                    "You may continue with subsequent cells."
                )
        except Exception as exc:
            self._handle_error("save", exc)
