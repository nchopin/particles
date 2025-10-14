"""Where datasets live.

This module gives access to several useful datasets. A dataset is represented
as a class that inherits from base class `Dataset`. When instantiating such a
class, you get an object with attributes:

* ``raw_data``: data in the original file;
* ``data`` : data obtained after a pre-processing step was applied to the raw
  data.

The pre-processing step is performed by method `preprocess` of the class. For
instance, for a regression dataset, the pre-processing steps normalises the
predictors and adds an intercept. The pre-processing step of base class
`Dataset` does nothing (``raw_data`` and ``data`` point to the same object).

Here a quick example::

    from particles import datasets as dts

    dataset = dts.Pima()
    help(dataset)  # basic info on dataset
    help(dataset.preprocess)  # info on how data was pre-processed
    data = dataset.data  # typically a numpy array

And here is a table of the available datasets; see the documentation of each
sub-class for more details on the preprocessing step.

================= ===================== =======================
Dataset           parent class          typical use/model
================= ===================== =======================
`Boston`          `RegressionDataset`   regression
`Concrete`        `RegressionDataset`   regression
`Eeg`             `BinaryRegDataset`    binary regression
`GBP_vs_USD_9798` `LogReturnsDataset`   stochastic volatility
`Liver`           `BinaryRegDataset`    binary regression
`Nutria`          `Dataset`             population ecology
`Pima`            `BinaryRegDataset`    binary regression
`Sonar`           `BinaryRegDataset`    binary regression
`Neuro`           `Dataset`             neuroscience ssm
================= ===================== =======================

See also utility function `prepare_predictors`, which prepares (rescales,
adds an intercept) predictors/features for a regression or classification task.

"""

import collections
from pathlib import Path

import numpy as np


def get_path(file_name: str) -> Path:
    return Path(__file__).parent / "datasets" / file_name


class Dataset:
    """Base class for datasets.

    The pre-processing step does nothing: attributes ``raw_data`` and ``data``
    point to the same object.
    """

    load_opts = {"delimiter": ","}

    def preprocess(self, raw_data, **kwargs):
        return raw_data

    def __init__(self, **kwargs):
        self.raw_data = np.loadtxt(
            get_path(self.file_name), **self.load_opts, encoding="utf-8"
        )
        self.data = self.preprocess(self.raw_data, **kwargs)


class Nutria(Dataset):
    """Nutria dataset.

    Time series of female nutria abundance in East Anglia at monthly intervals,
    obtained by retrospective census for a feral population. See Peters et al
    (2010) for various state-space models that may be applied to this dataset,
    such as `state_space_models.ThetaLogistic`.

    Source
    ------
    Data set 9833 in the Global Population Database [NERC Centre for
    Population Biology, Imperial College, 1999].

    Reference
    ---------
    * Peters et al. (2010). Ecological  non-linear  state  space  model
      selection  via  adaptive particle Markov chain Monte Carlo, arXiv:1005.2238

    """

    file_name = "nutria.txt"


class Neuro(Dataset):
    r"""Neuroscience experiment data from Temereanca et al (2008).

    Time series of number of activated neurons over 50 experiments. A potential
    state-space model for this dataset is:

    .. math ::
        Y_t | X_t = x     \sim Bin(50, logit^{-1}(x))
        X_t = \rho * X_{t-1} + \sigma * U_t,\quad U_t \sim N(0, 1)


    Reference
    ---------

    * Temereanca et al (2008).  Rapid changes in thalamic firing synchrony during
      repetitive whisker stimulation, J. of Neuroscience.

    """

    file_name = "thaldata.csv"


#######################################
# Log-return datasets


class LogReturnsDataset(Dataset):
    """Log returns dataset.

    For data on e.g. daily prices of a stock or some financial index.

    The pre-processing step simply consists in differentiating each row,
    taking the log, and multiplying by 100 (to get per-cent points).
    """

    def preprocess(self, raw_data):
        """compute log-returns."""
        return 100.0 * np.diff(np.log(raw_data), axis=0)


class GBP_vs_USD_9798(LogReturnsDataset):
    """GBP vs USD daily rates in 1997-98.

    A time-series of 751 currency rates.

    Source: I forgot, sorry!
    """

    file_name = "GBP_vs_USD_9798.txt"
    load_opts = {"skiprows": 2, "usecols": (3,), "comments": "(C)"}


#######################################
# Regression datasets


def prepare_predictors(predictors, add_intercept=True, scale=0.5):
    """Rescale predictors and (optionally) add an intercept.

    Standard pre-processing step in any regression/classification task.

    Parameters
    ----------
    predictors: numpy array
            a (n,) or (n,p) array containing the p predictors
    scale: float (default=0.5)
        rescaled predictors have mean 0 and std dev *scale*
    add_intercept: bool (default=True)
        whether to add a row filled with 1.

    Returns
    -------
    out: numpy array
        the rescaled predictors
    """
    preds = np.atleast_2d(predictors)  # in case predictors is (n,)
    rescaled_preds = scale * (preds - np.mean(preds, axis=0)) / np.std(preds, axis=0)
    if add_intercept:
        n, p = preds.shape
        out = np.empty((n, p + 1))
        out[:, 0] = 1.0  # intercept
        out[:, 1:] = rescaled_preds
    else:
        out = rescaled_preds
    return out


class RegressionDataset(Dataset):
    """Regression dataset.

    A regression dataset contains p predictors, and one scalar response.
    The pre-processing step consists of:

    1. rescaling the predictors (mean=0, std dev=0.5)
    2. adding an intercept (constant predictor)

    The ``data`` attribute is tuple (preds, response), where first (resp. second)
    element is a 2D (resp. 1D) numpy array.
    """

    def preprocess(self, raw_data):
        response = raw_data[:, -1]
        preds = prepare_predictors(raw_data[:, :-1])
        return preds, response


class Boston(RegressionDataset):
    """Boston house-price data of Harrison et al (1978).

    A dataset of 506 observations on 13 predictors.

    Reference
    ---------
    `UCI archive <https://archive.ics.uci.edu/ml/machine-learning-databases/housing/>`__

    """

    predictor_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSAT",
    ]
    response_name = ["MEDV"]
    file_name = "boston_house_prices.csv"
    load_opts = {"delimiter": ",", "skiprows": 2}


class Concrete(RegressionDataset):
    """Concrete compressive strength data of Yeh (1998).

    A dataset with 1030 observations and 9 predictors.

    Reference
    ---------
    `UCI archive <https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/>`__

    """

    predictor_names = [
        "cement",
        "blast",
        "fly ash",
        "water",
        "superplasticizer",
        "coarse aggregate",
        "fine aggregate",
        "age",
    ]
    response_name = ["strength"]
    file_name = "concrete.csv"
    load_opts = {"delimiter": ",", "skiprows": 1}


class BinaryRegDataset(Dataset):
    """Binary regression (classification) dataset.

    Such a dataset contains p predictors, and one binary response.
    The pre-processing step consists of:

        1. rescaling the predictors (mean=0, std dev=0.5)
        2. adding an intercept (constant predictor)
        3. optionally, applying the "flip sign" trick.

    Point 3 refers to the fact that the likelihood of a binary regression
    models depends only on the vectors:

    .. math:: y_i * x_i

    where :math:`y_i=-1/1` is the response, and :math:`x_i` is the vector of p
    predictors.

    Hence, by default, the pre-processing steps returns a single array,
    obtained by flipping the sign of each row such that the response is -1.
    If you'd rather get the predictors and the (not flipped) response as two
    separate arrays, set option ``return_y`` to ``True``, when instantiating
    the class.

    """

    def preprocess(self, raw_data, return_y=False):
        response = 2 * raw_data[:, -1] - 1  # 0/1 -> -1/1
        preds = prepare_predictors(raw_data[:, :-1])
        if return_y:
            return preds, response
        else:
            return preds * response[:, np.newaxis]


class Pima(BinaryRegDataset):
    r"""Pima Indians Diabetes.

    A dataset with 768 observations and 8 predictors.

    Response: diabetes test.

    Predictors:
        * pregnant: Number of times pregnant
        * glucose: Plasma glucose concentration (glucose tolerance test)
        * pressure: Diastolic blood pressure (mm Hg)
        * triceps: Triceps skin fold thickness (mm)
        * insulin: 2-Hour serum insulin (mu U/ml)
        * mass: Body mass index (weight in kg/(height in m)\^2)
        * pedigree: Diabetes pedigree function
        * age: Age (years)

    `Source: <https://cran.r-project.org/web/packages/mlbench/index.html>`__

    """

    file_name = "pima-indians-diabetes.data"


class Liver(BinaryRegDataset):
    """Indian liver patient dataset (ILPD).

    A dataset with 579 observations and 10 predictors.

    Response: liver patient or not.

    Predictors:
        * Age of the patient
        * Gender of the patient (0 / 1 = Male / Female)
        * TB Total Bilirubin
        * DB Direct Bilirubin
        * Alkphos Alkaline Phosphotase
        * Sgpt Alamine Aminotransferase
        * Sgot Aspartate Aminotransferase
        * TP Total Protiens
        * ALB Albumin
        * A/G Ratio Albumin and Globulin Ratio

    Reference
    ---------
    `UCI: <https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29#>`__
    """

    file_name = "indian_liver_patient.csv"


class Eeg(BinaryRegDataset):
    """EEG dataset from UCI repository.

    A dataset with 122 observations and 64 predictors.

    * Response: alcoholic vs control
    * predictors: EEG measurements

    Reference
    ---------
    `UCI: <https://archive.ics.uci.edu/ml/datasets/eeg+database>`__
    """

    file_name = "eeg_eye_state.data"
    load_opts = {"delimiter": ",", "skiprows": 19}


class Sonar(BinaryRegDataset):
    """Sonar dataset from UCI repository.

    A dataset with 110 observations and 60 predictors.

    * Response: rock vs mine
    * predictors: numbers in range [0, 1] representing the energy within a
      particular frequency band.

    `Link <https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)>`__

    """

    def convert_last_col(x):
        if x == "R":
            return 1.0
        elif x == "M":
            return 0.0
        else:
            return np.nan

    file_name = "sonar.all-data"
    load_opts = {"delimiter": ",", "converters": {60: convert_last_col}}
