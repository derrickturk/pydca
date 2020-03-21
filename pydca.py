import sys
import csv

import numpy as np
import matplotlib.pyplot as plt # type: ignore
from scipy.optimize import minimize # type: ignore

import dataclasses as dc

from typing import List, Optional, TextIO, Tuple

YEAR_DAYS: float = 365.25 # days
DOWNTIME_CUTOFF = 1.0 # daily rate

_GUESS_DI_NOM: float = 1.0 # nominal annual decline
_GUESS_B = 1.5

_FIT_BOUNDS: List[Tuple[Optional[float], Optional[float]]] = [
    (0.0, None), # initial rate
    (0.0, None), # nominal annual decline
    (0.0, 2.0),  # b
]

@dc.dataclass(frozen=True)
class ArpsDecline:
    qi: float # daily rate
    Di_nom: float # nominal annual decline
    b: float # unitless exponent

    def __post_init__(self):
        if self.qi < 0.0:
            raise ValueError('Negative qi')
        if self.Di_nom < 0.0:
            raise ValueError('Negative Di_nom')
        if self.b < 0.0 or self.b > 2.0:
            raise ValueError(f'Invalid b: {self.b}')

    # time (years)
    # returns (daily rate)
    def rate(self, time: np.ndarray) -> np.ndarray:
        if self.b == 0:
            return self.qi * np.exp(-self.Di_nom * time)
        elif self.b == 1.0:
            return self.qi / (1.0 + self.Di_nom * time)
        else:
            return (
              self.qi / (1.0 + self.b * self.Di_nom * time) ** (1.0 / self.b)
            )

    @staticmethod
    def clamped(qi: float, Di_nom: float, b: float) -> 'ArpsDecline':
        return ArpsDecline(
            max(qi, 0.0),
            max(Di_nom, 0.0),
            max(min(b, 2.0), 0.0),
        )

@dc.dataclass(frozen=True)
class WellProduction:
    well_name: str
    days_on: np.ndarray # time (days)
    oil: np.ndarray # daily rate

    def __post_init__(self):
        if len(self.days_on) != len(self.oil):
            raise ValueError('Different lengths for days on and oil rate')

    def best_fit(self) -> ArpsDecline:
        initial_guess = np.array([
            np.max(self.oil), # guess qi = peak rate
            _GUESS_DI_NOM,
            _GUESS_B,
        ])

        filtered = self.peak_forward().no_downtime()

        fit = minimize(
                lambda params: filtered._sse(ArpsDecline.clamped(*params)),
                initial_guess, method='L-BFGS-B', bounds=_FIT_BOUNDS)
        return ArpsDecline.clamped(*fit.x)

    # filter this data set to only peak-forward production
    def peak_forward(self) -> 'WellProduction':
        peak_idx = np.argmax(self.oil)
        return WellProduction(
                self.well_name, self.days_on[peak_idx:], self.oil[peak_idx:])

    # filter this data set to drop "downtime" (zero-production days)
    def no_downtime(self, cutoff: float = DOWNTIME_CUTOFF) -> 'WellProduction':
        keep_idx = self.oil > cutoff
        return WellProduction(
                self.well_name, self.days_on[keep_idx], self.oil[keep_idx])

    # sum of squared error for a given fit to this data
    def _sse(self, fit: ArpsDecline) -> float:
        time_years = self.days_on / YEAR_DAYS
        forecast = fit.rate(time_years)
        return np.sum((forecast - self.oil) ** 2)

def read_production_csv(csv_file: TextIO) -> List[WellProduction]:
    reader = csv.reader(csv_file)
    next(reader) # skip header row

    result = list()
    last_well_name: Optional[str] = None
    days_on: List[float] = list()
    oil: List[float] = list()

    for (w, d, o) in reader:
        if w != last_well_name:
            if last_well_name is not None:
                result.append(WellProduction(
                    last_well_name, np.array(days_on), np.array(oil)))
            days_on = list()
            oil = list()
            last_well_name = w

        days_on.append(float(d))
        oil.append(float(o))

    if last_well_name is not None:
            result.append(WellProduction(
                last_well_name, np.array(days_on), np.array(oil)))

    return result

def plot_examples():
    exp_decl = ArpsDecline(1000.0, 0.9, 0.0)
    hyp_decl = ArpsDecline(1000.0, 0.9, 1.5)
    hrm_decl = ArpsDecline(1000.0, 0.9, 1.0)

    time = np.array([m / 12.0 for m in range(0, 5 * 12)])
    exp_rate = exp_decl.rate(time)
    hyp_rate = hyp_decl.rate(time)
    hrm_rate = hrm_decl.rate(time)

    plt.semilogy(time, exp_rate)
    plt.semilogy(time, hyp_rate)
    plt.semilogy(time, hrm_rate)

    plt.savefig('examples.png')
    plt.close()

def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(f'Usage: {argv[0]} production-csv', file=sys.stderr)
        return 1

    plot_examples()

    with open(argv[1], 'r', newline='') as production_csv:
        data = read_production_csv(production_csv)

    for well in data:
        filtered = well.peak_forward().no_downtime()
        plt.semilogy(filtered.days_on, filtered.oil)
        best_fit = well.best_fit()
        plt.semilogy(well.days_on, best_fit.rate(well.days_on / YEAR_DAYS))
        plt.savefig(f'plot_{well.well_name}.png')
        plt.close()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
