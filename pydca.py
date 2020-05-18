#  Copyright 2020 terminus data science, LLC

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import sys
import csv
import calendar
from datetime import date, timedelta

import numpy as np
import matplotlib.pyplot as plt # type: ignore
from scipy.optimize import minimize # type: ignore

import dataclasses as dc

from typing import Iterable, Iterator, List, Optional, TextIO, Tuple

YEAR_DAYS: float = 365.25 # days
DOWNTIME_CUTOFF = 1.0 # daily rate

MIN_PTS_FIT: int = 3 # minimum # of points for fitting
PEAK_SHIFT_MAX: int = 6 # maximum # of months to peak for fitting

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
class DailyOil:
    api: str
    days_on: np.ndarray # time (days)
    oil: np.ndarray # daily rate
    prior_cum: Optional[float] # prior cumulative as of 1993/01 if available

    def __post_init__(self):
        if len(self.days_on) != len(self.oil):
            raise ValueError('Different lengths for days on and oil rate')

    def best_fit(self) -> ArpsDecline:
        initial_guess = np.array([
            np.max(self.oil), # guess qi = peak rate
            _GUESS_DI_NOM,
            _GUESS_B,
        ])

        fit = minimize(
                lambda params: self._sse(ArpsDecline.clamped(*params)),
                initial_guess, method='L-BFGS-B', bounds=_FIT_BOUNDS)
        return ArpsDecline.clamped(*fit.x)

    # filter this data set to only peak-forward production
    def peak_forward(self) -> Tuple[int, 'DailyOil']:
        if len(self.oil) == 0:
            return 0, self
        peak_idx = np.argmax(self.oil)
        return peak_idx, DailyOil(self.api,
                self.days_on[peak_idx:], self.oil[peak_idx:], self.prior_cum)

    # filter this data set to drop "downtime" (zero-production days)
    def no_downtime(self, cutoff: float = DOWNTIME_CUTOFF) -> 'DailyOil':
        keep_idx = self.oil > cutoff
        return DailyOil(self.api, self.days_on[keep_idx], self.oil[keep_idx],
                self.prior_cum)

    # sum of squared error for a given fit to this data
    def _sse(self, fit: ArpsDecline) -> float:
        time_years = self.days_on / YEAR_DAYS
        forecast = fit.rate(time_years)
        return np.sum((forecast - self.oil) ** 2)

@dc.dataclass(frozen=True)
class MonthlyRecord:
    api: str
    year: int
    month: int
    oil: Optional[float]
    gas: Optional[float]
    water: Optional[float]

def month_days(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]

def mid_month(year: int, month: int) -> date:
    return date(year, month, 1) + timedelta(days=month_days(year, month) / 2)

# precondition: monthly is sorted by API then date
def from_monthly(monthly: Iterable[MonthlyRecord]) -> Iterator[DailyOil]:
    last_api: Optional[str] = None
    first_prod: Optional[date] = None
    prior_cum: Optional[float] = None
    days_on: List[float] = list()
    oil: List[float] = list()

    for m in monthly:
        if m.api != last_api:
            if last_api is not None:
                yield DailyOil(last_api, np.array(days_on), np.array(oil),
                        prior_cum)
                days_on = list()
                oil = list()
            last_api = m.api
            first_prod = None

        # NM OCD reports cumulative prior to 1993-01-01 as 1992/12 monthly
        if m.year == 1992 and m.month == 12:
            prior_cum = m.oil
            continue

        if first_prod is None:
            first_prod = date(m.year, m.month, 1)

        if m.oil is not None: # skip months with missing oil data
            days_on.append((mid_month(m.year, m.month) - first_prod).days)
            oil.append(m.oil / month_days(m.year, m.month))

    if last_api is not None:
        yield DailyOil(last_api, np.array(days_on), np.array(oil), prior_cum)

def float_or_none(val: str) -> Optional[float]:
    if val == '':
        return None
    return float(val)

def read_production_file(prod_file: TextIO, header: bool = True, **csvkw
        ) -> Iterator[MonthlyRecord]:
    reader = csv.reader(prod_file, **csvkw)
    if header:
        next(reader) # skip header row

    for (api, yr, mo, o, g, w) in reader:
        yield MonthlyRecord(api, int(yr), int(mo),
                float_or_none(o), float_or_none(g), float_or_none(w))

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
        print(f'Usage: {argv[0]} production-file', file=sys.stderr)
        return 1

    plot_examples()

    with open(argv[1], 'r', newline='') as production_file:
        data = read_production_file(production_file, delimiter='\t')
        for well in from_monthly(data):
            if well.prior_cum is not None:
                print(f'{well.api}: production prior to 1993',
                        file=sys.stderr)
                continue

            shift, filtered = well.peak_forward()
            if shift > PEAK_SHIFT_MAX:
                print(f'{well.api}: peak occurs too late for fitting',
                        file=sys.stderr)
                continue

            filtered = filtered.no_downtime()
            if len(filtered.days_on) < MIN_PTS_FIT:
                print(f'{well.api}: not enough data', file=sys.stderr)
                continue

            plt.semilogy(filtered.days_on, filtered.oil)
            best_fit = filtered.best_fit()
            plt.semilogy(well.days_on, best_fit.rate(well.days_on / YEAR_DAYS))
            plt.savefig(f'plots/{well.api}.png')
            plt.close()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
