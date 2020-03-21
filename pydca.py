import sys
import csv

import numpy as np
import matplotlib.pyplot as plt # type: ignore
from scipy.optimize import minimize # type: ignore

import dataclasses as dc

from typing import List, Optional, TextIO

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
            raise ValueError('Invalid b')

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

@dc.dataclass(frozen=True)
class WellProduction:
    well_name: str
    days_on: np.ndarray # time (days)
    oil: np.ndarray # daily rate

    def __post_init__(self):
        if len(self.days_on) != len(self.oil):
            raise ValueError('Different lengths for days on and oil rate')

def best_fit(well_production):
    pass

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

    time = [m / 12.0 for m in range(0, 5 * 12)]
    exp_rate = [exp_decl.rate(t) for t in time]
    hyp_rate = [hyp_decl.rate(t) for t in time]
    hrm_rate = [hrm_decl.rate(t) for t in time]

    plt.semilogy(time, exp_rate)
    plt.semilogy(time, hyp_rate)
    plt.semilogy(time, hrm_rate)

    plt.savefig('examples.png')

def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(f'Usage: {argv[0]} production-csv', file=sys.stderr)
        return 1

    plot_examples()

    with open(argv[1], 'r', newline='') as production_csv:
        data = read_production_csv(production_csv)

    for well in data:
        plt.semilogy(well.days_on, well.oil)
    plt.savefig('plot.png')

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
