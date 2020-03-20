import sys
import csv
from math import exp

import matplotlib.pyplot as plt
from scipy.optimize import minimize

class ArpsDecline:
    # qi (daily rate)
    # Di_nom (nominal annual decline)
    # b (unitless)
    def __init__(self, qi, Di_nom, b):
        if qi < 0.0:
            raise ValueError('Negative qi')
        if Di_nom < 0.0:
            raise ValueError('Negative Di_nom')
        if b < 0.0 or b > 2.0:
            raise ValueError('Invalid b')

        self.qi = qi
        self.Di_nom = Di_nom
        self.b = b

    # time (years)
    # returns (daily rate)
    def rate(self, time):
        if self.b == 0:
            return self.qi * exp(-self.Di_nom * time)
        elif self.b == 1.0:
            return self.qi / (1.0 + self.Di_nom * time)
        else:
            return self.qi / (1.0 + self.b * self.Di_nom * time) ** (1.0 / self.b)

class WellProduction:
    def __init__(self, well_name, days_on, oil):
        self.well_name = well_name
        self.days_on = days_on
        self.oil = oil

def best_fit(well_production):

def read_production_csv(csv_file):
    reader = csv.reader(csv_file)
    next(reader) # skip header row

    result = list()
    last_well_name = None
    days_on = list()
    oil = list()

    for (w, d, o) in reader:
        if w != last_well_name:
            if last_well_name is not None:
                result.append(WellProduction(last_well_name, days_on, oil))
            days_on = list()
            oil = list()
            last_well_name = w

        days_on.append(float(d))
        oil.append(float(o))

    if last_well_name is not None:
            result.append(WellProduction(last_well_name, days_on, oil))

    return result

def main(argv):
    if len(argv) != 2:
        print(f'Usage: {argv[0]} production-csv', file=sys.stderr)
        return 1

    with open(argv[1], 'r', newline='') as production_csv:
        data = read_production_csv(production_csv)

    for well in data:
        plt.semilogy(well.days_on, well.oil)

    # exp_decl = ArpsDecline(1000.0, 0.9, 0.0)
    # hyp_decl = ArpsDecline(1000.0, 0.9, 1.5)
    # hrm_decl = ArpsDecline(1000.0, 0.9, 1.0)

    # time = [m / 12.0 for m in range(0, 5 * 12)]
    # exp_rate = [exp_decl.rate(t) for t in time]
    # hyp_rate = [hyp_decl.rate(t) for t in time]
    # hrm_rate = [hrm_decl.rate(t) for t in time]

    # plt.semilogy(time, exp_rate)
    # plt.semilogy(time, hyp_rate)
    # plt.semilogy(time, hrm_rate)
    plt.savefig('plot.png')

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
