import os
import pandas as pd
import numpy as np
from multicopula import EllipticalCopula
import pickle
import time

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Background-load scenario pool
# ---------------------------------------------------------------------------
# Drawing a day of feeder background load from the fitted copula costs 0.1-40 s
# (it is a rejection loop around ~17k frozen scipy distributions), and PowerGrid
# .reset() draws two of them on EVERY env reset. With 64 vectorised envs that is
# by far the dominant cost of training - orders of magnitude more than stepping
# the simulator.
#
# The pool draws each (weekday, slot) profile ONCE and reuses it, so the cost is
# paid at most POOL_SIZE * 7 times per process instead of twice per episode.
# Training still sees 7 * POOL_SIZE distinct background-load days, on top of the
# EV arrivals, prices, inflexible loads and DR events that keep varying freely.
#
#   EV2GYM_SCENARIO_POOL=0   -> pool disabled, original (slow) behaviour
#   EV2GYM_SCENARIO_POOL=n   -> n profiles per weekday (default 64)
#
# A profile is a deterministic function of (weekday, slot), because the fill is
# run under a seed derived from those two numbers with the global numpy state
# saved and restored around it. So the pool is identical in every process and in
# every run, and an episode remains reproducible from the env seed alone.
SCENARIO_POOL_SIZE = int(os.environ.get("EV2GYM_SCENARIO_POOL", 64))
_SCENARIO_POOL = {}


def clear_scenario_pool() -> None:
    '''Drop every cached background-load profile (frees ~POOL_SIZE*7*96*n_buses floats).'''
    _SCENARIO_POOL.clear()


class DataGenerator:
    def __init__(self):

        self._fit_load_profiles()

    def _fit_load_profiles(self):
        """
        Load active power data from the data manager.
        """
        data_path = "./ev2gym/data/original_train_data.csv"

        df = pd.read_csv(data_path, parse_dates=['date_time'])
        print(f'df initial shape is {df.shape}')

        # Drop the "price" column and any columns related to renewable generation.
        # cols_to_drop = [col for col in df.columns
        #                 if col.startswith('price') or col.startswith('renewable_active_power')]
        cols_to_drop = [col for col in df.columns
                        if col.startswith('price')]
        df.drop(columns=cols_to_drop, inplace=True)

        df['date_time'] = pd.to_datetime(df['date_time'])

        # Extract the date and time from the date_time column.
        df['day'] = df['date_time'].dt.day_of_week
        df.drop(columns=['date_time'], inplace=True)
        df['timestep'] = df.index % 96

        print(f'df shape is {df.values.shape}')
        # df = df[:96*7]
        new_df = pd.DataFrame()
        print(f'number of days is {int(len(df.values)//96)}')

        for j in range(int(len(df.values)//96)):

            for i in range(1, 34):
                day = df.values[j*96, -2].astype(int)
                active_power_96_node_i = df.values[j *
                                                   96:(j+1)*96, i]
                # - df.values[j*96:(j+1)*96, i+34]
                # active_power_96_node_i = -df.values[j*96:(j+1)*96, i+34]
                entry = {'day': [day],
                         }

                for k in range(96):
                    entry[f'active_power_{k}'] = active_power_96_node_i[k]

                new_df = pd.concat(
                    [new_df, pd.DataFrame(entry)], ignore_index=True)

        print(f'new_df shape is {new_df.shape}')

        dataset = new_df.values.T
        print(f'dataset shape is {dataset.shape}')

        self.copula_model = EllipticalCopula(dataset)
        self.copula_model.fit()

    def _draw_day(self, n_buses: int, day: int) -> np.ndarray:
        '''One valid 96 x n_buses day of background load, straight from the copula.

        drop_inf=True is what makes this affordable. The columns of a draw are
        independent samples, and EllipticalCopula.sample already discards the
        ones that come out NaN or +inf and redraws them - but it keeps -inf
        unless asked. On Sundays (x1=6) about 9 of 123 columns land on -inf, so
        rejecting the whole block for containing one meant ~200 full draws
        (~19 s) per profile instead of one (~0.1 s). Dropping per column instead
        of per block applies the same finiteness test the loop below applies,
        just at the granularity the samples are actually independent at - the
        marginals are unchanged.
        '''
        while True:
            augmented_data = np.asarray(
                self.copula_model.sample(n_buses,
                                         conditional=True,
                                         variables={
                                             'x1': day,
                                             },
                                         drop_inf=True,
                                         ))
            if augmented_data.shape[1] == n_buses and np.isfinite(augmented_data).all():
                return augmented_data

    def _pooled_day(self, n_buses: int, day: int, slot: int) -> np.ndarray:
        '''Cached variant of _draw_day - see the note at the top of this file.'''
        key = (n_buses, day, slot)
        cached = _SCENARIO_POOL.get(key)
        if cached is None:
            # Fill under a seed fixed by (day, slot) so the pool is identical in
            # every process/run, then hand the caller's RNG stream back untouched.
            state = np.random.get_state()
            try:
                np.random.seed((0x5AFE * (day + 1) + 7919 * slot) % (2**32))
                cached = self._draw_day(n_buses, day)
            finally:
                np.random.set_state(state)
            _SCENARIO_POOL[key] = cached
        return cached

    def sample_data(self,
                    n_buses: int,
                    n_steps: int,
                    start_day: int,
                    start_step: int = 0,
                    variant: int = None,
                    ):
        '''
        variant: index of the scenario to draw from the pool. Pass the episode
        seed to keep an episode reproducible; None draws one at random. Ignored
        when SCENARIO_POOL_SIZE is 0, which restores the uncached behaviour.
        '''

        n_days = int(np.ceil((start_step + n_steps)/96))
        data = np.zeros((n_days*96, n_buses))

        pool_size = SCENARIO_POOL_SIZE
        if pool_size > 0:
            slot = (np.random.randint(pool_size) if variant is None
                    else int(variant) % pool_size)

        for j in range(n_days):
            day = (start_day + j) % 7

            if pool_size > 0:
                data[j*96:(j+1)*96, :] = self._pooled_day(n_buses, day, slot)
            else:
                data[j*96:(j+1)*96, :] = self._draw_day(n_buses, day)

        return data[start_step:start_step+n_steps, :]

def get_pv_load(data, env):

    dataset_starting_date = '2019-01-01 00:00:00'
    simulation_length = env.simulation_length + 24
    simulation_date = env.sim_starting_date.strftime('%Y-%m-%d %H:%M:%S')

    # find year of the data
    year = int(dataset_starting_date.split('-')[0])
    # replace the year of the simulation date with the year of the data
    simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'

    simulation_index = data[data['date'] == simulation_date].index[0]

    # select the data for the simulation date
    data = data[simulation_index:simulation_index+simulation_length]
    
    return data['electricity'].values.reshape(-1, 1)

if __name__ == "__main__":

    # Code for fitting the copula model and saving it to a file so it can be quickly loaded later.

    augmentor = DataGenerator()
    pickle.dump(augmentor, open('augmentor.pkl', 'wb'))

    augmentor = pickle.load(open('augmentor.pkl', 'rb'))

    start_time = time.time()    
    augmented_data = augmentor.sample_data(n_buses=34,
                                           n_steps=96*5,
                                           start_day=5,
                                           start_step=0,
                                           )
    print(f'Elapsed time is {time.time() - start_time}')
    
    # plot the data
    # plt.plot(augmented_data[:,11:16])
    plt.plot(augmented_data)
    # plt.legend([f'Node {i}' for i in range(1, 6)])
    plt.show()
