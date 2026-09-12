'''
This file contains the loaders for the EV City environment.
'''

import numpy as np
import pandas as pd
import math
import datetime
import pkg_resources
import json
from typing import List, Tuple

from ev2gym.models.ev_charger import EV_Charger
from ev2gym.models.ev import EV
from ev2gym.models.transformer import Transformer
from ev2gym.models.grid import PowerGrid

from ev2gym.utilities.utils import EV_spawner, generate_power_setpoints, EV_spawner_GF


# ---------------------------------------------------------------------------
# Static-dataset caches
# ---------------------------------------------------------------------------
# The bundled csv files never change, but load_transformers() /
# load_electricity_prices() re-read and re-shape them on EVERY env reset. With
# 64 vectorised envs resetting every 96 steps that is pure repeated work, so the
# parsed frames are memoised per process. Every cached object is treated as
# read-only; callers get a copy of anything they might mutate.
_DATASET_CACHE = {}


def _cached(key, build):
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = build()
    return _DATASET_CACHE[key]


def clear_dataset_cache() -> None:
    '''Drop the memoised csv datasets (use after editing a bundled data file).'''
    _DATASET_CACHE.clear()


def load_ev_spawn_scenarios(env) -> None:
    '''Loads the EV spawn scenarios of the simulation'''

    # Load the EV specs
    if env.config['heterogeneous_ev_specs']:

        if "ev_specs_file" in env.config:
            ev_specs_file = env.config['ev_specs_file']
        else:
            ev_specs_file = pkg_resources.resource_filename('ev2gym', 'data/ev_specs.json')

        with open(ev_specs_file) as f:
            env.ev_specs = json.load(f)

        registrations = np.zeros(len(env.ev_specs.keys()))
        for i, ev_name in enumerate(env.ev_specs.keys()):
            # sum the total number of registrations
            registrations[i] = env.ev_specs[ev_name]['number_of_registrations']

        env.normalized_ev_registrations = registrations/registrations.sum()

    if env.scenario == 'GF':
        env.df_arrival = np.load('./GF_data/time_of_arrival.npy')  # weekdays
        env.time_of_connection_vs_hour_weekday = np.load(
            './GF_data/weekday_time_of_stay.npy')
        env.time_of_connection_vs_hour_weekend = np.load(
            './GF_data/weekend_time_of_stay.npy')
        env.df_req_energy_weekday = np.load('./GF_data/weekday_volumeKWh.npy')
        env.df_req_energy_weekend = np.load('./GF_data/weekend_volumeKWh.npy')

        return

    df_arrival_week_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-arrival.csv')
    df_arrival_weekend_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-arrival-weekend.csv')
    df_connection_time_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-connection-time.csv')
    df_energy_demand_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-energy-demand.csv')
    time_of_connection_vs_hour_file = pkg_resources.resource_filename(
        'ev2gym', 'data/time_of_connection_vs_hour.npy')

    df_req_energy_file = pkg_resources.resource_filename(
        'ev2gym', 'data/mean-demand-per-arrival.csv')
    df_time_of_stay_vs_arrival_file = pkg_resources.resource_filename(
        'ev2gym', 'data/mean-session-length-per.csv')

    env.df_arrival_week = pd.read_csv(df_arrival_week_file)  # weekdays
    env.df_arrival_weekend = pd.read_csv(df_arrival_weekend_file)  # weekends
    env.df_connection_time = pd.read_csv(
        df_connection_time_file)  # connection time
    env.df_energy_demand = pd.read_csv(df_energy_demand_file)  # energy demand
    env.time_of_connection_vs_hour = np.load(
        time_of_connection_vs_hour_file)  # time of connection vs hour

    env.df_req_energy = pd.read_csv(
        df_req_energy_file)  # energy demand per arrival
    # replace column work with workplace
    env.df_req_energy = env.df_req_energy.rename(columns={'work': 'workplace',
                                                          'home': 'private'})
    env.df_req_energy = env.df_req_energy.fillna(0)

    env.df_time_of_stay_vs_arrival = pd.read_csv(
        df_time_of_stay_vs_arrival_file)  # time of stay vs arrival
    env.df_time_of_stay_vs_arrival = env.df_time_of_stay_vs_arrival.fillna(0)
    env.df_time_of_stay_vs_arrival = env.df_time_of_stay_vs_arrival.rename(columns={'work': 'workplace',
                                                                                    'home': 'private'})


def load_power_setpoints(env) -> np.ndarray:
    '''
    Loads the power setpoints of the simulation based on the day-ahead prices
    '''

    if env.load_from_replay_path:
        return env.replay.power_setpoints
    else:
        if not env.config['power_setpoint_enabled']:
            return np.zeros(env.simulation_length)
        else:
            return generate_power_setpoints(env)


def generate_residential_inflexible_loads(env) -> np.ndarray:
    '''
    This function loads the inflexible loads of each transformer
    in the simulation.
    '''

    desired_timescale = env.timescale
    simulation_length = env.simulation_length
    simulation_date = env.sim_starting_date.strftime('%Y-%m-%d %H:%M:%S')
    number_of_transformers = env.number_of_transformers

    dataset_timescale = 15
    dataset_starting_date = '2022-01-01 00:00:00'

    def _build():
        # Load the data
        data_path = pkg_resources.resource_filename(
            'ev2gym', 'data/residential_loads.csv')
        data = pd.read_csv(data_path, header=None)

        if desired_timescale > dataset_timescale:
            data = data.groupby(
                data.index // (desired_timescale/dataset_timescale)).max()
        elif desired_timescale < dataset_timescale:
            # extend the dataset to data.shape[0] * (dataset_timescale/desired_timescale)
            # by repeating the data every (dataset_timescale/desired_timescale) rows
            data = data.loc[data.index.repeat(
                dataset_timescale/desired_timescale)].reset_index(drop=True)

        # duplicate the data to have two years of data
        data = pd.concat([data, data], ignore_index=True)

        # add a date column to the dataframe
        data['date'] = pd.date_range(
            start=dataset_starting_date, periods=data.shape[0], freq=f'{desired_timescale}min')
        return data

    data = _cached(('residential_loads', desired_timescale), _build)

    # find year of the data
    year = int(dataset_starting_date.split('-')[0])
    # replace the year of the simulation date with the year of the data
    simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'

    simulation_index = data[data['date'] == simulation_date].index[0]

    # With a single random_state every transformer draws the *same* 10
    # households, i.e. one perfectly correlated load shape for the whole city.
    # That is harmless for a single-transformer parking lot but wrong at city
    # scale, where substations peak at different times. Opt in per config.
    decorrelate = env.config['inflexible_loads'].get(
        'decorrelate_transformers', False)

    # The column draw is a pure function of (tr_seed, transformer index), and the
    # slice a pure function of the date, so the whole (n_tr x sim_length) block
    # is deterministic - cache it per date instead of resampling on every reset.
    def _build_block():
        block = data[simulation_index:simulation_index +
                     simulation_length].drop(columns=['date'])
        new_data = pd.DataFrame()
        for i in range(number_of_transformers):
            new_data['tr_'+str(i)] = block.sample(
                10, axis=1,
                random_state=env.tr_seed + i if decorrelate else env.tr_seed
            ).sum(axis=1)

        # return the "tr_" columns
        return new_data.to_numpy().T

    key = ('residential_block', desired_timescale, simulation_length,
           simulation_index, number_of_transformers, env.tr_seed, decorrelate)
    # copy: Transformer.normalize_inflexible_loads writes into its row
    return _cached(key, _build_block).copy()


def _build_pv_frame(desired_timescale, dataset_timescale, dataset_starting_date):
    '''Parse, resample, smooth and date-index pv_netherlands.csv (cached).'''

    # Load the data
    data_path = pkg_resources.resource_filename(
        'ev2gym', 'data/pv_netherlands.csv')
    data = pd.read_csv(data_path, sep=',', header=0)
    data.drop(['time', 'local_time'], inplace=True, axis=1)

    if desired_timescale > dataset_timescale:
        data = data.groupby(
            data.index // (desired_timescale/dataset_timescale)).max()
    elif desired_timescale < dataset_timescale:
        # extend the dataset to data.shape[0] * (dataset_timescale/desired_timescale)
        # by repeating the data every (dataset_timescale/desired_timescale) rows
        data = data.loc[data.index.repeat(
            dataset_timescale/desired_timescale)].reset_index(drop=True)
        # data = data/ (dataset_timescale/desired_timescale)

    # smooth data by taking the mean of every 5 rows
    data['electricity'] = data['electricity'].rolling(
        window=60//desired_timescale, min_periods=1).mean()
    # use other type of smoothing
    data['electricity'] = data['electricity'].ewm(
        span=60//desired_timescale, adjust=True).mean()

    # duplicate the data to have two years of data
    data = pd.concat([data, data], ignore_index=True)

    # add a date column to the dataframe
    data['date'] = pd.date_range(
        start=dataset_starting_date, periods=data.shape[0], freq=f'{desired_timescale}min')

    return data


def generate_pv_generation(env) -> np.ndarray:
    '''
    This function loads the PV generation of each transformer by loading the data from a file
    and then adding minor variations to the data
    '''

    desired_timescale = env.timescale
    simulation_length = env.simulation_length
    simulation_date = env.sim_starting_date.strftime('%Y-%m-%d %H:%M:%S')
    number_of_transformers = env.number_of_transformers

    dataset_timescale = 60
    dataset_starting_date = '2019-01-01 00:00:00'

    # Static file + deterministic smoothing -> parse and smooth once per process.
    data = _cached(('pv_generation', desired_timescale),
                   lambda: _build_pv_frame(desired_timescale,
                                           dataset_timescale,
                                           dataset_starting_date))

    # find year of the data
    year = int(dataset_starting_date.split('-')[0])
    # replace the year of the simulation date with the year of the data
    simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'

    simulation_index = data[data['date'] == simulation_date].index[0]

    # select the data for the simulation date
    data = data[simulation_index:simulation_index+simulation_length]

    # drop the date column
    data = data.drop(columns=['date'])
    new_data = pd.DataFrame()

    for i in range(number_of_transformers):
        new_data['tr_'+str(i)] = data * env.tr_rng.uniform(0.9, 1.1)

    return new_data.to_numpy().T


def resolve_transformer_ratings(env) -> np.ndarray:
    '''Resolves the max_power (in kW) of every transformer of the simulation.

    There are four ways of sizing the transformers, in order of precedence:

      1. `transformer.size_from_feeder`: size each substation from the peak load
         of the bus it sits on, the way a DSO actually sizes one - take the node
         peak from `network_info.bus_info_file`, divide by
         `transformer.target_peak_loading`, and round UP to the next standard
         rating on `transformer.rating_ladder` (IEC 60076 ladder by default).
         This is the only option that makes the transformer model and the power
         flow model describe the same physical system: every other option sizes
         the transformer independently of the load actually sitting on its node.
      2. `transformer.ratings_file`: a csv with columns `rating_kva,share`. One
         nameplate rating is drawn per transformer, which is what a real
         distribution grid looks like - secondary substations are not all the
         same size - but the draw is uncorrelated with the node load.
      3. `transformer.max_power` given as a list: used verbatim, tiled if it is
         shorter than the number of transformers.
      4. `transformer.max_power` given as a scalar: every transformer gets it.
         This is the historical behaviour and stays the default.

    In every case `transformer.kw_per_kva` converts nameplate kVA into the kW
    budget handed to the Transformer model (use it for the power factor of the
    mixed load and/or a planning derating).

    The result is cached on the env because `load_transformers` runs again on
    every `reset()`, and a substation does not change size between episodes.

    Returns:
        - ratings: an array of size (number_of_transformers,) in kW
    '''

    n_transformers = env.number_of_transformers

    cached = getattr(env, '_transformer_ratings', None)
    if cached is not None and len(cached) == n_transformers:
        return cached

    tr_config = env.config['transformer']
    ratings_file = tr_config.get('ratings_file', None)

    if tr_config.get('size_from_feeder', False):
        bus_info = pd.read_csv(env.config['network_info']['bus_info_file'])

        # Row 0 is the slack bus; the transformers are the remaining nodes, in
        # the same order load_grid uses (number_of_transformers = node_num - 1).
        node_peak_kw = bus_info['PD'].to_numpy(dtype=float)[1:]

        if len(node_peak_kw) != n_transformers:
            raise ValueError(
                f'transformer.size_from_feeder needs one bus per transformer, but '
                f'{env.config["network_info"]["bus_info_file"]} has {len(node_peak_kw)} '
                f'non-slack buses for {n_transformers} transformers. This option only '
                f'makes sense with simulate_grid: True.')

        target_loading = float(tr_config.get('target_peak_loading', 0.55))
        ladder = np.asarray(tr_config.get(
            'rating_ladder', [100, 160, 250, 400, 630, 800, 1000, 1250, 1600]),
            dtype=float)
        ladder.sort()

        # Round the required capacity UP to the next size you can actually buy.
        required_kva = node_peak_kw / target_loading
        kva = ladder[np.clip(np.searchsorted(ladder, required_kva),
                             0, len(ladder) - 1)]

        oversized = required_kva > ladder[-1]
        if oversized.any():
            print(f'Warning: {oversized.sum()} bus(es) need more than the largest '
                  f'rating on the ladder ({ladder[-1]:.0f} kVA) and were capped.')

        ratings = kva * float(tr_config.get('kw_per_kva', 1.0))

    elif ratings_file not in (None, 'None', ''):
        mix = pd.read_csv(ratings_file, comment='#')

        for column in ('rating_kva', 'share'):
            if column not in mix.columns:
                raise ValueError(
                    f'{ratings_file} must have columns rating_kva,share (missing {column})')

        kva = mix['rating_kva'].to_numpy(dtype=float)
        share = mix['share'].to_numpy(dtype=float)
        share = share / share.sum()

        # A dedicated generator seeded from tr_seed: which substation is how big
        # must not move when the EV-arrival RNG is re-seeded.
        rng = np.random.default_rng(env.tr_seed)
        ratings = rng.choice(kva, size=n_transformers, p=share) * \
            float(tr_config.get('kw_per_kva', 1.0))

    else:
        max_power = tr_config['max_power']

        if isinstance(max_power, (list, tuple, np.ndarray)):
            ratings = np.asarray(max_power, dtype=float)
            if len(ratings) < n_transformers:
                ratings = np.resize(ratings, n_transformers)
            ratings = ratings[:n_transformers]
        else:
            ratings = np.full(n_transformers, float(max_power))

    env._transformer_ratings = ratings
    return ratings


def resolve_cs_transformer_mapping(env) -> list:
    '''Decides which transformer every charging station hangs off.

    `cs_transformer_mapping` in the config selects the strategy:

      * 'round_robin' (default): spread the charging stations evenly over the
        transformers. Historical behaviour, kept bit-for-bit.
      * 'clustered': draw a per-transformer "attractiveness" weight from a
        Gamma(alpha) distribution and allocate stations by it, so a few
        substations host a handful of chargers and many host none. Real public
        charging is clustered like this instead of being perfectly spread.
        `cs_clustering_alpha` sets how hard: smaller is more clustered, large
        approaches round robin.
      * a path to a csv with columns `cs_id,transformer_id`: real, measured
        assignments. This is the option to use once an actual charger register
        has been joined against an actual substation register.

    Returns:
        - cs_transformers: a list of size (number_of_charging_stations,)
    '''

    n_transformers = env.number_of_transformers
    n_cs = env.cs
    mode = env.config.get('cs_transformer_mapping', 'round_robin')

    if mode in (None, 'None', 'round_robin'):
        cs_transformers = [*np.arange(n_transformers)] * \
            (n_cs // n_transformers)
        cs_transformers += np.arange(n_cs % n_transformers).tolist()
        return cs_transformers

    if mode == 'clustered':
        alpha = float(env.config.get('cs_clustering_alpha', 0.7))
        rng = np.random.default_rng(env.tr_seed)

        weights = rng.gamma(shape=alpha, scale=1.0, size=n_transformers)
        if weights.sum() <= 0:
            weights = np.ones(n_transformers)
        weights = weights / weights.sum()

        cs_transformers = rng.choice(n_transformers, size=n_cs, p=weights)
        # Sorting keeps consecutive charging station ids on the same substation,
        # which is what a street-by-street roll-out looks like, and keeps the
        # cs_ids of a transformer contiguous in the plots.
        return np.sort(cs_transformers).tolist()

    # Anything else is treated as a path to a file of real assignments
    table = pd.read_csv(mode, comment='#')

    for column in ('cs_id', 'transformer_id'):
        if column not in table.columns:
            raise ValueError(
                f'{mode} must have columns cs_id,transformer_id (missing {column})')

    if len(table) != n_cs:
        raise ValueError(f'{mode} maps {len(table)} charging stations but the '
                         f'config asks for {n_cs}')

    cs_transformers = table.sort_values(
        'cs_id')['transformer_id'].to_numpy(dtype=int)

    if cs_transformers.min() < 0 or cs_transformers.max() >= n_transformers:
        raise ValueError(f'{mode} references transformer ids outside '
                         f'[0, {n_transformers - 1}]')

    return cs_transformers.tolist()


def load_transformers(env) -> List[Transformer]:
    '''Loads the transformers of the simulation
    If load_from_replay_path is None, then the transformers are created randomly

    Returns:
        - transformers: a list of transformer objects
    '''

    if env.load_from_replay_path is not None:
        return env.replay.transformers

    transformers = []

    if env.config['inflexible_loads']['include']:

        if env.scenario == 'private':
            inflexible_loads = generate_residential_inflexible_loads(env)

        # TODO add inflexible loads for public and workplace scenarios
        else:
            inflexible_loads = generate_residential_inflexible_loads(env)

    else:
        inflexible_loads = np.zeros((env.number_of_transformers,
                                    env.simulation_length))

    if env.config['solar_power']['include']:
        solar_power = generate_pv_generation(env)
    else:
        solar_power = np.zeros((env.number_of_transformers,
                                env.simulation_length))

    if env.charging_network_topology:
        # parse the topology file and create the transformers
        cs_counter = 0
        for i, tr in enumerate(env.charging_network_topology):
            cs_ids = []
            for cs in env.charging_network_topology[tr]['charging_stations']:
                cs_ids.append(cs_counter)
                cs_counter += 1
            transformer = Transformer(id=i,
                                      env=env,
                                      cs_ids=cs_ids,
                                      max_power=env.charging_network_topology[tr]['max_power'],
                                      inflexible_load=inflexible_loads[i, :],
                                      solar_power=solar_power[i, :],
                                      simulation_length=env.simulation_length
                                      )

            transformers.append(transformer)

    else:
        # if env.number_of_transformers > env.cs:
        #     raise ValueError(
        #         'The number of transformers cannot be greater than the number of charging stations')
        ratings = resolve_transformer_ratings(env)

        for i in range(env.number_of_transformers):
            # get indexes where the transformer is connected
            transformer = Transformer(id=i,
                                      env=env,
                                      cs_ids=np.where(
                                          np.array(env.cs_transformers) == i)[0],
                                      max_power=ratings[i],
                                      inflexible_load=inflexible_loads[i, :],
                                      solar_power=solar_power[i, :],
                                      simulation_length=env.simulation_length
                                      )

            transformers.append(transformer)
    env.n_transformers = len(transformers)
    return transformers


def load_ev_charger_profiles(env) -> List[EV_Charger]:
    '''Loads the EV charger profiles of the simulation
    If load_from_replay_path is None, then the EV charger profiles are created randomly

    Returns:
        - ev_charger_profiles: a list of ev_charger_profile objects'''

    charging_stations = []
    if env.load_from_replay_path is not None:
        return env.replay.charging_stations

    v2g_enabled = env.config['v2g_enabled']

    if env.charging_network_topology:
        # parse the topology file and create the charging stations
        cs_counter = 0
        for i, tr in enumerate(env.charging_network_topology):
            for cs in env.charging_network_topology[tr]['charging_stations']:
                ev_charger = EV_Charger(id=cs_counter,
                                        connected_bus=i,
                                        connected_transformer=i,
                                        min_charge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['min_charge_current'],
                                        max_charge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['max_charge_current'],
                                        min_discharge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['min_discharge_current'],
                                        max_discharge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['max_discharge_current'],
                                        voltage=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['voltage'],
                                        n_ports=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['n_ports'],
                                        charger_type=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['charger_type'],
                                        phases=env.charging_network_topology[tr]['charging_stations'][cs]['phases'],
                                        timescale=env.timescale,
                                        verbose=env.verbose,)
                cs_counter += 1
                charging_stations.append(ev_charger)
        env.cs = len(charging_stations)
        return charging_stations

    else:
        if v2g_enabled:
            max_discharge_current = env.config['charging_station']['max_discharge_current']
            min_discharge_current = env.config['charging_station']['min_discharge_current']
        else:
            max_discharge_current = 0
            min_discharge_current = 0

        for i in range(env.cs):
            ev_charger = EV_Charger(id=i,
                                    connected_bus=env.cs_transformers[i],
                                    connected_transformer=env.cs_transformers[i],
                                    n_ports=env.number_of_ports_per_cs,
                                    max_charge_current=env.config['charging_station']['max_charge_current'],
                                    min_charge_current=env.config['charging_station']['min_charge_current'],
                                    max_discharge_current=max_discharge_current,
                                    min_discharge_current=min_discharge_current,
                                    phases=env.config['charging_station']['phases'],
                                    voltage=env.config['charging_station']['voltage'],
                                    timescale=env.timescale,
                                    verbose=env.verbose,)

            charging_stations.append(ev_charger)
        return charging_stations


def load_ev_profiles(env) -> List[EV]:
    '''Loads the EV profiles of the simulation
    If load_from_replay_path is None, then the EV profiles are created randomly

    Returns:
        - ev_profiles: a list of ev_profile objects'''

    if env.load_from_replay_path is None:

        if env.scenario == 'GF':
            ev_profiles = EV_spawner_GF(env)
            while len(ev_profiles) == 0:
                ev_profiles = EV_spawner_GF(env)
            return ev_profiles

        ev_profiles = EV_spawner(env)
        while len(ev_profiles) == 0:
            ev_profiles = EV_spawner(env)

        return ev_profiles
    else:
        return env.replay.EVs


def _build_price_data() -> pd.DataFrame:
    '''Parse the day-ahead price csv and split the timestamp into columns (cached).'''

    file_path = pkg_resources.resource_filename(
        'ev2gym', 'data/Netherlands_day-ahead-2015-2024.csv')
    price_data = pd.read_csv(file_path, sep=',', header=0)

    drop_columns = ['Country', 'Datetime (Local)']

    price_data.drop(drop_columns, inplace=True, axis=1)
    stamps = pd.DatetimeIndex(price_data['Datetime (UTC)'])
    price_data['year'] = stamps.year
    price_data['month'] = stamps.month
    price_data['day'] = stamps.day
    price_data['hour'] = stamps.hour

    return price_data


def load_electricity_prices(env) -> Tuple[np.ndarray, np.ndarray]:
    '''Loads the electricity prices of the simulation
    If load_from_replay_path is None, then the electricity prices are created randomly

    Returns:
        - charge_prices: a matrix of size (number of charging stations, simulation length) with the charge prices
        - discharge_prices: a matrix of size (number of charging stations, simulation length) with the discharge prices'''

    if env.load_from_replay_path is not None:
        return env.replay.charge_prices, env.replay.discharge_prices

    if env.price_data is None:
        env.price_data = _cached('price_data', _build_price_data)

    # assume charge and discharge prices are the same
    # assume prices are the same for all charging stations

    # {(year, month, day, hour): price} - the old four-mask .loc scan over 87k
    # rows ran 96 times per reset, i.e. ~9M row comparisons per episode.
    price_by_hour = _cached('price_by_hour',
                            lambda: dict(zip(zip(env.price_data['year'],
                                                 env.price_data['month'],
                                                 env.price_data['day'],
                                                 env.price_data['hour']),
                                             env.price_data['Price (EUR/MWhe)'])))

    charge_prices = np.zeros((env.cs, env.simulation_length))
    discharge_prices = np.zeros((env.cs, env.simulation_length))
    # for every simulation step, take the price of the corresponding hour
    sim_temp_date = env.sim_date
    for i in range(env.simulation_length):

        year = sim_temp_date.year
        month = sim_temp_date.month
        day = sim_temp_date.day
        hour = sim_temp_date.hour
        # find the corresponding price
        price = price_by_hour.get((year, month, day, hour))

        if price is None:
            print(
                'Error: no price found for the given date and hour. Using 2022 prices instead.')

            year = 2022
            if day > 28:
                day -= 1
            price = price_by_hour[(year, month, day, hour)]

        charge_prices[:, i] = -price/1000  # €/kWh
        discharge_prices[:, i] = price/1000  # €/kWh

        # step to next
        sim_temp_date = sim_temp_date + \
            datetime.timedelta(minutes=env.timescale)

    discharge_prices = discharge_prices * env.config['discharge_price_factor']
    return charge_prices, discharge_prices


def load_grid(env):
    '''Loads the grid of the simulation'''

    if env.load_from_replay_path is not None:
        env.cs_transformers = env.replay.cs_transformers
        return env.replay.grid

    # Simulate grid
    if env.simulate_grid:
        if env.load_from_replay_path is None:
            pv_profile = load_pv_profiles(env)
            
        grid = PowerGrid(env.config,
                         env=env,
                         pv_profile=pv_profile,
                         )

        assert env.charging_network_topology is None, "Charging network topology is not supported with grid simulation."

        # The feeder decides how many transformers there are: one per node minus
        # the slack bus. This overrides config['number_of_transformers'].
        env.number_of_transformers = grid.node_num-1
        env.cs_transformers = resolve_cs_transformer_mapping(env)
        # print(f'cs_transformers: {env.cs_transformers}')

        return grid

    if env.charging_network_topology is None:
        env.cs_transformers = resolve_cs_transformer_mapping(env)

    return None


def load_pv_profiles(env) -> np.ndarray:

    desired_timescale = env.timescale

    dataset_timescale = 60
    dataset_starting_date = '2019-01-01 00:00:00'

    # Same parse+smooth as generate_pv_generation, so share its cache entry.
    # Read-only downstream (get_pv_load only slices it).
    return _cached(('pv_generation', desired_timescale),
                   lambda: _build_pv_frame(desired_timescale,
                                           dataset_timescale,
                                           dataset_starting_date))


