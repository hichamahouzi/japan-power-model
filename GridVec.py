#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.optimize import linprog
from datetime import datetime, date, timedelta

# Configuration
np.set_printoptions(suppress=True)


class DataLoader:
    """Handles loading and initial cleaning of raw CSV data."""

    def __init__(self, file_config):
        self.config = file_config
        self.path = Path(file_config['Path'])
        self.step = int(file_config['step'])

    def load_numpy(self, filename, dtype=str, delimiter=',', encoding='utf-8-sig'):
        """
        Loads CSV data into a numpy array.
        Note: encoding='utf-8-sig' handles BOM (Byte Order Mark) common in Excel CSVs.
        """
        full_path = self.path / filename
        if not full_path.exists():
            print(f"Warning: File not found {full_path}")
            return None

        try:
            # autostrip=True removes leading/trailing whitespace
            data = np.genfromtxt(full_path, dtype=dtype, delimiter=delimiter, encoding=encoding, autostrip=True)
            # Extra safety: if string, strip again to be sure
            if np.issubdtype(data.dtype, np.str_) or np.issubdtype(data.dtype, np.object_):
                data = np.char.strip(data)
            return data
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def get_time_series_data(self, shape_ref):
        """Loads Demand, Wind, Solar and aligns dimensions."""
        print("Loading Time Series Data...")
        demand = self.load_numpy(self.config['Demand'])
        wind = self.load_numpy(self.config['Wind'])
        solar = self.load_numpy(self.config['Solar'])

        rows = shape_ref[0]

        def clean(arr):
            if arr is None: return np.zeros(shape_ref)
            arr[arr == ''] = 0
            # Resize to match Availability shape (skip header row and date col)
            return np.round(arr[1:rows + 1, 1:].astype(float), 0)

        return clean(demand), clean(wind), clean(solar)

    def get_prices(self, shape_ref):
        prices_daily = self.load_numpy(self.config['Prices'])
        if prices_daily is None: return np.zeros(shape_ref)

        # Expand daily prices to hourly (repeat rows)
        repeats = int(np.ceil(shape_ref[0] / self.step))
        prices = np.repeat(prices_daily[1:repeats + 1, 1:], self.step, axis=0)

        # Trim to exact length
        prices = prices[:shape_ref[0], :]
        return np.round(prices.astype(float), 1)

    def get_interconnectors(self, shape_ref):
        interco = {'Import': {'data': None, 'index': {}}, 'Export': {'data': None, 'index': {}}}
        for direction in ['Import', 'Export']:
            key = f'Transmission_{direction}'
            if key in self.config and (self.path / self.config[key]).exists():
                raw = self.load_numpy(self.config[key])
                if raw is not None:
                    raw[raw == ''] = 0
                    data = np.round(raw[1:shape_ref[0] + 1, 1:].astype(float), 0)
                    idx_map = dict(zip(raw[0, 1:], range(len(raw[0, 1:]))))
                    interco[direction] = {'data': data, 'index': idx_map}
        return interco


class PowerMarketModel:
    """Core logic for Japanese Power Market Optimization."""

    REGIONS = ['HOK', 'TOH', 'TOK', 'CHU', 'HKU', 'KAN', 'CHK', 'SHI', 'KYU']

    # Topology matrix (Rows: Regions, Cols: Lines)
    # 1 = Export from Region, -1 = Import to Region
    # Order matches typical JEPX line ordering: HOK-TOH, TOH-TOK, TOK-CHU, etc.
    BINARY_MATRIX = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # HOK
        [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # TOH
        [0, -1, 1, 0, 0, 0, 0, 0, 0, 0],  # TOK
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 1],  # CHU (Connected to TOK and KYU)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, -1],  # HKU
        [0, 0, 0, -1, -1, 1, 1, 0, 0, 0],  # KAN
        [0, 0, 0, 0, 0, -1, 0, 1, 1, 0],  # CHK
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0],  # SHI
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0]  # KYU
    ])*-1 # Adjust direction convention if needed

    def __init__(self, files):
        self.loader = DataLoader(files)
        self.run_name = files.get('run_name', 'JP')

        print("\n--- 1. LOADING DATA ---")

        # 1. Load Availability first to determine shapes
        raw_avai = self.loader.load_numpy(files['Availability'])
        if raw_avai is None:
            raise ValueError("Availability file could not be loaded.")

        self.date_index = raw_avai[1:, 0]
        self.datetime_index = pd.to_datetime(self.date_index, format='%Y-%m-%d %H:%M:%S')

        # Helper indices
        self.avai_cols = dict(zip(np.char.upper(raw_avai[0, 1:]), range(len(raw_avai[0, 1:]))))

        # Clean Availability
        avai_data = raw_avai[1:, 1:]
        avai_data[avai_data == ''] = 0
        self.Avai = np.round(avai_data.astype(float), 0)

        # 2. Load other Time Series
        self.Demand, self.Wind, self.Solar = self.loader.get_time_series_data(self.Avai.shape)
        self.Prices = self.loader.get_prices(self.Avai.shape)
        self.Interco = self.loader.get_interconnectors(self.Avai.shape)

        # 3. Load Static Data
        self.plantlist = np.char.upper(self.loader.load_numpy(files['Plantlist']))
        self.plant_cols = dict(zip(self.plantlist[0, :], range(self.plantlist.shape[1])))

        # Helper Indices for DataFrames
        raw_demand = self.loader.load_numpy(files['Demand'])
        self.demand_cols = dict(zip(np.char.upper(raw_demand[0, 1:]), range(len(raw_demand[0, 1:]))))
        self.prices_cols = dict(
            zip(np.char.upper(self.loader.load_numpy(files['Prices'])[0, 1:]), range(self.Prices.shape[1])))

        # Initialize Costs
        self.Costs = np.zeros_like(self.Avai)

        # Results Storage
        self.Q_sorted = {}
        self.MC_sorted = {}
        self.MAP = {}
        self.R = {}
        self.results_prices = None
        self.results_flows = None

        # --- PRINT DATA SHAPES ---
        print("\n--- DATA SHAPE REPORT ---")
        print(f"Time Steps (Rows):       {self.Avai.shape[0]}")
        print(f"Availability Matrix:     {self.Avai.shape} (Time x Units)")
        print(f"Demand Matrix:           {self.Demand.shape} (Time x Regions)")
        print(f"Wind Matrix:             {self.Wind.shape}")
        print(f"Solar Matrix:            {self.Solar.shape}")
        print(f"Prices Matrix:           {self.Prices.shape}")
        print(f"Plant List:              {self.plantlist.shape} (Units x Attributes)")

        if self.Interco['Import']['data'] is not None:
            print(f"Interco Import:          {self.Interco['Import']['data'].shape}")
        else:
            print(f"Interco Import:          Not Loaded / None")

        print("-------------------------\n")

    def process_plant_parameters(self):
        """Expands bands, applies availability coefficients and calculates marginal costs."""
        print("Processing plant parameters...")

        # DEBUG: Print sample IDs to help diagnosis
        print(f"DEBUG: First 5 Plantlist IDs: {self.plantlist[1:6, self.plant_cols['ID_EN']]}")
        print(f"DEBUG: First 5 Avai Headers: {list(self.avai_cols.keys())[:5]}")

        # Indices
        idx_id = self.plant_cols['ID_EN']
        idx_fh = self.plant_cols['FUELHUB']
        idx_bands = self.plant_cols['BANDS']
        idx_q = self.plant_cols['QCOEFF']
        idx_p = self.plant_cols['PCOEFF']
        idx_eff = self.plant_cols['EFFCY']
        idx_mr = self.plant_cols['MUSTRUN']

        # Base Costs from Fuel Hubs
        for unit_row in self.plantlist[1:]:
            uid = unit_row[idx_id]
            hub = unit_row[idx_fh]
            if uid in self.avai_cols and hub in self.prices_cols:
                self.Costs[:, self.avai_cols[uid]] = self.Prices[:, self.prices_cols[hub]]

        # Expand Bands logic
        new_avai_list = []
        new_costs_list = []
        new_names_list = []

        sorted_unit_keys = sorted(self.avai_cols.keys(), key=lambda x: self.avai_cols[x])

        match_count = 0
        for uid in sorted_unit_keys:
            # Get unit data
            row_idx = np.where(self.plantlist[:, idx_id] == uid)[0]
            if len(row_idx) == 0:
                continue

            match_count += 1
            row = self.plantlist[row_idx[0]]

            try:
                nbands = int(row[idx_bands])
                q_coeffs = [float(x) for x in row[idx_q].split(';')]
                p_coeffs = [float(x) for x in row[idx_p].split(';')]
                eff = float(row[idx_eff])
            except ValueError as e:
                print(f"Error parsing data for unit {uid}: {e}")
                continue

            # Original Column Data
            col_idx = self.avai_cols[uid]
            avai_col = self.Avai[:, col_idx:col_idx + 1]
            cost_col = self.Costs[:, col_idx:col_idx + 1]

            # Create Bands
            for b in range(nbands):
                # Quantity
                band_avai = avai_col * q_coeffs[b]
                new_avai_list.append(band_avai)

                # Cost
                safe_eff = eff if eff > 0 else 1.0
                band_cost = (cost_col / safe_eff) * p_coeffs[b]
                new_costs_list.append(np.round(band_cost, 2))

                new_names_list.append(f"{uid}_{b + 1:02d}")

        print(f"DEBUG: Matched {match_count} units out of {len(sorted_unit_keys)} in Availability file.")

        if new_avai_list:
            self.Avai = np.hstack(new_avai_list)
            self.Costs = np.hstack(new_costs_list)
            self.avai_cols = {name: i for i, name in enumerate(new_names_list)}
        else:
            print("CRITICAL WARNING: No units matched! Check your IDs.")

    def build_merit_order(self):
        """Constructs the Q and MC sorted stacks and solves for unconstrained regional prices (R)."""
        print("\n--- 2. BUILDING REGIONAL STACKS ---")

        # Helper Indices
        wind_idx_map = dict(
            zip(np.char.upper(self.loader.load_numpy(self.loader.config['Wind'])[0, 1:]), range(self.Wind.shape[1])))
        solar_idx_map = dict(
            zip(np.char.upper(self.loader.load_numpy(self.loader.config['Solar'])[0, 1:]), range(self.Solar.shape[1])))

        idx_reg = self.plant_cols['REGION']
        idx_id = self.plant_cols['ID_EN']

        max_price_idx = self.prices_cols.get('MAX', 0)
        MAX_PRICE_VEC = self.Prices[:, [max_price_idx]] + 15000

        scenarios = self.loader.config.get('sensitivity', [0])

        for reg in self.REGIONS:
            print(f"\nProcessing Region: {reg}")
            self.R[reg] = {}

            # --- STEP 1: Identify Units ---
            reg_units_raw = self.plantlist[self.plantlist[:, idx_reg] == reg][:, idx_id]

            reg_indices = []
            reg_names = []

            for col_name, col_idx in self.avai_cols.items():
                base_name = col_name.rsplit('_', 1)[0]
                if base_name in reg_units_raw:
                    reg_indices.append(col_idx)
                    reg_names.append(col_name)

            # --- STEP 2: Extract Data ---
            if reg_indices:
                Avai_reg = self.Avai[:, reg_indices]
                Costs_reg = self.Costs[:, reg_indices]
                Names_base = reg_names
            else:
                print(f"  > Warning: No thermal units found.")
                Avai_reg = np.zeros((self.Avai.shape[0], 0))
                Costs_reg = np.zeros((self.Avai.shape[0], 0))
                Names_base = []

            # --- STEP 3: Add Renewables & Slack ---
            w_col = wind_idx_map.get(reg)
            s_col = solar_idx_map.get(reg)

            wind_reg = self.Wind[:, [w_col]] if w_col is not None else np.zeros((self.Avai.shape[0], 1))
            solar_reg = self.Solar[:, [s_col]] if s_col is not None else np.zeros((self.Avai.shape[0], 1))
            max_reg_cap = np.full_like(wind_reg, 10 ** 6)

            # Concatenate
            AvaiPlus = np.concatenate((Avai_reg, wind_reg, solar_reg, max_reg_cap), axis=1)

            # Costs
            wind_price = np.full_like(wind_reg, -2.0)
            solar_price = np.full_like(solar_reg, -3.0)
            CostsPlus = np.concatenate((Costs_reg, wind_price, solar_price, MAX_PRICE_VEC), axis=1)

            print(f"  > Final Stack:   {AvaiPlus.shape} (Includes Wind/Solar/Slack)")

            NamesPlus = np.array(Names_base + ['WIND', 'SOLAR', 'LOAD_SHEDDING'])
            NamesPlusMatrix = np.tile(NamesPlus, (self.Avai.shape[0], 1))

            # --- STEP 4: Sort (Merit Order) ---
            sort_indices = np.argsort(CostsPlus, axis=1)

            self.MC_sorted[reg] = np.take_along_axis(CostsPlus, sort_indices, axis=1)
            self.Q_sorted[reg] = np.take_along_axis(AvaiPlus, sort_indices, axis=1)
            self.MAP[reg] = np.take_along_axis(NamesPlusMatrix, sort_indices, axis=1)

            # --- STEP 5: Solve Unconstrained (R) ---
            Q_cumu = np.cumsum(self.Q_sorted[reg], axis=1)
            D_reg = self.Demand[:, [self.demand_cols[reg]]]
            I_flow = np.zeros((self.Avai.shape[0], 1))

            # Prepare DataFrame for Prices
            df_prices_iso = pd.DataFrame(index=self.datetime_index)

            for scen in scenarios:
                Target_Load = (D_reg - I_flow) * 1.035 + scen
                DEL = Q_cumu - Target_Load
                intersection_indices = np.argmax(DEL > 0, axis=1).reshape(-1, 1)
                # Extract Price
                price_vec = np.take_along_axis(self.MC_sorted[reg], intersection_indices, axis=1)
                self.R[reg][scen] = price_vec

                # Add to Export DataFrame
                col_name = f"Price_{reg}_Scen{scen}"
                df_prices_iso[col_name] = price_vec.flatten()

            # --- STEP 6: EXPORT RESULTS ---
            price_file = self.loader.path / f"PRICES_ISOLATED_{reg}_{self.run_name}.csv"
            df_prices_iso.to_csv(price_file)
            print(f"  > Exported isolated prices to {price_file.name}")

        print("\n--- REGIONAL STACKS BUILT AND EXPORTED ---")

    # =========================================================================
    # LP FORMULATION
    # =========================================================================

    def _get_lp_vectors(self, i):
        """Constructs c, A_eq, b_eq, and bounds for a specific time step i."""
        if self.Interco['Import']['data'] is None:
            nb_lines = 10
            lines_keys = [f"Line_{k}" for k in range(nb_lines)]
            imp_data = np.zeros((len(self.datetime_index), nb_lines))
            exp_data = np.zeros((len(self.datetime_index), nb_lines))
            line_indices = {k: k for k in lines_keys}
        else:
            lines_keys = list(self.Interco['Import']['index'].keys())
            nb_lines = len(lines_keys)
            imp_data = self.Interco['Import']['data']
            exp_data = self.Interco['Export']['data']
            line_indices = self.Interco['Import']['index']
            exp_indices = self.Interco['Export']['index']

        # 1. GENERATION VARIABLES
        QQQ_list = []
        PPP_list = []
        Names_list = []
        Region_list = []  # NEW: Track region for every variable

        for reg in self.REGIONS:
            # Quantity from Q_sorted
            QQQ_list.append(self.Q_sorted[reg][i, :-1])

            # Price from MC_sorted
            PPP_list.append(self.MC_sorted[reg][i, :-1])

            # Names directly from MAP
            Names_list.append(self.MAP[reg][i, :-1])

            # Create vector of region names ["HOK", "HOK"...]
            Region_list.append(np.full(len(self.Q_sorted[reg][i, :-1]), reg))

        QQQ = np.hstack(QQQ_list)
        PPP = np.hstack(PPP_list)
        unit_names = np.hstack(Names_list)
        unit_regions = np.hstack(Region_list)  # NEW

        # 2. TRANSMISSION VARIABLES
        QQQ = np.hstack((QQQ, np.ones(nb_lines)))
        PPP = np.hstack((PPP, np.zeros(nb_lines)))
        unit_names = np.hstack((unit_names, np.array(lines_keys)))
        # Mark lines as "INTERCO" region
        unit_regions = np.hstack((unit_regions, np.full(nb_lines, "INTERCO")))

        # 3. OBJECTIVE FUNCTION (c)
        obj = PPP * QQQ

        # 4. EQUALITY CONSTRAINTS (A_eq) -> Supply = Demand
        total_gen_vars = len(obj) - nb_lines
        lhs_gen = np.zeros((len(self.REGIONS), total_gen_vars))

        current_col = 0
        for r_idx, reg in enumerate(self.REGIONS):
            q_reg = self.Q_sorted[reg][i, :-1]
            n_units = len(q_reg)
            lhs_gen[r_idx, current_col: current_col + n_units] = q_reg
            current_col += n_units

        lhs_flow = self.BINARY_MATRIX
        lhs_eq = np.hstack((lhs_gen, lhs_flow))

        # 5. RHS (Demand)
        rhs_eq = np.array([self.Demand[i, self.demand_cols[reg]] * 1.035 for reg in self.REGIONS])

        # 6. BOUNDS
        bnd_gen = [(0, 1) for _ in range(total_gen_vars)]
        bnd_flow = []
        for line in lines_keys:
            if self.Interco['Import']['data'] is None:
                bnd_flow.append((-0, 0))
            else:
                imp_idx = line_indices[line]
                exp_idx = exp_indices[line]
                imp_lim = imp_data[i, imp_idx]
                exp_lim = exp_data[i, exp_idx]
                bnd_flow.append((-imp_lim, exp_lim))

        bnd = bnd_gen + bnd_flow

        return obj, lhs_eq, rhs_eq, bnd, unit_names, PPP, unit_regions

    # =========================================================================
    # SOLVER LOOP
    # =========================================================================
    def run_optimization(self):
        """Iterates through all time steps, solves LP, and identifies Price Setters & Congestion."""
        print("Starting Linear Programming Optimization...")

        T = len(self.datetime_index)

        # Setup Lines
        if self.Interco['Import']['index']:
            line_names = list(self.Interco['Import']['index'].keys())
            nb_lines = len(line_names)
            line_map_imp = self.Interco['Import']['index']
            line_map_exp = self.Interco['Export']['index']
            imp_data_matrix = self.Interco['Import']['data']
            exp_data_matrix = self.Interco['Export']['data']
        else:
            nb_lines = 10
            line_names = [f"Line_{k}" for k in range(nb_lines)]
            imp_data_matrix = np.zeros((T, nb_lines))
            exp_data_matrix = np.zeros((T, nb_lines))
            line_map_imp = {n: i for i, n in enumerate(line_names)}
            line_map_exp = {n: i for i, n in enumerate(line_names)}

        # Results Arrays
        prices_all = np.zeros((T, len(self.REGIONS)))
        flows_all = np.zeros((T, nb_lines))
        setters_all = np.empty((T, len(self.REGIONS)), dtype=object)

        failure_count = 0

        # --- SOLVER LOOP ---
        for i in range(T):
            if i % 500 == 0: print(f"Solving timestep {i}/{T}")

            try:
                obj, lhs_eq, rhs_eq, bnd, unit_names, PPP_vector, unit_regions_vec = self._get_lp_vectors(i)

                if np.isnan(obj).any(): raise ValueError("NaNs in Objective Function")

                opt = linprog(c=obj, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method="highs")

                if opt.success:
                    lmps = opt.eqlin.marginals
                    prices_all[i, :] = lmps
                    flows_all[i, :] = opt.x[-nb_lines:]

                    # Marginal Unit Identification
                    gen_costs = PPP_vector[:-nb_lines]
                    gen_names = unit_names[:-nb_lines]
                    gen_regions = unit_regions_vec[:-nb_lines]

                    lmps_rounded = np.round(lmps, 2)

                    for r_idx, reg in enumerate(self.REGIONS):
                        current_lmp = lmps_rounded[r_idx]
                        coupled_mask = (lmps_rounded == current_lmp)
                        coupled_regions_list = [self.REGIONS[k] for k in range(len(self.REGIONS)) if coupled_mask[k]]
                        valid_units_mask = np.isin(gen_regions, coupled_regions_list)
                        relevant_costs = gen_costs[valid_units_mask]
                        relevant_names = gen_names[valid_units_mask]
                        matches = np.where(np.abs(relevant_costs - lmps[r_idx]) < 0.1)[0]

                        if len(matches) > 0:
                            setter_name = relevant_names[matches[0]]
                            if "_" in setter_name and setter_name[-2:].isdigit():
                                setter_name = setter_name.rsplit('_', 1)[0]
                            setters_all[i, r_idx] = setter_name
                        else:
                            if lmps[r_idx] > 1000:
                                setters_all[i, r_idx] = "LOAD_SHEDDING"
                            elif lmps[r_idx] < -1.0:
                                setters_all[i, r_idx] = "RE_CURTAILMENT"
                            else:
                                setters_all[i, r_idx] = "CONGESTION/MIX"
                else:
                    failure_count += 1
                    prices_all[i, :] = 15000.0
                    setters_all[i, :] = "INFEASIBLE"

            except Exception as e:
                print(f"Error at step {i}: {e}")
                prices_all[i, :] = np.nan
                setters_all[i, :] = "ERROR"

        # --- DATAFRAME CREATION ---
        self.results_prices = pd.DataFrame(prices_all, index=self.datetime_index, columns=self.REGIONS).round(2)
        self.results_flows = pd.DataFrame(flows_all, index=self.datetime_index, columns=line_names).round()
        self.results_setters = pd.DataFrame(setters_all, index=self.datetime_index, columns=self.REGIONS)

        # --- CONGESTION CALCULATION ---
        print("Calculating congestion status...")
        congestion_matrix = np.zeros((T, nb_lines), dtype=int)
        TOL = 1e-3

        for idx, line in enumerate(line_names):
            flow_ts = flows_all[:, idx]
            limit_imp_idx = line_map_imp[line]
            limit_exp_idx = line_map_exp[line]
            limit_imp = imp_data_matrix[:, limit_imp_idx]
            limit_exp = exp_data_matrix[:, limit_exp_idx]

            is_maxed_exp = np.abs(flow_ts - limit_exp) < TOL
            is_maxed_imp = np.abs(flow_ts - (-limit_imp)) < TOL
            congestion_matrix[:, idx] = (is_maxed_exp | is_maxed_imp).astype(int)

        self.results_congestion = pd.DataFrame(congestion_matrix, index=self.datetime_index, columns=line_names).round()

        print(f"Optimization Complete. Failures: {failure_count}")
        return self.results_prices

    def calculate_physical_connectivity(self):
        """
        Determines physically coupled 'Super Groups' based on unconstrained lines.
        Returns a DataFrame where columns=Regions, Values=Group Name (e.g. 'HOK-TOH').
        """
        print("Calculating physical connectivity (Super Groups)...")

        if self.results_congestion is None:
            print("Error: Run optimization first to generate congestion results.")
            return None

        line_connections = {}
        for line_idx in range(self.BINARY_MATRIX.shape[1]):
            connected_regions = np.where(self.BINARY_MATRIX[:, line_idx] != 0)[0]
            if len(connected_regions) == 2:
                line_connections[line_idx] = (connected_regions[0], connected_regions[1])

        coupling_data = []
        congestion_values = self.results_congestion.values

        for t in range(len(self.datetime_index)):
            adj = {i: [] for i in range(len(self.REGIONS))}
            for line_idx, (r_a, r_b) in line_connections.items():
                is_congested = congestion_values[t, line_idx]
                if is_congested == 0:
                    adj[r_a].append(r_b)
                    adj[r_b].append(r_a)

            visited = [False] * len(self.REGIONS)
            row_map = {}

            for i in range(len(self.REGIONS)):
                if not visited[i]:
                    component = []
                    stack = [i]
                    visited[i] = True
                    while stack:
                        node = stack.pop()
                        component.append(self.REGIONS[node])
                        for neighbor in adj[node]:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                stack.append(neighbor)
                    group_name = "-".join(sorted(component))
                    for reg_name in component:
                        row_map[reg_name] = group_name

            coupling_data.append(row_map)

        self.results_physical_groups = pd.DataFrame(coupling_data, index=self.datetime_index)
        self.results_physical_groups = self.results_physical_groups[self.REGIONS]
        return self.results_physical_groups


class ResultVisualizer:
    """Handles plotting and exporting."""

    def __init__(self, output_path):
        self.path = Path(output_path)

    def export_csv(self, df, prefix, run_name):
        if df is None: return
        ts = datetime.now().strftime("%y%m%d%H%M")
        filename = f"{prefix}_{run_name}_{ts}.csv"
        df.to_csv(self.path / filename)
        print(f"Exported {filename}")

    def export_daily_split(self, df, folder_suffix, filename_prefix, run_name):
        """
        Generic method to split a DataFrame by day and save to individual files.
        """
        if df is None: return
        folder_name = f"{folder_suffix}_{run_name}"
        save_path = self.path / folder_name
        save_path.mkdir(exist_ok=True)
        print(f"Exporting daily files to: {folder_name}...")
        count = 0
        for date_key, group in df.groupby(df.index.date):
            date_str = date_key.strftime("%Y-%m-%d")
            filename = f"{filename_prefix}_{run_name}_{date_str}.csv"
            group.to_csv(save_path / filename)
            count += 1
        print(f"  > Saved {count} daily files.")


# ==========================================
# Execution Block
# ==========================================

if __name__ == "__main__":
    # --- USER SETTINGS ---
    # Update this path to your local data folder
    BASE_PATH = Path(r'')
    YEAR = 2025
    RUN_NAME = 'JP'

    # Setup Dictionary
    files_config = {
        'run_name': RUN_NAME,
        'Path': BASE_PATH,
        'step': 24,
        'Demand': f'Demand_{YEAR}.csv',
        'Solar': f'Solar_{YEAR}.csv',
        'Wind': f'Wind_{YEAR}.csv',
        'Plantlist': 'plantlist_JP.csv',
        'Availability': f'Availability_{YEAR}.csv',
        'Prices': f'Prices_{YEAR}_2025-04-17.csv',
        'Transmission_Import': f'Interconnectors_Import_{YEAR}.csv',
        'Transmission_Export': f'Interconnectors_Export_{YEAR}.csv'
    }

    # 1. Initialize
    model = PowerMarketModel(files_config)

    # 2. Process Inputs
    model.process_plant_parameters()

    # 3. Build Stacks & Regional Prices
    model.build_merit_order()

    # 4. Run Coupled Solver
    model.run_optimization()

    # 5. Calculate Physical Connectivity (The new DFS Logic)
    model.calculate_physical_connectivity()

    # 6. Export Results
    viz = ResultVisualizer(BASE_PATH)

    # Standard Exports (Prices, Flows)
    if model.results_prices is not None:
        viz.export_csv(model.results_prices, "MODEL_PRICES", RUN_NAME)
    if model.results_flows is not None:
        viz.export_csv(model.results_flows, "MODEL_FLOWS", RUN_NAME)
    if hasattr(model, 'results_congestion'):
        viz.export_csv(model.results_congestion, "MODEL_CONGESTION", RUN_NAME)

    # --- DAILY SPLIT EXPORTS ---

    # 1. Physical Coupled Groups (The DFS Result)
    if hasattr(model, 'results_physical_groups'):
        viz.export_daily_split(
            model.results_physical_groups,
            folder_suffix="GROUPS_PHYSICAL",
            filename_prefix="GROUPS",
            run_name=RUN_NAME
        )

    # 2. Setters (Marginal Units)
    if hasattr(model, 'results_setters'):
        viz.export_daily_split(
            model.results_setters,
            folder_suffix="SETTERS_DAILY",
            filename_prefix="SETTERS",
            run_name=RUN_NAME
        )

    print("Run Complete. Results exported.")