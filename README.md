# ⚡ Japan Power Market Optimization Model

## 🎯 Overview

This repository contains a robust Python-based optimization model for the **Japanese wholesale electricity market (JEPX)**. 

The model uses **Linear Programming (LP)** to simulate day-ahead market clearing across all 9 regional grids in Japan (HOK, TOH, TOK, CHU, HKU, KAN, CHK, SHI, KYU). It accounts for:

* **Merit Order Dispatch:** Optimizes generation from Thermal (Coal, Gas, Oil), Nuclear, Hydro, and Renewables based on marginal costs.
* **Market Coupling:** Explicitly models interconnector transmission limits between regions to solve for coupled or split market prices.
* **Congestion Analysis:** Identifies congested lines and "super groups" (physically coupled price zones).
* **Marginal Price Setters:** Tracks exactly which generation unit sets the clearing price in each region for every hour.

## ⚙️ Key Features

* **Object-Oriented Design:** Modular `PowerMarketModel` class for easy extension.
* **Vectorized Data Processing:** Fast processing of 8,760 hourly time steps using NumPy/Pandas.
* **Robust Solver:** Uses `scipy.optimize.linprog` (Highs method) with automatic failure handling and fallback logic.
* **Detailed Analytics:** Exports granular data:
    * `MODEL_PRICES`: Locational Marginal Prices (LMP) per region.
    * `MODEL_FLOWS`: Physical power flows on interconnectors.
    * `MODEL_CONGESTION`: Binary flags indicating constrained lines.
    * `SETTERS`: Daily files identifying the price-setting unit.
    * `GROUPS`: Daily files visualizing physically coupled price zones.

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Dependencies listed in `requirements.txt`

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/japan-power-model.git](https://github.com/your-username/japan-power-model.git)
    cd japan-power-model
    ```

2.  Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Data Requirements

The model expects the following CSV files in your data directory (dummy files provided in `example_data/`):

* `Demand_YYYY.csv`: Hourly demand profiles.
* `Solar_YYYY.csv`: Hourly solar generation profiles.
* `Wind_YYYY.csv`: Hourly wind generation profiles.
* `Availability_YYYY.csv`: Hourly availability factors for thermal units.
* `Prices_YYYY_Date.csv`: Daily fuel/carbon prices.
* `Plantlist_JP.csv`: Master list of generation units (Capacity, Efficiency, Fuel Hub).
* `Interconnectors_Import/Export_YYYY.csv`: Transmission limits.

### Running the Model

1.  Open `market_model.py`.
2.  Update the `BASE_PATH` variable in the `if __name__ == "__main__":` block to point to your data folder (e.g., `example_data`).
3.  Run the script:
    ```bash
    python market_model.py
    ```

## 📊 Output

Results are saved to your data folder:

* **CSVs:** `MODEL_PRICES_JP_...csv`, `MODEL_FLOWS_JP_...csv`, etc.
* **Daily Folders:** `SETTERS_DAILY_JP` and `GROUPS_PHYSICAL_JP` containing daily breakdowns.

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.