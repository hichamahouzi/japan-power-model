# ⚡ GridVec

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**GridVec** is a high-performance, vectorized solver for nodal power markets.

Originally built for the **Japanese (JEPX)** grid, it uses **NumPy vectorization** and **Linear Programming** to solve year-long economic dispatch problems significantly faster than traditional iterative Unit Commitment tools.

## 🚀 Why GridVec?

Commercial tools like **PLEXOS** or **Aurora** are powerful but often computationally heavy, expensive, and slow for long-term scenario analysis. This solver bridges the gap:

* **⚡ Blazing Fast:** Solves unconstrained market clearing for 8,760 hours effectively instantaneously using matrix vectorization.
* **🔗 Coupled Optimization:** Solves transmission-constrained LMPs (Locational Marginal Prices) using `scipy.highs` with robust handling of congestion.
* **📉 Transparent Logic:** Pure Python/Pandas codebase—no black-box algorithms. Easy to audit, modify, and integrate into ML pipelines.
* **🌏 Topology Agnostic:** While configured for Japan (9 regions), the **Binary Topology Matrix** design makes it trivial to adapt for:
    * **Australia (NEM):** 5 regions (QLD, NSW, VIC, SA, TAS).
    * **New Zealand (NZEM):** North/South Island nodes.
    * **Europe:** Cross-border zonal flows.

## ⚙️ Core Methodology

### 1. Vectorized Pre-Solving
Unlike standard loops that calculate intersections hour-by-hour, this model builds 3D Supply Stacks (Time × Units × Price) and solves the **Unconstrained Market Intersection** using vectorized NumPy operations.


* *Result:* Immediate calculation of isolated regional prices for all 8,760 hours in sub-second time.

### 2. Linear Programming (LP) for Coupling
For coupled simulations, the model formulates a dispatch optimization problem:
$$\text{Minimize } C = \sum_{g, t} (MC_{g} \cdot P_{g,t})$$
Subject to:
* **Nodal Balance:** Demand must be met at every node.
* **Transmission Limits:** Interconnector flows constrained by dynamic NTC (Net Transfer Capacity).
* **Unit Bounds:** Generation limited by available capacity (thermal/hydro/VRE).

## 📊 Key Output Metrics

The model automatically generates granular CSVs and visualization charts:

* **LMP Prices:** Nodal prices for every region.
* **Physical Flows:** Net flow on every interconnector.
* **Congestion Flags:** Binary indicators for constrained lines.
* **Price Setters:** Identifies exactly which unit (Nuclear, Coal, Gas, Load Shedding) sets the marginal price per hour.
* **Coupled Groups:** Daily reports identifying physically coupled price zones (e.g., "East Japan vs West Japan").

## 🛠️ Getting Started

### Prerequisites
* Python 3.8+
* `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

### Installation

```bash
# Clone the repository
git clone [https://github.com/hichamahouzi/GridVec.git](https://github.com/hichamahouzi/GridVec.git)
cd GridVec

# Install dependencies
pip install -r requirements.txt