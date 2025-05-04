## 📊 Strategy Summary

The strategy is based on a set of traditional momentum factors, calculated over short-term windows from high-frequency order book data:

### 🔍 Selected Factors:
- **VWAP Momentum**
- **Price Momentum**
- **Order Imbalance**
- **Order Acceleration**

These are standardized (Z-score) and summed to generate a trading signal. A **long** or **short** position is triggered when the signal exceeds a threshold.

---

## 🧠 Threshold Design

### 1. Fixed Threshold (Baseline)
- A simple upper/lower bound.
- Weak performance on low-volatility days.
- Risk of overtrading on volatile days.

### 2. Dynamic Threshold via PPO (Advanced)
- Uses **Proximal Policy Optimization** to dynamically adjust the trading threshold.
- Inputs: Statistical summaries (max, mean, std, etc.) of the last 5000-tick chunk.
- Target: Maximize cumulative return over that chunk.
- Reinforcement Learning Objective: Learn an adaptive thresholding strategy to capture persistent patterns.

---

## 🧪 Backtest Settings

- **Training**: 3 days of high-frequency BTC data
- **Testing**: 2 additional days
- **Chunk size**: 5000 ticks (~4-5 minutes)
- **Evaluation**: Strategy return before/after threshold optimization

---

## 📁 Project Structure

simple-momentum-driven-btc-trading/
├── real_run.py # 🚀 Auto-trading main script (fixed threshold)
├── train_ppo.py # PPO training for dynamic thresholding
├── strategy/
│ └── signal_generator.py # Signal computation and threshold logic
├── features/
│ └── factors.py # Momentum and OFI factor calculation
├── data/
│ └── *.parquet # BTC tick-level and feature data
├── report.pdf # 📄 Strategy documentation
└── README.md
