"""dcforecast: GPU-accelerated probabilistic forecasting of data-center power demand.

The package generates a physically-motivated synthetic telemetry dataset for the fictional
cloud operator *Helios Cloud*, cleans it, trains and backtests a family of forecasters
(seasonal naive, Holt-Winters, gradient boosting, LSTM and TCN) and turns the forecasts into
energy, cost and carbon decisions, including carbon-aware scheduling of deferrable compute.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
