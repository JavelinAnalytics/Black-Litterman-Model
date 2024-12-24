"""
Created on Wed Aug 14 00:50:51 2024

@author: a_hoj
"""
import numpy as np
import yfinance as yf
from pypfopt import expected_returns, risk_models, black_litterman, efficient_frontier

class BlackLittermanModel:
    def __init__(self, portfolio, absolute_views, relative_views):
        """
        Initialize the Black-Litterman model with portfolio and user-defined views.

        param portfolio: List of tickers in the portfolio.
        param absolute_views: List of dicts, each specifying an asset, view return, and confidence level.
        param relative_views: List of dicts, each specifying outperforming/underperforming assets, 
                               view return, and confidence level.
        """
        self.portfolio = portfolio
        self.absolute_views = absolute_views
        self.relative_views = relative_views

        self.prices = None
        self.mcaps = None
        self.market_prior = None
        self.cov_matrix = None
        self.weights = None
        self.performance = None

    def fetch_data(self):
        """Fetch historical price data and market capitalizations dynamically."""
        # Download historical prices
        self.prices = yf.download(self.portfolio, start="2020-01-01")["Close"]

        # Download market capitalizations dynamically
        self.mcaps = {}
        for ticker in self.portfolio:
            stock = yf.Ticker(ticker)
            self.mcaps[ticker] = stock.info.get("marketCap", 0)  

    def compute_pi(self):
        """Compute the implied equilibrium returns."""
        # Retrieve the 10-year U.S. Treasury yield as the risk-free rate
        us_10yr = yf.Ticker("^TNX")
        R_f = us_10yr.history(period="1d")["Close"].iloc[-1] / 100

        # Compute market capitalization weights
        total_market_cap = sum(self.mcaps.values())
        market_weights = {ticker: mcap / total_market_cap for ticker, mcap in self.mcaps.items()}
        weights_vector = np.array(list(market_weights.values()))

        # Compute the covariance matrix
        self.cov_matrix = risk_models.sample_cov(self.prices)

        # Compute expected returns for individual assets
        individual_expected_returns = expected_returns.mean_historical_return(self.prices)

        # Compute the market portfolio's expected return
        R_m = sum(weights_vector[i] * individual_expected_returns[ticker] for i, ticker in enumerate(self.portfolio))

        # Compute risk aversion coefficient/delta
        market_risk_premium = R_m - R_f
        market_variance = weights_vector.T @ self.cov_matrix.values @ weights_vector
        delta = market_risk_premium / market_variance

        # Compute implied equilibrium returns
        self.market_prior = black_litterman.market_implied_prior_returns(self.mcaps, delta, self.cov_matrix)

    def compute_weights(self):
        """Compute weights for each relative view based on mini portfolio return differential."""

        # Define the function for weighted return of mini portfolios
        def weighted_return(assets):
            total_mcap = sum(self.mcaps[asset] for asset in assets)
            weights = {asset: self.mcaps[asset] / total_mcap for asset in assets}
            return sum(weights[asset] * self.market_prior[asset] for asset in assets)

        # Iterate through relative views
        for view in self.relative_views:
            # Extract outperforming and underperforming assets
            outperforming = view["outperforming"]
            underperforming = view["underperforming"]
            view_return = view["view_return"]

            # Calculate total market caps for each group
            total_outperforming_mcap = sum(self.mcaps[asset] for asset in outperforming)
            total_underperforming_mcap = sum(self.mcaps[asset] for asset in underperforming)

            # Calculate weighted implied returns for each group
            outperforming_return = weighted_return(outperforming)
            underperforming_return = weighted_return(underperforming)

            # Compute the differential
            differential = outperforming_return - underperforming_return

            # Assign long/short positions based on comparison of view return and differential
            if view_return > differential:
                weights = {asset: self.mcaps[asset] / total_outperforming_mcap for asset in outperforming}
                weights.update({asset: -self.mcaps[asset] / total_underperforming_mcap for asset in underperforming})
            else:
                weights = {asset: -self.mcaps[asset] / total_outperforming_mcap for asset in outperforming}
                weights.update({asset: self.mcaps[asset] / total_underperforming_mcap for asset in underperforming})

            # Update view with calculated weights
            view["weights"] = weights

    def construct_matrices(self):
        """Construct the combined P matrix, Q vector, and Ω matrix."""
        P = []

        # Add rows for absolute views
        for view in self.absolute_views:
            row = [1 if asset == view["asset"] else 0 for asset in self.portfolio]
            P.append(row)

        # Add rows for relative views
        for view in self.relative_views:
            row = [0] * len(self.portfolio)
            for asset, weight in view["weights"].items():
                asset_index = self.portfolio.index(asset)
                row[asset_index] = weight
            P.append(row)

        P = np.array(P)
        Q = np.array(
            [view["view_return"] for view in self.absolute_views] + 
            [view["view_return"] for view in self.relative_views]
        )  

        # Combine confidence levels from absolute and relative views
        confidence_levels = [view["confidence"] for view in self.absolute_views] + [
            view["confidence"] for view in self.relative_views
        ]
        avg_confidence = np.mean(confidence_levels)

        # Compute omega diagonal covariance matrix (Ω)
        P_Sigma_Pt = P @ self.cov_matrix.values @ P.T
        P_Sigma_Pt = np.trace(P_Sigma_Pt)
        calibration_factor = P_Sigma_Pt / (1 / avg_confidence)
        omega = np.diag([(1 / confidence) * calibration_factor for confidence in confidence_levels])

        return P, Q, omega

    def run_model(self):
        """Run the Black-Litterman model and return results."""
        self.fetch_data()
        self.compute_pi()
        self.compute_weights()

        # Construct P, Q, and Ω matrices
        P, Q, omega = self.construct_matrices()

        # Initialize the Black-Litterman model
        bl = black_litterman.BlackLittermanModel(
            cov_matrix=self.cov_matrix,
            P=P,
            Q=Q,
            pi=self.market_prior,
            omega=omega
        )

        # Compute posterior returns
        bl_returns = bl.bl_returns()

        # Portfolio optimization
        ef = efficient_frontier.EfficientFrontier(bl_returns, self.cov_matrix)
        self.weights = ef.max_sharpe()
        self.performance = ef.portfolio_performance(verbose=False)

        return self.weights, self.performance


# Example Usage:
portfolio = ["AAPL", "AMZN", "NVDA", "TLSA", "GOOGL", "META", "MSFT"]
absolute_views = [
    {"asset": "AAPL", "view_return": 0.08, "confidence": 0.5}
]
relative_views = [
    {"outperforming": ["NVDA"], "underperforming": ["META"], "view_return": 0.02, "confidence": 0.6},
    {"outperforming": ["GOOGL", "AMZN"], "underperforming": ["TLSA", "MSFT"], "view_return": 0.01, "confidence": 0.4}
]

blm = BlackLittermanModel(portfolio, absolute_views, relative_views)
weights, performance = blm.run_model()

print("Optimized Portfolio Weights:", weights)
print("Portfolio Performance Metrics: expected annual return, annual volatility, sharpe ratio", performance)


