"""
Enhanced Option Pricing Models
Ensemble of Black-Scholes and Binary Tree (Binomial) Models
"""

import numpy as np
import math
from scipy.stats import norm
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OptionPricingEnsemble:
    """
    Ensemble option pricing using Black-Scholes and Binary Tree models
    """
    
    def __init__(self, risk_free_rate=0.05):
        self.risk_free_rate = risk_free_rate
        self.weights = {
            'black_scholes': 0.6,  # 60% weight to Black-Scholes
            'binary_tree': 0.4     # 40% weight to Binary Tree
        }
    
    def black_scholes_price(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option price using Black-Scholes model
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiry (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        """
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                logger.error(f"Invalid parameters: S={S}, K={K}, T={T}, sigma={sigma}")
                return None
            
            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            elif option_type.lower() == 'put':
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
            return max(0, price)  # Option price can't be negative
            
        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {e}")
            return None
    
    def binary_tree_price(self, S, K, T, r, sigma, option_type='call', steps=100):
        """
        Calculate option price using Binary Tree (Binomial) model
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiry (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        steps: Number of time steps in the tree
        """
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or steps <= 0:
                logger.error(f"Invalid parameters: S={S}, K={K}, T={T}, sigma={sigma}, steps={steps}")
                return None
            
            # Calculate time step
            dt = T / steps
            
            # Calculate up and down factors
            u = math.exp(sigma * math.sqrt(dt))  # Up factor
            d = 1 / u  # Down factor
            
            # Risk-neutral probability
            p = (math.exp(r * dt) - d) / (u - d)
            
            if p < 0 or p > 1:
                logger.error(f"Invalid risk-neutral probability: {p}")
                return None
            
            # Initialize asset prices at maturity
            asset_prices = np.zeros(steps + 1)
            for i in range(steps + 1):
                asset_prices[i] = S * (u ** (steps - i)) * (d ** i)
            
            # Initialize option values at maturity
            option_values = np.zeros(steps + 1)
            for i in range(steps + 1):
                if option_type.lower() == 'call':
                    option_values[i] = max(0, asset_prices[i] - K)
                elif option_type.lower() == 'put':
                    option_values[i] = max(0, K - asset_prices[i])
                else:
                    raise ValueError("option_type must be 'call' or 'put'")
            
            # Work backwards through the tree
            for j in range(steps - 1, -1, -1):
                for i in range(j + 1):
                    # Calculate option value at current node
                    option_values[i] = math.exp(-r * dt) * (
                        p * option_values[i] + (1 - p) * option_values[i + 1]
                    )
                    
                    # For American options, check early exercise
                    # (For simplicity, we're implementing European-style here)
                    # Current asset price at this node
                    current_asset_price = S * (u ** (j - i)) * (d ** i)
                    
                    if option_type.lower() == 'call':
                        intrinsic_value = max(0, current_asset_price - K)
                    else:
                        intrinsic_value = max(0, K - current_asset_price)
                    
                    # For European options, don't check early exercise
                    # option_values[i] = max(option_values[i], intrinsic_value)
            
            return max(0, option_values[0])  # Return option price at root
            
        except Exception as e:
            logger.error(f"Error in Binary Tree calculation: {e}")
            return None
    
    def monte_carlo_price(self, S, K, T, r, sigma, option_type='call', simulations=10000):
        """
        Calculate option price using Monte Carlo simulation (additional model)
        """
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return None
            
            # Generate random paths
            np.random.seed(42)  # For reproducibility
            Z = np.random.standard_normal(simulations)
            
            # Calculate final stock prices
            ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
            
            # Calculate payoffs
            if option_type.lower() == 'call':
                payoffs = np.maximum(ST - K, 0)
            elif option_type.lower() == 'put':
                payoffs = np.maximum(K - ST, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
            # Discount back to present value
            option_price = math.exp(-r * T) * np.mean(payoffs)
            
            return max(0, option_price)
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo calculation: {e}")
            return None
    
    def ensemble_price(self, S, K, T, r, sigma, option_type='call', include_monte_carlo=False):
        """
        Calculate option price using ensemble of models
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiry (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        include_monte_carlo: Whether to include Monte Carlo in ensemble
        """
        try:
            logger.info(f"Calculating ensemble option price: S={S}, K={K}, T={T}, r={r}, sigma={sigma}, type={option_type}")
            
            # Calculate prices using different models
            bs_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            bt_price = self.binary_tree_price(S, K, T, r, sigma, option_type)
            
            prices = {}
            weights = {}
            
            if bs_price is not None:
                prices['black_scholes'] = bs_price
                weights['black_scholes'] = self.weights['black_scholes']
                logger.info(f"Black-Scholes price: {bs_price}")
            
            if bt_price is not None:
                prices['binary_tree'] = bt_price
                weights['binary_tree'] = self.weights['binary_tree']
                logger.info(f"Binary Tree price: {bt_price}")
            
            # Optional: Include Monte Carlo
            if include_monte_carlo:
                mc_price = self.monte_carlo_price(S, K, T, r, sigma, option_type)
                if mc_price is not None:
                    prices['monte_carlo'] = mc_price
                    weights['monte_carlo'] = 0.2
                    # Adjust other weights
                    weights['black_scholes'] = 0.5
                    weights['binary_tree'] = 0.3
                    logger.info(f"Monte Carlo price: {mc_price}")
            
            if not prices:
                logger.error("No valid prices calculated from any model")
                return None
            
            # Calculate weighted average
            total_weight = sum(weights.values())
            if total_weight <= 0:
                logger.error("Invalid total weight")
                return None
            
            ensemble_price = sum(prices[model] * weights[model] for model in prices) / total_weight
            
            # Calculate model agreement (how close the models are)
            if len(prices) > 1:
                price_values = list(prices.values())
                price_std = np.std(price_values)
                price_mean = np.mean(price_values)
                agreement_score = 1 - (price_std / price_mean) if price_mean > 0 else 0
                agreement_score = max(0, min(1, agreement_score))  # Clamp between 0 and 1
            else:
                agreement_score = 1.0
            
            logger.info(f"Ensemble price: {ensemble_price}, Agreement score: {agreement_score}")
            
            return {
                'ensemble_price': round(ensemble_price, 2),
                'individual_prices': {model: round(price, 2) for model, price in prices.items()},
                'model_weights': weights,
                'agreement_score': round(agreement_score, 3),
                'confidence': 'High' if agreement_score > 0.8 else 'Medium' if agreement_score > 0.6 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble pricing: {e}")
            return None
    
    def get_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option Greeks using Black-Scholes model
        """
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return None
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Delta
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            
            # Theta
            if option_type.lower() == 'call':
                theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) 
                        - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) 
                        + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            
            # Rho
            if option_type.lower() == 'call':
                rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
            
            return {
                'delta': round(delta, 4),
                'gamma': round(gamma, 4),
                'theta': round(theta, 4),
                'vega': round(vega, 4),
                'rho': round(rho, 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return None

def calculate_implied_volatility(market_price, S, K, T, r, option_type='call', max_iterations=100, tolerance=1e-6):
    """
    Calculate implied volatility using Newton-Raphson method
    """
    try:
        pricing_model = OptionPricingEnsemble(r)
        
        # Initial guess for volatility
        sigma = 0.2
        
        for i in range(max_iterations):
            # Calculate option price and vega
            price = pricing_model.black_scholes_price(S, K, T, r, sigma, option_type)
            if price is None:
                return None
            
            # Calculate vega (sensitivity to volatility)
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T)
            
            if abs(vega) < tolerance:
                break
            
            # Newton-Raphson update
            price_diff = price - market_price
            sigma_new = sigma - price_diff / vega
            
            if abs(sigma_new - sigma) < tolerance:
                return max(0.01, sigma_new)  # Minimum volatility of 1%
            
            sigma = max(0.01, sigma_new)  # Ensure positive volatility
        
        return max(0.01, sigma)
        
    except Exception as e:
        logger.error(f"Error calculating implied volatility: {e}")
        return None
