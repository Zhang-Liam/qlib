# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Risk management for live trading.
"""

import logging
from typing import Dict, Optional
from qlib.production.broker import Order, Position, Account
from qlib.production.config import RiskConfig
from datetime import datetime

class RiskManager:
    """
    Enforces risk limits before orders are sent to the broker.
    Checks position size, daily loss, exposure, and other constraints.
    """
    def __init__(self, risk_config: RiskConfig):
        self.risk_config = risk_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.account: Optional[Account] = None
        self.daily_loss: float = 0.0
        self.last_check_date: Optional[datetime] = None

    def update_account(self, account: Account):
        self.account = account
        self._reset_daily_loss_if_new_day(account.timestamp)

    def _reset_daily_loss_if_new_day(self, now: datetime):
        if self.last_check_date is None or now.date() != self.last_check_date.date():
            self.daily_loss = 0.0
            self.last_check_date = now

    def record_trade_pnl(self, realized_pnl: float):
        self.daily_loss += -realized_pnl  # Loss is positive if PnL is negative

    def validate_order(self, order: Order) -> bool:
        """
        Validate an order against all risk limits.
        Returns True if order is allowed, False otherwise.
        """
        if self.account is None:
            self.logger.warning("No account info available for risk checks.")
            return False

        # 1. Position size limit
        symbol = order.symbol
        qty = order.quantity
        price = order.price or self._estimate_market_price(symbol)
        order_value = qty * price
        max_pos = self.risk_config.max_position_size
        current_pos = self._get_position_value(symbol)
        if current_pos + order_value > max_pos:
            self.logger.warning(f"Order exceeds max position size for {symbol}: {current_pos + order_value} > {max_pos}")
            return False

        # 2. Daily loss limit
        if self.daily_loss > self.risk_config.max_daily_loss:
            self.logger.warning(f"Daily loss limit exceeded: {self.daily_loss} > {self.risk_config.max_daily_loss}")
            return False

        # 3. Exposure limits (single stock, sector, leverage)
        # (For now, only single stock exposure is checked)
        max_single = self.risk_config.max_single_stock_exposure
        total_equity = self.account.equity
        if total_equity > 0 and (current_pos + order_value) / total_equity > max_single:
            self.logger.warning(f"Order exceeds single stock exposure: {(current_pos + order_value) / total_equity:.2%} > {max_single:.2%}")
            return False

        # 4. Leverage
        max_leverage = self.risk_config.max_leverage
        if total_equity > 0 and (self.account.buying_power / total_equity) > max_leverage:
            self.logger.warning(f"Order exceeds leverage limit: {(self.account.buying_power / total_equity):.2f} > {max_leverage:.2f}")
            return False

        # 5. Stop-loss (not enforced pre-trade, but can be used for monitoring)
        # 6. Other custom checks can be added here

        return True

    def _get_position_value(self, symbol: str) -> float:
        if self.account is None:
            return 0.0
        for pos in self.account.positions:
            if pos.symbol == symbol:
                return pos.market_value
        return 0.0

    def _estimate_market_price(self, symbol: str) -> float:
        # Placeholder: In production, use live data provider or broker quote
        self.logger.warning(f"Estimating price for {symbol} as $100.0 (override in production)")
        return 100.0 