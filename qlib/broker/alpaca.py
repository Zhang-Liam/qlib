from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from qlib.backtest.executor import BaseExecutor
from qlib.utils import get_date_by_shift
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class AlpacaExecutor(BaseExecutor):
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        super().__init__()
        self.client = TradingClient(api_key, secret_key, paper=paper_trading)
        
    def execute(self, trade_decision):
        orders = trade_decision.order_list
        executed_orders = []
        
        for order in orders:
            try:
                # Create market order request
                order_request = MarketOrderRequest(
                    symbol=order.stock,
                    qty=order.amount,
                    side=OrderSide.BUY if order.side == 1 else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                # Place order
                result = self.client.submit_order(order_request)
                executed_orders.append(result)
                
            except Exception as e:
                logger.error(f"Failed to execute order for {order.stock}: {str(e)}")
                
        return executed_orders

    def get_position(self):
        positions = self.client.get_all_positions()
        position_dict = {}
        
        for pos in positions:
            position_dict[pos.symbol] = {
                'amount': float(pos.qty),
                'cost': float(pos.avg_entry_price),
                'value': float(pos.market_value)
            }
        
        return position_dict

    def get_account_info(self):
        account = self.client.get_account()
        return {
            'cash': float(account.cash),
            'total_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power)
        }
