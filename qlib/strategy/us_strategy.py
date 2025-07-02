from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import TradeDecisionWO
from qlib.backtest.executor import BaseExecutor
from qlib.data.dataset import DatasetH
from qlib.model.base import Model
from qlib.utils import get_date_by_shift
import pandas as pd
import numpy as np

class USStockStrategy(BaseStrategy):
    def __init__(
        self,
        model: Model,
        dataset: DatasetH,
        freq: str = 'day',
        topk: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.dataset = dataset
        self.freq = freq
        self.topk = topk

    def generate_trade_decision(self, execute_result=None):
        # Get current trading date
        current_date = self.common_infra.get("calendar").current_date
        
        # Get predictions
        pred_df = self.model.predict(self.dataset)
        pred_df = pred_df.loc[current_date]
        
        # Select topk stocks
        selected_stocks = pred_df.nlargest(self.topk, 'score').index
        
        # Create orders
        orders = []
        for stock in selected_stocks:
            orders.append(
                Order(
                    stock=stock,
                    amount=np.floor(self.trade_position.cash / len(selected_stocks) / pred_df.loc[stock, 'close']),
                    side=OrderSide.BUY,
                    price_type=OrderPriceType.LIMIT,
                    price=pred_df.loc[stock, 'close'] * 1.01,  # Add small margin
                )
            )
        
        return TradeDecisionWO(orders, self)

    def _reset(self, **kwargs):
        super()._reset(**kwargs)
        self.model.reset()
        self.dataset.reset()

    def get_data_cal_avail_range(self, rtype="full"):
        return self.dataset.get_available_time_range(rtype)
