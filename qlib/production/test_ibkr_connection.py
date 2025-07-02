# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.production.config import ProductionConfig
from qlib.production.broker import create_broker_connector, Order, OrderType, OrderSide

def main():
    # Load config (edit the path as needed)
    config = ProductionConfig('path/to/your_config.yaml')
    broker = create_broker_connector(config.broker)

    print("Connecting to Interactive Brokers...")
    if not broker.connect():
        print("Failed to connect to broker.")
        return

    print("Connected!")
    print("Fetching account info...")
    account = broker.get_account_info()
    print("Account info:", account)

    print("Fetching positions...")
    positions = broker.get_positions()
    print("Positions:", positions)

    # --- Place a test order (UNCOMMENT TO USE) ---
    # order = Order(
    #     symbol="AAPL",
    #     quantity=1,
    #     side=OrderSide.BUY,
    #     order_type=OrderType.MARKET
    # )
    # order_id = broker.place_order(order)
    # print(f"Placed order with ID: {order_id}")
    # status = broker.get_order_status(order_id)
    # print(f"Order status: {status}")

    broker.disconnect()
    print("Disconnected.")

if __name__ == "__main__":
    main() 