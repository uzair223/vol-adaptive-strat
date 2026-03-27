import os
from src.config import Config
from src.broker import AlpacaBroker
from src.data import YahooDataProvider
from src.trader import Trader
from src.util import setup_logging

def main(config_path: str = "config.yaml") -> None:
    setup_logging()
    
    config = Config.from_yaml(config_path)
    api_key, secret_key = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("Missing Alpaca API credentials")
    
    broker = AlpacaBroker(api_key, secret_key, paper=config.trading.paper_trading)
    data_provider = YahooDataProvider()
    trader = Trader(config, broker, data_provider)
    trader.run()

if __name__ == "__main__":
    main()
