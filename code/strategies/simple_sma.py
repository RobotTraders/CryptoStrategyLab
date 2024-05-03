import sys
import ta
import pandas as pd
from datetime import datetime

from . import tools as ut


class Strategy:
    def __init__(self, params, ohlcv) -> None:
        self.params = params
        self.data = ohlcv.copy()

        self.populate_indicators()
        self.set_trade_mode()

    # --- Trade Mode ---
    def set_trade_mode(self):
        self.params.setdefault('mode', 'both')

        valid_modes = ('long', 'short', 'both')
        if self.params['mode'] not in valid_modes:
            raise ValueError(f"Wrong strategy mode. Can either be {', '.join(valid_modes)}.")

        self.ignore_shorts = self.params['mode'] == 'long'
        self.ignore_longs = self.params['mode'] == 'short'

        if not self.ignore_longs:
            self.populate_long_signals()
        if not self.ignore_shorts:
            self.populate_short_signals()

    # --- Indicators ---
    def populate_indicators(self):
        # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html
        self.data['fastMA'] = ta.trend.sma_indicator(self.data['close'], self.params['fast_ma_period']).shift(1)
        self.data['slowMA'] = ta.trend.sma_indicator(self.data['close'], self.params['slow_ma_period']).shift(1)
        self.data['trend'] = ta.trend.sma_indicator(self.data['close'], self.params['trend_ma_period']).shift(1)

    # --- Long Rules ---
    def populate_long_signals(self):
        self.data['open_long'] = (
                (self.data['fastMA'] > self.data['slowMA']) &
                (self.data['fastMA'].shift(1) <= self.data['slowMA'].shift(1)) &  # "to check on previous candle"
                (self.data['close'] > self.data['trend'])
        )
        self.data['close_long'] = self.data['fastMA'] < self.data['slowMA']

    def calculate_long_tp_price(self, row):
        return row['close'] * 1.3

    def calculate_long_sl_price(self, row):
        return row['close'] * 0.85

    # --- Short Rules ---
    def populate_short_signals(self):
        self.data['open_short'] = (
                (self.data['fastMA'] < self.data['slowMA']) &
                (self.data['fastMA'].shift(1) >= self.data['slowMA'].shift(1)) &  # "to check on previous candle"
                (self.data['close'] < self.data['trend'])
        )
        self.data['close_short'] = self.data['fastMA'] > self.data['slowMA']

    def calculate_short_tp_price(self, row):
        return row['close'] * 0.7

    def calculate_short_sl_price(self, row):
        return row['close'] * 1.15

    # --- Position size ---
    def calculate_initial_margin(self, balance, price, stop_loss_price):
        if 'position_size_percentage' in self.params:  # total wallet percentage position size
            return balance * self.params['position_size_percentage'] / 100

        elif 'position_size_exposure' in self.params:  # risk adjusted (sl) position size
            amount_at_risk = balance * self.params['exposure'] / 100
            size = amount_at_risk * price / abs(price - stop_loss_price)
            return size

        elif 'position_size_fixed_amount' in self.params:  # fixed amount position size
            return self.params['position_size_fixed_amount']

    # --- Positions ---
    def evaluate_orders(self, time, row):
        if self.position.side == 'long':
            if self.position.check_for_sl(row):
                self.close_trade(time, self.position.sl_price, "SL long")

            elif self.position.check_for_liquidation(row):
                print(f'Your long was liquidated on the {time} (price = {self.position.liquidation_price})')
                sys.exit()
                
            elif self.position.check_for_tp(row):
                self.close_trade(time, self.position.tp_price, "TP long")

            elif row["close_long"]:
                self.close_trade(time,  row['close'], "Exit long")

        elif self.position.side == 'short':
            if self.position.check_for_sl(row):
                self.close_trade(time, self.position.sl_price, "SL short")
                
            elif self.position.check_for_liquidation(row):
                print(f'Your short was liquidated on the {time} (price = {self.position.liquidation_price})')
                sys.exit()

            elif self.position.check_for_tp(row):
                self.close_trade(time, self.position.tp_price, "TP short")

            elif row["close_short"]:
                self.close_trade(time, row['close'], "Exit short")

        else:
            price = row["close"]

            if not self.ignore_longs and row["open_long"]:
                sl_price = self.calculate_long_sl_price(row)
                tp_price = self.calculate_long_tp_price(row)
                initial_margin = self.calculate_initial_margin(self.balance, price, sl_price)
                self.balance -= initial_margin
                self.position.open(time, 'long', initial_margin, price, 'Open long', sl_price, tp_price)

            elif not self.ignore_shorts and row["open_short"]:
                sl_price = self.calculate_short_sl_price(row)
                tp_price = self.calculate_short_tp_price(row)
                initial_margin = self.calculate_initial_margin(self.balance, price, sl_price)
                self.balance -= initial_margin
                self.position.open(time, 'short', initial_margin, price, 'Open short', sl_price, tp_price)

    def close_trade(self, time, price, reason):
        self.position.close(time, price, reason)
        open_balance = self.balance
        self.balance += self.position.initial_margin + self.position.net_pnl
        trade_info = self.position.info()
        trade_info["open_balance"] = open_balance
        trade_info["close_balance"] = self.balance
        self.trades_info.append(trade_info)

    # --- Backtest ---
    def run_backtest(self, initial_balance=1000, leverage=1, fee_rate=0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = ut.Position(leverage=leverage, fee_rate=fee_rate)
        self.equity_update_interval = pd.Timedelta(days=1)
        self.previous_equity_update_time = datetime(1900, 1, 1)
        self.trades_info = []
        self.equity_record = []

        for time, row in self.data.iterrows():
            self.evaluate_orders(time, row)
            self.previous_equity_update_time = ut.update_equity_record(
                time,
                self.position,
                self.balance,
                row["close"],
                self.previous_equity_update_time,
                self.equity_update_interval,
                self.equity_record
            )

        self.trades_info = pd.DataFrame(self.trades_info)
        self.equity_record = pd.DataFrame(self.equity_record).set_index("time")
        self.final_equity = round(self.equity_record.iloc[-1]["equity"], 2)

    # --- Save results ---
    def save_equity_record(self, path):
        self.equity_record.to_csv(path + '_equity_record.csv', header=True, index=True)


    def save_trades_info(self, path):
        self.trades_info.to_csv(path + '_trades_info.csv', header=True, index=True)
