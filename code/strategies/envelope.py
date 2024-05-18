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
        self.good_to_trade = True
        self.position_was_closed = False
        self.n_bands_hit = 0

    # --- Trade Mode ---
    def set_trade_mode(self):
        self.params.setdefault("mode", "both")

        valid_modes = ("long", "short", "both")
        if self.params["mode"] not in valid_modes:
            raise ValueError(f"Wrong strategy mode. Can either be {', '.join(valid_modes)}.")

        self.ignore_shorts = self.params["mode"] == "long"
        self.ignore_longs = self.params["mode"] == "short"

        if not self.ignore_longs:
            self.populate_long_signals()
        if not self.ignore_shorts:
            self.populate_short_signals()

    # --- Indicators ---
    def populate_indicators(self):
        # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html
        if "DCM" == self.params["average_type"]:
            ta_obj = ta.volatility.DonchianChannel(self.data["high"], self.data["low"], self.data["close"], window=self.params["average_period"])
            self.data["average"] = ta_obj.donchian_channel_mband().shift(1)
        elif "SMA" == self.params["average_type"]:
            self.data["average"] = ta.trend.sma_indicator(self.data["close"], window=self.params["average_period"]).shift(1)
        elif "EMA" == self.params["average_type"]:
            self.data["average"] = ta.trend.ema_indicator(self.data["close"], window=self.params["average_period"]).shift(1)
        elif "WMA" == self.params["average_type"]:
            self.data["average"] = ta.trend.wma_indicator(self.data["close"], window=self.params["average_period"]).shift(1)
        else:
            raise ValueError(f"The average type {self.params["average_type"]} is not supported")

        for i, e in enumerate(self.params["envelopes"]):
            self.data[f"band_high_{i + 1}"] = self.data["average"] / (1 - e)
            self.data[f"band_low_{i + 1}"] = self.data["average"] * (1 - e)

    # --- Long Rules ---
    def populate_long_signals(self):
        self.data["close_long"] = self.data["high"] >= self.data["average"]
        for i in range(len(self.params["envelopes"])):
            self.data[f"open_long_{i + 1}"] = self.data["low"] <= self.data[f"band_low_{i + 1}"]

    def calculate_long_sl_price(self, avg_open_price):
        return avg_open_price * (1 - self.params["stop_loss_pct"])

    # --- Short Rules ---
    def populate_short_signals(self):
        self.data["close_short"] = self.data["low"] <= self.data["average"]
        for i in range(len(self.params["envelopes"])):
            self.data[f"open_short_{i + 1}"] = self.data["high"] >= self.data[f"band_high_{i + 1}"]

    def calculate_short_sl_price(self, avg_open_price):
        return avg_open_price * (1 + self.params["stop_loss_pct"])

    # --- Positions ---
    def evaluate_orders(self, time, row):
        self.position_was_closed = False
        if not self.good_to_trade:
            if self.last_position_side == 'long' and row["close"] > row["average"]:
                self.good_to_trade = True
            elif self.last_position_side == 'short' and row["close"] < row["average"]:
                self.good_to_trade = True

        if self.position.side == "long":
            if "price_jump_pct" in self.params and row['open'] <= self.position.open_price * (1 - self.params['price_jump_pct']):
                self.close_trade(time, row['open'], "CA long")
                self.good_to_trade = False
                self.n_bands_hit = 0
                
            elif self.position.check_for_sl(row):
                self.close_trade(time, self.position.sl_price, "SL long")
                self.good_to_trade = False
                self.n_bands_hit = 0
                
            elif self.position.check_for_liquidation(row):
                print(f"Your long was liquidated on the {time} (price = {self.position.liquidation_price})")
                sys.exit()

            elif row["close_long"]:
                self.close_trade(time,  row["average"], "Exit long")
                self.position_was_closed = True
                self.n_bands_hit = 0

        elif self.position.side == "short":
            if "price_jump_pct" in self.params and row['open'] >= self.position.open_price * (1 + self.params['price_jump_pct']):
                self.close_trade(time, row['open'], "CA short")
                self.good_to_trade = False
                self.n_bands_hit = 0
                
            elif self.position.check_for_sl(row):
                self.close_trade(time, self.position.sl_price, "SL short")
                self.good_to_trade = False
                self.n_bands_hit = 0
                
            elif self.position.check_for_liquidation(row):
                print(f"Your short was liquidated on the {time} (price = {self.position.liquidation_price})")
                sys.exit()

            elif row["close_short"]:
                self.close_trade(time, row["average"], "Exit short")
                self.position_was_closed = True
                self.n_bands_hit = 0

        if self.good_to_trade and not self.position_was_closed:
            balance = self.balance
            for i in range(self.n_bands_hit, len(self.params["envelopes"])):
                if self.position.side != "short" and row[f"open_long_{i + 1}"]:
                    side = "long"
                    price_key = f"band_low_{i + 1}"
                    sl_price_calc = self.calculate_long_sl_price
                elif self.position.side != "long" and row[f"open_short_{i + 1}"]:
                    side = "short"
                    price_key = f"band_high_{i + 1}"
                    sl_price_calc = self.calculate_short_sl_price
                else:
                    continue

                self.last_position_side = side
                price = row[price_key]

                if 'position_size_percentage' in self.params:  # total wallet fraction position size
                    initial_margin = balance * round(self.params['position_size_percentage'] / 100 / len(self.params["envelopes"]), 4)

                elif 'position_size_fixed_amount' in self.params:  # fixed amount position size
                    initial_margin = round(self.params['position_size_fixed_amount'] / len(self.params["envelopes"]), 4)

                self.balance -= initial_margin
                self.n_bands_hit += 1

                if i == 0:
                    self.position.open(
                        time,
                        side,
                        initial_margin,
                        price,
                        f"Open {side} {i + 1}",
                        sl_price=sl_price_calc(price),
                    )
                else:
                    self.position.add(initial_margin, price, f"Open {side} {i + 1}")
                    self.position.sl_price = sl_price_calc(self.position.open_price)


    def close_trade(self, time, price, reason):
        self.position.close(time, price, reason)
        open_balance = self.balance
        self.balance += self.position.initial_margin + self.position.net_pnl
        trade_info = self.position.info()
        trade_info["open_balance"] = open_balance
        trade_info["close_balance"] = self.balance
        del trade_info["tp_price"]
        self.trades_info.append(trade_info)

    # --- Backtest ---
    def run_backtest(self, initial_balance, leverage, open_fee_rate, close_fee_rate):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = ut.Position(leverage=leverage, open_fee_rate=open_fee_rate, close_fee_rate=close_fee_rate)
        self.equity_update_interval = pd.Timedelta(hours=6)

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
        self.equity_record.to_csv(path+'_equity_record.csv', header=True, index=True)

    def save_trades_info(self, path):
        self.trades_info.to_csv(path+'_trades_info.csv', header=True, index=True)
