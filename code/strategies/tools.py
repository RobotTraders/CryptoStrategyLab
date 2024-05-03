from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


class PositionBehavior(ABC):
    @abstractmethod
    def calculate_pnl(self, position, close_price, amount):
        pass

    @abstractmethod
    def calculate_liquidation_price(self, position, price):
        pass

    @abstractmethod
    def check_for_sl(self, position, row):
        pass

    @abstractmethod
    def check_for_tp(self, position, row):
        pass

    @abstractmethod
    def check_for_liquidation(self, position, row):
        pass


class LongPositionBehavior(PositionBehavior):
    def calculate_pnl(self, position, close_price, amount):
        return amount * (close_price - position.open_price)

    def calculate_liquidation_price(self, position, price):  ### approximated computation, check exchange specifics.
        return price * (1 - 1 / position.leverage)

    def check_for_sl(self, position, row):
        return row['low'] <= position.sl_price

    def check_for_tp(self, position, row):
        return row['high'] >= position.tp_price

    def check_for_liquidation(self, position, row):
        return row['low'] <= position.liquidation_price


class ShortPositionBehavior(PositionBehavior):
    def calculate_pnl(self, position, close_price, amount):
        return amount * (position.open_price - close_price)

    def calculate_liquidation_price(self, position, price):  ### approximated computation, check exchange specifics.
        return price * (1 + 1 / position.leverage)

    def check_for_sl(self, position, row):
        return row['high'] >= position.sl_price

    def check_for_tp(self, position, row):
        return row['low'] <= position.tp_price

    def check_for_liquidation(self, position, row):
        return row['high'] >= position.liquidation_price


class Position:
    def __init__(
            self,
            leverage: Optional[int] = 1,
            fee_rate: Optional[float] = 0.001,
            open_fee_rate: Optional[float] = None,
            close_fee_rate: Optional[float] = None,
    ) -> None:
        self.leverage = leverage
        self.open_fee_rate = fee_rate if open_fee_rate is None else open_fee_rate
        self.close_fee_rate = fee_rate if close_fee_rate is None else close_fee_rate
        self.side = None
        self.open_time = None
        self.close_time = None
        self.open_price = None
        self.close_price = None
        self.open_reason = None
        self.close_reason = None
        self.open_fee = None
        self.close_fee = None
        self.initial_margin = None
        self.open_notional_value = None
        self.close_notional_value = None
        self.amount = None
        self.net_pnl = None
        self.net_pnl_pct = None
        self.sl_price = None
        self.tp_price = None
        self.liquidation_price = None
        self.behavior = None

    def _calculate_opening_metrics(self, initial_margin: float, open_price: float):
        open_notional_value = initial_margin * self.leverage
        open_fee = open_notional_value * self.open_fee_rate
        open_notional_value -= open_fee
        amount = open_notional_value / open_price

        return {
            'open_notional_value': open_notional_value,
            'open_fee': open_fee,
            'amount': amount,
        }

    def open(
            self,
            open_time: datetime,
            side: str,
            initial_margin: float,
            open_price: float,
            open_reason: str,
            sl_price: Optional[float] = None,
            tp_price: Optional[float] = None,
    ):
        self.open_time = open_time
        self.side = side
        self.initial_margin = initial_margin
        self.open_price = open_price
        self.open_reason = open_reason
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.behavior = LongPositionBehavior() if side == "long" else ShortPositionBehavior()
        metrics = self._calculate_opening_metrics(self.initial_margin, self.open_price)
        self.open_fee = metrics['open_fee']
        self.open_notional_value = metrics['open_notional_value']
        self.amount = metrics['amount']
        self.liquidation_price = self.calculate_liquidation_price(self.open_price)

    def close(self, time: datetime, price: float, reason: str) -> None:
        self.close_time = time
        self.close_price = price
        self.close_reason = reason
        pnl = self.calculate_pnl(price, self.amount)
        self.close_notional_value = self.open_notional_value + pnl
        self.close_fee = self.close_notional_value * self.close_fee_rate
        self.net_pnl = pnl - self.open_fee - self.close_fee
        self.net_pnl_pct = self.net_pnl / self.initial_margin * 100
        self.side = None

    def add(
            self,
            initial_margin: float,
            price: float,
            reason: str,
    ) -> None:
        self.open_reason = reason
        metrics = self._calculate_opening_metrics(initial_margin, price)
        self.open_price = (self.open_price * self.amount + price * metrics['amount']) / (self.amount + metrics['amount'])
        self.initial_margin += initial_margin
        self.open_fee += metrics['open_fee']
        self.open_notional_value += metrics['open_notional_value']
        self.amount += metrics['amount']
        self.liquidation_price = self.calculate_liquidation_price(self.open_price)

    def info(self):
        return {
            "open_time": self.open_time,
            "close_time": self.close_time,
            "open_reason": self.open_reason,
            "close_reason": self.close_reason,
            "open_price": self.open_price,
            "close_price": self.close_price,
            "initial_margin": self.initial_margin,
            "net_pnl": self.net_pnl,
            "net_pnl_pct": self.net_pnl_pct,
            "open_notional_value": self.open_notional_value,
            "close_notional_value": self.close_notional_value,
            "amount": self.amount,
            "open_fee": self.open_fee,
            "close_fee": self.close_fee,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "liquidation_price": self.liquidation_price,
        }

    def calculate_pnl(self, price: float, amount: float) -> float:
        return self.behavior.calculate_pnl(self, price, amount)

    def calculate_liquidation_price(self, price: float) -> float:
        return self.behavior.calculate_liquidation_price(self, price)

    def check_for_liquidation(self, row: Dict[str, float]) -> bool:
        return self.behavior.check_for_liquidation(self, row)

    def check_for_sl(self, row: Dict[str, float]) -> bool:
        return self.behavior.check_for_sl(self, row)

    def check_for_tp(self, row: Dict[str, float]) -> bool:
        return self.behavior.check_for_tp(self, row)


def update_equity_record(
    time: datetime,
    position: Optional[Position],
    balance: float,
    price: float,
    previous_equity_update_time: datetime,
    equity_update_interval: timedelta,
    equity_record: List[Dict[str, Any]]
) -> datetime:

    if time - previous_equity_update_time >= equity_update_interval:
        equity = balance
        if position.side:
            unrealized_pnl = position.calculate_pnl(price, position.amount)
            close_fee = (position.open_notional_value + unrealized_pnl) * position.close_fee_rate
            equity += position.initial_margin + unrealized_pnl - position.open_fee - close_fee

        equity_record.append({
            "time": time,
            "price": price,
            "equity": equity
        })

        return time
    return previous_equity_update_time
