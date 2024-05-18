import pandas as pd
import io
import datetime
import matplotlib.pyplot as plt
from lightweight_charts import JupyterChart
from typing import Optional

class BacktestAnalysis:

    def __init__(self, strategy) -> None:
        self.data = strategy.data.copy()
        self.trades = strategy.trades_info.copy()
        self.wallet = strategy.equity_record.copy()

        if self.trades.empty:
            raise ValueError('trades_info is empty, probably need to run the backtest first')

        if self.wallet.empty:
            raise ValueError('equity_record is empty, probably need to run the backtest first')

        self.compute_metrics()

    def compute_metrics(self) -> None:

        # --- Trades Info ---
        self.trades['duration'] = self.trades['close_time'] - self.trades["open_time"]

        self.mean_trade_duration = self.trades['duration'].mean()
        self.total_trades = len(self.trades)

        self.good_trades = self.trades.loc[self.trades["net_pnl"] > 0]
        if self.good_trades.empty:
            print("/!\\ No winning trades were found !")
            self.total_good_trades = 0
        else:
            self.total_good_trades = len(self.good_trades)
            self.avg_pnl_pct_good_trades = self.good_trades["net_pnl_pct"].mean()
            self.mean_good_trades_duration = self.good_trades["duration"].mean()

        self.bad_trades = self.trades.loc[self.trades["net_pnl"] < 0]
        if self.bad_trades.empty:
            print("/!\\ No losing trades were found !")
            self.total_bad_trades = 0
        else:
            self.total_bad_trades = len(self.bad_trades)
            self.avg_pnl_pct_bad_trades = self.bad_trades["net_pnl_pct"].mean()
            self.mean_bad_trades_duration = self.bad_trades["duration"].mean()

        self.best_trade = self.trades.loc[self.trades["net_pnl"].idxmax()]
        self.worst_trade = self.trades.loc[self.trades["net_pnl_pct"].idxmin()]
        self.global_win_rate = self.total_good_trades / self.total_trades
        self.avg_pnl_pct = self.trades["net_pnl_pct"].mean()
        self.profit_factor = abs(self.trades.loc[self.trades['net_pnl'] > 0, 'net_pnl'].sum() / self.trades.loc[self.trades['net_pnl'] < 0, 'net_pnl'].sum())

        self.trades['balance_ath'] = self.trades['close_balance'].cummax()
        self.trades['drawdown'] = self.trades['balance_ath'] - self.trades['close_balance']
        self.trades['drawdown_pct'] = self.trades['drawdown'] / self.trades['balance_ath']
        self.max_drawdown_trades = self.trades["drawdown_pct"].max()

        fees = self.trades['open_fee'] + self.trades['close_fee']
        self.total_fee = round(fees.sum(), 2)
        self.biggest_fee = round(fees.max(), 2)
        self.avg_fee = round(fees.mean(), 2)

        try:
            self.trades['win'] = self.trades['net_pnl'] > 0
            streak_changes = self.trades['win'].ne(self.trades['win'].shift()).cumsum()
            streaks = self.trades.groupby(streak_changes).agg({'win': ['sum', 'count']})
            streaks.columns = streaks.columns.map('_'.join)
            self.max_win_streak = streaks.loc[streaks['win_sum'] == streaks['win_count'], 'win_count'].max()
            self.max_lose_streak = streaks.loc[streaks['win_sum'] == 0, 'win_count'].max()
        except Exception as e:
            pass

        # --- Equity Record ---
        returns = self.wallet["equity"].pct_change()
        equity_ath = self.wallet["equity"].cummax()
        self.wallet["drawdown"] = equity_ath - self.wallet["equity"]
        self.wallet["drawdown_pct"] = self.wallet["drawdown"] / equity_ath

        self.max_drawdown_equity = self.wallet["drawdown_pct"].max()
        self.initial_balance = self.wallet.iloc[0]["equity"]
        self.final_balance = self.wallet.iloc[-1]["equity"]
        self.roi = self.final_balance / self.initial_balance - 1

        self.sharpe_ratio = 365 ** 0.5 * returns.mean() / returns.std()
        self.sortino_ratio = 365 ** 0.5 * returns.mean() / returns[returns < 0].std()
        self.calmar_ratio = returns.mean() * 365 / self.max_drawdown_equity

        self.hodl_pct = self.wallet.iloc[-1]['price'] / self.wallet.iloc[0]['price'] - 1
        hodl = self.initial_balance * (1 + self.hodl_pct)
        self.performance_vs_hodl = (self.final_balance - hodl) / hodl

        total_days = (self.wallet.index[-1] - self.wallet.index[0]).days + 1
        self.mean_trades_per_day = self.total_trades / total_days

        total_time_in_position = self.trades['duration'].sum()
        total_backtest_period = self.wallet.index.max() - self.wallet.index.min()
        self.time_in_position_ratio = (total_time_in_position.total_seconds() / total_backtest_period.total_seconds())
        self.return_over_max_drawdown = self.roi / abs(self.max_drawdown_equity)

    def plot_equity(self, path: Optional[str] = None, plot_price: bool = True) -> None:
        plot_equity(self.wallet, plot_price, path)

    def plot_drawdown(self, path: Optional[str] = None) -> None:
        plot_drawdown(self.wallet, path)

    def plot_monthly_performance(self, path: Optional[str] = None, year: str = "all") -> None:
        if year == "all":
            years = self.wallet.index.year.unique()
            for yr in years:
                plot_monthly_performance(self.wallet, year=yr, path=path)
        else:
            plot_monthly_performance(self.wallet, year=year, path=path)

    def plot_candlestick(self, indicators: Optional[dict] = None, show_volume: bool = False) -> None:
        plot_candlestick(self.trades, self.data, indicators, show_volume)

    def print_metrics(self, path: Optional[str] = None) -> None:

        result_io = io.StringIO()

        print("--- General ---", file=result_io)
        print(f"Period: [{self.wallet.index[0]}] -> [{self.wallet.index[-1]}]", file=result_io)
        print(f"Initial balance: {round(self.initial_balance, 2)} $", file=result_io)
        print(f"Final balance: {round(self.final_balance, 2)} $", file=result_io)
        print(f"Performance: {round(self.roi * 100, 2)} %", file=result_io)
        print(f"Hodl performance: {round(self.hodl_pct * 100, 2)}%", file=result_io)
        print(f"Performance/Hodl: {round(self.performance_vs_hodl * 100, 2)} %", file=result_io)
        print(f"Total trades: {self.total_trades}", file=result_io)
        print(f"Time in position: {round(self.time_in_position_ratio * 100, 2)} %", file=result_io)

        print("\n--- Health ---", file=result_io)
        print(f"Win rate: {round(self.global_win_rate * 100, 2)} %", file=result_io)
        print(f"Max drawdown at trade close: -{round(self.max_drawdown_trades * 100, 2)} %", file=result_io)
        print(f"Max drawdown at equity update: -{round(self.max_drawdown_equity * 100, 2)} %", file=result_io)
        print(f"Profit factor: {round(self.profit_factor, 2)}", file=result_io)
        print(f"Return over max drawdown: {round(self.return_over_max_drawdown, 2)}", file=result_io)
        print(f"Sharpe ratio: {round(self.sharpe_ratio, 2)}", file=result_io)
        print(f"Sortino ratio: {round(self.sortino_ratio, 2)}", file=result_io)
        print(f"Calmar ratio: {round(self.calmar_ratio, 2)}", file=result_io)

        print("\n--- Trades ---", file=result_io)
        print(f"Average net PnL: {round(self.avg_pnl_pct, 2)} %", file=result_io)
        print(f"Average trades per day: {round(self.mean_trades_per_day, 3)}", file=result_io)
        print(f"Average trades duration: {self.mean_trade_duration}", file=result_io)
        print(
            f"Best trade: +{round(self.best_trade['net_pnl_pct'], 2)} % entered {self.best_trade['open_time']} exited {self.best_trade['close_time']}", file=result_io)
        print(
            f"Worst trade: {round(self.worst_trade['net_pnl_pct'], 2)} % entered {self.worst_trade['open_time']} exited {self.worst_trade['close_time']}", file=result_io)
        print(f"Total winning trades: {self.total_good_trades}", file=result_io)
        print(f"Total loosing trades: {self.total_bad_trades}", file=result_io)
        print(f"Average net PnL winning trades: {round(self.avg_pnl_pct_good_trades, 2)} %", file=result_io)
        print(f"Average net PnL loosing trades: {round(self.avg_pnl_pct_bad_trades, 2)} %", file=result_io)
        print(f"Mean winning trades duration: {self.mean_good_trades_duration}", file=result_io)
        print(f"Mean loosing trades duration: {self.mean_bad_trades_duration}", file=result_io)
        print(f"Max win streak: {self.max_win_streak}", file=result_io)
        print(f"Max lose streak: {self.max_lose_streak}", file=result_io)
        print("Open reasons:", file=result_io)
        print(self.trades["open_reason"].value_counts().to_string(header=False), file=result_io)
        print("Close reasons:", file=result_io)
        print(self.trades["close_reason"].value_counts().to_string(header=False), file=result_io)

        print("\n--- Fees in Quote ---", file=result_io)
        print(f"Total: {self.total_fee}", file=result_io)
        print(f"Biggest: {self.biggest_fee}", file=result_io)
        print(f"Average: {self.avg_fee}\n", file=result_io)

        results = result_io.getvalue()
        result_io.close()

        if path:
            with open(path+'_metrics.txt', 'w') as file:
                file.write(results)
        else:
            print(results)


# --- Utilities ---
def plot_equity(equity_record: pd.DataFrame, plot_price: bool = True, path: Optional[str] = None) -> None:
    config = {
        'fig_size': (8, 4),
        'color': {
            'price': '#800080',
            'equity': '#089981',
        },
        'font_size': {
            'title': 13,
            'axis_label': 11,
            'tick_params': 10,
            'legend': 10
        },
        'line_width': 1,
        'alpha': 0.2
    }

    data = equity_record
    fig, ax_price = plt.subplots(figsize=config['fig_size'])
    ax_equity = ax_price.twinx()

    plt.title("Equity Value" + (" vs Asset Price" if plot_price else ""), fontsize=config['font_size']['title'])

    if plot_price:
        ax_price.plot(data.index, data['price'], color=config['color']['price'], lw=config['line_width'],
                      label="Asset Price")
        ax_price.set_ylabel("Asset Price in Quote", color=config['color']['price'],
                            fontsize=config['font_size']['axis_label'])
        ax_price.tick_params(axis='y', colors=config['color']['price'], labelsize=config['font_size']['tick_params'])
        price_range = data['price'].max() - data['price'].min()
        ax_price.set_ylim(data['price'].min() - price_range * 0.1, data['price'].max() + price_range * 0.1)
        ax_equity.tick_params(axis='y', colors=config['color']['equity'], labelsize=config['font_size']['tick_params'])
        ax_equity.set_ylabel("Equity Value in Quote", color=config['color']['equity'],
                             fontsize=config['font_size']['axis_label'])
    else:
        ax_price.get_yaxis().set_visible(False)
        ax_equity.tick_params(axis='y', labelsize=config['font_size']['tick_params'])
        ax_equity.set_ylabel("Equity Value in Quote", fontsize=config['font_size']['axis_label'])

    ax_equity.plot(data.index, data['equity'], color=config['color']['equity'], lw=config['line_width'],
                   label="Equity Value")
    ax_equity.fill_between(data.index, data['equity'], alpha=config['alpha'], color=config['color']['equity'])
    ax_equity.axhline(y=data.iloc[0]['equity'], color='black', lw=config['line_width'], ls='--')
    equity_range = data['equity'].max() - data['equity'].min()
    ax_equity.set_ylim(data['equity'].min() - equity_range * 0.1, data['equity'].max() + equity_range * 0.1)
    ax_equity.set_xlim(data.index.min(), data.index.max())

    if plot_price:
        handles, labels = [], []
        for ax in [ax_equity, ax_price]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.append(handle)
                labels.append(label)
        ax_equity.legend(handles, labels, loc="upper left", fontsize=config['font_size']['legend'])

    plt.tight_layout()
    if path:
        plt.savefig(f"{path}_plot_equity.png")
    else:
        plt.show()


def plot_drawdown(equity_record: pd.DataFrame, path: Optional[str] = None) -> None:
    config = {
        'fig_size': (8, 4),
        'color': {
            'drawdown': 'indianred',
            'line': 'black'
        },
        'font_size': {
            'title': 13,
            'axis_label': 11,
            'tick_params': 10,
            'legend': 10
        },
        'line_width': 1,
        'alpha': 0.2
    }

    fig, ax = plt.subplots(figsize=config['fig_size'])
    data = equity_record

    ax.title.set_text("Drawdown")
    ax.plot(-data['drawdown_pct'] * 100, color=config['color']['drawdown'], lw=config['line_width'])
    ax.fill_between(data.index, -data['drawdown_pct'] * 100, alpha=config['alpha'], color=config['color']['drawdown'])
    ax.axhline(y=0, color=config['color']['line'], alpha=0.3, lw=config['line_width'])
    ax.set_ylabel("%", color=config['color']['line'], fontsize=config['font_size']['axis_label'])
    ax.tick_params(axis='y', labelsize=config['font_size']['tick_params'])
    ax.set_xlim(data.index.min(), data.index.max())

    plt.tight_layout()

    if path:
        plt.savefig(f"{path}_plot_drawdown.png")
    else:
        plt.show()


def plot_monthly_performance(equity_record: pd.DataFrame, year: int, path: Optional[str] = None) -> None:
    config = {
        'fig_size': (8, 4),
        'colors': {
            'positive': '#089981',
            'negative': '#F23645'
        },
        'title_font_size': 13,
        'axis_label_font_size': 11,
        'tick_params_font_size': 10,
        'bar_text_font_size': 9,
        'text_color': 'black',
        'line_color': 'black',
        'alpha': 0.5,
        'rotation': 45,
    }

    year = int(year)
    yearly_data = equity_record[equity_record.index.year == year]

    if yearly_data.empty:
        print(f"No data available for the year {year}.")
        return

    yearly_performance = 0
    if not yearly_data.empty:
        yearly_performance = ((yearly_data['equity'].iloc[-1] - yearly_data['equity'].iloc[0]) /
                              yearly_data['equity'].iloc[0]) * 100

    monthly_performances = []
    for month in range(1, 13):
        monthly_data = yearly_data[yearly_data.index.month == month]

        if not monthly_data.empty:
            monthly_performance = ((monthly_data['equity'].iloc[-1] - monthly_data['equity'].iloc[0]) /
                                   monthly_data['equity'].iloc[0]) * 100
        else:
            monthly_performance = 0

        monthly_performances.append(monthly_performance)

    months = [datetime.date(1900, month, 1).strftime('%B') for month in range(1, 13)]

    fig, ax = plt.subplots(figsize=config['fig_size'])
    bars = ax.bar(months, monthly_performances,
                  color=[config['colors']['positive'] if x >= 0 else config['colors']['negative'] for x in
                         monthly_performances])

    plt.title(f"{year} (Cumulative Performance: {round(yearly_performance, 2)}%)", fontsize=config['title_font_size'])
    ax.axhline(0, color=config['line_color'], alpha=config['alpha'])
    ax.set_ylabel('Performance (%)', fontsize=config['axis_label_font_size'])

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{round(height, 2)}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 0 if height >= 0 else -2),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=config['bar_text_font_size'],
                    color=config['text_color'])

    plt.xticks(rotation=config['rotation'])
    ax.tick_params(axis='x', which='major', labelsize=config['tick_params_font_size'])
    ax.tick_params(axis='y', which='major', labelsize=config['tick_params_font_size'])

    performance_range = max(monthly_performances) - min(monthly_performances)
    lower_limit = min(monthly_performances) - (performance_range * 0.15)
    upper_limit = max(monthly_performances) + (performance_range * 0.15)
    ax.set_ylim(lower_limit, upper_limit)

    plt.tight_layout()

    if path:
        plt.savefig(f"{path}_plot_performance_{year}.png")
    else:
        plt.show()


def plot_candlestick(trades: pd.DataFrame, ohlcv: pd.DataFrame, indicators: Optional[dict] = None, show_volume: bool = False) -> None:
    chart = JupyterChart(width=900, height=400)
    if not show_volume:
        ohlcv = ohlcv.copy()
        del ohlcv['volume']

    chart.set(ohlcv)

    if indicators is not None:
        for name, ind in indicators.items():
            line = chart.create_line(name, ind['color'], price_line=False)
            line.set(ind['df'])

    for index, row in trades.iterrows():
        chart.marker(time=row['open_time'], position="below", shape="arrow_up", color="white", text=row['open_reason'])
        chart.marker(time=row['close_time'], position="above", shape="arrow_down", color="white", text=row['close_reason'])

    chart.load()
