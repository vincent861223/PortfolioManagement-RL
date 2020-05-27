import gym
from gym import spaces
import pandas as pd
import numpy as np 
import time
import os

class StockEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}

    def __init__(self, dataDir, period=1, time_interval=10, fund=100000):
        super(StockEnv, self).__init__()
        self.dataDir = dataDir
        self.period = period
        self.time_interval = time_interval
        self.fund = fund

        self.stockDir = os.path.join(dataDir, "Stocks")
        # Read selcted stocks to list -> [aapl, googl, ... ,v]
        self.selectedStocks = self._readSelectedStocks(os.path.join(dataDir, "selected_stocks.txt"))
        self.stock_dfs = [pd.read_csv(os.path.join(self.stockDir, selectedStock + ".us.txt")).sort_values('Date') for selectedStock in self.selectedStocks]
        self.startDate, self.endDate = self._getTimeInterval()

        self.currentDate = self.startDate
        #self.step = self._getDfIndex(self.currentDate)
        self.portfolioValue = fund
        self.stockWeights = np.array([1.0] + [0.0] * len(self.selectedStocks))
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([0] * len(self.stockWeights)), high=np.array([1] * len(self.stockWeights)), dtype=np.float16)
        # # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=300, shape=(4, len(self.selectedStocks), self.time_interval), dtype=np.float16)

        #price = self._getStockPrices(self.currentDate)
        #print(price)
        #print(price.shape)
        #print(self.currentDate)

    def step(self, action):
      # Execute one time step within the environment
      self.currentDate = self._nextNDate(self.currentDate, self.period)
      self.stockWeights = action
      obs = self._getStockPrices(self.currentDate)
      reward = self._netWorth(self.currentDate)
      self.portfolioValue = reward
      done = self.portfolioValue > (2 * self.fund) or self.portfolioValue < (self.fund * 0.9) or self._nextNDate(self.currentDate, self.period) > self.endDate 

      return obs, reward, done, {}


    def reset(self):
      # Reset the state of the environment to an initial state
      #self.step = self.startDate
      self.currentDate = self.startDate
      self.portfolioValue = self.fund
      self.stockWeights = np.array([1.0] + [0.0] * len(self.selectedStocks))
      return self._getStockPrices(self.currentDate)



    def render(self, mode='human', close=False):
      # Render the environment to the screen
      print(self._netWorth(self.currentDate))

    def __str__(self):
        desc = "---------------StockEnv----------------\n"
        desc += "selectedStocks: " + str(self.selectedStocks) + "\n"
        return desc

    def _readSelectedStocks(self, filepath):
        with open(os.path.join(self.dataDir, "selected_stocks.txt")) as f:
            lines = f.readlines()
            selectedStocks = [line.strip() for line in lines if (len(line.strip()) > 0)]
        return selectedStocks

    def _getTimeInterval(self):
        # This function will return the most begining time of all selected stocks
        dates = [pd.to_datetime(df['Date'].iloc[0]) for df in self.stock_dfs]
        min_date = dates[0]
        for date in dates:
            if date > min_date: min_date = date
        #print(dates)
        #print(min_date.strftime("%Y-%m-%d"))

        dates = [pd.to_datetime(df['Date'].iloc[-1]) for df in self.stock_dfs]
        max_date = dates[0]
        for date in dates:
            if date < max_date: max_date = date
        #print(dates)
        #print(max_date.strftime("%Y-%m-%d"))
        return self._nextNDate(min_date, self.period * self.time_interval), max_date

    def _getDfIndex(self, date):
        ids = [df[df["Date"] == date.strftime("%Y-%m-%d")].index.tolist() for df in self.stock_dfs]
        ids = [i[0] if len(i) > 0 else -1 for i in ids]
        if(not -1 in ids): return ids
        else: return None

    def _nextDate(self, date):
        currentDate = date + pd.DateOffset(days=1)
        while(self._getDfIndex(currentDate) == None): currentDate = currentDate + pd.DateOffset(days=1)
        return currentDate

    def _nextNDate(self, date, days):
        currentDate = date
        for i in range(days):
            currentDate = self._nextDate(currentDate)
        return currentDate

    def _getStockPrices(self, date):
        # current_date = date
        currentIds = self._getDfIndex(date)
        if(currentIds == None): return None
        open_prices = []
        high_prices = []
        close_prices = []
        low_prices = []
        # # Read the stock prices of from (selected date - time_interval) to (selected date)
        for i, stock_df in enumerate(self.stock_dfs):
            open_price = stock_df[currentIds[i] - (self.period * self.time_interval) + 1: currentIds[i] + 1]["Open"].values
            high_price = stock_df[currentIds[i] - (self.period * self.time_interval) + 1: currentIds[i] + 1]["High"].values
            close_price = stock_df[currentIds[i] - (self.period * self.time_interval) + 1: currentIds[i] + 1]["Close"].values
            low_price = stock_df[currentIds[i] - (self.period * self.time_interval) + 1: currentIds[i] + 1]["Low"].values
            #print(open_price, low_price, close_price, low_price)
            open_price = np.array([open_price[i] for i in range(0, open_price.shape[0], self.period)])
            close_price = np.array([close_price[i] for i in range(self.period-1, close_price.shape[0], self.period)])
            high_price = np.array([high_price[i: i + self.period].max() for i in range(0, high_price.shape[0], self.period)])
            low_price = np.array([low_price[i: i + self.period].max() for i in range(0, low_price.shape[0], self.period)])
            #print(open_price, low_price, close_price, low_price)
            open_prices += [open_price / close_price[-1]]
            high_prices += [high_price / close_price[-1]]
            close_prices += [close_price / close_price[-1]]
            low_prices += [low_price / close_price[-1]]

        open_prices = np.asarray(open_prices)
        high_prices = np.asarray(high_prices)
        close_prices = np.asarray(close_prices)
        low_prices = np.asarray(low_prices)

        prices = np.asarray([open_prices, high_prices, close_prices, low_prices])

        return prices

    def _priceReleativeVector(self, date):
        # current_date = date
        currentIds = self._getDfIndex(date)
        if(currentIds == None): return None
        close_prices = [1.0]
        # # Read the stock prices of from (selected date - time_interval) to (selected date)
        for i, stock_df in enumerate(self.stock_dfs):
            close_price_current = stock_df[currentIds[i] : currentIds[i]+1]["Close"].values
            close_price_last = stock_df[currentIds[i] - (self.period): currentIds[i] - (self.period) + 1]["Close"].values
            close_prices += [close_price_current/close_price_last]

        close_prices = np.asarray(close_prices)

        return close_prices

    def _netWorth(self, date):
        worth = self.portfolioValue * self._priceReleativeVector(date).dot(self.stockWeights);

        return worth
