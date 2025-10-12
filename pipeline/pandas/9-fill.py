#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# Fill missing Close values with previous row value
df['Close'] = df['Close'].ffill()

# Fill missing High, Low, Open with same row's Close value
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# Fill missing Volume values with 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

print(df.head())
print(df.tail())
