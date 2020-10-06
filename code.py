# -*- coding: utf-8 -*-
from collections import Counter
import itertools, functools, operator
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class File:
  """
  Class that wraps the file reader.
  """

  @staticmethod
  def from_html(path,
                result_range=None,
                header=0,
                remove_duplicated_index=None):
    """
    Reads an HTML table and return a pandas dataframe.
    path = the HTML file path.
    result_range = the columns that have the results.
    header = the row header.
    remove_duplicated_index = the index of an unique identifier 
      to remove the duplicate entries that an HTML nested table can generate.
    """
    df = pd.read_html(open(path, "r"), header=header)[0]

    if remove_duplicated_index != None:
      df = df.drop_duplicates(
          subset=[df.columns[remove_duplicated_index]], keep="first")

    if result_range != None:
      df = df.iloc[:, result_range[0]:result_range[1]]

    return df.reset_index(drop=True)

  @staticmethod
  def from_csv(path, result_range=None, header=0):
    """
    Read a CSV file and returns a pandas dataframe.
    path = the CSV file path.
    result_range = the columns that have the results.
    header = the row header.
    """
    df = pd.read_csv(open(path, "r"), header=header)

    if result_range != None:
      return df.iloc[:, result_range[0]:result_range[1]]

    return df

  @staticmethod
  def rename(df, cols_name=None):
    """
    Rename the dataframe's columns and return the renamed dataframe.
    df = pandas dataframe.
    cols_name = (optional) a list containing the columns name, 
      must be the same number of elements as the dataframe's columns.
    """
    rename = []
    if cols_name != None:
      rename = cols_name
    else:
      rename = ["rst_{}".format(x) for x in range(len(df.columns))]

    return df.set_axis(rename, axis=1, inplace=False)

  @staticmethod
  def to_csv(df, path):
    """
    Save to csv file.
    df = pandas dataframe.
    path = save location.
    """
    df.to_csv(path, index=False)


class Analysis:
  """
  Wraps the functions that analizes the given dataframe.
  """

  @staticmethod
  def frequency(df):
    """
    Returns a Counter object containing the numbers frequency.
    df = pandas dataframe.
    """
    counter = Counter()
    for _, row in df.iterrows():
      for number in row:
        counter[number] += 1

    return counter

  @staticmethod
  def group_frequency(df, group_size):
    """
    Returns a Counter object containing the 
      most common pair, trio, etc in the dataframe.
    df = pandas dataframe
    group_size = the group's desired size.
    """
    uniq = [set() for _ in range(len(df.index))]
    freq = [{} for _ in range(len(df.index))]

    for index, line in df.iterrows():
      for j in line:
        freq[index][j] = freq[index].get(j, 0) + 1
        uniq[index].add(j)

    counter = Counter()

    for i in range(len(df.index)):
      for j in itertools.combinations(uniq[i], group_size):
        freqp = [freq[i].get(j[x], 0) for x in range(group_size)]
        counter[j] = counter.get(j, 0) + functools.reduce(operator.mul, freqp)

    return counter


class Plot:
  """
  Plots the given counters.
  """

  @staticmethod
  def bar_plot(counter, n_items=None):
    """
    Bar plot using the seaborn one.
    counter = a counter object containing the values to plot.
    n_items = number of items to plot.
    """
    labels, values = zip(*counter.most_common(n_items))
    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width * 0.6)
    plt.xticks(indexes + width * 0, labels)
    plt.show()

  @staticmethod
  def sns_bar_plot(counter, n_items=None):
    """
    Bar plot using the seaborn one.
    counter = a counter object containing the values to plot.
    n_items = number of items to plot.
    """
    labels, values = zip(*counter.most_common(n_items))
    indexes = np.arange(len(labels))
    width = 1

    sns.barplot(x=indexes, y=values)
    plt.xticks(indexes + width * 0, labels)
    plt.show()
