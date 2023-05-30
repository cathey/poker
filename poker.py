import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def filter_data(df, form='cash', game='nlh', stakes=None, location=None, time=None):
    if form:
        df = df[df.format == form]
        
    if form == 'cash':
        if stakes:
            if '/' in stakes:   # reject tournament buy ins
                if game == 'nlh':
                    blinds = [stakes, 'USD ' + stakes]
                else:
                    blinds = [game.upper() + ' ' + stakes,
                              game.upper() + ' USD ' + stakes]
        else:
            blinds_all = df['blinds / buy in'].unique().tolist()
            blinds = []
            if game == 'nlh':
                for blind in blinds_all:
                    if 'PLO' not in blind:
                        blinds.append(blind)
            elif game == 'plo':
                for blind in blinds_all:
                    if 'PLO' in blind:
                        blinds.append(blind)
        df = df[df['blinds / buy in'].isin(blinds)]
        
    if location:
        df = df[df.location == location]
    df = df.reset_index(drop=True)
    
    if time:
        if type(time) is list:  # range
            year1, month1, day1 = time_conv(time[0])
            year2, month2, day2 = time_conv(time[1])
            if not month1:
                month1 = 1
            if not day1:
                day1 = 1
            if not month2:
                month2 = 12
            if not day2:
                day2 = (datetime(year2 + month2 // 12, 
                  month2 % 12 + 1, 1) - timedelta(1)).day
        else:
            year, month, day = time_conv(time)        
            if day: # specific date
                df = df[df.date == datetime(year, month, day)]
                df = df.reset_index(drop=True)
                return df
            else:   # range
                year1 = year2 = year
                if month:
                    month1 = month2 = month
                else:
                    month1 = 1
                    month2 = 12
                day1 = 1
                day2 = (datetime(year2 + month2 // 12, 
                  month2 % 12 + 1, 1) - timedelta(1)).day
        i1 = i2 = 0
        for i in range(1, len(df)):
            if df.loc[i, 'date'] < datetime(year1, month1, day1):
                i1 = i
            elif df.loc[i, 'date'] > datetime(year2, month2, day2):
                i2 = i
                break
        if i2 == 0:
            i2 = len(df)
        df = df[i1:i2]
        df = df.reset_index(drop=True)
        
    return df


def time_conv(time):
    year, month, day = None, None, None
    if type(time) is str:
        try:
            date = datetime.strptime(time, '%Y/%m/%d')
            year = date.year
            month = date.month
            day = date.day
        except:
            try:
                date = datetime.strptime(time, '%Y/%m')
                year = date.year
                month = date.month
            except:
                year = int(time)
    elif type(time) is int: # only year
        year = time
    return year, month, day


def recalc_winnings(df):
    df.loc[0, 'bankroll'] = df.loc[0, 'results']
    for i in range(1, len(df)):
        df.loc[i, 'bankroll'] = df.loc[i-1, 'bankroll'] \
            + df.loc[i, 'results']
    return df


def plot_winnings(df, tag=None):
    # total winnings
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    total_winnings = df['bankroll'].to_numpy()
    plt.plot(total_winnings)
    if np.argmax(total_winnings) < len(df) - 10:
        x = [np.argmax(total_winnings), len(df)-1]
        y = [np.max(total_winnings), total_winnings[-1]]
    else:
        x = [len(df)-1]
        y = [total_winnings[-1]]
    for i,j in zip(x,y):
        ax1.annotate('%s)' %j, xy=(i,j), xytext=(30,0), textcoords='offset points')
        ax1.annotate('(%s,' %i, xy=(i,j))
    wps = total_winnings[-1]//len(total_winnings)
    plt.legend(['per session = ' + str(wps)], loc = 'lower right', \
               handlelength=0)
    plt.grid()
    if tag:
        plt.title('bankroll - ' + tag)
    else:
        plt.title('bankroll')
    
    # histogram
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    session_winnings = df['results'].to_numpy()
    N_win = len(session_winnings[session_winnings > 0])
    N_lose = len(session_winnings[session_winnings < 0])
    N_draw = len(session_winnings[session_winnings == 0])
    win_rate = (N_win + N_draw/2)/len(session_winnings)
    std = int(np.round(np.std(session_winnings)))
    plt.hist(session_winnings, bins = range(-1500, 2500, 100))
    plt.grid()
    ax2.axvline(x=0,  color='k')
    plt.legend(["#win = {:d}\n#lose = {:d}\nwin rate = {:.0%}\nstd = {:d}".format(\
                N_win, N_lose, win_rate, std)], handlelength=0)
    if tag:
        plt.title('session winnings - ' + tag)
    else:
        plt.title('session winnings')


def session_ranking(df, sess=None):
    results = df.results.to_numpy()
    msg = ""
    if sess is None: # final session
        sess = results[-1]
        msg = "last "
    results = np.sort(results)[::-1]
    try:
        idx = np.where(results <= sess)[0][0]
    except:
        idx = len(results)
    print(msg+"session: {:d}, rank {:d} of {:d}, top {:.0%}".format(\
          sess, idx+1, len(results), (idx+1)/len(results)))


def streak_count(df):
    results = df.results.to_numpy()
    results_wl = np.sign(results)
    streak = 1
    money = results[-1]
    record = "W" if results_wl[-1] == 1 else "L"
    for i in range(len(results_wl)-1, 1, -1):
        if results_wl[i-1] == results_wl[i]:
            streak += 1
            money += results[i-1]
        else:
            break
    print("current streak: " + record + str(streak) + ', ' + str(money))


def downswing_calc(df):
    total_winnings = df['bankroll'].to_numpy()
    downswing = np.max(total_winnings) - total_winnings[-1]
    if downswing > 0:
        idx = np.argmax(total_winnings)
        day = df.loc[idx, 'date']
        print("current downswing: " + str(downswing) + ', since ' + str(day)[:10])
    

os.chdir('C:\Travel')
df = pd.read_excel('poker.xlsx')
df_nlh_cash = filter_data(df, form='cash', game='nlh') #, location = 'TPTS')#, stakes='1/3')#, location='TPTS')
df_nlh_cash = recalc_winnings(df_nlh_cash)
plot_winnings(df_nlh_cash, tag='NLH cash')
session_ranking(df_nlh_cash)
streak_count(df_nlh_cash)
downswing_calc(df_nlh_cash)