import os,sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

sns.set_style("white")


def get_arrivals(arrivals_input_file, gdi_per_month_dict, cpi_per_month_dict, exchange_rates):
    """
    Read the arrivals file
    """
    arrivals_dict = {} 
    
    with open(arrivals_input_file, 'r') as inpf:
        header = inpf.readline().strip().split('\t')[1::]
        #read the input matrix (years in rows, months in columns and numbers in the cells
        for l in inpf.readlines():
            sl = l.strip().split('\t')
            for i, s in enumerate(sl[1:-1]):
                if s!="NA": #skip months that have missing value
                    iyear = int(sl[0])
                    imonth = int(header[i])
                    date = sl[0] + "-" + header[i] + "-01"
                    
                    num_arrivals = int(s.replace(',','').replace('.', ''))
                    num_fatalities = 0
                    incident_type = 'None'
                    war = 0
                    season = 0
                    conflict = 0
                    
                    if int(header[i])>=3 and int(header[i]) < 10: #between march and october is defined as high season
                        season = 1
                    
                    #Wars in 2003 and also in June 2013
                    if  (iyear>=2003 and iyear <= 2006) or (iyear>=2014 and iyear < 2017):
                        war = 1
                    
                    #conflict column
                    if  (iyear>=2011 and iyear <= 2011+1):
                        if  ((iyear == 2011 and imonth>=2) or (iyear==2012 and imonth <= 2)):#spring demonstrations):
                            conflict = 1
                    
                    gdi = 0
                    try:
                        gdi = gdi_per_month_dict[date]
                    except KeyError:
                        pass
                    cpi = -1
                    try:
                        cpi= cpi_per_month_dict[date]
                    except KeyError:
                        pass
                    exchange_rate = 0.0
                    try:
                        exchange_rate = exchange_rates[date]
                    except KeyError:
                        pass
                    arrivals_dict[date] = [date, num_arrivals, num_fatalities, incident_type, war, season, conflict, gdi, cpi, exchange_rate]
                    
    return arrivals_dict
    
def get_incidents(incidents_input_file, arrivals_dict):
    """
    Read the GTD data and add the records to the arrivals dataset
    """
    df_incidents = pd.read_table(incidents_input_file, sep='\t')
    cols = df_incidents.columns.tolist()
    iyear_index = cols.index('iyear')
    nkill_index = cols.index('nkill')
    
    imonth_index = cols.index('imonth')
    attack_type_index = cols.index('catgory_attacktype')
    c = 0 #num incidents
    f = 0 #num incidents with fatalities
    k = 0 #numb of fatalities
    for row in df_incidents.iterrows():
        
        index, data_row = row
        data  = data_row.tolist()
        date = str(data[iyear_index]) +'-'+ str(data[imonth_index]) + "-01"
        if data[iyear_index]>=2003:
            c +=1
        try:
            arrivals_dict[date][2] += int(data[nkill_index])
            if data[iyear_index]>=2003 and int(data[nkill_index])>0:
                f+=1
                k+=int(data[nkill_index])
        except ValueError:
            arrivals_dict[date][2] += 0
        except KeyError:
            pass
        #binarize nkills
        try:
            arrivals_dict[date][3] = data[attack_type_index].replace('Bombing/', '').replace('Hostage Taking (Kidnapping)', 'kidnapping').replace("Armed Assault", "armed_assault").replace('Explosion', 'explosion')
        except KeyError:
            pass
    print("number of incidents: ", c, "number of incidents with fatalities", f, "number of deaths", k)
    
    return df_incidents, arrivals_dict

   
def make_figures(df):
    
    tick_positions =[]
    years=[]
    for i in range(0,len(df['date'])):
        if i%12 == 0:
            tick_positions.append(i)
            years.append(df['date'][i].split('-')[0])
    from pandas.plotting import lag_plot, autocorrelation_plot
    
    plt.figure(figsize=(8,6))
    plt.subplot(2,2,1)
    lag_plot(df['n'], lag=1)
    plt.subplot(2,2,2)
    lag_plot(df['n'], lag=3, ax=plt.gca())
    plt.subplot(2,2,3)
    lag_plot(df['n'], lag=6, ax=plt.gca())
    plt.subplot(2,2,4)
    lag_plot(df['n'], lag=12, ax=plt.gca())
    plt.tight_layout()
    plt.savefig("correlation_lag.png")
    
    plt.figure()
    autocorrelation_plot(df['n'])
    plt.tight_layout()
    plt.savefig("autocor_lag.png")
    
    plt.figure(figsize=(8,5))
    new_df = df[['n']]
    new_df['n'] = new_df['n']
    new_df.columns = ['Number of arrivals']
    sns_plot = sns.lineplot(data=new_df, legend=False)
    sns_plot.set(xlabel='Years', ylabel='Number of Arrivals')
    
    sns_plot.set(xticklabels = years, xticks = tick_positions)
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig("num_arrivals.png")
    
    plt.figure(figsize=(8,5))
    new_df = df[['nkill', 'n']]
    new_df['nkill'] = new_df['nkill'].apply(np.log2)
    new_df['n'] = new_df['n'].apply(np.log10)
    new_df.columns = ['Number of fatalities (log2)', 'Number of arrivals (log10)']
    sns_plot = sns.lineplot(data=new_df)
    #sns_plot = sns.lineplot(x = df.i, y = df.nkill, label = 'Tourist arrivals', color='grey')
    sns_plot.set(xlabel='Years', ylabel='Number of fatalities (log2)')
    sns_plot.set(xticklabels = years, xticks = tick_positions)
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig("nkill_num.png")
    
    plt.figure(figsize=(8,5))
    sns_plot = sns.boxplot(x="type", y="n", data=df)
    sns_plot.set(xlabel='Atack Type', ylabel='Number of arrivals')
    fig = sns_plot.get_figure() 
    plt.tight_layout()
    fig.savefig("type_num.png")
    
    plt.figure(figsize=(14,5))
    fig, axs = plt.subplots(ncols=2)
    sns_plot = sns.boxplot(x = "season", y="n", data=df, color='grey', ax=axs[0])
    sns_plot.set(xlabel='', ylabel='Number of arrivals')
    sns_plot.set(xticklabels = ['Low season', 'High season'])
    sns_plot = sns.boxplot(x = "war", y = "n", data = df, color='grey', ax=axs[1])
    sns_plot.set_yticks([])
    sns_plot.set(xlabel="", ylabel="")
    sns_plot.set(xticklabels = ['No War', 'War'])
    
    plt.tight_layout()
    plt.savefig("season_war.png")
    

def analysis_fatalities(df, include_cpi=False):
    """
    Implement the model
    """
    dummies = pd.get_dummies(df['type'])
    df_dummies = pd.concat([df, dummies], axis=1)
    df_dummies.drop(['type', 'None'], inplace=True, axis=1)#remove the orginal column and the None column
    df_dummies.loc[df_dummies.nkill > 1, 'bin_nkill'] = 1
    df_dummies.loc[df_dummies.nkill <= 0, 'bin_nkill'] = 0
    cols = df_dummies.columns.tolist()
    window_size = 6
    #df_dummies['n'] = df_dummies['n'].apply(np.log10)
    struct_df = pd.concat([df_dummies, 
                           df_dummies['armed_assault'].rolling(window=window_size).sum(),
                           df_dummies['kidnapping'].rolling(window=window_size).sum(),
                           df_dummies['explosion'].rolling(window=window_size).sum(),
                           df_dummies['nkill'].rolling(window=window_size).mean(),
                           #df_dummies['bin_nkill'].rolling(window=window_size).sum(),
                           ], axis=1).dropna()
    lag_cols = ['armed_assault_{}'.format(window_size), 'kidnapping_{}'.format(window_size), 'explosion_{}'.format(window_size), 'nkill_{}'.format(window_size)]#, 'bin_nkill_{}'.format(window_size)]
    cols.extend(lag_cols)
    struct_df.columns = cols
    for col in ['nkill_{}'.format(window_size)]:
        vals = []
        for n in struct_df[col]:
            if n==0.0:
                vals.append(n)
            else:
                vals.append(np.log2(n))
        struct_df[col] = vals
    #struct_df['gdi'] = struct_df['gdi'].apply(np.log)
    
    if include_cpi:
        result = ols("n ~ season + armed_assault_{w} + explosion_{w} + war + nkill_{w} + conflict + gdi + cpi + exchange_rate".format(w=window_size), data = struct_df).fit()
    else:
        result = ols("n ~ season + armed_assault_{w} + explosion_{w} + war + nkill_{w} + conflict + gdi + exchange_rate".format(w=window_size), data = struct_df).fit()
    print(result.summary())
    
    #get correlations
    print("correlation between number of arrivals and season: ", struct_df['n'].corr(struct_df['season']))
    print("correlation between number of arrivals and war: ", struct_df['n'].corr(struct_df['war']))
    print("correlation between number of arrivals and number of deaths in the previous {} months: ".format(window_size), struct_df['n'].corr(struct_df['nkill_{}'.format(window_size)]))
    
    #save the output dataset to a file
    if include_cpi:
        struct_df.to_csv("structured_dataset_used_for_modeling_nkill_include_cpi.csv", sep = '\t')
    else:
        struct_df.to_csv("structured_dataset_used_for_modeling_nkill.csv", sep = '\t')
    return struct_df

def read_quarterly_GDI(file_input):
    "reads the input tsv and writes gdi per month from the given quartely data"
    gdi_per_month_dict = {}
    with open(file_input) as fn:
        header = fn.readline()
        for l in fn.readlines():
            sl = l.strip().split('\t')
            y = int(sl[0].split('-')[0])
            m = int(sl[0].split('-')[1])
            n = int(sl[1])
            if y==2003 and m==1:
                gdi_per_month_dict['{}-{}-01'.format(y, m)] =  n
                continue
            m_1 = m-1
            m_2 = m-2
            y_1 = y
            y_2 = y
            if m == 1:
                m_1 = 12
                m_2 = 11
                y_1 = y-1
                y_2 = y-1
            gdi_per_month_dict['{}-{}-01'.format(y, m)] =  n
            gdi_per_month_dict['{}-{}-01'.format(y_1, m_1)] =  n
            gdi_per_month_dict['{}-{}-01'.format(y_2, m_2)] =  n
    return gdi_per_month_dict
    
def read_CPI(input_file):
    "reads the input tsv and writes gdi per month from the given quartely data"
    cp_per_month_dict = {}
    with open(input_file) as fn:
        header = fn.readline()
        for l in fn.readlines():
            sl = l.strip().split('\t')
            y = sl[0]
            m = sl[1]
            n = float(sl[2].replace(',', '.'))
            cp_per_month_dict["{}-{}-01".format(y,m)] = n
    return cp_per_month_dict
    
def get_exchange_rates(input_file):
    val_per_month_dict = {}
    with open(input_file) as fn:
        header = fn.readline()
        for l in fn.readlines():
            sl = l.strip().split('\t')
            y = int(sl[0])
            m = int(sl[1])
            n = float(sl[2])
            if y==2003:
                n=n*150
            val_per_month_dict["{}-{}-01".format(y,m)] = n
    return val_per_month_dict
    

if __name__ == '__main__':
    
    arrivals_input_file = "peryearmonth.txt"
    incidents_input_file = "gtd_export13Nov2018_filteredCol13ErbilSulaymaniahDihok_groupedIncidents.tsv"
    
    gdi_quarterly_input_file = "quarterly_gdi.tsv"
    gdi_per_month_dict = read_quarterly_GDI(gdi_quarterly_input_file)
    
    cpi_input_file = "monthly_CPI.tsv"
    cpi_per_month_dict = read_CPI(cpi_input_file)
    
    exchange_rates = get_exchange_rates("exchange_rates_IQD_USD.tsv")
    print(exchange_rates)
    arrivals_dict = get_arrivals(arrivals_input_file, gdi_per_month_dict, cpi_per_month_dict, exchange_rates)
    
    df_incidents, incidents_dict = get_incidents(incidents_input_file, arrivals_dict)
    df = pd.DataFrame.from_dict(incidents_dict, orient='index', columns='date,n,nkill,type,war,season,conflict,gdi,cpi,exchange_rate'.split(','))
    df.to_csv('dataset_Jan2003Sep2018.csv', sep='\t')
    make_figures(df)
    
    print("OLS Resuls (2003-2018)")
    struct_df = analysis_fatalities(df)
    #print(df.describe())
    
    #skip observations where no cpi info exist
    print("OLS Results with CPI (2009-2018)")
    df_cpi = df[df.cpi>=0]
    struct_df = analysis_fatalities(df_cpi, include_cpi=True)
    #print(df_cpi.cpi.describe())
    
    
