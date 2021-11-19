
# Import of libraries that we will use in this script
import os
import time
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime as dt
from datetime import timedelta, date
import bar_chart_race as bcr

#Initialization of the useful paths o
path_data = "C:/Users/Elliott Joliman/Desktop/EFREI/M1/S7/Data visualisation/projet/data/usefull/"
path_useless = "C:/Users/Elliott Joliman/Desktop/EFREI/M1/S7/Data visualisation/projet/data/useless/"
path_img = "C:/Users/Elliott Joliman/Desktop/EFREI/M1/S7/Data visualisation/projet/img/"

#Initialisation of the list whitch contains all the dataset
dataSets = os.listdir(path_data)
uselessSets = os.listdir(path_useless)
nbDataSetsI = len(dataSets)+len(uselessSets)

#Definition of the get functions useful for the 
def get_dom(df):
    return df.day
    
def get_weekday(df): 
    return df.weekday()
    
def get_hour(df):
    return df.hour

def get_month(df):
    return df.month

def get_year(df):
    return df.year

def find_year(df):
    """this function find all the years in a dataset"""
    dates = df['yr']
    years = []
    for i in range(len(dates)):
        if dates[i] not in years:
            years.append(dates[i])
    return years

def find_month_of_year(df,year):
    """this function find all the months of a year in a dataset"""
    dates = df['mth']
    months = []
    for i in range(len(dates)):
        if dates[i] not in months:
            if df['yr'][i] == np.int64(year):
                months.append(dates[i].item())
    months = sorted(months)
    return months

def find_day_month_year(df,month,year):
    """this function find all the days of a given month and year"""
    dates = df['dom']
    days = []
    for i in range(len(dates)):
        if dates[i] not in days:
            if df['yr'][i] == np.int64(year):
                if df['mth'][i] == np.int64(month):
                    days.append(dates[i].item())
    days = sorted(days)
    return days

# ****************************************     STREAMLIT APP SCRIPT     *******************************************

left, right = st.columns(2)
with left: 
    st.title('Visualisation project of my health data')
with right:
    health_logo = Image.open(path_img+"health_app.png")
    st.image(health_logo,width=90)

url = "https://www.linkedin.com/in/elliott-joliman/"
st.sidebar.header("Elliott JOLIMAN")
st.sidebar.write("[My Linkedin](%s)" % url)

toc = st.sidebar.selectbox(
    "Navigation bar :",
    ("1 - Where these data come from", 
    "2 - First look on the data", 
    "3 - Around my HearthRate",
    "4 - My Audio exposure",
    "5 - My stepcount")
)

if toc == "1 - Where these data come from":
    
    # ----------------------------   Part 1   --------------------------------
    st.header('1 - Where these data come from')
    par = "Indeed before starting the project I had to find this data set and I want to explain how to find your health data set. \n\nFirst of all and unfortunatly for the others this dataset come from my Iphone Health application. This app gather the data from my phone captors, tierces apps on my phone and other devices like an apple watch and a connected scale. \n\nYou can export your data in the 'Apple Health' app. But the raw export is in XML format. So, our first task is to convert it into something more useable, like CSV.\n\n To do it I folowed this tutorial :"
    link = "http://www.markwk.com/data-analysis-for-apple-health.html"
    st.write(par)
    st.write("[Tutorial to export and convert from XML to CSV](%s)" % link)
if toc == "2 - First look on the data":
    # -----------------------------   Part 2   -------------------------------

    st.header('2 - First look on the data')
    par = "Indeed before starting the project I had to find this data set and I want to explain how to find your health data set."+"\n\nFirst of all and unfortunatly for the others this dataset come from my Iphone Health application.\nFirst of all and unfortunatly for the others this dataset come from my Iphone Health application."

    st.subheader('2.1 - somes figures')
    st.write("The health app gave me a lot of raw data, to be accurate, there where :\n\n - ", nbDataSetsI ,"datasets from steps count to handwashing event passing through my headphone audio exposure.")

    dataSetSize = 0
    for file in dataSets:
        dataSetSize += os.path.getsize(path_data+file)
    dataSetSize = round(dataSetSize *(10**-6),2)   
    uselessSize = 0
    for file in uselessSets :
        uselessSize += os.path.getsize(path_useless+file)
    uselessSize = round(uselessSize *(10**-6),2)  

    totalSize = (dataSetSize + uselessSize)

    st.write("\n\n - ",totalSize,"MB of raw data.")
    st.write("After a quick manual exploration of these csv files I decided to keep {} datasets whitch represents {} MB i.e. {}% of the initial data.".format(len(dataSets),round(dataSetSize,2),round((dataSetSize*100)/totalSize,2)))
    
    #building a dictionary that contain the name of the datasets and their associate weith in MB and number of rows
    pie_dict = {}
    for ds in dataSets :
        pie_dict[ds] = round((os.path.getsize(path_data+ds)*(10**-6)*100)/dataSetSize,2)
    #round(os.path.getsize(path_data+ds)*(10**-6),3) (size in MB)
    #st.write(pie_dict)

    st.subheader('2.2 - raw data')
    #select_dataset = st.sidebar.select_slider("Choose the dataset you want to see with the slider",[k+1 for k in range(len(dataSets))])
    select_dataset = st.sidebar.selectbox("Choose the dataset :",dataSets) 
    i = dataSets.index(select_dataset)
    p = path_data+dataSets[i]
    data = pd.read_csv(p)
    
    pourcentage = pie_dict.get(dataSets[i])
    pie_labels = [dataSets[i],'Other']
    pie_sizes = [pourcentage,100-pourcentage]
    colors = ['#4895EF','#e5e5e5']
    
    fig, ax = plt.subplots()
    ax.pie(pie_sizes,labels=pie_labels,colors=colors,radius=1,autopct='%1.0f%%', pctdistance=1.1, labeldistance=0.5)

    st.write("The {} dataset".format(dataSets[i]),data.head(5))
    left, right = st.columns(2)
    with left: 
        st.write("A quick description of the {} dataset".format(dataSets[i]))
        st.write(data.describe())
    with right:
        st.write("Proprotion of the {} among the whole dataSets".format(dataSets[i]))
        st.pyplot(fig)
    st.write("As you can see it's not easy to visualize and understand the dataset, to interprete these data we have to vizualize them")

if toc == "3 - Around my HearthRate": 
    # ----------------------------   Part 3 : HeartRate   --------------------------------

    st.header('3 - Around my HearthRate')

    data1 = pd.read_csv(path_data+"HeartRate.csv")
    data1["startDate"] = data1["startDate"].map(pd.to_datetime)

    data1['yr'] = data1['startDate'].map(get_year)
    data1['mth'] = data1['startDate'].map(get_month)
    data1['dom'] = data1['startDate'].map(get_dom)
    data1['weekday']= data1['startDate'].map(get_weekday)
    data1['hour'] = data1['startDate'].map(get_hour)

    ticks = [0,1,2,3,4,5,6]
    labels = ['Mon','Tue','wed','Thu','Fri','Sat','Sun']

    left1, right1 = st.columns(2)
    with left1: 
        def julyHeatMap():
            #Calculate a matrix of the average heartrate by hour and weekday in July
            mean_matrix = np.zeros((7,24))
            for w in range(7):
                for h in range(24):
                    w_h=data1.loc[(data1['weekday']==w) & (data1['hour']==h) & (data1['startDate'] >= '2021-07-01') & (data1['startDate'] <= '2021-07-31')]
                    mean_matrix[w,h] = w_h['value'].mean()
            #Display an heatmap of my average heartrate by hour and day on a month in July
            fig, ax = plt.subplots()
            sns.heatmap(julyHeatMap(), linewidths = .5, ax=ax)
            plt.yticks(ticks, labels)
            st.write('Average HeartRate by day and hour of July 2021')
            st.write(fig)

        st.write('Average HeartRate by day and hour of July 2021')
        julyHM = Image.open(path_img+"julyHM.png")
        st.image(julyHM,width=360)

    with right1:
        def augustHeatMap():
            #Calculate a matrix of the average heartrate by hour and weekday in July
            mean_matrix = np.zeros((7,24))
            for w in range(7):
                for h in range(24):
                    w_h=data1.loc[(data1['weekday']==w) & (data1['hour']==h) & (data1['startDate'] >= '2021-08-01') & (data1['startDate'] <= '2021-08-31')]
                    mean_matrix[w,h] = w_h['value'].mean()
            #Display an heatmap of my average heartrate by hour and day on a month in August
            fig, ax = plt.subplots()
            sns.heatmap(augustHeatMap(), linewidths = .5, ax=ax)
            plt.yticks(ticks, labels)
            st.write('Average HeartRate by day and hour of August 2021')
            st.write(fig)

        st.write('Average HeartRate by day and hour of August 2021')
        augustHM = Image.open(path_img+"augustHM.png")
        st.image(augustHM,width=360)

    st.write("These heatmaps are showing my average heartrate during the 2021 summer holydays. \n\nThanks to this visualisation we can notice that I was most active in August than in July and indeed it's true, I made more activity in august like hike with my family during the afternoon. Furthemore, the dark cases show a low heartrate, they represent time of low activity or also my sleeptime, my average hour of wake up during this holydays was near to 10:00 am and after this hour the case begining to be more colorful.")

    st.write("Now if you want to see closer my heartrate, choose the date and the hour range to visualize it :")

    #Let's find the start date of our dataset
    #s_year = find_year(data1)[0]
    #s_month = find_month_of_year(data1,s_year)[0]
    #s_day = find_day_month_year(data1,s_month,s_year)[0]

    #Let's find the end date of our dataset
    e_year = find_year(data1)[len(find_year(data1))-1]
    e_month = find_month_of_year(data1,e_year)[len(find_month_of_year(data1,e_year))-1]
    e_day = find_day_month_year(data1,e_month,e_year)[len(find_day_month_year(data1,e_month,e_year))-1]

    #format = 'MMM DD, YYYY'  # format output
    start_date = dt.date(year=2021,month=7,day=13) #to keep only my applewatch data a avoid the Iris Apple watch
    end_date = dt.date(year=e_year,month=e_month,day=e_day)

    
    d = st.sidebar.date_input(
        min_value = start_date,
        max_value = end_date,
        label = "Select a date between {} and {}".format(start_date,end_date),
        value = start_date,
    )

    t1 = st.sidebar.number_input(label = 'Select the start time',min_value = 0, max_value = 23, value = 0)
    t2 = st.sidebar.number_input(label = 'Select the end time',min_value = 0, max_value = 23, value = 23)

    #display the heartrate data of the selected date
    fig, ax1 = plt.subplots()
    hr = data1.loc[(data1['startDate'] >= "{}".format(d)) & (data1['startDate'] < "{}".format(d + timedelta(days=1))) & (data1['hour'] >= t1) & (data1['hour'] < t2)]
    index = hr.index
    values = []
    for i in range(len(index)):
        values.append(hr['value'][index[i]])
    
    import datetime

    time=[]
    for i in range(len(index)):
        time.append(hr['startDate'][index[i]].time())
    time_dt = [datetime.datetime.combine(datetime.date.today(), t) for t in time]

    import matplotlib.dates as mdates

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        from math import factorial

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except (ValueError, msg):
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')

    from scipy.interpolate import make_interp_spline

    #valuesModified = savitzky_golay(values, 51, 3) # window size 51, polynomial order 3

    #df1 = pd.DataFrame({'date': time_dt,'value': valuesModified})
    #df1 = df1.set_index('date')
    df1 = pd.DataFrame({'date': time_dt,'value': values})
    df1 = df1.set_index('date')

    #st.line_chart(df2)
    st.line_chart(df1)
    
if toc == "4 - My Audio exposure": 
    # ------------------------------    Part 4 : AudioExposure    ----------------------------------
    st.header('4 - My Audio exposure')

    #import the data
    hae = pd.read_csv(path_data+"HeadphoneAudioExposure.csv")
    st.write(hae.head())
    hae["startDate"] = hae["startDate"].map(pd.to_datetime)
    hae["endDate"] = hae["endDate"].map(pd.to_datetime)

    hae['yr'] = hae['startDate'].map(get_year)
    hae['mth'] = hae['startDate'].map(get_month)
    hae['dom'] = hae['startDate'].map(get_dom)
    hae['weekday']= hae['startDate'].map(get_weekday)
    hae['hour'] = hae['startDate'].map(get_hour)

    haeValues = hae['value'].tolist()

    st.write("After a quick research, I found that the threshold of danger of listening is from 85 dB. Let's see how many time is was above this threshold :")
    nb_threshold = hae[hae['value'] >= 85]['value'].count()
    total = len(hae)
    percentage = round((nb_threshold * 100)/total,2)
    st.write("On ",total," values my headphone audio exposure was above of the limit ",nb_threshold," times, which represent ",percentage," % of the reccords.")
    nb_max = hae[hae['value'] >= 100]['value'].count()
    percentage2 = round((nb_max * 100)/total,3)
    st.write("More over I have also found the maximal limit impose by the law in France is 100 dB for a HeadPhone. this limit has been reached ",nb_max," times. That is ",percentage2,"% of the data set")
    st.write("Let's see the frequency of each slice range 5 dB :")

    plt.figure(figsize=(8,4))
    ranges = [k*5 for k in range(22)]
    h = plt.hist(
        hae['value'],
        bins=ranges,
        edgecolor='black',
        color='#4895EF'
    )

    plt.xlabel('range of dB')
    plt.xticks(ranges)
    plt.ylabel('frequency')
    plt.title('frequency of range of Headphone audio exposure in dB')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    #Let's find the start date of our dataset
    s_year = find_year(hae)[0]
    s_month = find_month_of_year(hae,s_year)[0]
    s_day = find_day_month_year(hae,s_month,s_year)[0]

    #Let's find the end date of our dataset
    e_year = find_year(hae)[len(find_year(hae))-1]
    e_month = find_month_of_year(hae,e_year)[len(find_month_of_year(hae,e_year))-1]
    e_day = find_day_month_year(hae,e_month,e_year)[len(find_day_month_year(hae,e_month,e_year))-1]

    #format = 'MMM DD, YYYY'  # format output
    start_date = dt.date(year=s_year,month=s_month,day=s_day) #to keep only my applewatch data a avoid the Iris Apple watch
    end_date = dt.date(year=e_year,month=e_month,day=e_day)

    
    d = st.sidebar.date_input(
        min_value = start_date,
        max_value = end_date,
        label = "Select a date between {} and {}".format(start_date,end_date),
        value = start_date,
    )

    t1 = st.sidebar.number_input(label = 'Select the start time',min_value = 0, max_value = 23, value = 0)
    t2 = st.sidebar.number_input(label = 'Select the end time',min_value = 0, max_value = 23, value = 23)

    fig, ax1 = plt.subplots()
    hae_dt = hae.loc[(hae['startDate'] >= "{}".format(d)) & (hae['startDate'] < "{}".format(d + timedelta(days=1))) & (hae['hour'] >= t1) & (hae['hour'] < t2)]
    index = hae_dt.index

    values = []
    for i in range(len(index)):
        values.append(hae_dt['value'][index[i]])
    
    import datetime

    time=[]
    for i in range(len(index)):
        time.append(hae_dt['startDate'][index[i]].time())
    time_dt = [datetime.datetime.combine(datetime.date.today(), t) for t in time]

    import matplotlib.dates as mdates

    color = ['#4895EF' if x<85 else 'red' for x in values]

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.scatter(time_dt,values, c=color)
    plt.grid()
    fig

if toc == "5 - My stepcount": 
    # ------------------------------    Part 5 : stepcount    ----------------------------------
    st.header('5 - My stepcount')
    #import the data
    data2 = pd.read_csv(path_data+"StepCount.csv")

    #select only the iPhone sourceName for the data
    data2 = data2.loc[data2['sourceName'] == 'iPhone']

    #delete the useless columns, create a year "yr" column, modify the creation date from datetime to date
    del data2['sourceName']
    del data2['sourceVersion']
    del data2['device']
    del data2['type']
    del data2['unit']
    del data2['startDate']
    del data2['endDate']

    data2["creationDate"] = data2["creationDate"].map(pd.to_datetime)
    data2['yr'] = data2['creationDate'].map(get_year)
    data2["creationDate"] = data2["creationDate"].apply(lambda x : x.date())

    #save the usefull cols and make a hot encoding trick to have a columns for each value of 'yr', finaly add the two saves cols
    cd = data2['creationDate']
    v = data2['value']
    data2 = pd.get_dummies(data2.yr, prefix ='y')
    data2['creationDate'] = cd
    data2['value'] = v

    #groupby the date to have a row by day
    data2 = data2.groupby(['creationDate']).sum()

    #for each rows, i set the stepcount value in the yr_col which have a >0 value and finaly delete the value col
    data2['y_2016'] = np.where(data2.y_2016 !=0,data2.value,data2.y_2016)
    data2['y_2017'] = np.where(data2.y_2017 !=0,data2.value,data2.y_2017)
    data2['y_2019'] = np.where(data2.y_2019 !=0,data2.value,data2.y_2019)
    data2['y_2020'] = np.where(data2.y_2020 !=0,data2.value,data2.y_2020)
    data2['y_2021'] = np.where(data2.y_2021 !=0,data2.value,data2.y_2021)
    del data2['value']

    #i shuffle the dataset to "simulate" a barchart race
    data2 = data2.sample(frac=1)

    #In order to perform a bar chart race i have to transform this data set. The purpose is that for each columns each cells is the sum of the previous ones
    data2['y_2016'] = data2.y_2016.cumsum()
    data2['y_2017'] = data2.y_2017.cumsum()
    data2['y_2019'] = data2.y_2019.cumsum()
    data2['y_2020'] = data2.y_2020.cumsum()
    data2['y_2021'] = data2.y_2021.cumsum()
    st.write('To perform a bar chart race, I create a dataset index by date, each columns will be the future bar and the values of the cells is the cumulative sum of the previous one in the column.')
    st.write("My bar char race is special because, my data doesn't fit well to this animation, to simulate a good one I sort the date because, otherwhise the bars will increase one after one.")
    
    #we take only the even rows to improve the performance
    data3 = data2.iloc[::2]
    st.write(data3.head())

    #now we have the dataset let's perform a barchart race
    import bar_chart_race as bcr

    import cv2 as cv
    import tempfile

    f = st.file_uploader("Upload file")

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())


    vf = cv.VideoCapture(tfile.name)

    stframe = st.empty()

    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        stframe.image(frame)

    #st.write(bcr.bar_chart_race(
    #    df = data3,
    #    figsize = (6/1.2, 3.5/1.2),
    #    steps_per_period = 10,
    #    period_length = 50,
    #    filename = None,#"test.mp4",
    #    title = 'StepCount by year'
    #))

if toc == "6 - My sleep Analysis": 
    # ------------------ Part 6 : Sleep analysis ----------------------

    st.header('6 - My sleep Analysis')
    sa_data = pd.read_csv(path_data+"SleepAnalysis.csv")
    
    sa_data["startDate"] = sa_data["startDate"].map(pd.to_datetime)
    sa_data["endDate"] = sa_data["endDate"].map(pd.to_datetime)
    sa_data["duration"] = sa_data["endDate"] - sa_data["startDate"]
    sa_data["duration"] = sa_data["duration"]/np.timedelta64(1,'s')
    sa_data["duration"] = sa_data["duration"]/3600

    #record by my apple watch 
    sleep_time = sa_data.loc[(sa_data['startDate'] >= '2021-08-01') & (sa_data['startDate'] <= '2021-08-31') & (sa_data['value'] == 'HKCategoryValueSleepAnalysisAsleep')]
    #record by my iphone
    bed_time = sa_data.loc[(sa_data['startDate'] >= '2021-08-01') & (sa_data['startDate'] <= '2021-08-31') & (sa_data['value'] == 'HKCategoryValueSleepAnalysisInBed')]

    p1 = "sub dataSet of the sleepAnalysis data, the sleep_time dataset record by my apple watch"
    p2 = "sub dataSet of the sleepAnalysis data, the bed_time dataset record by my iphone"
    st.write(p1,sleep_time.describe())
    st.write(p2,bed_time.describe())