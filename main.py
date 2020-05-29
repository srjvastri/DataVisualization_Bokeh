# Pandas for data management
import pandas as pd
import numpy as np

# Using MySQL connector to import data from MySQL database
import mysql.connector

# Using scikitlearn methods for Doing linear regression
from sklearn.linear_model import LinearRegression

# Using pretty html table to construct HTML table from pandas dataframe
from pretty_html_table import build_table
import panel as pn

# Read required datasets into dataframes
# I will use SQL connector to load the MySQL databases
mydb1 = mysql.connector.connect(user='root', host='127.0.0.1', database='factor_models')
df1 = pd.read_sql('SELECT * FROM daily_ledger', con=mydb1)
df1['Daily_Returns'] = df1['PNL']/df1['NotionalCapital']*100 
df2 = pd.read_sql('SELECT * FROM factors', con=mydb1)
df = pd.merge(df1, df2, how='inner', on=['Date'])

ndays = len(df)
factor_variables = ['India_IIMA_MarketExcess', 'India_IIMA_Size', 'India_IIMA_Value', 'India_IIMA_Momentum']
factor_riskfree = ['India_IIMA_RiskFree']

factors = factor_variables + ['Alpha']
factor_selection = pn.widgets.CrossSelector(name='Factors', value=[factor_variables[0]],
                                            options=factor_variables) #,inline=False

# Widgets

to_plot = pn.widgets.RadioButtonGroup(
    name='Waterfall Chart Group', options=['%Return' , 'Annual %Return'], button_type='primary')

factor_selection2 = pn.widgets.CrossSelector(name='Factors', value=[factor_variables[0]],
                                            options=factor_variables) #,inline=False
rolling_window2 = pn.widgets.IntSlider(name='Rolling Window - For %Annual Ret ALPHA', start=50, end=ndays-100, step=50, value=250)

factor_selection3 = pn.widgets.CrossSelector(name='Factors', value=[factor_variables[0]],
                                            options=factor_variables) #,inline=False
rolling_window3 = pn.widgets.IntSlider(name='Rolling Window - For Betas', start=50, end=ndays-100, step=50, value=250)

factor_selection4 = pn.widgets.CrossSelector(name='Factors', value=[factor_variables[0]],
                                            options=factor_variables) #,inline=False
rolling_window4 = pn.widgets.IntSlider(name='Rolling Window - For Cumulative Factorwise Returns', start=50, end=ndays-100, step=50, value=250)



def make_dataset(factors_to_plot):
    if (not factors_to_plot):
        ret=df['Daily_Returns'].sum().round(decimals=1)
        data = {'Factor Variables': ['Alpha','Total Returns'],
                '%Return': [ret,ret],
                'Annual %Return': [(ret/ndays*256).round(decimals=1),(ret/ndays*256).round(decimals=1)]}
        df_ret_att = pd.DataFrame(data)
        return df_ret_att
    factors = factors_to_plot + ['Alpha']

    X = df[factors_to_plot].values
    Y = (df['Daily_Returns']-df[factor_riskfree[0]]).values # Calculating Excess Daily returns by subtracting Risk free returns 
    dates = df['Date'].values
    regressor = LinearRegression().fit(X, Y)

    ###### Determining strategy returns attribution for the entire trading period
    strategy_returns = float(sum(df['PNL']/df['NotionalCapital'])*100)
    factor_returns = df[factors_to_plot].sum(axis=0).values # Calculating factor returns for the entire period

    return_attribution = np.multiply(regressor.coef_,factor_returns) # Multiplying factor returns with corresponding betas from regression
    return_attribution = np.append(return_attribution, strategy_returns - sum(return_attribution)) # Calculating the alpha part and appending
    values = return_attribution.round(decimals=3) # values to be used for plotting the pie chart
    return_attribution = np.append(return_attribution, sum(return_attribution)).round(decimals=3) # Appending total strategy returns to the array

    # Constructing a dataframe of attribution analysis on strategy returns for entire period
    df_ret_att = pd.DataFrame(return_attribution.round(decimals=1), columns=['%Return'])
    df_ret_att.insert(loc=0, column='Factor Variables', value=factors + ['Total Returns'])
    df_ret_att.insert(loc=2, column='Annual %Return', value=(return_attribution/ndays*256).round(decimals=1)) # Calculating and Inserting annual percentage returns column to the dataframe
    df_ret_att.insert(loc=3, column='%age Attribution', value=(return_attribution/strategy_returns*100).round(decimals=1)) # Calculating and  inserting hte %age attribution  column to the dataframe
    return df_ret_att
    
    
@pn.depends(factor_selection.param.value)
def attribution_table(factor_selection):
    return build_table(make_dataset(factor_selection).drop('%age Attribution',axis=1),'green_light')
    
    
@pn.depends(factor_selection.param.value)
def attribution_pie_chart(factor_selection):
    import plotly.express as px
    df=make_dataset(factor_selection)
    df = df.drop(len(df)-1, axis=0)
    attribution_piechart = px.pie(df, values='%age Attribution', title = 'Returns %Attribution: Factor-wise ',
                                  opacity=0.85,names='Factor Variables',
                                  color_discrete_sequence=px.colors.sequential.Magenta) #title = 'Strategy Returns Attribution',
    attribution_piechart.update_layout(
    font=dict(
#         family="Courier New, monospace",
        size=18
#         color="#7f7f7f"
    ))
    attribution_piechart.update_layout(title={'y':0.97,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    attribution_piechart.update_layout( font=dict(size=18,),margin = dict(t=60,r=0,pad=0,b=0),height = 350, width = 700)
    return attribution_piechart #height = 520, width = 650, 
    
    
    
@pn.depends(factor_selection.param.value,to_plot.param.value)
def attribution_waterfall_chart(factor_selection,to_plot):
    n = len(factor_selection)
    str1 = to_plot
    import plotly.graph_objects as go
    df4=make_dataset(factor_selection)
    fig = go.Figure(go.Waterfall(
    name = "Absolute " + str1 + " Attribution", orientation = "h", measure = ["relative"]*(n+1) +  ["total"],
    y = factor_selection + ["Alpha","Total Returns"],
    x = list(df4[str1]),
    connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}}
    ))
    fig.update_layout(title={'text': "Absolute " + str1 + " Attribution", 'y':0.97, 'x':0.5,'xanchor': 'center','yanchor': 'top'}, 
                      font=dict(size=18,),
                      margin = dict(t=60,r=0,b=0,pad=0),
                      height = 350, width = 700,
                      xaxis= {'showspikes': True, 'title': '%age returns'},
                      yaxis= {'showspikes': True},
                      spikedistance = 10)
    return fig
    
    
    
    
@pn.depends(factor_selection.param.value)
def factor_sensitivity(factor_selection):
    if (not factor_selection):
        ret=df['Daily_Returns'].sum().round(decimals=1)
        data = {'Factors':['Alpha'], 'Coefficient':[ret/ndays]}
        coeff_df = pd.DataFrame(data)
        return build_table(coeff_df, 'blue_dark')
    ###### Doing a linear regression on factor_variables and daily returns on the entire period
    X = df[factor_selection].values
    Y = (df['Daily_Returns']-df[factor_riskfree[0]]).values # Calculating Excess Daily returns by subtracting Risk free returns
    dates = df['Date'].values
    regressor = LinearRegression().fit(X, Y)

    ###### Constructing dataframe of Alphas and Betas - Sensitivity Analysis
    results = np.append(regressor.coef_,regressor.intercept_)
    coeff_df = pd.DataFrame(results.round(decimals=2), columns=['Coefficient'])
    coeff_df.insert(loc=0, column='Factors', value=factor_selection+['Alpha'])
    return build_table(coeff_df, 'blue_dark')
    
    
    
    
from bokeh.layouts import column
from bokeh.models import RangeTool, Legend, LegendItem, HoverTool
from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Spectral11

# input is dates,result_alpha
@pn.depends(rolling_window2.param.value,factor_selection2.param.value)
def plot_rolling_alpha(rolling_window,factors_to_plot):
    
    result_array = np.zeros((1,len(factors_to_plot)+1)) # Initializing the results array
    factors1 = factors_to_plot + ['Alpha']
    X = df[factors_to_plot].values
    Y = (df['Daily_Returns']-df[factor_riskfree[0]]).values # Calculating Excess Daily returns by subtracting Risk free returns
    dates = df['Date'].values
    for i in range(X.shape[0]-rolling_window+1): #Loop to do regression for each time period of rolling window
        x = X[i:rolling_window+i] # The X variables for regression based on rolling window
        y = Y[i:rolling_window+i] # The Y variable for regression based on rolling window
        regressor = LinearRegression().fit(x, y) # Doing the linear regression
        temp = np.append(regressor.coef_,regressor.intercept_) # Constructing the results of the regression numpy arrray by appending the alpha to the betas
        result_array = np.vstack((result_array, np.resize(temp,(1,len(factors_to_plot)+1)))) # Appending the Results of regression to result_array

    result_array = np.delete(result_array, 0, 0) # Deleing the first initialized row of the result_array
    dates = dates[rolling_window-1:] # Determing the dates of all the days on which rolling window regression is done
    result_alpha = pd.DataFrame(result_array, index=dates) # Values are not rounded to make the plots of annual alpha
    result_alpha.columns = factors1
    
    # To insert date column in the result_df dataframe
    df3 = pd.DataFrame(data=list(dates), columns=['int_date'])
    df3[['str_date']] = df3[['int_date']].applymap(str).applymap(lambda s: "{}-{}-{}".format(s[0:4], s[4:6], s[6:] ))
    rolling_dates=list(df3['str_date'])
    # df[['str_date']] = df[['int_date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[6:], s[4:6], s[0:4]))
    dates = np.array(rolling_dates, dtype=np.datetime64)
    rolling_ndays = len(dates)
    
    source = ColumnDataSource(data=dict(date=dates, close=list(result_alpha['Alpha'].values*256)))
    mypalette=Spectral11[:4]+Spectral11[8:] 

    p = figure(title = f'Rolling Annual %Alpha: {rolling_window} days window - Betas: {factors_to_plot}', plot_height=550, plot_width=1500, 
               toolbar_location='above', x_axis_type="datetime", x_axis_location="above", background_fill_color="#efefef", 
               x_range=(dates[(ndays-rolling_window)//3], dates[((ndays-rolling_window)*2)//3])) #, tools="xpan" ,tooltips=tooltips,
    p.toolbar.logo = None
    p.line('date', 'close', source=source,line_width=3, legend_label="% ALPHA", name="% ALPHA", color=mypalette[0])
    p.legend.background_fill_alpha = 0
    p.yaxis.axis_label = '%age Returns'

    p.yaxis.axis_label_text_font_size= "18px"
    p.axis.major_label_text_font_size = '16px'
    p.legend.label_text_font_size = '16px'

    hover_tool1 = HoverTool(
        tooltips=[
            ( 'Date',   '$x{%F}'),
            ( 'ALPHA',  '@close%' ) # use @{ } for field names with spaces @{% ALPHA}
        ],
        formatters={
            '$x':'datetime', # use 'datetime' formatter for '@date' field
        },
        names = ["% ALPHA"],
        mode='vline' # display a tooltip whenever the cursor is vertically in line with a glyph
    )
    p.tools.append(hover_tool1)

    p.title.text_color = mypalette[-1]
    p.title.text_alpha = 0.6
    p.title.text_font = "antiqua"
    p.title.text_font_size = "22px"
    p.title.align = "center"

    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    plot_height=140, plot_width=1500, y_range=p.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line('date', 'close', source=source, color=mypalette[0])
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
#     rolling_alpha = file_html([p,select], CDN)
    return column(p,select)
 
 
 
 
 


#input - dates,result_df
@pn.depends(rolling_window3.param.value,factor_selection3.param.value)
def plot_rolling_beta(rolling_window,factors_to_plot):
    
    result_array = np.zeros((1,len(factors_to_plot)+1)) # Initializing the results array
    factors1 = factors_to_plot + ['Alpha']
    X = df[factors_to_plot].values
    Y = (df['Daily_Returns']-df[factor_riskfree[0]]).values # Calculating Excess Daily returns by subtracting Risk free returns
    dates = df['Date'].values
    for i in range(X.shape[0]-rolling_window+1): #Loop to do regression for each time period of rolling window
        x = X[i:rolling_window+i] # The X variables for regression based on rolling window
        y = Y[i:rolling_window+i] # The Y variable for regression based on rolling window
        regressor = LinearRegression().fit(x, y) # Doing the linear regression
        temp = np.append(regressor.coef_,regressor.intercept_) # Constructing the results of the regression numpy arrray by appending the alpha to the betas
        result_array = np.vstack((result_array, np.resize(temp,(1,len(factors_to_plot)+1)))) # Appending the Results of regression to result_array

    result_array = np.delete(result_array, 0, 0) # Deleing the first initialized row of the result_array
    dates = dates[rolling_window-1:] # Determing the dates of all the days on which rolling window regression is done
    result_df = pd.DataFrame(result_array.round(decimals=2), index=dates) # Values rounded to make the plots of beta
    result_df.columns = factors1
    
    # To insert date column in the result_df dataframe
    df3 = pd.DataFrame(data=list(dates), columns=['int_date'])
    df3[['str_date']] = df3[['int_date']].applymap(str).applymap(lambda s: "{}-{}-{}".format(s[0:4], s[4:6], s[6:] ))
    rolling_dates=list(df3['str_date'])
    # df[['str_date']] = df[['int_date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[6:], s[4:6], s[0:4]))
    dates = np.array(rolling_dates, dtype=np.datetime64)
    result_df.insert(loc=0, column='Date', value=dates)
    rolling_ndays = len(dates)
    
    mypalette=Spectral11[:4]+Spectral11[8:]
    numlines=len(factors_to_plot)

    f = figure(title = f'Rolling Beta of Factors: {rolling_window} days window', plot_height=550, plot_width=1500, 
               toolbar_location='above',x_axis_type="datetime", x_axis_location="above", background_fill_color="#efefef", 
               x_range=(dates[(rolling_ndays)//3], dates[((rolling_ndays)*2)//3])) #tooltips=tooltips,tools="xpan", 
    select1 = figure(title="Drag the middle and edges of the selection box to change the range above",
                    plot_height=140, plot_width=1500, y_range=f.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")
    date = list(result_df['Date'])

    f.title.text_color = mypalette[-1]
    f.title.text_alpha = 0.6
    f.title.text_font = "antiqua"
    f.title.text_font_size = "32px"
    f.title.align = "center"
    f.yaxis.axis_label = 'Rolling Beta'


    for i in range(numlines):
        f.line(x=date, y=list(result_df[factors_to_plot[i]]), legend_label=factors_to_plot[i], color=mypalette[i],line_width=3,name=factors_to_plot[i])

    hover_tool = HoverTool(
        tooltips=[
            ( 'Date',   '$x{%F}'),
            ( 'Beta',   '@y' ), # use @{ } for field names with spaces
            ( 'Factor', '$name')
        ],
        formatters={
            '$x'        : 'datetime', # use 'datetime' formatter for '@date' field
        },
        names = factors_to_plot,
        mode='vline' # display a tooltip whenever the cursor is vertically in line with a glyph
    )
    f.tools.append(hover_tool)    
    f.legend.click_policy="hide"
    f.legend.background_fill_alpha = 0
    f.legend.border_line_color = None
    f.legend.margin = 10
    f.legend.padding = 18
    f.legend.spacing = 10
    f.toolbar.logo = None

    f.yaxis.axis_label_text_font_size= "18px"
    f.axis.major_label_text_font_size = '16px'
    f.legend.label_text_font_size = '16px'

    range_tool = RangeTool(x_range=f.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    for i in range(numlines):
        select1.line(x=date, y=list(result_df[factors_to_plot[i]]), color=mypalette[i],line_width=1,name=factors_to_plot[i])

    select1.ygrid.grid_line_color = None
    select1.add_tools(range_tool)
    select1.toolbar.active_multi = range_tool
    return column(f,select1)
    
    
    
    
    
@pn.depends(rolling_window4.param.value,factor_selection4.param.value)
def plot_returns_stack(rolling_window,factors_to_plot):
    result_array = np.zeros((1,len(factors_to_plot)+1)) # Initializing the results array
    factors1 = factors_to_plot + ['Alpha']
    X = df[factors_to_plot].values
    Y = (df['Daily_Returns']-df[factor_riskfree[0]]).values # Calculating Excess Daily returns by subtracting Risk free returns
    dates = df['Date'].values
    for i in range(X.shape[0]-rolling_window+1): #Loop to do regression for each time period of rolling window
        x = X[i:rolling_window+i] # The X variables for regression based on rolling window
        y = Y[i:rolling_window+i] # The Y variable for regression based on rolling window
        regressor = LinearRegression().fit(x, y) # Doing the linear regression
        temp = np.append(regressor.coef_,regressor.intercept_) # Constructing the results of the regression numpy arrray by appending the alpha to the betas
        result_array = np.vstack((result_array, np.resize(temp,(1,len(factors_to_plot)+1)))) # Appending the Results of regression to result_array

    result_array = np.delete(result_array, 0, 0) # Deleing the first initialized row of the result_array
    dates = dates[rolling_window-1:] # Determing the dates of all the days on which rolling window regression is done
    result_df = pd.DataFrame(result_array.round(decimals=2), index=dates) # Values rounded to make the plots of beta
    result_df.columns = factors1

    # To insert date column in the result_df dataframe
    df3 = pd.DataFrame(data=list(dates), columns=['int_date'])
    df3[['str_date']] = df3[['int_date']].applymap(str).applymap(lambda s: "{}-{}-{}".format(s[0:4], s[4:6], s[6:] ))
    rolling_dates=list(df3['str_date'])
    # df[['str_date']] = df[['int_date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[6:], s[4:6], s[0:4]))
    dates = np.array(rolling_dates, dtype=np.datetime64)
    result_df.insert(loc=0, column='Date', value=dates)
    rolling_ndays = len(dates)
    # result_df.insert(loc=0, column='Date', value=list(df['str_date']))

    betas = result_array[:,0:len(factors_to_plot)]
    alphas = result_array[:,-1]
    factor_values = df[factors_to_plot].values[rolling_window-1:,:]
    returns = (df['Daily_Returns']-df[factor_riskfree[0]]).values[rolling_window-1:]
    daily_att = (factor_values)*(betas)
    factor_returns = np.column_stack((daily_att,returns-daily_att.sum(axis=1)))
    factor_returns=factor_returns.cumsum(axis=0)
    factor_returns_df = pd.DataFrame(factor_returns, index=dates) # Values rounded to make the plots of beta
    factor_returns_df.columns = factors1
    
    numlines=len(factors_to_plot)
    mypalette=Spectral11[:4]+Spectral11[8:]
    source = ColumnDataSource(factor_returns_df)
    tooltips=[
            ( 'Date',   '$x{%F}'),
        ]
    f=figure(title = f'Cumulative Returns - Factorwise Rolling: {rolling_window} days window', plot_height=550, plot_width=1500, x_axis_location="above", 
               toolbar_location='above',x_axis_type="datetime", background_fill_color="#efefef", 
               x_range=(dates[(rolling_ndays)//3], dates[((rolling_ndays)*2)//3]), tooltips=tooltips) #
    select2 = figure(title="Drag the middle and edges of the selection box to change the range above",
                    plot_height=140, plot_width=1500, y_range=f.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    f.varea_stack(stackers=factors1, x='index', color=tuple(mypalette[0:numlines+1]), source=factor_returns_df, alpha = 0.85, legend_label=factors1)

    f.legend.click_policy="hide"
    f.legend.background_fill_alpha = 0
    f.legend.border_line_color = None
    f.legend.margin = 10
    f.legend.padding = 18
    f.legend.spacing = 10
    f.toolbar.logo = None

    f.title.text_color = mypalette[-1]
    f.title.text_alpha = 0.6
    f.title.text_font = "antiqua"
    f.title.text_font_size = "32px"
    f.title.align = "center"
    f.yaxis.axis_label = '%age Returns ITD'
    f.yaxis.axis_label_text_font_size= "18px"
    f.axis.major_label_text_font_size = '16px'
    f.legend.label_text_font_size = '16px'

    range_tool = RangeTool(x_range=f.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select2.varea_stack(stackers=factors1, x='index', color=tuple(mypalette[0:numlines+1]), source=factor_returns_df, alpha = 0.85)
    select2.ygrid.grid_line_color = None
    select2.add_tools(range_tool)
    select2.toolbar.active_multi = range_tool
    return column(f,select2)
    
    
    
    
 tab1 = pn.GridSpec(sizing_mode='stretch_width') #, max_height=800, 

tab1[0,0:8] = pn.Row(f'<h1 style="color:#808080;">For entire period: {ndays} trading days </h1>')
tab1[2:9,0:8] = factor_selection
tab1[9,3:18] = "<br/>"
tab1[9,0:3] = pn.Row(to_plot)
tab1[10:20,9:18] = pn.Row(attribution_pie_chart)
tab1[10:20,0:9] = pn.Row(attribution_waterfall_chart)

tab1[1,9:13] = '<h2 style="color:#6b8e23;">Strategy Returns Attribution</h2>'
tab1[2:9,9:13] = pn.Column(attribution_table)
tab1[1,14:18] = '<h2 style="color:#00008b;">Factor Sensitivity Analysis</h2>'
tab1[2:9,14:18] = pn.Column(factor_sensitivity)

dashboard = pn.Tabs()
dashboard.append(('Returns Attribution',tab1))

tab2 = pn.GridSpec(sizing_mode='stretch_both') #, max_height=800,
tab2[0:2,0:5] = factor_selection2
tab2[2,0:3] = rolling_window2
tab2[3:14,0:10] = plot_rolling_alpha

tab2[15,0:10] = '<br/>'
tab2[16:18,0:5] = factor_selection3
tab2[18,0:3] = rolling_window3
tab2[19:30,0:10] = plot_rolling_beta

tab2[31,0:10] = '<br/>'
tab2[32:34,0:5] = factor_selection4
tab2[34,0:3] = rolling_window4
tab2[35:46,0:10] = plot_returns_stack

dashboard.append(('Rolling Attribution',tab2))
dashboard.show()









   
