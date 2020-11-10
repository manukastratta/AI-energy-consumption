import pandas as pd
import datetime as dt
import dateutil as dateutil


def ExcelDateToDateTime(xlDate):
    xlDateTime = xlDate.to_pydatetime()
    return (xlDateTime.replace(second=0, microsecond=0, minute=0, hour=xlDateTime.hour)
            + dt.timedelta(hours=xlDateTime.minute//30))


def incrementDate(xlDate):
    month = int(xlDate[:2])
    day = int(xlDate[3:5])
    day = day + 1
    if day > 31:
        month = month + 1
        day = 1
    elif day > 30 and (month == 4 or month == 6 or month == 9 or month == 11):
        month = month + 1
        day = 1
    elif day > 28 and month == 2:
        month = month + 1
        day = 1
    monthString = str(month).zfill(2)
    dayString = str(day).zfill(2)
    return monthString + '/' + dayString + '/' + xlDate[6:]


def DateToDateTime2(xlDate):
    hourSub = xlDate[11: 13]
    if hourSub == '24':
        incrementedDate = incrementDate(xlDate[:10])
        xlDate = incrementedDate + ' 00:00'
    xlDateTime = dateutil.parser.parse(xlDate)
    xlDateTimeString = xlDateTime.isoformat()
    xlDateTime = dateutil.parser.parse(xlDateTimeString)
    return (xlDateTime.replace(second=0, microsecond=0, minute=0, hour=xlDateTime.hour)
            + dt.timedelta(hours=xlDateTime.minute//30))


def calc_texas():
    temp_column_names = ['uvIndex', 'HeatIndexC',
                         'WindChillC', 'humidity', 'tempC']
    tx_temp_df = pd.DataFrame(index=None, columns=temp_column_names)

    tx_san_antonio_df = pd.read_csv(
        'texasdata/san_antonio_temp_2009_2019.csv', header=0,  names=temp_column_names, usecols=[5, 13, 14, 17, 20], dtype={'uvIndex': int, 'HeatIndexC': int, 'WindChillC': int, 'humidity': int, 'tempC': int})
    tx_austin_df = pd.read_csv(
        'texasdata/austin_temp_2009_2019.csv', header=0, names=temp_column_names, usecols=[5, 13, 14, 17, 20], dtype={'uvIndex': int, 'HeatIndexC': int, 'WindChillC': int, 'humidity': int, 'tempC': int})

    for col_name in tx_san_antonio_df:
        tx_san_antonio_df[col_name] = 0.6 * tx_san_antonio_df[col_name]
        tx_austin_df[col_name] = 0.4 * tx_austin_df[col_name]
        tx_temp_df[col_name] = tx_san_antonio_df[col_name] + \
            tx_austin_df[col_name]
        tx_temp_df[col_name] = tx_temp_df[col_name].astype(int)

    power_column_names = ['Date', 'MWh']
    tx_res_df = pd.DataFrame(index=None, columns=power_column_names)

    for i in range(9, 16):
        tx_load_df = pd.read_excel(
            f'texasdata/native_load_{i}.xls', names=power_column_names, header=0, index_col=None, usecols='A,E', dtype={'MWh': float})
        tx_load_df['Date'] = tx_load_df['Date'].apply(ExcelDateToDateTime)
        tx_res_df = tx_res_df.append(tx_load_df)

    tx_load_df1 = pd.read_excel('texasdata/native_load_16.xlsx', names=power_column_names,
                                header=0, index_col=None, usecols='A,E', dtype={'MWh': float})
    tx_load_df1['Date'] = tx_load_df1['Date'].apply(ExcelDateToDateTime)
    tx_res_df = tx_res_df.append(tx_load_df1)

    for i in range(17, 20):
        tx_load_df = pd.read_excel(
            f'texasdata/native_load_{i}.xlsx', names=power_column_names, header=0, index_col=None, usecols='A,E', dtype={'MWh': float})
        tx_load_df['Date'] = tx_load_df['Date'].apply(DateToDateTime2)
        tx_res_df = tx_res_df.append(tx_load_df)
    for col_name in tx_temp_df:
        tx_res_df[col_name] = tx_temp_df[col_name]

    print(tx_res_df)

    tx_res_df.to_csv('texas_2009_to_2019_dataset01.csv', index=False)


calc_texas()
