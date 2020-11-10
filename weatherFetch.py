from wwo_hist import retrieve_hist_data
import urllib

start_date = '1-JAN-2009'
end_date = '12-31-2019'
api_key = '6fc75e9cd9ef4499bf551057202410'
location_list = ['San+Antonio,TX']
frequency = 1

output_csv = retrieve_hist_data(
    api_key, location_list, start_date, end_date, frequency)
