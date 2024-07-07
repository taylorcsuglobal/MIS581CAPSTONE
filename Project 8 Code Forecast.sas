/* Import the dataset */
proc import datafile='/home/u63831335/sasuser.v94/Productlevel_Sales_Transactions_Dataset_Weekly.csv'
    out=sales_data
    dbms=csv
    replace;
    getnames=yes;
run;

/* Restructure the dataset to have week and sales columns */
data sales_data_long;
    set sales_data;
    array weeks {*} Wk:;
    do i = 1 to dim(weeks);
        week = i - 1;
        sales = weeks[i];
        output;
    end;
    drop i Wk:;
run;

/* Aggregate sales data to weekly totals */
proc sql;
    create table weekly_sales as
    select week, sum(sales) as total_sales
    from sales_data_long
    group by week;
quit;

/* Train ARIMA model */
proc arima data=weekly_sales;
    identify var=total_sales;
    estimate p=1 q=1;
    forecast lead=260 interval=week out=forecast_results;
run;

/* Separate the actual and forecasted data */
data actual forecast;
    set forecast_results;
    if _N_ <= 104 then output actual;  /* Actual data */
    else output forecast;              /* Forecasted data */
run;

/* Print the actual data */
proc print data=actual;
    title "Actual Sales Data";
run;

/* Print the forecast data */
proc print data=forecast;
    title "Sales Forecast for the Next Year and Five Years";
    var FORECAST STD L95 U95;
run;
