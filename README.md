# CLTV Prediction with BG/NBD and Gamma-Gamma Models

This repository contains code and data for predicting Customer Lifetime Value (CLTV) using the BG/NBD and Gamma-Gamma models. The goal is to estimate the potential value that existing customers will bring to the company in order to support sales and marketing activities.

## Dataset

The dataset consists of historical transactional data from customers who made purchases through both online and offline channels between 2020 and 2021. It includes information such as customer IDs, order channels, order dates, total order numbers, total customer value, and interested categories.

## Business Problem

FLO, the company, aims to develop a roadmap for its sales and marketing activities. To plan for the medium to long term, it needs to predict the potential value that existing customers will generate in the future.

## Tasks

### Task 1: Data Preparation

- Read and load the "flo_data_20k.csv" dataset.
- Handle outliers by defining functions to set outlier thresholds and replace outliers with the thresholds.
- Apply outlier handling to the variables: "order_num_total_ever_online," "order_num_total_ever_offline," "customer_value_total_ever_offline," and "customer_value_total_ever_online."
- Create new variables to capture the total order numbers and total customer value for omnichannel customers.
- Examine variable types and convert date variables to the date format.

### Task 2: Creating the CLTV Data Structure

- Set the analysis date as two days after the last purchase date in the dataset.
- Create a new dataframe called cltv_df with columns for customer ID, recency (weekly), T (weekly), frequency, and monetary average.

### Task 3: Building the BG/NBD and Gamma-Gamma Models, Calculating CLTV

- Fit the BG/NBD model and predict expected sales for customers in the next 3 months (exp_sales_3_month) and 6 months (exp_sales_6_month).
- Fit the Gamma-Gamma model and predict the expected average value per transaction (exp_average_value).
- Calculate the 6-month CLTV using the fitted models and add it to the cltv_df dataframe.
- Identify the top 20 customers with the highest CLTV.

### Task 4: Segmenting Customers Based on CLTV

- Divide all customers into 4 segments (A, B, C, D) based on their 6-month CLTV.
- Add the segment labels to the cltv_df dataframe.
- Analyze the segment averages for recency, frequency, and monetary value.

## Bonus

The entire process can be further automated by implementing a function.

Please note that this is a brief summary of the code and its tasks. For a more detailed explanation and code implementation, please refer to the provided code file.


