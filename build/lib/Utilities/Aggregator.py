import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import re
import spacy
import seaborn as sns

class AggregateandPlot():

    def __init__(self):
        self.text = ""



    def aggregate(self, df, group_columns_by, aggregate_column):
        '''
        INPUT
        :param df: The dataframe where we want to aggregate columns
        :param group_by_columns: What column we need to goup the data by
        :param aggregate_column: What values we want to aggregate
        OUTPUT

        An aggregated dataframe


        '''

        ##new_df = df[df["available"] == 't']
        new_df = df.copy()



        ##print(new_df)

        # for i in range(len(df)):
        #     ##print(df.loc[i, "date"])
        #     ##print(str(df.loc[i, "date"])[0:4])
        #     new_df.loc[i, "date"] =  str(df.loc[i, "date"])[0:4]
        agg_functions = {

            'listing_id':
                ['nunique'],

            'listing_id':
                ['count']
        }

        agg_functions2 = {
            'is_available':
                ['sum']
        }

        agg_functions3 = {

            ##'listing_id':
              ##  ['nunique'],

            'num_price':
                ['mean']

        }

        agg_functions4 = {

            ##'reviewer_id':
            ##   ['nunique'],

            'reviewer_id':
                ['nunique']
        }




        if group_columns_by == "month"  and aggregate_column == "num_price":
            ##split = pd.DataFrame()
           ##split['month'] = pd.to_datetime(df['date']).dt.month
            ##new_df = df.join(split, on=None, how='left', sort=False)
            new_df['month'] = pd.to_datetime(df['date']).dt.month
            new_df = new_df.loc[new_df['available'] == 't']
            avg_price_listing = pd.DataFrame(new_df.groupby(['month']).mean(), \
                                             columns=['num_price'])

            new_df = avg_price_listing
            ##new_df = new_df.groupby('month').mean()

        if group_columns_by == "month" and aggregate_column == "listing_id":
            new_df['month'] = pd.to_datetime(df['date']).dt.month
            new_df = new_df.loc[new_df['available'] == 't'].groupby(['month']).agg(agg_functions)


        if group_columns_by == "date" and aggregate_column == "listing_id":
            new_df['month_year'] = pd.to_datetime(new_df['date']).dt.strftime('%m-%Y')
            new_df['month_year'] = pd.to_datetime(new_df['month_year'], format='%m-%Y')
            new_df = new_df.loc[new_df['available'] == 't'].groupby(["month_year"]).agg(agg_functions)
            new_df = new_df.sort_values(by=['month_year'])

        if group_columns_by == "date" and aggregate_column == "num_price":
            new_df['month_year'] = pd.to_datetime(new_df['date']).dt.strftime('%m-%Y')
            new_df['month_year'] = pd.to_datetime(new_df['month_year'], format='%m-%Y')
            # new_df = new_df.loc[new_df['available'] == 't'].groupby(["month_year"]).agg(agg_functions3)

            new_df = new_df.loc[new_df['available'] == 't']
            avg_price_listing = pd.DataFrame(new_df.groupby(['month_year', 'listing_id']).mean(), \
                                             columns=['num_price'])
            new_df = pd.DataFrame(avg_price_listing.groupby(['month_year']).mean(), columns=['num_price'])

            # new_df.columns = ["_".join(col).strip() for col in new_df.columns.values]
            # new_df.reset_index(inplace=True)
            new_df = new_df.sort_values(by=['month_year'])

        if group_columns_by == "available" and aggregate_column == 'is_available':
            new_df = new_df.groupby(["listing_id"]).agg(agg_functions2)

        if group_columns_by == "host_listings_count" and aggregate_column == 'num_price' :
            new_df = new_df.groupby(["host_listings_count"]).agg(agg_functions3)


        if group_columns_by == "listing_id" and aggregate_column == "reviewer_id":
            ##new_df = new_df.groupby([group_columns_by ]).agg(agg_functions)
            ##new_df = new_df.groupby([group_columns_by]).agg(agg_functions4)
            ##new_df = pd.DataFrame(new_df.groupby([group_columns_by]).count(), \
                                           # columns=['reviewer_id'])
            ##new_df = new_df.groupby([group_columns_by]).agg(unique_reviewer_counts = ('reviewer_id', 'nunique'))
            new_df =pd.DataFrame(new_df.groupby([group_columns_by]).agg(unique_reviewer_counts=('reviewer_id', 'nunique')).\
                                 reset_index(), \
                    columns = ['listing_id', 'unique_reviewer_counts'])

        ### JAN 21 2022
        if group_columns_by == "listing_id" and aggregate_column == "id":
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(review_counts=('id', 'count')). \
                reset_index(), \
                columns=['listing_id', 'review_counts'])



        #### UPDATED USES agg and creates the name for yhe aggregated data
        if group_columns_by == "neighbourhood_cleansed" and aggregate_column == "reviewer_id":
            ##new_df = pd.DataFrame(new_df.groupby([group_columns_by]).count(), \
                                  ##columns=['reviewer_id'])
            ### I FOUND THIS WAY TO AGGREGATE AND RENAME THE COLUMN
            ### https://stackoverflow.com/questions/19078325/naming-returned-columns-in-pandas-aggregate-function
            # new_df =new_df.groupby([group_columns_by]).agg(unique_reviewer_counts =\
            #                                                    ('reviewer_id', 'nunique'))
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(unique_reviewer_counts=('reviewer_id', 'nunique')). \
                    reset_index(), \
                columns=['neighbourhood_cleansed', 'unique_reviewer_counts'])


        if group_columns_by == "neighbourhood_cleansed" and aggregate_column == "listing_id":
            new_df = pd.DataFrame(new_df.groupby([group_columns_by]).nunique(), \
                                columns=['listing_id'])
            # new_df = pd.DataFrame(
            #     new_df.groupby([group_columns_by]).agg(listing_id=('listing_id', 'count')). \
            #         reset_index(), \
            #     columns=['listing_id'])


        ### JAN 22 20:04 pm 2022
        if group_columns_by == "neighbourhood_cleansed" and aggregate_column == "id":
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(listings =('id', 'count')). \
                    reset_index(), \
                columns=[group_columns_by, 'listings'])

        if group_columns_by == "neighbourhood_cleansed" and aggregate_column == 'host_experience_numeric':
            new_df = pd.DataFrame(new_df.groupby([group_columns_by]).mean(), \
                                  columns=['host_experience_numeric'])

        #### UPDATED USES agg and creates the name for yhe aggregated data
        if group_columns_by == "neighbourhood_cleansed" and aggregate_column == 'number_of_reviews' :
            ##new_df = pd.DataFrame(new_df.groupby([group_columns_by]).sum(), \
                            ##      columns=[aggregate_column])
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(total_reviews=('number_of_reviews', 'sum')). \
                reset_index(), \
                columns=['neighbourhood_cleansed', 'total_reviews'])

        if group_columns_by == 'host_identity_verified_t' and aggregate_column == 'number_of_reviews':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(total_reviews=('number_of_reviews', 'sum')). \
                    reset_index(), \
                columns=['host_identity_verified_t', 'total_reviews'])

        if group_columns_by == 'host_has_profile_pic_t' and aggregate_column == 'number_of_reviews':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(total_reviews=('number_of_reviews', 'sum')). \
                    reset_index(), \
                columns=['host_has_profile_pic_t', 'total_reviews'])



        if group_columns_by == "host_id" and aggregate_column == 'number_of_reviews':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(total_reviews=('number_of_reviews', 'sum')). \
                    reset_index(), \
                columns=['host_id', 'total_reviews'])

        ### JAN 23 2022 11:43

        if group_columns_by == "host_id" and aggregate_column == 'id':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(total_listings=('id', 'count')). \
                    reset_index(), \
                columns=['host_id', 'total_listings'])


        if group_columns_by == "host_id" and aggregate_column == 'host_about':
            new_df = new_df.drop_duplicates(keep = 'last')

        if group_columns_by == 'sentiment_above_average' and aggregate_column == 'total_reviews':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(total_reviews=('total_reviews', 'sum')). \
                    reset_index(), \
                columns=['sentiment_above_average', 'total_reviews'])

        if group_columns_by == 'neighbourhood_cleansed'  and aggregate_column == 'sentiment_value':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(mean_sentiment=('sentiment_value', 'mean')). \
                    reset_index(), \
                columns=['neighbourhood_cleansed', 'mean_sentiment'])
            # new_df = pd.DataFrame(new_df.groupby(['neighbourhood_cleansed']).mean(), \
            #                                  columns=['sentiment_value'])

        if group_columns_by == 'neighbourhood_cleansed' and aggregate_column ==  'mean_sentiment_above_average':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(listings_above_average=('mean_sentiment_above_average', 'sum')). \
                    reset_index(), \
                columns=['neighbourhood_cleansed', 'listings_above_average'])

        if group_columns_by == 'neighbourhood_cleansed'  and aggregate_column == 'review_scores_location':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(mean_review_scores_location=('review_scores_location', 'mean')). \
                    reset_index(), \
                columns=['neighbourhood_cleansed', 'mean_review_scores_location'])

        ### JAN 21 2022
        if group_columns_by == 'neighbourhood_cleansed' and   aggregate_column == 'num_price':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(mean_price=('num_price', 'mean')). \
                    reset_index(), \
                columns=['neighbourhood_cleansed', 'mean_price'])


        if group_columns_by == 'city' and aggregate_column == 'id':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(listings_in_city=('id', 'count')). \
                    reset_index(), \
                columns=[group_columns_by, 'listings_in_city'])

        ### JAN 24 2022
        if group_columns_by == 'neighbourhood_cleansed' and   aggregate_column == 'total_listings':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(mean_hosts=('total_listings', 'mean')). \
                    reset_index(), \
                columns=['neighbourhood_cleansed', 'mean_hosts'])

       ### MARCH 11 2022
       ###
        if group_columns_by == 'ubigeo' and   aggregate_column == 'pct_validos':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(perc_cast=('pct_emitidos', 'sum')). \
                    reset_index(), \
                columns=['ubigeo', 'perc_cast'])
        if group_columns_by == 'ubigeo' and   aggregate_column == 'pct_emitidos':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(perc_valid=('pct_validos', 'sum')). \
                    reset_index(), \
                columns=['ubigeo', 'perc_valid'])
        if group_columns_by == 'ubigeo' and aggregate_column == 'cod_partido':
                new_df = pd.DataFrame(
                    new_df.groupby([group_columns_by]).agg(party_votes=('pct_emitidos', 'sum')). \
                        reset_index(), \
                    columns=['ubigeo', 'perc_cast'])
        if group_columns_by == 'ubigeo' and aggregate_column == 'num_electores':
                new_df = pd.DataFrame(
                    new_df.groupby([group_columns_by]).agg(registered_voters=('num_electores', 'sum')). \
                        reset_index(), \
                    columns=['ubigeo', 'registered_voters'])
                print("now")
        if group_columns_by == 'Geolocation' and aggregate_column == 'all':
                new_df = pd.DataFrame(
                    new_df.groupby([group_columns_by]).agg(Votes_P16=('Votes_P16', 'sum'), Votes_P13 = ('Votes_P13', 'sum'), \
                        Votes_P11 = ('Votes_P11', 'sum'), Votes_Cast = ('Votes_Cast', 'sum'), Registered_voters = ('Registered_voters', 'sum') ).
                        reset_index(), \
                    columns=['Geolocation', 'Votes_P16', 'Votes_P13', 'Votes_P11','Votes_Cast', 'Registered_voters'  ])

        if group_columns_by == 'Province' and aggregate_column == 'all':
                new_df = pd.DataFrame(
                    new_df.groupby([group_columns_by]).agg(Votes_P16=('Votes_P16', 'sum'), Votes_P13 = ('Votes_P13', 'sum'), \
                        Votes_P11 = ('Votes_P11', 'sum'), Votes_Cast = ('Votes_Cast', 'sum'), Registered_voters = ('Registered_voters', 'sum') ).
                        reset_index(), \
                    columns=['Province', 'Votes_P16', 'Votes_P13', 'Votes_P11','Votes_Cast', 'Registered_voters'  ])

        if group_columns_by == 'Geolocation' and aggregate_column == 'all_1':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(Votes_P16=('Votes_P16', 'sum'), Votes_P13=('Votes_P13', 'sum'), \
                                                       Votes_P11=('Votes_P11', 'sum'), Votes_Cast=('Total_Votes_Three_Main_Candidates', 'sum'),
                                                       Registered_voters=('Registered_voters', 'sum')).
                    reset_index(), \
                columns=['Geolocation', 'Votes_P16', 'Votes_P13', 'Votes_P11', 'Total_Votes_Three_Main_Candidates', 'Registered_voters'])

        if group_columns_by == 'Department' and aggregate_column == 'all_1':
            new_df = pd.DataFrame(
                new_df.groupby([group_columns_by]).agg(Votes_P16=('Votes_P16', 'sum'), Votes_P13=('Votes_P13', 'sum'), \
                                                       Votes_P11=('Votes_P11', 'sum'), Votes_Cast=('Total_Votes_Three_Main_Candidates', 'sum'),
                                                       Registered_voters=('Registered_voters', 'sum')).
                    reset_index(), \
                columns=['Department', 'Votes_P16', 'Votes_P13', 'Votes_P11', 'Total_Votes_Three_Main_Candidates', 'Registered_voters'])








        return new_df

    def aggregate_by_province(self, df):

        votes_p11 = {}
        votes_p13 = {}
        votes_p16 = {}
        votes_cast = {}
        reg_voters = {}
        print(df.head())
        data = []
        counter = 0

        ##from IPython.display import display, HTML

        for province in df['Province'].unique():

            tempdf = df[df['Province']  == province]





            Votes_P11 = tempdf['Votes_P11'].sum()
            Votes_P16 = tempdf['Votes_P16'].sum()

            Votes_P13 = tempdf['Votes_P13'].sum()
            Votes_Cast = tempdf['Votes_Cast'].sum()

            Registered_voters = tempdf['Registered_voters'].sum()

            ## AUG 29 2022
            votes = {"Province": province, "votes_p11": Votes_P11}
            votes_p11[counter] =  votes

            votes = {"Province": province, "votes_p16": Votes_P16}
            votes_p16[counter] = votes

            votes = {"Province": province, "votes_p13": Votes_P13}
            votes_p13[counter] = votes

            votes = {"Province": province, "votes_cast": Votes_Cast}
            votes_cast[counter] = votes

            votes = {"Province": province, "reg_voters": Registered_voters}
            reg_voters[counter] = votes





            counter += 1






        dataframe_elections_by_province_p11 = pd.DataFrame.from_dict(votes_p11, orient='index')
        dataframe_elections_by_province_p16 = pd.DataFrame.from_dict(votes_p16, orient='index')
        dataframe_elections_by_province_p13 = pd.DataFrame.from_dict(votes_p13, orient='index')
        dataframe_elections_by_province_cast = pd.DataFrame.from_dict(votes_cast, orient='index')
        dataframe_elections_by_province_registered = pd.DataFrame.from_dict(reg_voters, orient='index')





        dataframe_elections_by_province = pd.merge(dataframe_elections_by_province_p11, dataframe_elections_by_province_p16, \

                                                on = 'Province' )

        dataframe_elections_by_province = pd.merge(dataframe_elections_by_province, dataframe_elections_by_province_p13, \
                                                on='Province'  )

        dataframe_elections_by_province = pd.merge(dataframe_elections_by_province, dataframe_elections_by_province_cast, \
                                                on='Province'  )

        dataframe_elections_by_province = pd.merge(dataframe_elections_by_province, dataframe_elections_by_province_registered, \
                                                on='Province'  )


        return dataframe_elections_by_province






    def plot_line(self, df):
        df.plot(kind='line')
        plt.show()

    def plot_scatter(self, df, x_title, y_title):

        sns.scatterplot(x= x_title, y= y_title, data=df)
        plt.show()

    def plot_stacked_bar_chart(self, x_label, y_label, df):
        # ax = df.plot(kind='bar', stacked=True, figsize=(15, 10), rot='horizontal', xlabel='x_label', ylabel='y_label')
        # for rect in ax.patches:
        #     # Find where everything is located
        #     height = rect.get_height()
        #     width = rect.get_width()
        #     x = df['neighbourhood_cleansed']
        #     y = rect.get_y()
        #
        #     # The height of the bar is the data value and can be used as the label
        #     label_text = f'{height}'  # f'{height:.2f}' to format decimal values
        #
        #     # ax.text(x, y, text)
        #     label_x = x  + width / 2
        #     label_y = y + height / 2
        #
        #     # plot only when height is greater than specified value
        #     if height > 0:
        #         ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8)
        #
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # ax.set_ylabel(y_label, fontsize=18)
        # ax.set_xlabel(x_label, fontsize=18)
        fig, ax = plt.subplots()
        N = df.shape[0]
        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars: can also be len(x) sequence

        p1 = ax.bar(ind, df['listings_below_average'].tolist(),  width, label='Listings With < Average Sentiment')
        p2 = ax.bar(ind, df['listings_above_average'].tolist(), width, label='Listings With > Average Sentiment', \
                    bottom = df['listings_below_average'].tolist())
        plt.xticks(ind, df['neighbourhood_cleansed'], fontsize=7)
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_ylabel(y_label)
        plt.rc('xtick', labelsize=8)
        ax.set_title('Sentiment in Neighbourhoods with Below Average Listing Concentration')

        ##ax.set_xticklabels( df['neighbourhood_cleansed'].tolist(), fontsize=7)



        ax.legend()

        plt.show()
