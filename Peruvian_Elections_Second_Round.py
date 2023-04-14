import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from translate import Translator
import seaborn as sns
from scipy import stats
from utilities_project1 import Empirical_distribution
from tabulate import tabulate
from utilities_project1 import ConvertTextToNumber
import missingno as mno


PATH = '../downloads/'


## Loading the file into a pandas datafram
file_peruvian_elections = "elections_second_round.xlsx"

load_peruian_elections_dataset = os.path.join(PATH, file_peruvian_elections)

df_data_peruvian_elections = pd.read_excel(load_peruian_elections_dataset)

## Taking a look at the data
print(df_data_peruvian_elections.info())


### Loading the data dictionary
data_dictionart_peruvian_elections = "data_dictionary_peru_elections_2nd_round.xlsx"
load_peruian_elections_data_dictionary= os.path.join(PATH, data_dictionart_peruvian_elections)
df_data_dictionary_peruvian_elections = pd.read_excel(load_peruian_elections_data_dictionary)
pd.options.display.max_colwidth = 100
print(tabulate(df_data_dictionary_peruvian_elections, showindex=False, \
               headers=df_data_dictionary_peruvian_elections.columns))


from googletrans import Translator
translator = Translator()
df_data_dictionary_peruvian_elections_en = df_data_dictionary_peruvian_elections.copy()
df_data_dictionary_peruvian_elections.rename(columns=lambda x: translator.translate(x).text, inplace=True)

print(df_data_dictionary_peruvian_elections_en)






row = 0
english_spanish_column_dictionary = {}
for column in df_data_dictionary_peruvian_elections.columns:

    row = 0
    # The list of unique values in a dictionary column
    values_in_column = df_data_dictionary_peruvian_elections[column].unique()
    # We will create a spanish-english translation of each entry in the data dictionary
    for each_value in values_in_column:
        english_spanish_column_dictionary[each_value] = \
                 translator.translate(each_value).text  # Then add the translated variable name to the data dictionary
        row += 1


# Checking the accuracy of the trasnslations
print(">>>Printing the translations...")
import textwrap

wrapper = textwrap.TextWrapper(width=50)

for key in english_spanish_column_dictionary:
    print(key + " : " + wrapper.fill(text=english_spanish_column_dictionary[key]))
    print("=========================================================")




# Correcting the errors in the Spanish - English dictionaty

english_spanish_column_dictionary.update({'UBIGEO':'GEOLOCATION', 'TIPO_ELECCION':'ELECTION_TYPE', \
                                          'DESCRIP_ESTADO_ACTA':'POLLING_STATION_MINUTES', \
                                          'N_CVAS':'VOTES_CAST', \
                                          'N_ELEC_HABIL':'REGISTERED_VOTERS',\
                                          'VOTOS_P1':"VOTES_WINNER", \
                                          'VOTOS_P2':'VOTES_LOSER', \
                                          'VOTOS_VB':'BLANK_VOTES', \
                                          'VOTOS_VN':'NULL_VOTES', \
                                          'VOTOS_VI':'CONTESTED_VOTES'})

# Making sure the corrections to the Spanish - English are there
print(">>>Printing the corrected translations...")
wrapper = textwrap.TextWrapper(width=50)
for key in english_spanish_column_dictionary:
    print(key + " : " + wrapper.fill(text=english_spanish_column_dictionary[key]))
    print("=========================================================")


# Using the corrected Spanish - English dictionary to translate the Data Dictionary
for key in english_spanish_column_dictionary:
    df_data_dictionary_peruvian_elections.replace(key, english_spanish_column_dictionary[key], inplace = True)

print(">>>Printing translated data dictionary ... ")
print(df_data_dictionary_peruvian_elections)



# Using the Spanhsh - English dictionary  to rename the elections dataset column names

for key in english_spanish_column_dictionary:
    df_data_peruvian_elections.rename(columns = {key: english_spanish_column_dictionary[key]}, inplace = True)



polling_station_minutes_dictionary = {'ANULADA':'ANNULLED', 'COMPUTADA RESUELTA':'RESOLVED', 'CONTABILIZADA':'ENTERED',
                                      'EN PROCESO':'IN_PROCESS', 'SIN INSTALAR':'NOT_INSTALLED'}
df_data_peruvian_elections = df_data_peruvian_elections.replace({"POLLING_STATION_MINUTES": polling_station_minutes_dictionary})


pd.options.display.max_colwidth = 100
pd.set_option('display.max_columns', None)

print(">>Data frame with transated columns...")
print(df_data_peruvian_elections.info())

# Dropping irrelevant columns
df_data_peruvian_elections.drop(['ELECTION_TYPE',  'OBSERVATION_TYPE'], axis=1, inplace=True)


print(">>>Print the trimmed data frame without irrelevant columns: ")
print(df_data_peruvian_elections)
print(df_data_peruvian_elections.info())

# Cross checking the vote count
df_data_peruvian_elections["TOTAL_VOTES"] = df_data_peruvian_elections.iloc[:,8:12].sum(axis=1)
df_data_peruvian_elections_validate_total_votes = df_data_peruvian_elections.filter(["GEOLOCATION", "TOTAL_VOTES", "VOTES_CAST", "POLLING_STATION_MINUTES"])

print("Total Votes and Votes Cast: \n", df_data_peruvian_elections_validate_total_votes.head(50))
total_votes_nationally = df_data_peruvian_elections_validate_total_votes["TOTAL_VOTES"].sum(axis=0)
total_votes_cast_nationally = df_data_peruvian_elections_validate_total_votes["VOTES_CAST"].sum(axis=0)

if(total_votes_cast_nationally > total_votes_nationally):
    print("VOTES_CAST is greater than the total number of votes in the ballot.")
elif(total_votes_cast_nationally < total_votes_nationally):
    print("VOTES_CAST is less than the total number of votes in the ballot.")
else:
    print("VOTES_CAST is equal to the total number of votes in the ballot.")

# Counting null values in the df_data_peruvian_elections dataframe
print(df_data_peruvian_elections.isnull().sum()/len(df_data_peruvian_elections) )
pd.options.display.float_format = '{:.2%}'.format
print(df_data_peruvian_elections.isnull().sum()/len(df_data_peruvian_elections))

pd.options.display.float_format = '{:}'.format
# Detecting outliers with the describe() function
print(df_data_peruvian_elections.describe())

# Plotting histograms using Plotly library
import plotly.express as px
fig = px.histogram(df_data_peruvian_elections["NULL_VOTES"], x='NULL_VOTES')
fig.show()

fig = px.histogram(df_data_peruvian_elections["BLANK_VOTES"], x='BLANK_VOTES')
fig.show()


# Add two dummy variables "Annuled" or "Counted"

df3 = pd.get_dummies(df_data_peruvian_elections,
                     columns=['POLLING_STATION_MINUTES'])
print(df3.head())

df_data_peruvian_elections = pd.get_dummies(df_data_peruvian_elections,
                     columns=['POLLING_STATION_MINUTES'])

# Creating a binary variable to quantify the nulls in the VOTES_CAST variable
df_data_peruvian_elections['NULLS_IN_VOTES_CAST'] =  df_data_peruvian_elections['VOTES_CAST'].apply(lambda x: 0 if x>=0 else 1)

print(df_data_peruvian_elections.filter(['POLLING_STATION_MINUTES_NOT_INSTALLED', 'NULLS_IN_VOTES_CAST' ]))

# Creating a data frame of the variables we want to see the correlation between the
        #      quantifued nulls in Votes_Cast (NULLS_IN_VOTES_CAST) and the polling station minutes status
df_correlation_of_nulls_VotesCast_and_PollingStationsStatus = df_data_peruvian_elections.filter(['POLLING_STATION_MINUTES_NOT_INSTALLED',  'GEOLOCATION',
              'POLLING_STATION_MINUTES_IN_PROCESS','POLLING_STATION_MINUTES_ENTERED', 'POLLING_STATION_MINUTES_ANNULLED',
              'POLLING_STATION_MINUTES_RESOLVED', 'NULLS_IN_VOTES_CAST'])
# Drawing the correlation matrix
f = plt.figure(figsize=(9, 9))
plt.matshow(df_correlation_of_nulls_VotesCast_and_PollingStationsStatus.corr(), fignum=f.number)
plt.xticks(range(df_correlation_of_nulls_VotesCast_and_PollingStationsStatus.select_dtypes(['number']).shape[1]), df_correlation_of_nulls_VotesCast_and_PollingStationsStatus.\
           select_dtypes(['number']).columns, \
           fontsize=5, rotation=60)
plt.yticks(range(df_correlation_of_nulls_VotesCast_and_PollingStationsStatus.select_dtypes(['number']).shape[1]), df_correlation_of_nulls_VotesCast_and_PollingStationsStatus.\
           select_dtypes(['number']).columns, fontsize=5, rotation=60)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Correlation Matrix', fontsize=10)
plt.show()


# Imputing Nulls in Votes_Cast with 0
df_data_peruvian_elections['VOTES_CAST'] = \
    df_data_peruvian_elections['VOTES_CAST'].fillna(0)


# Creating two binary variables to quantify the nulls in the VOTES_WINNER and VOTES_LOSER variables
df_data_peruvian_elections['NULLS_IN_VOTES_WINNER'] =  df_data_peruvian_elections['VOTES_WINNER'].apply(lambda x: 0 if x>=0 else 1)
df_data_peruvian_elections['NULLS_IN_VOTES_LOSER'] =  df_data_peruvian_elections['VOTES_LOSER'].apply(lambda x: 0 if x>=0 else 1)
# Creating a data frame of the variables we want to see the correlation between the
# quantifued nulls in Votes_Cast (NULLS_IN_VOTES_CAST) and the polling station minutes status
df_correlation_of_nulls_VotesWinner_and_Loser_and_PollingStationsStatus = df_data_peruvian_elections.filter(['POLLING_STATION_MINUTES_NOT_INSTALLED',  'GEOLOCATION',
              'POLLING_STATION_MINUTES_IN_PROCESS','POLLING_STATION_MINUTES_ENTERED', 'POLLING_STATION_MINUTES_ANNULLED',
              'POLLING_STATION_MINUTES_RESOLVED', 'NULLS_IN_VOTES_WINNER','NULLS_IN_VOTES_LOSER' ])
# Drawing the correlation matrix
f = plt.figure(figsize=(9, 9))
plt.matshow(df_correlation_of_nulls_VotesWinner_and_Loser_and_PollingStationsStatus.corr(), fignum=f.number)
plt.xticks(range(df_correlation_of_nulls_VotesWinner_and_Loser_and_PollingStationsStatus.select_dtypes(['number']).shape[1]),
           df_correlation_of_nulls_VotesWinner_and_Loser_and_PollingStationsStatus.\
           select_dtypes(['number']).columns, \
           fontsize=5, rotation=60)
plt.yticks(range(df_correlation_of_nulls_VotesWinner_and_Loser_and_PollingStationsStatus.select_dtypes(['number']).shape[1]),
           df_correlation_of_nulls_VotesWinner_and_Loser_and_PollingStationsStatus.\
           select_dtypes(['number']).columns, fontsize=5, rotation=60)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Correlation Matrix', fontsize=10)
plt.show()


# Imputing Nulls in Votes_Winner and Votes_Loser with 0
df_data_peruvian_elections['VOTES_WINNER'] = \
    df_data_peruvian_elections['VOTES_WINNER'].fillna(0)

df_data_peruvian_elections['VOTES_LOSER'] = \
    df_data_peruvian_elections['VOTES_LOSER'].fillna(0)



print("Non-Null Contested_Votes\n---------------")
print(df_data_peruvian_elections[df_data_peruvian_elections['CONTESTED_VOTES'].notnull()])

print("Null Contested_Votes\n---------------")
print(df_data_peruvian_elections[df_data_peruvian_elections['CONTESTED_VOTES'].isnull()])


# Dropping Contested_Votes, Blank_Votes, Null_Votes and Total_Votes
df_data_peruvian_elections.drop(['BLANK_VOTES', 'NULL_VOTES',
                                 'CONTESTED_VOTES', 'TOTAL_VOTES' ], axis=1, inplace=True)

print(df_data_peruvian_elections.head())



