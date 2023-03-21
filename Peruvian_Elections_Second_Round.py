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
##dfStyler = df_data_dictionary_peruvian_elections.style.set_properties(**{'text-align': 'left'})
## dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
## print(df_data_dictionary_peruvian_elections)

print(tabulate(df_data_dictionary_peruvian_elections, showindex=False, headers=df_data_dictionary_peruvian_elections.columns))

from googletrans import Translator
translator = Translator()
df_data_dictionary_peruvian_elections_en = df_data_dictionary_peruvian_elections.copy()
df_data_dictionary_peruvian_elections_en.rename(columns=lambda x: translator.translate(x).text, inplace=True)

print(df_data_dictionary_peruvian_elections_en)






translations = {}
row = 0
english_spanish_column_dictionary = {}
for column in df_data_dictionary_peruvian_elections_en.columns:
    # unique elements of the column
    row = 0

    column_set = df_data_dictionary_peruvian_elections_en[column].unique()

    print("printing column:", column)

    for each_column in column_set:
        # add translation to the dictionary
        ##print(df_data_dictionary_peruvian_elections_en[column])
        ##print(element)

        translations[each_column] = translator.translate(each_column).text

        ### TESTING MARCH 2023
        ##df_data_dictionary_peruvian_elections_en.loc[row:, column] = \
        ##    translator.translate(each_column).text

        ### Creating a Dictionary of Spanish and English column names
          ### Since we only want the Variable Name Translations, we will exit when the column = "Description"

        if (column != "Description"):

            print(each_column)
            df_data_peruvian_elections.rename(columns={each_column:translator.translate(each_column).text}, inplace = True)
            english_spanish_column_dictionary[each_column] = translator.translate(each_column).text



        if (column == 'Description'):
            df_data_dictionary_peruvian_elections_en.loc[row:, column] = \
                translator.translate(each_column).text
        # df_data_dictionary_peruvian_elections_en.loc[row:,column] = translations[each_column]
        print("translation of element: ", translations[each_column])

        row += 1

print(df_data_peruvian_elections)
exit(0)
### Printing the Dictionary With Spanish and English Columns for Testing
for key in english_spanish_column_dictionary:
    print(key + ":" + english_spanish_column_dictionary[key])

print(df_data_dictionary_peruvian_elections_en)


# create excel writer object
writer = pd.ExcelWriter('output.xlsx')
# write dataframe to excel
df_data_dictionary_peruvian_elections_en.to_excel(writer)
# save the excel
writer.save()

### Printing the Dataframe with the Translated Columns
print( df_data_dictionary_peruvian_elections_en)


### Correcting the Errors In Translation
### First, the Dictionary of Spanish and English Columns
english_spanish_column_dictionary.update({'UBIGEO':'GEOLOCATION', 'TIPO_ELECCION':'ELECTION_TYPE', \
                                          'DESCRIP_ESTADO_ACTA':'VOTING_ACT_STATUS', \
                                          'N_CVAS':'TURNOUT', \
                                          'N_ELEC_HABIL':'REGISTERED_VOTERS',\
                                          'VOTOS_P1':"VOTES_WINNER", \
                                          'VOTOS_P2':'VOTES_LOSER', \
                                          'VOTOS_VB':'BLANK_VOTES', \
                                          'VOTOS_VN':'NULL_VOTES', \
                                          'VOTOS_VI':'CONTESTED_VOTES'})

df_data_peruvian_elections.rename({'UBIGEO':'GEOLOCATION', 'TIPO_ELECCION':'ELECTION_TYPE', \
                                          'DESCRIP_ESTADO_ACTA':'VOTING_ACT_STATUS', \
                                          'N_CVAS':'TURNOUT', \
                                          'N_ELEC_HABIL':'REGISTERED_VOTERS',\
                                          'VOTOS_P1':"VOTES_WINNER", \
                                          'VOTOS_P2':'VOTES_LOSER', \
                                          'VOTOS_VB':'BLANK_VOTES', \
                                          'VOTOS_VN':'NULL_VOTES', \
                                          'VOTOS_VI':'CONTESTED_VOTES'}, inplace = True)

print(english_spanish_column_dictionary)
for key in english_spanish_column_dictionary:
    print(key + ":" + english_spanish_column_dictionary[key])


print(df_data_dictionary_peruvian_elections_en['Description'])



df_data_dictionary_peruvian_elections_en.replace({'Variable' : english_spanish_column_dictionary}, inplace = True)

print(df_data_dictionary_peruvian_elections_en)


## df3.rename(columns = {'DiabetesPedigreeFunction':'DPF'}, inplace = True)


# df_data_peruvian_elections.rename(columns = {'UBIGEO':'GEOLOCATION', 'TIPO_ELECCION':'ELECTION_TYPE', \
#                                           'DESCRIP_ESTADO_ACTA':'VOTING_ACT_STATUS', \
#                                           'N_CVAS':'TURNOUT', \
#                                           'N_ELEC_HABIL':'REGISTERED_VOTERS',\
#                                           'VOTOS_P1':"VOTES_WINNER", \
#                                           'VOTOS_P2':'VOTES_LOSER', \
#                                           'VOTOS_VB':'BLANK_VOTES', \
#                                           'VOTOS_VN':'NULL_VOTES', \
#                                           'VOTOS_VI':'CONTESTED_VOTES'}, inplace = True)

print(df_data_peruvian_elections.info())


### Dropping the irrelevant variables

df_data_peruvian_elections.drop(['ELECTION_TYPE', 'VOTING_ACT_STATUS', 'OBSERVATION_TYPE'], axis=1, inplace=True)




exit(0)
##  Renaming the dictionary of terms in English

#df_data_dictionary_peruvian_elections_en.replace('BIGEO','GEOLOCATION', inplace = True)


print(df_data_dictionary_peruvian_elections_en)

exit(0)




exit(0)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

translated_columns = ["Geolocation", "Department", "Province", "District", "Type_of_Election", \
               "Polling_Station", "Voting_Act_Desc", "Observ_Type", "Votes_Cast", "Registered_voters",\
               "Votes_Winner",  "Votes_Loser", "Blank_Votes", "Nulled_Votes", \
               "Contested_Votes"]

translated_descriptions = ["Code of geographic location", "A Department is the equivalent of a State", \
                           "Province", "District", "Type of Election: Presidential/Congressional/Municipal", "Pollling station id",
                           "Voting Act Status (Opened, Installed,  Votes Connted, Votes Computed)", \
                           "Observation Code", "Number of Voters Casting a Vote",  "Number of Eligible / Registered Voters", \
                           "Votes winning party", "Votes losing party", "Blank_Votes", "Nulled_Votes", \
                           "Contested_Votes"]

translated_dictionary = pd.DataFrame(
    {
        'Column Names':translated_columns,
        'Description': translated_descriptions
    }
)

print("=================================Translated Dictionary=================================")

print(translated_dictionary.head(50))



print("=======================================================================================")
