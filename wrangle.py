import pandas as pd
import numpy as np
import os
import explore as ex

# acquire
from env import host, user, password
from pydataset import data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, RFE, SelectKBest
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

##########################################################################################

# Aquire Data

##########################################################################################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'    
       

        
#----------------------------------------------------------------------------------------#
        
###### Zillow Database        
        
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
    
select *
from properties_2017

join (
			select parcelid, max(logerror) as logerror, max(transactiondate) as transactiondate
			from predictions_2017
			group by parcelid
		) as pred_17
	using(parcelid)
left join airconditioningtype using(airconditioningtypeid)
left join architecturalstyletype using(architecturalstyletypeid)
left join buildingclasstype using(buildingclasstypeid)
left join heatingorsystemtype using(heatingorsystemtypeid)
left join storytype using(storytypeid)
left join typeconstructiontype using(typeconstructiontypeid)

where year(transactiondate) = 2017
;
                '''
    
    return pd.read_sql(sql_query, get_connection('zillow'))
        
    
        
def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from Codeup database and writes data to a csv file if cached == False or if cached == True reads in titanic df from a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    return df

#----------------------------------------------------------------------------------------#

###### mall_customers Database        

def new_mall_data():
    '''
    This function reads in the mall_customers data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
                    select *
                    from customers;
                '''
    
    return pd.read_sql(sql_query, get_connection('mall_customers'))
        
    
        
def get_mall_data(cached=False):
    '''
    This function reads in zillow data from Codeup database and writes data to a csv file if cached == False or if cached == True reads in titanic df from a csv file, returns df.
    '''
    if cached == False or os.path.isfile('mall_customers.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_mall_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('mall_customers.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('mall_customers.csv', index_col=0)
        
    return df

##########################################################################################

# Wrangle Zillow Data

##########################################################################################

def wrangle_zillow(cached=False):
    '''
    This functions creates a dataframe from the zillow dataset and returns a cleaned and imputed version of the dataframe.
    ''' 
    if cached or os.path.isfile('wrangle_zillow.csv') == False:
        
        # Takes in data from Codeup SQL server
        df = get_zillow_data(cached=True)
        
        # Remove rows with nulls in lat and long
        df = df[df.latitude.notnull()]
        df = df[df.longitude.notnull()]
        
        # Isolated Single Unit Properties
        df.propertylandusetypeid = df.propertylandusetypeid.astype(int)
        x = 'propertylandusetypeid'
        df = df[df[x] != 246]
        df = df[df[x] != 247]
        df = df[df[x] != 248]
        df = df[df[x] != 269]
        
          # If "fireplaceflag" is "True" and "fireplacecnt" is "NaN", we will set "fireplacecnt" equal to the median value of "1".
        df.loc[(df['fireplaceflag'] == True) & (df['fireplacecnt'].isnull()), ['fireplacecnt']] = 1
        # If 'fireplacecnt' is "NaN", replace with "0"
        df.fireplacecnt.fillna(0,inplace = True)
        # If "fireplacecnt" is 1 or larger "fireplaceflag" is "NaN", we will set "fireplaceflag" to "True".
        df.loc[(df['fireplacecnt'] >= 1.0) & (df['fireplaceflag'].isnull()), ['fireplaceflag']] = True
        df.fireplaceflag.fillna(0,inplace = True)
        # Convert "True" to 1
        df.fireplaceflag.replace(to_replace = True, value = 1,inplace = True)
        
        # NULLs in poolcnt = No pools
        df.poolcnt.fillna(0, inplace=True)
        
        # Removing column s and rows outside specifies threshhold
        df = handle_missing_values(df, 0.6, 0.75)
        
        # Fill unitcnt NULLs with the mode (1.0) and delete the rest.
        df.unitcnt.fillna(1, inplace=True)
        df = df[df.unitcnt == 1.0]
        
        # Dropped null from the property value features
        df = df[df.taxvaluedollarcnt.notnull()]
        df = df[df.structuretaxvaluedollarcnt.notnull()]
        df = df[df.taxamount.notnull()]
        df = df[df.calculatedfinishedsquarefeet.notnull()]
        
        # Fill in beds and baths nulls with median because the range of values is small
        df['bedroomcnt'].replace(to_replace=0, value=df.bedroomcnt.median(), inplace=True)
        df['bathroomcnt'].replace(to_replace=0, value=df.bathroomcnt.median(), inplace=True)
    
        # Convert to int and replace NULLs with mode for 'heatingorsystemdesc','heatingorsystemtypeid
        df['heatingorsystemtypeid'].fillna(2, inplace=True)
        df['heatingorsystemdesc'].fillna('Central', inplace=True)
        df.heatingorsystemtypeid = df.heatingorsystemtypeid.astype(int)
        
         # Convert latitude and longitude to positonal data points using lambda funtion (i.e. putting a decimal in the correct place)
        df['latitude'] = df['latitude'].apply(lambda x: x / 10 ** (len((str(x))) - 2))
        df['longitude'] = df['longitude'].apply(lambda x: x / 10 ** (len((str(x))) - 4))
        
        # Impute kNN, for lotsizesquarefeet and regionidcity using 'latitude', 'longitude', 'fips', 'calculatedfinishedsquarefeet' and 'taxvaluedollarcnt' with 5 neighbors
        features = ['lotsizesquarefeet', 'latitude', 'longitude', 'fips', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt']
        df = impute_knn(df, features, 5)
        df.lotsizesquarefeet = df.lotsizesquarefeet.astype(int)
        
        features = ['regionidcity', 'fips', 'latitude', 'longitude']
        df = impute_knn(df, features, 2)
        df.regionidcity = df.regionidcity.astype(int)
        
        # changed fips to an int
        df.fips = df.fips.astype(int)
        
        # Dummy FiPs
        dummy_df =  pd.get_dummies(df['fips'])
        dummy_df.columns = ['la_cnty', 'orange_cnty', 'ventura_cnty']
        df = pd.concat([df, dummy_df], axis=1)    # clean up column names
        
        # list of columns not being used
        dropcols = ['id', 'buildingqualitytypeid', 'finishedsquarefeet12', 'calculatedbathnbr', 'fullbathcnt', 'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc', 'censustractandblock', 'rawcensustractandblock', 'regionidcounty', 'regionidzip', 'roomcnt', 'unitcnt', 'assessmentyear', 'transactiondate']
        df = df.drop(columns=dropcols)
    
        # create categorical log error column into 5 sections
        df['log_error_class'] = pd.qcut(df.logerror, q=4, labels=['l1', 'l2', 'l3', 'l4'])
        
        # rename columns
        df.columns = ['heating_system_type_id', 'parcelid', 'bathrooms', 'bedrooms', 'prop_sqft', 'fips', 'fireplace_cnt', 'latitude', 'longitude', 'lot_sqft', 'pool_cnt', 'region_id_city', 'year_built', 'fireplace_flag', 'struct_tax_value', 'tax_value', 'land_tax_value', 'tax_amount', 'log_error', 'heating_system_desc', 'la_cnty', 'orange_cnty', 'ventura_cnty', 'log_error_class']
        
        # Set index
        df.set_index('parcelid', inplace=True)
        
        # drop the last of the null values
        df = df.dropna()
        
        
        # Outliers
    
#         df = remove_outliers(df, 'tax_value', 3)
#         df = remove_outliers(df, 'lot_sqft', 3)
#         df = remove_outliers(df, 'log_error', 1.5)
        
        df.to_csv('wrangled_zillow.csv')
    
    else: 
        
        # pull cached data
        df = pd.read_csv('wrangled_zillow.csv', index_col=0)
        

    return df




##########################################################################################

###### Wrangle Mall

##########################################################################################

# Wrangle Mall_customers

def wrangle_mall():
    df = get_mall_data(cached=True)
    df = ex.add_upper_outlier_columns(df, 1.5)
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)
    quant_vars = ['annual_income','spending_score']
    train, validate, test = train_validate_test_split(df, 'annual_income', 42)
    train, validate, test = scale_my_data(train, validate, test, quant_vars)
    df = df.drop(columns=['customer_id_outliers'])
    return train, validate, test


##########################################################################################

# Zero's and NULLs

##########################################################################################



#----------------------------------------------------------------------------------------#
###### Identifying Zeros and Nulls in columns and rows

def missing_zero_values_table(df):
    '''
    This function tales in a dataframe and counts number of Zero values and NULL values. Returns a Table with counts and percentages of each value type.
    '''
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'NULL Values', 2 : '% of Total NULL Values'})
    mz_table['Total Zero\'s plus NULL Values'] = mz_table['Zero Values'] + mz_table['NULL Values']
    mz_table['% Total Zero\'s plus NULL Values'] = 100 * mz_table['Total Zero\'s plus NULL Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] >= 0].sort_values(
    '% of Total NULL Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
        "There are " + str((mz_table['NULL Values'] != 0).sum()) +
          " columns that have NULL values.")
    #       mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
    return mz_table



def missing_columns(df):
    '''
    This function takes a dataframe, counts the number of null values in each row, and converts the information into another dataframe. Adds percent of total columns.
    '''
    missing_cols_df = pd.Series(data=df.isnull().sum(axis = 1).value_counts().sort_index(ascending=False))
    missing_cols_df = pd.DataFrame(missing_cols_df)
    missing_cols_df = missing_cols_df.reset_index()
    missing_cols_df.columns = ['total_missing_cols','num_rows']
    missing_cols_df['percent_cols_missing'] = round(100 * missing_cols_df.total_missing_cols / df.shape[1], 2)
    missing_cols_df['percent_rows_affected'] = round(100 * missing_cols_df.num_rows / df.shape[0], 2)
    
    return missing_cols_df


#----------------------------------------------------------------------------------------#
###### Do things to the above zeros and nulls ^^

def handle_missing_values(df, prop_to_drop_col, prop_to_drop_row):
    '''
    This function takes in a dataframe, 
    a number between 0 and 1 that represents the proportion, for each column, of rows with non-missing values required to keep the column, 
    a another number between 0 and 1 that represents the proportion, for each row, of columns/variables with non-missing values required to keep the row, and returns the dataframe with the columns and rows dropped as indicated.
    '''
    # drop cols > thresh, axis = 1 == cols
    df = df.dropna(axis=1, thresh = prop_to_drop_col * df.shape[0])
    # drop rows > thresh, axis = 0 == rows
    df = df.dropna(axis=0, thresh = prop_to_drop_row * df.shape[1])
    return df



# def impute_mode(df, col, strategy):
#     '''
#     impute mode for column as str
#     '''
#     train, validate, test = train_validate_test_split(df, seed=42)
#     imputer = SimpleImputer(strategy=strategy)
#     train[[col]] = imputer.fit_transform(train[[col]])
#     validate[[col]] = imputer.transform(validate[[col]])
#     test[[col]] = imputer.transform(test[[col]])
#     return train, validate, test

def impute_mode(df, cols):
    ''' 
    Imputes column mode for all missing data
    '''
    for col in cols:
        df = df.fillna(df[col].value_counts().index[0])
        return df


def impute_knn(df, list_of_features, knn):
    '''
    This function performs a kNN impute on a single column and returns an imputed df.
    Parameters: df: dataframee
    list_of_features: a List of features, place the feature intended for impute first, then supporting features after.
    knn: an integer, indicates number of neighbors to find prior to selecting imputed value.
    '''
    knn_cols_df = df[list_of_features]
    imputer = KNNImputer(n_neighbors=knn)
    imputed = imputer.fit_transform(knn_cols_df)
    imputed = pd.DataFrame(imputed, index = df.index)
    df[list_of_features[0]] = imputed[[0]]
    return df


#----------------------------------------------------------------------------------------#
###### Removing outliers


def remove_outliers(df, col, multiplier):
    '''
    The function takes in a dataframe, column as str, and an iqr multiplier as a float. Returns dataframe with outliers removed.
    '''
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3 - q1
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    df = df[df[col] > lower_bound]
    df = df[df[col] < upper_bound]
    return df


##########################################################################################

# Split Data

##########################################################################################



####### PICK ONE OF THE METHODS OF SPLITTING DATA BELOW, NOT BOTH



# 1.)
#----------------------------------------------------------------------------------------#
def train_validate_test_split(df, target, seed=42):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            # stratify=df[target]
                                           )
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       # stratify=train_validate[target]
                                      )
    return train, validate, test



def scale_my_data(train, validate, test, quant_vars):
    scaler = MinMaxScaler()
    scaler.fit(train[quant_vars])
    
    X_train_scaled = scaler.transform(train[quant_vars])
    X_validate_scaled = scaler.transform(validate[quant_vars])
    X_test_scaled = scaler.transform(test[quant_vars])

    train[quant_vars] = X_train_scaled
    validate[quant_vars] = X_validate_scaled
    test[quant_vars] = X_test_scaled
    return train, validate, test
#----------------------------------------------------------------------------------------#




# 2.)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test



def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")
        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols



def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols



def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).


    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])
    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])
    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#