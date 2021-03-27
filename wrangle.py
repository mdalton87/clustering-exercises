import pandas as pd
import numpy as np
import os
import explore as ex

# acquire
from env import host, user, password
from pydataset import data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, RFE, SelectKBest
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler 


# Aquire Data

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'    
       
    # Zillow Database        
        
def new_zillow_data():
    '''
    This function reads in the teclo_churn data from the Codeup db
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


# Prepare Data


def wrangle_zillow():
    '''
This functions creates a dataframe from the zillow dataset.           
    '''
    df = get_zillow_data(cached=True)
    

    
    return df


def tax_rate_distribution():
    '''
This function creates the dataframe used to calculate the tax distribution rate per county. It takes in the cached zillow dataset, sets the parcelid as the index, makes the features list in order to limit the dataframe, renames the columns for clarity, drops null values, creates the tax_rate feature, and removed outliers from tax_rate and tx_value.
    '''
    df = get_zillow_data(cached=True)
    df.set_index('parcelid', inplace=True)
    features = ['fips', 'taxvaluedollarcnt', 'taxamount']
    df = df[features]
    
    df.columns = ['fips', 'tax_value', 'tax_amount']
    
    df = df.dropna()
    
    df['tax_rate'] = (df.tax_amount / df.tax_value)
    
    df = remove_outliers(df, 'tax_rate', 2.5)
    df = remove_outliers(df, 'tax_value', 2.5)
    
    return df


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



# Split Data

def impute_mode(df, col, strategy):
    '''
    impute mode for column as str
    '''
    train, validate, test = train_validate_test_split(df, seed=123)
    imputer = SimpleImputer(strategy=strategy)
    train[[col]] = imputer.fit_transform(train[[col]])
    validate[[col]] = imputer.transform(validate[[col]])
    test[[col]] = imputer.transform(test[[col]])
    return train, validate, test


def train_validate_test_split(df, target, seed):
    
    # Train, Validate, and test
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed)
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed)
    
    # Split
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


def prep_zillow_data():
    df = wrangle_zillow()
    train_validate, test = train_test_split(df, test_size=.2, random_state=42)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=42)
    train, validate, test = impute_mode()
    return train, validate, test




# Feature engineering

def select_kbest(x, y, k):
    '''
    This function takes in a dataframe, a target, and a number that is <= total number of features. The dataframe is split and scaled, the features are separated into objects and numberical columns, Finally the Select KBest test is run and returned.
    Parameters:
        x = dataframe
        y = target,
        k = # features to return
    '''
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(x, y)
    object_cols = get_object_cols(x)
    numeric_cols = get_numeric_X_cols(X_train, object_cols)
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    f_selector = SelectKBest(f_regression, k)
    f_selector.fit(X_train_scaled, y_train)
    feature_mask = f_selector.get_support()
    f_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    return f_feature


def rfe(x, y, k):
    '''
    This function takes in a dataframe, a target, and a number that is <= total number of features. The dataframe is split and scaled, the features are separated into objects and numberical columns, Finally the RFE test is run and returned.
    Parameters:
    x = dataframe
    y = target,
    k = # features to return
    '''
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(x, y)
    object_cols = get_object_cols(x)
    numeric_cols = get_numeric_X_cols(X_train, object_cols)
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    lm = LinearRegression()
    rfe = RFE(lm, k)
    rfe.fit(X_train_scaled,y_train)
    feature_mask = rfe.support_
    rfe_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    return rfe_feature


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



