import datawig

df = datawig.utils.generate_df_numeric( num_samples=200,
                                        data_column_name='x',
                                        label_column_name='y')
df_train, df_test = datawig.utils.random_split(df)

#Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns=['x'], # column(s) containing information about the column we want to impute
    output_column='y', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer.fit(train_df=df_train, num_epochs=50)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)

ololo = 5