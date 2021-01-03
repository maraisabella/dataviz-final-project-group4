# %%
# Dependencies
import numpy as np
import pandas as pd
import datetime as dt

# Python SQL toolkit and Object Relational Mapper
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func

# %%
# create engine
engine = create_engine("sqlite:///covid.sqlite")

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)


# %%
# Save references to the table
Covid = Base.classes.owid_covid_data

# %%
# Create our session (link) from Python to the DB
session = Session(engine)

# %%
# 1. Import the sqlalchemy extract function.
from sqlalchemy import extract

# 2. Write a query that filters the Measurement table to retrieve the 'USA' total_cases. 
#results = session.query(Owid.date, Owid.iso_code, Owid.total_cases).filter(Owid.iso_code == 'USA').all()
results = session.query(Owid.date, Owid.total_cases).filter(Owid.iso_code == 'USA').all()
results

# %%
#  3. Convert the 'USA' total_cases to a list.
usa_total_cases_list = [result for result in results]

# %%
usa_total_cases_list

# %%
# 4. Create a DataFrame from the list of 'USA' total_cases. 
df = pd.DataFrame(usa_total_cases_list, columns=['Date', 'Total USA Cases'])
df

# %%


# %%
