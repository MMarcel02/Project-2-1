import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
from pandas import DataFrame

load_dotenv()

URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_KEY')

def get_data(URL:str, KEY:str, external:bool=False) -> DataFrame:

    supabase = create_client(URL, KEY)

    tables = ["our_run_data", "external_run_data"]

    response = (
        supabase.table(tables[int(external)])
        .select("*")
        .execute()
    )

    return pd.DataFrame(response.data)

print(get_data(URL,KEY))





