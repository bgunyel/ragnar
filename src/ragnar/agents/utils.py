import copy
import datetime
from typing import Any

from supabase import Client

from .enums import ColumnsBase


def insert_to_db(db_client: Client, input_dict: dict[str, Any], table_name: str):
    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

    row_dict = copy.deepcopy(input_dict)
    row_dict[ColumnsBase.CREATED_AT] = str(time_now)
    row_dict[ColumnsBase.UPDATED_AT] = str(time_now)
    # These two will be inputs after the system supports multiple users
    row_dict[ColumnsBase.CREATED_BY_ID] = 1
    row_dict[ColumnsBase.UPDATED_BY_ID] = 1

    response = (
        db_client.table(table_name=table_name)
        .insert(row_dict)
        .execute()
    )
    idx = response.data[0]['id']
    return idx