import copy
import datetime
from typing import Any

from supabase import Client

from .enums import ColumnsBase


def insert_entity_to_db(db_client: Client, input_dict: dict[str, Any], table_name: str):
    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

    row_dict = copy.deepcopy(input_dict)
    row_dict[ColumnsBase.UPDATED_AT] = str(time_now)
    row_dict[ColumnsBase.UPDATED_BY_ID] = 1 # This will be an input after the system supports multiple users
    row_dict[ColumnsBase.CREATED_AT] = str(time_now)
    row_dict[ColumnsBase.CREATED_BY_ID] = 1

    response = (
        db_client.table(table_name=table_name)
        .insert(row_dict)
        .execute()
    )
    idx = response.data[0]['id']
    return idx

def update_entity_in_db(db_client: Client, input_dict: dict[str, Any], table_name: str):
    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

    row_dict = copy.deepcopy(input_dict)
    row_dict[ColumnsBase.UPDATED_BY_ID] = 1  # This will be an input after the system supports multiple users
    row_dict[ColumnsBase.UPDATED_AT] = str(time_now)
    idx = row_dict.pop(ColumnsBase.ID)

    response = (
        db_client.table(table_name=table_name)
        .update(row_dict)
        .eq(ColumnsBase.ID, idx)
        .execute()
    )
    idx = response.data[0]['id']
    return idx

def fetch_entity_by_id(db_client: Client, table_name: str, entity_id: int) -> list[dict[str, Any]]:
    response = (
        db_client.table(table_name=table_name)
        .select("*")
        .eq(ColumnsBase.ID, entity_id)
        .execute()
    )
    return response.data

def fetch_entity_by_name(db_client: Client, entity_name: str, table_name: str) -> list[dict[str, Any]]:
    response = (
        db_client.table(table_name=table_name)
        .select("*")
        .eq(ColumnsBase.NAME, entity_name)
        .execute()
    )
    return response.data
