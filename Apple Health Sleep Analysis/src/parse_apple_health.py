import xml.etree.ElementTree as ET
import pandas as pd

def parse_sleep_records(xml_file: str) -> pd.DataFrame:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    records = []
    for record in root.findall('Record'):
        if record.get('type') == 'HKCategoryTypeIdentifierSleepAnalysis':
            records.append({
                'startDate': record.get('startDate'),
                'endDate': record.get('endDate'),
                'value': record.get('value')
            })
    df = pd.DataFrame(records)
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['endDate'] = pd.to_datetime(df['endDate'])
    df['duration_hours'] = (df['endDate'] - df['startDate']).dt.total_seconds() / 3600
    df['sleep_type'] = df['value'].apply(lambda x: x.split('.')[-1] if x else 'Unknown')
    return df
