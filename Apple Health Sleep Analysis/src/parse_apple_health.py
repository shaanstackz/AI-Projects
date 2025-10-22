import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime

def parse_sleep_records(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    sleep_data = []
    for record in root.findall('Record'):
        if record.get('type') == 'HKCategoryTypeIdentifierSleepAnalysis':
            start = record.get('startDate')
            end = record.get('endDate')
            value = record.get('value')
            sleep_data.append({'startDate': start, 'endDate': end, 'value': value})
    
    df = pd.DataFrame(sleep_data)
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['endDate'] = pd.to_datetime(df['endDate'])
    df['total_sleep_hours'] = (df['endDate'] - df['startDate']).dt.total_seconds() / 3600
    return df

def parse_health_metrics(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    metrics = []
    for record in root.findall('Record'):
        r_type = record.get('type')
        start = record.get('startDate')
        end = record.get('endDate')
        value = record.get('value')
        
        if r_type in ['HKQuantityTypeIdentifierStepCount',
                      'HKQuantityTypeIdentifierHeartRate',
                      'HKCategoryTypeIdentifierMindfulSession']:
            
            if r_type == 'HKQuantityTypeIdentifierStepCount':
                r_type_label = 'activity'
                value = float(value)
            elif r_type == 'HKQuantityTypeIdentifierHeartRate':
                r_type_label = 'heart_rate'
                value = float(value)
            elif r_type == 'HKCategoryTypeIdentifierMindfulSession':
                r_type_label = 'mindfulness'
                value = float((pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 60)  # minutes
            
            metrics.append({
                'startDate': start,
                'endDate': end,
                'type': r_type_label,
                'value': value
            })
    
    df = pd.DataFrame(metrics)
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['endDate'] = pd.to_datetime(df['endDate'])
    return df
