import pandas as pd
import os

EXCEL_FILE = 'Attendance_Log.xlsx'

def export_to_excel(daily_records, date_str):
    """
    daily_records: list of dicts from get_daily_attendance
    date_str: string representation of the date (YYYY-MM-DD format usually)
    """
    if not daily_records:
        return False
    
    df = pd.DataFrame(daily_records)
    # Ensure all required columns are present
    expected_cols = ['student_id', 'name', 'department', 'date', 'period', 'time', 'status', 'confidence']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ''
    
    # Reorder/select columns
    df = df[expected_cols]
    
    # Sheet name can be the date, max 31 chars for Excel
    sheet_name = str(date_str)[:31]
    
    # Append to existing excel or create new
    if os.path.exists(EXCEL_FILE):
        try:
            with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                # Check if sheet exists to decide whether to write header
                if sheet_name in writer.book.sheetnames:
                    startrow = writer.book[sheet_name].max_row
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=startrow)
                else:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        except Exception as e:
            print(f"Error appending to Excel: {e}")
            return False
    else:
        try:
            with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        except Exception as e:
            print(f"Error creating Excel: {e}")
            return False
            
    return True
