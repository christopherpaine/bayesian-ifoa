import os
import pandas as pd  # Ensure pandas is imported

with pd.ExcelWriter("output.xlsx", engine="openpyxl") as writer:

    df1.to_excel(writer, sheet_name="df1", index=False)
    df2.to_excel(writer, sheet_name="df2", index=False)
    df.to_excel(writer, sheet_name="df", index=False)
    df_numeric.to_excel(writer, sheet_name="df_numeric", index=False)
    df_converted.to_excel(writer, sheet_name="df_converted", index=False)
    mask_problem.to_excel(writer, sheet_name="mask_problem", index=False)
    filtered.to_excel(writer, sheet_name="filtered", index=False)
    col_names.to_excel(writer, sheet_name="col_names", index=False)
    col_names2.to_excel(writer, sheet_name="col_names2", index=False)

# Open the Excel file with LibreOffice Calc in the background
os.system("libreoffice --calc output.xlsx & disown")

