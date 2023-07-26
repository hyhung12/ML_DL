# ML_DL
1. Import library
2. Read csv
```
drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/heart_2020_cleaned.csv'
# file_path = 'heart_2020_cleaned.csv'
df = pd.read_csv(file_path)
df.head()
```
3. Inspect data
```
df[].value_counts, df[].nunique()
```
