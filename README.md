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
4. Split Dataset for Training and Testing
```
from sklearn.model_selection import train_test_split
heart_map = {'Yes':1, 'No':0}

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(df, df['HeartDisease'], test_size = 0.2, random_state = 1234)
y_train = np.vectorize(heart_map.get)(y_train_df).reshape(-1, 1) # Convert dataframe into numpy
y_test = np.vectorize(heart_map.get)(y_test_df).reshape(-1, 1)

print(f"X_train: {X_train_df.shape}")
print(f"X_test: {X_test_df.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
X_train_df.head()
```
5. Target Variable Visualization
- View number of instances & percentages
```
print(f"Number of No of test set: {len(y_test_df[y_test_df == 'No'])}")
print(f"Percentange of No of test set: {(len(y_test_df[y_test_df == 'No']) / len(y_test_df)):.3f}")

print(f"Number of Yes of test set: {len(y_test_df[y_test_df == 'Yes'])}")
print(f"Percentange of Yes of test set: {(len(y_test_df[y_test_df == 'Yes']) / len(y_test_df)):.3f}")
```



<- Categorical & Numerical Features>
