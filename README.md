# ML_DL
#### 1. Import library
#### 2. Read csv
```
drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/heart_2020_cleaned.csv'
# file_path = 'heart_2020_cleaned.csv'
df = pd.read_csv(file_path)
df.head()
```
#### 3. Inspect data
```
df[].value_counts(), df[].nunique()
```
#### 4. Split Dataset for Training and Testing
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
#### 5. Target Variable Visualization
- View number of instances & percentages
```
print(f"Number of No of test set: {len(y_test_df[y_test_df == 'No'])}")
print(f"Percentange of No of test set: {(len(y_test_df[y_test_df == 'No']) / len(y_test_df)):.3f}")

print(f"Number of Yes of test set: {len(y_test_df[y_test_df == 'Yes'])}")
print(f"Percentange of Yes of test set: {(len(y_test_df[y_test_df == 'Yes']) / len(y_test_df)):.3f}")
```
#### 6. Encode Categorical Features
```
n = list(range(2,len(df.columns)))
n = [2,3,4] + list(range(7,len(df.columns)))
n = [2,3,4] + [num for num in range(7,len(df.columns)) if num != 14]

ct = ColumnTransformer([('onehot', OneHotEncoder(), n)], remainder = 'passthrough', sparse_threshold = 0)
enc_train_arr = ct.fit_transform(X_train_df)
enc_test_arr = ct.fit_transform(X_test_df)
print(f"Encoded train set: {enc_train_arr.shape}")
print(f"Encoded test set: {enc_test_arr.shape}")
```
#### 7. Normalize Data
```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
removed_idx = -5
X_train = sc.fit_transform(np.delete(enc_train_arr, removed_idx, axis=1))
X_test = sc.transform(np.delete(enc_test_arr, removed_idx, axis=1))
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
```
#### 8. Train & Predict 
```
act_func = 'relu'
# act_func = 'sigmoid'
model = Sequential()
model.add(tf.keras.Input(shape=X_train.shape[1]))
# model.add(Dense(units = 128, activation = act_func))
# model.add(Dense(units = 64, activation = act_func))
# model.add(Dense(units = 32, activation = act_func))
# model.add(Dense(units = 16, activation = act_func))
# model.add(Dense(units = 1, activation = act_func))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics = ['accuracy']
)

# model.fit(X_train, y_train, epochs=10, batch_size = 32, verbose = 2)
# model.fit(X_val_data, y_val_data, epochs=10, batch_size = 32, verbose = 2)
model.fit(X_train, y_train, epochs = 12, batch_size = 512, verbose = 2, validation_data = (X_test, y_test))

from sklearn.metrics import accuracy_score
y_pred =( model.predict(X_test, verbose = 2) >= 0.5).astype(int)
accuracy_score(y_test, y_pred)
```
#### 9. Confusion Matrix (To debug if dataset is uneven, collect class 0 or class 1 more)
```
from sklearn.metrics import confusion_matrix
y_true = y_test
cm = confusion_matrix(y_true, y_pred)
classes = ['No', 'Yes']
# Step 4: Plot the confusion matrix with counts
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```
#### 10. Debug using Confusion Matrix
```
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

false_pred = [index for index, (true_label, predicted_label) in enumerate(zip(y_true, y_pred))
 if true_label == 1 and predicted_label == 0]

print("Indices of False predicted instances:", false_pred)
print(len(false_pred))
```







<- Categorical & Numerical Features>
