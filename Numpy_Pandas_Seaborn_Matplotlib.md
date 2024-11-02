Certainly! Here’s a compact revision guide for each library you mentioned: NumPy, Pandas, Seaborn, and Matplotlib. These notes cover key concepts, functions, and typical uses, designed to reinforce knowledge ahead of an interview.

---

### NumPy (Numerical Python)
- **Purpose**: Efficient handling of large arrays, matrices, and mathematical functions.
- **Core Concepts**:
  - **Array Creation**:
    - `np.array([1, 2, 3])`: Creates a 1D array.
    - `np.zeros((3, 3))`: 3x3 array of zeros.
    - `np.ones((2, 4))`: 2x4 array of ones.
    - `np.arange(0, 10, 2)`: Array with steps; e.g., [0, 2, 4, 6, 8].
    - `np.linspace(0, 1, 5)`: Evenly spaced values from 0 to 1.
  - **Array Manipulation**:
    - Reshape: `array.reshape((rows, cols))`.
    - Flatten: `array.flatten()`.
    - Concatenate: `np.concatenate([array1, array2], axis=0 or 1)`.
  - **Indexing and Slicing**:
    - Standard slicing: `array[1:5]`.
    - Boolean indexing: `array[array > 5]`.
  - **Mathematical Operations**:
    - Basic: `np.sum(array)`, `np.mean(array)`, `np.std(array)`.
    - Element-wise operations: `array1 + array2`, `np.multiply(array1, array2)`.
  - **Random**:
    - `np.random.rand(2, 3)`: Random numbers in [0, 1].
    - `np.random.randint(0, 10, size=(3, 3))`: Integers in a range.

---

### Pandas
- **Purpose**: Data manipulation and analysis using DataFrames and Series.
- **Core Concepts**:
  - **Data Structures**:
    - Series: `pd.Series([1, 2, 3])`.
    - DataFrame: `pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})`.
  - **Data Import/Export**:
    - `pd.read_csv('file.csv')`: Load a CSV file.
    - `df.to_csv('file.csv')`: Save a DataFrame to CSV.
  - **Indexing and Selection**:
    - `.loc[]`: Label-based indexing (e.g., `df.loc[0]`).
    - `.iloc[]`: Position-based indexing (e.g., `df.iloc[0]`).
  - **Data Cleaning**:
    - Handling Missing Values: `df.dropna()`, `df.fillna(value)`.
    - Duplicates: `df.drop_duplicates()`.
  - **Operations**:
    - Sorting: `df.sort_values(by='column')`.
    - Grouping: `df.groupby('column').mean()`.
    - Aggregation: `df.agg({'col1': 'sum', 'col2': 'mean'})`.
  - **Merging and Joining**:
    - `pd.merge(df1, df2, on='key')`.
    - `df.join(df2)` (by index).
  - **Apply and Lambda**:
    - Apply functions to rows/columns: `df['col'].apply(lambda x: x*2)`.

---

### Seaborn
- **Purpose**: Statistical data visualization, built on Matplotlib.
- **Core Concepts**:
  - **Basic Plots**:
    - `sns.scatterplot(data=df, x='col1', y='col2')`: Scatter plot.
    - `sns.lineplot(data=df, x='col1', y='col2')`: Line plot.
    - `sns.barplot(data=df, x='col1', y='col2')`: Bar plot.
  - **Distribution Plots**:
    - `sns.histplot(df['col'])`: Histogram.
    - `sns.kdeplot(df['col'])`: Kernel density plot.
    - `sns.boxplot(data=df, x='col1', y='col2')`: Box plot.
    - `sns.violinplot(data=df, x='col1', y='col2')`: Violin plot.
  - **Matrix Plots**:
    - `sns.heatmap(data)`: Heatmap for correlation matrices.
  - **Categorical Plots**:
    - `sns.countplot(data=df, x='col')`: Counts of unique values.
    - `sns.catplot(data=df, x='col1', y='col2', kind='swarm')`: Swarm plot for categorical data.
  - **Customizations**:
    - **Color Palettes**: `sns.color_palette('pastel')`.
    - **Styling**: `sns.set_style('darkgrid')`.

---

### Matplotlib
- **Purpose**: Fundamental plotting library for data visualization.
- **Core Concepts**:
  - **Basic Plotting**:
    - `plt.plot(x, y)`: Line plot.
    - `plt.scatter(x, y)`: Scatter plot.
    - `plt.bar(x, height)`: Bar plot.
  - **Subplots**:
    - `plt.subplot(1, 2, 1)`: Grid layout, e.g., 1 row, 2 columns.
    - `fig, ax = plt.subplots(2, 2)`: Multiple subplots with axes object.
  - **Titles and Labels**:
    - `plt.title("Title")`, `plt.xlabel("X-axis label")`, `plt.ylabel("Y-axis label")`.
  - **Legends and Annotations**:
    - `plt.legend()`: Add legend.
    - `plt.annotate("text", xy=(x, y))`: Annotation on plot.
  - **Customization**:
    - Colors: `plt.plot(x, y, color='red')`.
    - Line styles: `plt.plot(x, y, linestyle='--')`.
  - **Save Figure**:
    - `plt.savefig("filename.png")`: Save figure to a file.

---

These notes should help you quickly review the fundamentals of each library. Practice implementing a few small scripts or functions using these features to reinforce your understanding. Good luck with your interview preparation!
