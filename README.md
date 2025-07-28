
#  Iris Project

A simple Iris classifier that uses point-based feature voting and a custom optimizer to predict flower species.

---

##  Overview

This project implements a manual learning algorithm for the Iris dataset.  
Each feature (sepal length, sepal width, petal length, petal width) contributes to the final decision by voting for the nearest class label (`a`, `b`, or `c`) using distance scoring.  
Instead of using traditional machine learning libraries, the classifier uses a custom parameter tuning process and a simple scoring system.

---

##  Project Structure

```

Iris-project/
├── data/
│   ├── iris\_train.csv        # Training data with features + target
│   └── iris\_test.csv         # Test data for prediction
├── src/
│   └── main.py               # Core script: trains, predicts, prints accuracy
├── README.md                 # Project overview (this file)

````

---

##  How to Run

1. Clone this repository:
```bash
git clone https://github.com/Sunnygking/Iris-project.git
cd Iris-project
````

2. Run the classifier:

```bash
python src/main.py
```

3. You’ll see:

* Training accuracy with detailed evaluation
* Predictions on the test set
* Labels (`a`, `b`, `c`) mapped to numeric form (0, 1, 2)

---

##  Example Output

```
row 0: 'a' vs 'a' → right | points: {'a': 5, 'b': 2, 'c': 1}
...
Total correct: 47/50 | Accuracy: 94.00%
Total true correct: 43/50 | Accuracy: 86.00% (the either become false)

Prediction Results:
  sepal length (cm)  ...  predicted  predicted_numeric
0               6.1  ...          b                  1
1               5.7  ...          a                  0
...
```

---

##  Dependencies

This project only requires:

* `pandas`
* `numpy`

Install with:

```bash
pip install pandas numpy
```

---

##  Author

**Sunnygking**
GitHub: [@Sunnyngking](https://github.com/Sunnyngking)

---

##  License

This project is for educational and demonstration purposes. Feel free to explore, use, and improve it.

```
