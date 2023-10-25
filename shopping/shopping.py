import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        data = list(reader)
    
    evidence = []
    labels = []
    for row in data:
        evidence_row = [
            int(row[0]), float(row[1]), int(row[2]), float(row[3]),
            int(row[4]), float(row[5]), float(row[6]), float(row[7]),
            float(row[8]), float(row[9]), month_to_int(row[10]), int(row[11]),
            int(row[12]), int(row[13]), int(row[14]), visitor_type_to_int(row[15]),
            weekend_to_int(row[16])
        ]
        evidence.append(evidence_row)
        labels.append(1 if row[17] == "TRUE" else 0)
    
    return (evidence, labels)

def month_to_int(month):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return months.index(month)

def visitor_type_to_int(visitor_type):
    return 1 if visitor_type == "Returning_Visitor" else 0

def weekend_to_int(weekend):
    return 1 if weekend == "TRUE" else 0

def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    true_positive = sum(1 for label, prediction in zip(labels, predictions) if label == prediction == 1)
    false_negative = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 0)
    true_negative = sum(1 for label, prediction in zip(labels, predictions) if label == prediction == 0)
    false_positive = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 1)
    
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    
    return (sensitivity, specificity)



if __name__ == "__main__":
    main()
