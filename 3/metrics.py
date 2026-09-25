from sklearn.ensemble import *
import matplotlib

def main():

    error_matrix = {"#": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                    "fact": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0], 
                    "forecast": [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    "type": []}
    for i in range(len(error_matrix["#"])):
        if error_matrix["fact"][i] and error_matrix["forecast"][i]:
            error_matrix["type"].append("TP")
        elif error_matrix["fact"][i] and not error_matrix["forecast"][i]:
            error_matrix["type"].append("FN")
        elif not error_matrix["fact"][i] and error_matrix["forecast"][i]:
            error_matrix["type"].append("FP")
        else:
            error_matrix["type"].append("TN")

    print(f" #  | fact | forecast | type")
    print("-" * 30)
    for i in range(len(error_matrix["#"])):
        print(f" {error_matrix["#"][i]:2} | {error_matrix["fact"][i]:4} | {error_matrix["forecast"][i]:8} |  {error_matrix["type"][i]}")

    report = classification_report(y_test, lr.predict(X_test), target_names=['Non-churned', 'Churned'])
    print(report)

if __name__ == "__main__":
    main()