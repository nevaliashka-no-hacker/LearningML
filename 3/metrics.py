from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def main():
    print("METRICS\n")

    error_matrix = {"#": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                    "fact": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0], 
                    "forecast": [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    "program type": [],
                    "paper type": ['TP', 'TN', 'TP', 'FN', 'FP', 'TN', 'TP', 'TN', 'FN', 'TN']}
    for i in range(len(error_matrix["#"])):
        if error_matrix["fact"][i] and error_matrix["forecast"][i]:
            error_matrix["program type"].append("TP")
        elif error_matrix["fact"][i] and not error_matrix["forecast"][i]:
            error_matrix["program type"].append("FN")
        elif not error_matrix["fact"][i] and error_matrix["forecast"][i]:
            error_matrix["program type"].append("FP")
        else:
            error_matrix["program type"].append("TN")

    print(" #  | fact | forecast | program type | paper type")
    print("----|------|----------|--------------|-----------")
    for i in range(len(error_matrix["#"])):
        print(f" {error_matrix["#"][i]:2} | {error_matrix["fact"][i]:4} | {error_matrix["forecast"][i]:8} |  {error_matrix["program type"][i]:11} | {error_matrix["paper type"][i]}")
    print()

    TP = 32
    TN = 54
    FP = 6
    FN = 8
    y_pred = error_matrix["forecast"]
    y_true = error_matrix["fact"]

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Program accuracy: {accuracy}\nOn paper: 0.86")
    print()

    precision = precision_score(y_true, y_pred)
    print(f"Program precision: {precision}\nOn paper: 0.89")
    print()

    recall = recall_score(y_true, y_pred)
    print(f"Program recall: {recall}\nOn paper: 0.8")
    print()

    f1 = f1_score(y_true, y_pred)
    print(f"Program f1: {f1}\nOn paper: 0.84")
    print()

    data = {"#": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            "fact": [62, 65, 68, 70, 72, 75, 78, 80, 84, 88], 
            "forecast": [60, 66, 70, 69, 75, 73, 80, 79, 86, 85], 
            "error": []}
    for i in range(len(data["#"])):
        data["error"].append(data["fact"][i] - data["forecast"][i])

    print(" #  | fact | forecast | error")
    print("----|------|----------|------")
    for i in range(10):
        print(f" {data["#"][i]:2} | {data["fact"][i]:4} | {data["forecast"][i]:8} | {data["error"][i]:5}")
    print()
    

if __name__ == "__main__":
    main()